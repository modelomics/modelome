from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.openaire import OpenAireGraphSourceAdapter

API = "https://zenodo.org/api/records"
CONCEPT_ID = "3516917"
RELEASE_ID = "20428976"
NOW = datetime(2026, 9, 2, 18, 30, tzinfo=UTC)


class CatalogClient:
    def __init__(self, resolver: Callable[[str, int], Any]) -> None:
        self.resolver = resolver
        self.calls: list[tuple[str, Mapping[str, str], Any]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        redirect_validator: Any = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(headers or {}), redirect_validator))
        payload = self.resolver(url, len(self.calls))
        if isinstance(payload, HttpResponse):
            return payload
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return HttpResponse(status=200, headers={}, body=body, url=url)


def _file(
    key: str,
    *,
    record_id: str = RELEASE_ID,
    size: int = 1_000,
    checksum: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    encoded = quote(key, safe="")
    return {
        "id": f"file-{key}",
        "key": key,
        "size": size,
        "checksum": checksum or f"md5:{len(key):032x}",
        "links": {
            "self": url or f"{API}/{record_id}/files/{encoded}/content",
        },
    }


def _release(
    *,
    record_id: str = RELEASE_ID,
    files: list[dict[str, Any]] | None = None,
    version: str = "11.1.1",
) -> dict[str, Any]:
    record_api = f"{API}/{record_id}"
    return {
        "id": int(record_id),
        "conceptrecid": CONCEPT_ID,
        "doi": f"10.5281/zenodo.{record_id}",
        "conceptdoi": "10.5281/zenodo.3516917",
        "created": "2026-06-08T09:10:11+00:00",
        "updated": "2026-06-08T10:11:12+00:00",
        "metadata": {
            "title": "OpenAIRE Graph Dataset",
            "version": version,
            "publication_date": "2026-06-08",
            "description": "The complete graph release.",
            "creators": [
                {
                    "name": "OpenAIRE ScholeXplorer Service",
                    "affiliation": "OpenAIRE",
                }
            ],
            "rights": [{"title": "Creative Commons Attribution 4.0"}],
            "communities": [{"id": "openaire"}, {"id": "eu"}],
            "related_identifiers": [
                {
                    "identifier": "https://graph.openaire.eu/",
                    "relation": "isSupplementTo",
                }
            ],
        },
        "links": {
            "self": record_api,
            "self_html": f"https://zenodo.org/records/{record_id}",
            "versions": f"{record_api}/versions",
        },
        "files": files
        or [
            _file("publication_2.tar", record_id=record_id),
            _file("product_Cites_1.tar", record_id=record_id),
            _file("dataset_1.tar", record_id=record_id),
            _file("otherresearchproduct.tar", record_id=record_id),
            _file("publication_1.tar", record_id=record_id),
            _file("software.tar", record_id=record_id),
        ],
    }


def _client(payload: Mapping[str, Any]) -> CatalogClient:
    return CatalogClient(lambda _url, _call: deepcopy(payload))


def _adapter(client: CatalogClient, **kwargs: Any) -> OpenAireGraphSourceAdapter:
    return OpenAireGraphSourceAdapter(
        client=client,
        clock=lambda: NOW,
        **kwargs,
    )


def _scan_all(
    source: OpenAireGraphSourceAdapter,
    state: Mapping[str, Any] | None = None,
) -> list[Any]:
    pages = []
    current = dict(state or {})
    for _ in range(100):
        page = source.fetch_page(current)
        pages.append(page)
        assert not page.issues
        current = dict(page.next_state)
        if page.complete:
            return pages
    raise AssertionError("OpenAIRE scan did not complete")


def test_full_release_enumerates_every_upstream_partition_without_filtering() -> None:
    payload = _release()
    client = _client(payload)
    source = _adapter(client, page_size=2)

    pages = _scan_all(source)

    assert len(pages) == 4
    release = pages[0].records[0]
    assert release.source_record_id == f"openaire-graph:release:{RELEASE_ID}"
    assert release.kind is ArtifactKind.CATALOG_RECORD
    assert release.canonical_url == f"https://zenodo.org/records/{RELEASE_ID}"
    assert release.identifiers == (
        Identifier("openaire:graph-release", RELEASE_ID),
        Identifier("zenodo", RELEASE_ID),
        Identifier("doi", f"10.5281/zenodo.{RELEASE_ID}"),
        Identifier("doi", "10.5281/zenodo.3516917"),
    )
    assert release.raw["metadata"] == payload["metadata"]
    assert release.raw["selection"] == {
        "scope": "complete_graph_release",
        "partition_filter": None,
        "scientific_filter": None,
        "research_product_type_resolution": "each_payload_object.type",
    }
    assert release.raw["snapshot_semantics"]["explicit_tombstone_stream"] is False
    assert release.raw["payload_contract"]["storage"] == "parquet"
    assert release.models == ()

    shards = [record for page in pages[1:] for record in page.records]
    assert [record.raw["file_key"] for record in shards] == [
        "dataset_1.tar",
        "otherresearchproduct.tar",
        "product_Cites_1.tar",
        "publication_1.tar",
        "publication_2.tar",
        "software.tar",
    ]
    assert [record.raw["entity_partition"] for record in shards] == [
        "dataset",
        "otherresearchproduct",
        "product_Cites",
        "publication",
        "publication",
        "software",
    ]
    assert all(record.kind is ArtifactKind.CATALOG_RECORD for record in shards)
    assert all(record.models == () for record in shards)
    assert all(link.crawl is False for record in shards for link in record.links)
    assert pages[-1].upstream_count == 7
    assert pages[-1].next_state["watermark"] == RELEASE_ID
    assert pages[-1].next_state["file_count"] == 6
    assert pages[-1].next_state["release_version"] == "11.1.1"
    assert pages[-1].next_state["completed_at"] == "2026-09-02T18:30:00Z"
    assert [call[0] for call in client.calls] == [
        f"{API}/{CONCEPT_ID}",
        f"{API}/{RELEASE_ID}",
        f"{API}/{RELEASE_ID}",
        f"{API}/{RELEASE_ID}",
    ]
    assert all(call[1] == {"Accept": "application/json"} for call in client.calls)
    assert all(call[2] is not None for call in client.calls)


def test_large_relation_manifest_pages_preserve_every_exact_shard_identity() -> None:
    files = [
        _file(f"product_Cites_{index}.tar")
        for index in range(1, 81)
    ] + [
        _file(f"product_IsRelatedTo_{index}.tar")
        for index in range(1, 26)
    ] + [
        _file(f"product_IsSourceOf_{index}.tar")
        for index in range(1, 14)
    ]
    pages = _scan_all(_adapter(_client(_release(files=files)), page_size=11))

    shards = [record for page in pages[1:] for record in page.records]
    assert len(shards) == len(files)
    assert {record.raw["file_key"] for record in shards} == {
        item["key"] for item in files
    }
    assert len({record.source_record_id for record in shards}) == len(files)
    assert {
        record.source_record_id
        for record in shards
        if record.raw["entity_partition"] == "product_Cites"
    } == {
        f"openaire-graph:file:{RELEASE_ID}:{quote(f'product_Cites_{index}.tar', safe='')}"
        for index in range(1, 81)
    }
    assert [len(page.records) for page in pages[1:]] == [11] * 10 + [8]
    assert len(pages) == 1 + (len(files) + 10) // 11
    assert pages[-1].complete is True
    assert pages[-1].upstream_count == len(files) + 1
    assert pages[-1].next_state["file_count"] == len(files)


def test_unknown_future_partition_is_preserved_from_upstream_filename_metadata() -> None:
    payload = _release(files=[_file("quantum_biomed_neural_objects_17.tar")])
    pages = _scan_all(_adapter(_client(payload)))

    record = pages[-1].records[0]
    assert record.raw["entity_partition"] == "quantum_biomed_neural_objects"
    assert record.raw["partition_index"] == 17
    assert record.raw["partition_classification"] == "upstream_filename_metadata"
    assert "model" not in record.raw
    assert "topic" not in record.raw
    assert "journal" not in record.raw


def test_file_control_preserves_integrity_and_lossless_loading_contract() -> None:
    key = "all_research_products_3.tar"
    payload = _release(
        files=[
            _file(
                key,
                size=10_800_000_000,
                checksum=f"sha256:{'ab' * 32}",
            )
        ]
    )

    pages = _scan_all(_adapter(_client(payload)))
    record = pages[-1].records[0]

    assert record.canonical_url == (f"{API}/{RELEASE_ID}/files/{key}/content")
    assert record.identifiers == (
        Identifier("openaire:graph-file", f"{RELEASE_ID}/{key}"),
        Identifier("sha256", "ab" * 32),
    )
    assert record.raw["size"] == 10_800_000_000
    assert record.raw["checksum"] == {"algorithm": "sha256", "value": "ab" * 32}
    assert (
        record.raw["payload_contract"][
            "preserve_all_pids_relations_urls_creators_hosts_dates_rights"
        ]
        is True
    )
    assert record.raw["snapshot_semantics"]["partial_release_must_not_be_committed"]


def test_same_immutable_release_is_a_validated_complete_noop() -> None:
    payload = _release()
    client = _client(payload)
    source = _adapter(client, page_size=100)
    pages = _scan_all(source)
    complete_state = pages[-1].next_state
    call_count = len(client.calls)

    page = source.fetch_page(complete_state)

    assert page.complete is True
    assert page.records == ()
    assert page.upstream_count == 0
    assert page.next_state["watermark"] == RELEASE_ID
    assert len(client.calls) == call_count + 1
    assert client.calls[-1][0] == f"{API}/{CONCEPT_ID}"


def test_new_release_uses_prior_completed_state_as_retry_boundary() -> None:
    prior_payload = _release(record_id="20000000")
    prior_source = _adapter(_client(prior_payload), page_size=100)
    prior_state = _scan_all(prior_source)[-1].next_state
    source = _adapter(_client(_release()), page_size=100)

    first = source.fetch_page(prior_state)

    assert first.complete is False
    assert first.records[0].raw["release_id"] == RELEASE_ID
    assert first.retry_state == prior_state
    assert first.next_state["watermark"] == "20000000"
    assert first.next_state["target_release"] == RELEASE_ID


def test_manifest_drift_during_resume_emits_no_controls_and_restarts_from_watermark() -> None:
    original = _release()
    changed = _release()
    changed["files"][0]["size"] += 1

    def resolve(url: str, call: int) -> Any:
        if url.endswith(f"/{CONCEPT_ID}") or call == 2:
            return deepcopy(original)
        return deepcopy(changed)

    client = CatalogClient(resolve)
    source = _adapter(client, page_size=1)
    release_page = source.fetch_page({})
    first_file_page = source.fetch_page(release_page.next_state)

    drift = source.fetch_page(first_file_page.next_state)

    assert drift.complete is False
    assert drift.records == ()
    assert drift.issues[0].stage == "source_manifest"
    assert "contents changed" in drift.issues[0].error
    assert drift.next_state == {
        "repository_signature": source.checkpoint_signature,
    }
    assert drift.retry_state == drift.next_state


def test_set_like_metadata_order_does_not_create_false_manifest_drift() -> None:
    concept_payload = _release()
    immutable_payload = _release()
    immutable_payload["metadata"]["communities"].reverse()
    client = CatalogClient(
        lambda url, _call: deepcopy(
            concept_payload if url.endswith(f"/{CONCEPT_ID}") else immutable_payload
        )
    )
    source = _adapter(client, page_size=100)

    pages = _scan_all(source)

    assert pages[-1].complete is True
    assert pages[-1].next_state["watermark"] == RELEASE_ID
    assert pages[0].records[0].raw["metadata"]["communities"] == [
        {"id": "openaire"},
        {"id": "eu"},
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["files"].append(deepcopy(payload["files"][0])),
            "duplicate release file key",
        ),
        (
            lambda payload: payload["files"][0].update({"checksum": "md5:abcd"}),
            "wrong digest length",
        ),
        (
            lambda payload: payload["files"][0]["links"].update(
                {"self": "https://evil.example/file.tar"}
            ),
            "changed origin",
        ),
        (
            lambda payload: payload["files"][0].update({"key": "../escape.tar"}),
            "safe .tar filename",
        ),
    ],
)
def test_malformed_manifest_fails_closed(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payload = _release()
    mutate(payload)

    page = _adapter(_client(payload)).fetch_page({})

    assert page.records == ()
    assert page.complete is False
    assert page.next_state == {}
    assert page.retry_state == {}
    assert message in page.issues[0].error


def test_manifest_response_and_file_limits_are_enforced_with_injected_clients() -> None:
    payload = _release(files=[_file("publication.tar", size=1_001)])
    oversized_body = json.dumps(payload).encode()
    response_client = CatalogClient(
        lambda _url, _call: HttpResponse(
            status=200,
            headers={},
            body=oversized_body,
            url=f"{API}/{CONCEPT_ID}",
        )
    )

    body_page = _adapter(
        response_client,
        max_manifest_bytes=len(oversized_body) - 1,
    ).fetch_page({})
    file_page = _adapter(
        _client(payload),
        max_file_bytes=1_000,
    ).fetch_page({})

    assert "catalog exceeded" in body_page.issues[0].error
    assert "size 1001 exceeds limit 1000" in file_page.issues[0].error


@pytest.mark.parametrize(
    "state_change",
    [
        {"file_cursor": 6},
        {"control_records_seen": 3},
        {"target_manifest_signature": "not-a-signature"},
    ],
)
def test_corrupt_pagination_checkpoint_is_rejected(state_change: Mapping[str, Any]) -> None:
    payload = _release()
    source = _adapter(_client(payload), page_size=1)
    state = dict(source.fetch_page({}).next_state)
    state.update(state_change)

    with pytest.raises(ValueError, match="checkpoint|manifest signature|file_cursor"):
        source.fetch_page(state)


def test_repository_identity_prevents_cross_configuration_resume() -> None:
    source = _adapter(_client(_release()))

    with pytest.raises(ValueError, match="another repository"):
        source.fetch_page({"repository_signature": "0" * 64})

    assert source.repository_identity == {
        "api_url": API,
        "concept_record_id": CONCEPT_ID,
        "checkpoint_signature": source.checkpoint_signature,
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://zenodo.org/api/records",
        "https://user:secret@zenodo.org/api/records",
        "https://127.0.0.1/api/records",
        "https://zenodo.org/api/records?token=secret",
        "https://zenodo.org/",
    ],
)
def test_catalog_url_must_be_a_public_credential_free_https_api_path(url: str) -> None:
    with pytest.raises(ValueError):
        OpenAireGraphSourceAdapter(url=url)


def test_catalog_rollback_and_mutated_completed_release_fail_closed() -> None:
    payload = _release()
    source = _adapter(_client(payload), page_size=100)
    complete_state = dict(_scan_all(source)[-1].next_state)

    rolled_back = _adapter(
        _client(_release(record_id="20000000")),
        page_size=100,
    ).fetch_page(complete_state)
    mutated_state = {**complete_state, "release_signature": "0" * 64}
    mutated = _adapter(_client(payload), page_size=100).fetch_page(mutated_state)

    assert "moved backward" in rolled_back.issues[0].error
    assert rolled_back.retry_state == complete_state
    assert "manifest changed upstream" in mutated.issues[0].error
    assert mutated.retry_state == mutated_state


def test_non_json_and_non_object_catalogs_are_retryable_manifest_failures() -> None:
    invalid_json = CatalogClient(lambda _url, _call: b"{")
    array_json = CatalogClient(lambda _url, _call: [])

    invalid = _adapter(invalid_json).fetch_page({})
    wrong_shape = _adapter(array_json).fetch_page({})

    assert "not valid JSON" in invalid.issues[0].error
    assert "must be a JSON object" in wrong_shape.issues[0].error


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            HttpResponse(
                status=503,
                headers={},
                body=b"{}",
                url=f"{API}/{CONCEPT_ID}",
            ),
            "HTTP 503",
        ),
        (
            HttpResponse(
                status=200,
                headers={"content-type": "application/json"},
                body=b"{}",
                url="https://evil.example/api/records/3516917",
            ),
            "changed origin",
        ),
        (
            HttpResponse(
                status=200,
                headers={"content-type": "text/html"},
                body=b"{}",
                url=f"{API}/{CONCEPT_ID}",
            ),
            "did not return JSON content",
        ),
    ],
)
def test_http_status_final_url_and_media_type_are_validated(
    response: HttpResponse,
    message: str,
) -> None:
    page = _adapter(CatalogClient(lambda _url, _call: response)).fetch_page({})

    assert page.records == ()
    assert message in page.issues[0].error
