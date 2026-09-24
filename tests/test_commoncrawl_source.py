from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.commoncrawl import CommonCrawlWetSourceAdapter

CATALOG_URL = "https://index.commoncrawl.org/collinfo.json"
DATA_URL = "https://data.commoncrawl.org/"
NOW = datetime(2026, 9, 2, 20, 0, tzinfo=UTC)


class MappingClient:
    def __init__(self, bodies: Mapping[str, bytes]) -> None:
        self.bodies = dict(bodies)
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(headers or {})))
        if url not in self.bodies:
            raise AssertionError(f"unexpected GET {url}")
        body = self.bodies[url]
        response_headers = {
            "content-length": str(len(body)),
            "etag": f'"{len(body):x}"',
        }
        return HttpResponse(
            status=200,
            headers=response_headers,
            body=body,
            url=url,
        )


def _collection(collection_id: str, offset: int) -> dict[str, str]:
    return {
        "id": collection_id,
        "name": f"Crawl {offset}",
        "timegate": f"https://index.commoncrawl.org/{collection_id}/",
        "cdx-api": f"https://index.commoncrawl.org/{collection_id}-index",
        "from": f"20{offset:02}-01-01T00:00:00",
        "to": f"20{offset:02}-01-31T23:59:59",
    }


def _path(collection_id: str, index: int) -> str:
    return (
        f"crawl-data/{collection_id}/segments/{index:05}/wet/"
        f"part-{index:05}.warc.wet.gz"
    )


def _manifest(*paths: str) -> bytes:
    return gzip.compress("".join(f"{path}\n" for path in paths).encode(), mtime=0)


def _bodies(
    collections: list[dict[str, str]],
    paths: Mapping[str, list[str]],
) -> dict[str, bytes]:
    result = {CATALOG_URL: json.dumps(collections).encode()}
    for collection in collections:
        collection_id = collection["id"]
        result[
            f"{DATA_URL}crawl-data/{collection_id}/wet.paths.gz"
        ] = _manifest(*paths[collection_id])
    return result


def _adapter(client: MappingClient, **kwargs: Any) -> CommonCrawlWetSourceAdapter:
    return CommonCrawlWetSourceAdapter(
        client=client,
        page_size=2,
        manifests_per_page=2,
        clock=lambda: NOW,
        **kwargs,
    )


def _finish(
    source: CommonCrawlWetSourceAdapter,
    state: Mapping[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    records = []
    current = dict(state or {})
    for _ in range(30):
        page = source.fetch_page(current)
        assert not page.issues
        records.extend(page.records)
        current = dict(page.next_state)
        if page.complete:
            return records, current
    raise AssertionError("Common Crawl control scan did not finish")


def test_multi_collection_scan_resumes_inside_manifests_and_emits_exact_shards() -> None:
    historical = _collection("CC-MAIN-2008-2009", 8)
    current = _collection("CC-MAIN-2026-34", 26)
    paths = {
        historical["id"]: [_path(historical["id"], index) for index in range(3)],
        current["id"]: [_path(current["id"], index) for index in range(2)],
    }
    client = MappingClient(_bodies([historical, current], paths))
    source = _adapter(client)

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)
    third = source.fetch_page(second.next_state)
    records = [*first.records, *second.records, *third.records]

    assert first.complete is False
    assert first.next_state["collection_index"] == 0
    assert first.next_state["manifest_offset"] == 2
    assert second.complete is False
    assert third.complete is True
    assert len(records) == 5
    assert len({record.source_record_id for record in records}) == 5
    assert [record.raw["collection_id"] for record in records] == [
        historical["id"],
        historical["id"],
        historical["id"],
        current["id"],
        current["id"],
    ]
    assert [record.raw["manifest_index"] for record in records] == [0, 1, 2, 0, 1]
    assert [record.raw["total_shards"] for record in records] == [3, 3, 3, 2, 2]
    assert third.upstream_count == 5
    assert third.next_state["collection_count"] == 2
    assert third.next_state["completed_at"] == "2026-09-02T20:00:00Z"

    record = records[0]
    assert record.kind is ArtifactKind.CATALOG_RECORD
    assert record.identifiers == (
        Identifier("commoncrawl:wet-shard", record.source_record_id.rsplit(":", 1)[1]),
    )
    assert record.canonical_url == f"{DATA_URL}{paths[historical['id']][0]}"
    assert record.raw["stable_object_url"] == record.canonical_url
    assert record.raw["record_type"] == "commoncrawl_wet_shard"
    assert record.raw["collection_from"] == historical["from"]
    assert record.raw["collection_to"] == historical["to"]
    assert len(record.raw["manifest_digest"]) == 64
    assert record.raw["manifest_digest_algorithm"] == "sha256"
    assert {link.relation for link in record.links} == {
        "bulk_payload",
        "control_manifest",
    }
    assert all(link.crawl is False for link in record.links)
    assert record.models == ()
    assert all(not call[0].endswith(".warc.wet.gz") for call in client.calls)


def test_complete_repeat_is_an_exact_noop_and_does_not_refetch_manifests() -> None:
    collection = _collection("CC-MAIN-2024-10", 24)
    paths = {collection["id"]: [_path(collection["id"], 0)]}
    client = MappingClient(_bodies([collection], paths))
    source = _adapter(client)
    records, state = _finish(source)
    calls = len(client.calls)

    repeat = source.fetch_page(state)

    assert len(records) == 1
    assert repeat.complete is True
    assert repeat.records == ()
    assert repeat.upstream_count == 0
    assert repeat.next_state == state
    assert len(client.calls) == calls + 1
    assert client.calls[-1][0] == CATALOG_URL


def test_catalog_refresh_after_completion_emits_only_a_new_collection() -> None:
    first_collection = _collection("CC-MAIN-2025-51", 25)
    first_paths = {first_collection["id"]: [_path(first_collection["id"], 0)]}
    client = MappingClient(_bodies([first_collection], first_paths))
    source = _adapter(client)
    _, state = _finish(source)

    new_collection = _collection("CC-MAIN-2026-04", 26)
    new_path = _path(new_collection["id"], 0)
    client.bodies.update(
        _bodies(
            [new_collection, first_collection],
            {
                new_collection["id"]: [new_path],
                first_collection["id"]: first_paths[first_collection["id"]],
            },
        )
    )
    old_manifest_url = (
        f"{DATA_URL}crawl-data/{first_collection['id']}/wet.paths.gz"
    )
    calls_before = sum(call[0] == old_manifest_url for call in client.calls)

    records, refreshed = _finish(source, state)

    assert [record.raw["collection_id"] for record in records] == [new_collection["id"]]
    assert records[0].raw["stable_object_url"] == f"{DATA_URL}{new_path}"
    assert refreshed["collection_count"] == 2
    assert sum(call[0] == old_manifest_url for call in client.calls) == calls_before


def test_catalog_drift_mid_scan_fails_closed_with_a_safe_restart_boundary() -> None:
    first_collection = _collection("CC-MAIN-2026-30", 26)
    paths = {
        first_collection["id"]: [
            _path(first_collection["id"], 0),
            _path(first_collection["id"], 1),
            _path(first_collection["id"], 2),
        ]
    }
    client = MappingClient(_bodies([first_collection], paths))
    source = _adapter(client)
    first = source.fetch_page({})
    client.bodies[CATALOG_URL] = json.dumps(
        [_collection("CC-MAIN-2026-34", 27), first_collection]
    ).encode()

    drift = source.fetch_page(first.next_state)

    assert drift.records == ()
    assert drift.complete is False
    assert len(drift.issues) == 1
    assert "changed during the frozen scan" in drift.issues[0].error
    assert drift.retry_state == {}


@pytest.mark.parametrize(
    "catalog,error",
    [
        (
            [_collection("CC-MAIN-2026-30", 26)] * 2,
            "duplicate collection ID",
        ),
        (
            [
                {
                    **_collection("CC-MAIN-2026-30", 26),
                    "id": "../private",
                }
            ],
            "malformed collection ID",
        ),
        ([], "empty or truncated"),
    ],
)
def test_malformed_or_truncated_catalog_is_quarantined(
    catalog: list[dict[str, str]], error: str
) -> None:
    client = MappingClient({CATALOG_URL: json.dumps(catalog).encode()})

    page = _adapter(client).fetch_page({})

    assert page.records == ()
    assert page.complete is False
    assert error in page.issues[0].error
    assert page.retry_state == {}


@pytest.mark.parametrize(
    "manifest,error",
    [
        (b"not-gzip", "invalid or truncated gzip"),
        (gzip.compress(b"\xff\n", mtime=0), "not valid UTF-8"),
        (
            _manifest("crawl-data/CC-MAIN-2026-30/../wet/a.warc.wet.gz"),
            "path traversal",
        ),
        (
            _manifest(
                _path("CC-MAIN-2026-30", 0),
                _path("CC-MAIN-2026-30", 0),
            ),
            "duplicate WET path",
        ),
        (
            gzip.compress(_path("CC-MAIN-2026-30", 0).encode(), mtime=0),
            "overlong or truncated path line",
        ),
    ],
)
def test_invalid_wet_manifests_fail_before_emitting_controls(
    manifest: bytes, error: str
) -> None:
    collection = _collection("CC-MAIN-2026-30", 26)
    manifest_url = f"{DATA_URL}crawl-data/{collection['id']}/wet.paths.gz"
    client = MappingClient(
        {
            CATALOG_URL: json.dumps([collection]).encode(),
            manifest_url: manifest,
        }
    )

    page = _adapter(client).fetch_page({})

    assert page.records == ()
    assert page.complete is False
    assert error in page.issues[0].error


def test_overlong_manifest_and_content_length_truncation_are_rejected() -> None:
    collection = _collection("CC-MAIN-2026-30", 26)
    manifest_url = f"{DATA_URL}crawl-data/{collection['id']}/wet.paths.gz"
    client = MappingClient(
        {
            CATALOG_URL: json.dumps([collection]).encode(),
            manifest_url: _manifest(_path(collection["id"], 0)),
        }
    )
    source = CommonCrawlWetSourceAdapter(
        client=client,
        max_path_bytes=10,
        clock=lambda: NOW,
    )

    page = source.fetch_page({})

    assert page.records == ()
    assert "overlong" in page.issues[0].error

    class TruncatedClient(MappingClient):
        def get(
            self,
            url: str,
            *,
            headers: Mapping[str, str] | None = None,
        ) -> HttpResponse:
            response = super().get(url, headers=headers)
            return HttpResponse(
                status=response.status,
                headers={**response.headers, "content-length": str(len(response.body) + 1)},
                body=response.body,
                url=response.url,
            )

    truncated = CommonCrawlWetSourceAdapter(
        client=TruncatedClient({CATALOG_URL: json.dumps([collection]).encode()})
    ).fetch_page({})
    assert truncated.records == ()
    assert "Content-Length" in truncated.issues[0].error


def test_requests_are_unfiltered_control_plane_gets() -> None:
    collection = _collection("CC-MAIN-2013-20", 13)
    paths = {collection["id"]: [_path(collection["id"], 0)]}
    client = MappingClient(_bodies([collection], paths))

    records, _ = _finish(_adapter(client))

    assert len(records) == 1
    assert [url for url, _ in client.calls] == [
        CATALOG_URL,
        f"{DATA_URL}crawl-data/{collection['id']}/wet.paths.gz",
    ]
    assert all("?" not in url for url, _ in client.calls)
    assert all("search" not in url.casefold() for url, _ in client.calls)
