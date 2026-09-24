from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.semantic_scholar import SemanticScholarDatasetSourceAdapter

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
API = "https://api.semanticscholar.org/datasets/v1"
API_KEY = "s2-test-secret"


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


class FailingClient:
    def get(self, url: str, *, headers: Mapping[str, str] | None = None) -> HttpResponse:
        raise RuntimeError(f"request with {headers['x-api-key']} failed")


def fixture_response(name: str) -> HttpResponse:
    return HttpResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=(FIXTURES / name).read_bytes(),
        url=f"https://fixtures.test/{name}",
    )


def json_response(payload: Any) -> HttpResponse:
    return HttpResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=json.dumps(payload).encode(),
        url="https://api.semanticscholar.org/fixture",
    )


def adapter(client: Any, **kwargs: Any) -> SemanticScholarDatasetSourceAdapter:
    return SemanticScholarDatasetSourceAdapter(
        api_key=API_KEY,
        client=client,
        clock=lambda: NOW,
        **kwargs,
    )


def initial_responses() -> tuple[HttpResponse, HttpResponse]:
    return (
        fixture_response("semantic_scholar_releases.json"),
        fixture_response("semantic_scholar_release_2026-08-25.json"),
    )


def scan_all(
    source: SemanticScholarDatasetSourceAdapter,
    state: Mapping[str, Any] | None = None,
) -> list[Any]:
    pages = []
    next_state = dict(state or {})
    while True:
        page = source.fetch_page(next_state)
        pages.append(page)
        if page.complete:
            return pages
        next_state = dict(page.next_state)


def test_full_scan_enumerates_releases_pins_date_and_emits_every_required_shard() -> None:
    client = QueuedClient(
        *initial_responses(),
        fixture_response("semantic_scholar_papers_manifest.json"),
        fixture_response("semantic_scholar_abstracts_manifest.json"),
        fixture_response("semantic_scholar_paper_ids_manifest.json"),
    )
    source = adapter(client)

    pages = scan_all(source)

    assert len(pages) == 4
    release = pages[0].records[0]
    assert release.source_record_id == "semantic-scholar:release:2026-08-25"
    assert release.kind is ArtifactKind.CATALOG_RECORD
    assert release.identifiers == (
        Identifier("semantic-scholar:release", "2026-08-25"),
    )
    assert release.raw["available_releases"] == [
        "2026-08-11",
        "2026-08-18",
        "2026-08-25",
    ]
    assert release.raw["release_readme"] == "Release-wide license and usage terms."
    assert [item["name"] for item in release.raw["datasets"]] == [
        "papers",
        "abstracts",
        "paper-ids",
    ]
    assert release.raw["datasets"][0]["README"] == "Papers license: ODC-BY."
    assert pages[0].next_state["target_release"] == "2026-08-25"
    assert pages[0].next_state["stage"] == "snapshot"

    shards = [record for page in pages[1:] for record in page.records]
    assert len(shards) == 4
    assert [record.raw["dataset"] for record in shards] == [
        "papers",
        "papers",
        "abstracts",
        "paper-ids",
    ]
    assert {record.raw["operation"] for record in shards} == {"snapshot"}
    assert pages[-1].complete is True
    assert pages[-1].upstream_count == 5
    assert pages[-1].next_state == {
        "watermark": "2026-08-25",
        "completed_at": "2026-09-02T12:00:00Z",
    }

    assert [call[0] for call in client.calls] == [
        f"{API}/release/",
        f"{API}/release/2026-08-25",
        f"{API}/release/2026-08-25/dataset/papers",
        f"{API}/release/2026-08-25/dataset/abstracts",
        f"{API}/release/2026-08-25/dataset/paper-ids",
    ]
    assert all(call[1]["x-api-key"] == API_KEY for call in client.calls)


def test_shard_records_are_stable_and_do_not_persist_or_enqueue_presigned_urls() -> None:
    first_manifest = {
        "name": "papers",
        "description": "Core metadata",
        "README": "ODC-BY",
        "files": [
            "https://objects.example/releases/papers/part.gz?X-Amz-Signature=first-secret"
        ],
    }
    second_manifest = {
        **first_manifest,
        "files": [
            "https://objects.example/releases/papers/part.gz?X-Amz-Signature=rotated-secret"
        ],
    }
    state = {
        "stage": "snapshot",
        "target_release": "2026-08-25",
        "dataset_index": 0,
        "shard_offset": 0,
        "control_records_seen": 1,
        "release_signature": "pinned-release-signature",
    }

    first = adapter(QueuedClient(json_response(first_manifest))).fetch_page(state)
    second = adapter(QueuedClient(json_response(second_manifest))).fetch_page(state)
    first_record = first.records[0]
    second_record = second.records[0]

    assert first_record.source_record_id == second_record.source_record_id
    assert first_record.canonical_url == (
        "https://objects.example/releases/papers/part.gz"
    )
    assert first_record.links == ()
    serialized = json.dumps(first_record.raw, sort_keys=True)
    assert "first-secret" not in serialized
    assert "X-Amz-Signature" in serialized
    assert first_record.raw["temporary_download_url_persisted"] is False
    assert first_record.raw["download_resolution"] == (
        "refetch_manifest_and_match_stable_object_url"
    )


def test_incremental_scan_preserves_sequential_upserts_and_deletes() -> None:
    client = QueuedClient(
        *initial_responses(),
        fixture_response("semantic_scholar_papers_diff.json"),
        fixture_response("semantic_scholar_abstracts_diff.json"),
        fixture_response("semantic_scholar_paper_ids_diff.json"),
    )
    source = adapter(client)

    pages = scan_all(source, {"watermark": "2026-08-11"})

    release = pages[0].records[0]
    assert release.raw["transfer_mode"] == "diff"
    assert release.raw["base_release"] == "2026-08-11"
    records = [record for page in pages[1:] for record in page.records]
    assert [(record.raw["dataset"], record.raw["operation"]) for record in records] == [
        ("papers", "upsert"),
        ("papers", "delete"),
        ("papers", "upsert"),
        ("abstracts", "upsert"),
        ("paper-ids", "upsert"),
        ("paper-ids", "delete"),
    ]
    assert records[0].raw["from_release"] == "2026-08-11"
    assert records[0].raw["to_release"] == "2026-08-18"
    assert records[2].raw["from_release"] == "2026-08-18"
    assert records[2].raw["to_release"] == "2026-08-25"
    assert records[0].raw["license_metadata_record_id"] == (
        "semantic-scholar:release:2026-08-25"
    )
    assert pages[-1].upstream_count == 7
    assert pages[-1].next_state["watermark"] == "2026-08-25"
    assert all("/diffs/" in url for url, _ in client.calls[2:])


def test_same_pinned_release_is_a_complete_noop_after_metadata_validation() -> None:
    client = QueuedClient(*initial_responses())

    page = adapter(client).fetch_page({"watermark": "2026-08-25"})

    assert page.complete is True
    assert page.records == ()
    assert page.upstream_count == 0
    assert page.next_state["watermark"] == "2026-08-25"
    assert len(client.calls) == 2


def test_explicit_release_is_resolved_from_catalog_without_using_latest_alias() -> None:
    release = json.loads(
        (FIXTURES / "semantic_scholar_release_2026-08-25.json").read_text()
    )
    release["release_id"] = "2026-08-18"
    client = QueuedClient(
        fixture_response("semantic_scholar_releases.json"),
        json_response(release),
    )

    page = adapter(client, release_id="2026-08-18").fetch_page({})

    assert page.next_state["target_release"] == "2026-08-18"
    assert client.calls[1][0] == f"{API}/release/2026-08-18"
    assert all("latest" not in url for url, _ in client.calls)


def test_malformed_shard_urls_are_quarantined_without_exposing_credentials() -> None:
    manifest = {
        "name": "papers",
        "description": "Core metadata",
        "README": "ODC-BY",
        "files": [
            "https://objects.example/releases/papers/good.gz?X-Amz-Signature=good",
            "http://objects.example/releases/papers/plaintext.gz?token=leaked",
            f"https://objects.example/releases/papers/reflected.gz?token={API_KEY}",
            "https://objects.example/releases/papers/good.gz?X-Amz-Signature=duplicate",
            "https://127.0.0.1/internal/shard.gz?token=private",
        ],
    }
    state = {
        "stage": "snapshot",
        "target_release": "2026-08-25",
        "dataset_index": 0,
        "shard_offset": 0,
        "control_records_seen": 1,
        "release_signature": "pinned-release-signature",
    }

    page = adapter(QueuedClient(json_response(manifest))).fetch_page(state)

    assert len(page.records) == 1
    assert len(page.issues) == 4
    assert {issue.stage for issue in page.issues} == {"source_manifest"}
    assert page.retry_state == state
    assert API_KEY not in repr(page.issues)
    assert "leaked" not in repr(page.issues)


def test_manifest_count_change_resets_current_dataset_to_a_safe_boundary() -> None:
    original = {
        "name": "papers",
        "description": "Core metadata",
        "README": "ODC-BY",
        "files": [
            "https://objects.example/releases/papers/one.gz?sig=one",
            "https://objects.example/releases/papers/two.gz?sig=two",
        ],
    }
    changed = {
        **original,
        "files": [
            *original["files"],
            "https://objects.example/releases/papers/three.gz?sig=three",
        ],
    }
    client = QueuedClient(json_response(original), json_response(changed))
    source = adapter(client, page_size=1)
    state = {
        "stage": "snapshot",
        "target_release": "2026-08-25",
        "dataset_index": 0,
        "shard_offset": 0,
        "control_records_seen": 1,
        "release_signature": "pinned-release-signature",
    }

    first = source.fetch_page(state)
    second = source.fetch_page(first.next_state)

    assert first.next_state["manifest_count"] == 2
    assert second.issues[0].stage == "source_manifest"
    assert "manifest count changed from 2 to 3" in second.issues[0].error
    assert second.retry_state["shard_offset"] == 0
    assert "manifest_count" not in second.retry_state
    assert "manifest_signature" not in second.retry_state


def test_non_contiguous_diff_chain_is_quarantined_at_the_same_boundary() -> None:
    payload = {
        "dataset": "papers",
        "start_release": "2026-08-11",
        "end_release": "2026-08-25",
        "diffs": [
            {
                "from_release": "2026-08-18",
                "to_release": "2026-08-25",
                "update_files": [],
                "delete_files": [],
            }
        ],
    }
    state = {
        "stage": "diff",
        "target_release": "2026-08-25",
        "base_release": "2026-08-11",
        "watermark": "2026-08-11",
        "dataset_index": 0,
        "shard_offset": 0,
        "control_records_seen": 1,
        "release_signature": "pinned-release-signature",
    }

    page = adapter(QueuedClient(json_response(payload))).fetch_page(state)

    assert page.records == ()
    assert page.retry_state == state
    assert any("diff chain expected 2026-08-11" in issue.error for issue in page.issues)
    assert any("diff chain ended at 2026-08-11" in issue.error for issue in page.issues)


def test_release_catalog_and_required_dataset_shape_fail_closed() -> None:
    malformed_catalog = QueuedClient(
        json_response(["2026-08-25", "not-a-date"]),
        fixture_response("semantic_scholar_release_2026-08-25.json"),
    )
    page = adapter(malformed_catalog).fetch_page({})
    assert len(page.issues) == 1
    assert page.retry_state == {}

    missing = json.loads(
        (FIXTURES / "semantic_scholar_release_2026-08-25.json").read_text()
    )
    missing["datasets"] = [
        item for item in missing["datasets"] if item["name"] != "paper-ids"
    ]
    source = adapter(
        QueuedClient(
            fixture_response("semantic_scholar_releases.json"),
            json_response(missing),
        )
    )
    with pytest.raises(ValueError, match=r"missing required dataset\(s\): paper-ids"):
        source.fetch_page({})


def test_authentication_is_required_redacted_and_excluded_from_checkpoint_identity() -> None:
    with pytest.raises(ValueError, match="API key must not be empty"):
        SemanticScholarDatasetSourceAdapter(api_key=None)

    first = adapter(QueuedClient(), name="first")
    second = SemanticScholarDatasetSourceAdapter(
        name="second",
        api_key="a-different-secret",
        client=QueuedClient(),
        clock=lambda: NOW,
    )
    assert first.checkpoint_signature == second.checkpoint_signature
    assert API_KEY not in first.checkpoint_signature

    with pytest.raises(RuntimeError, match="Semantic Scholar request failed") as captured:
        adapter(FailingClient()).fetch_page({})
    assert API_KEY not in str(captured.value)


def test_list_releases_returns_sorted_complete_catalog() -> None:
    source = adapter(QueuedClient(fixture_response("semantic_scholar_releases.json")))

    assert source.list_releases() == (
        "2026-08-11",
        "2026-08-18",
        "2026-08-25",
    )
