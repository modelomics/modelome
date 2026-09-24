from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.pipeline import SyncEngine
from modelome.sources.biorxiv import BioRxivPublicationSourceAdapter, BioRxivSourceAdapter
from modelome.sources.catalog import create_source
from modelome.storage import Database

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def fixture_response(name: str) -> HttpResponse:
    return HttpResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=(FIXTURES / name).read_bytes(),
        url=f"https://fixtures.test/{name}",
    )


def json_response(payload: Mapping[str, Any]) -> HttpResponse:
    return HttpResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=json.dumps(payload).encode(),
        url="https://api.biorxiv.org/fixture",
    )


def test_details_source_pages_by_actual_collection_length_and_freezes_window() -> None:
    client = QueuedClient(
        fixture_response("source_biorxiv_details_page1.json"),
        fixture_response("source_biorxiv_details_page2.json"),
    )
    adapter = BioRxivSourceAdapter(
        name="biorxiv",
        url="https://api.biorxiv.org/details",
        server="biorxiv",
        initial_lookback_days=7,
        overlap_days=2,
        client=client,
        clock=lambda: NOW,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.upstream_count == 3
    assert first.next_state["cursor"] == 2
    assert first.next_state["raw_items_seen"] == 2
    assert first.next_state["scan_total"] == 3
    assert first.next_state["window_start"] == "2026-08-25"
    assert first.next_state["window_end"] == "2026-08-31"
    assert client.calls[0][0] == (
        "https://api.biorxiv.org/details/biorxiv/2026-08-25/2026-08-31/0/json"
    )
    assert client.calls[1][0] == (
        "https://api.biorxiv.org/details/biorxiv/2026-08-25/2026-08-31/2/json"
    )
    assert client.calls[0][1] == {}
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-08-31"

    first_record, revised_record = first.records
    assert first_record.kind is ArtifactKind.PAPER
    assert first_record.source_record_id == "biorxiv:10.1101/2026.08.30.123456:v1"
    assert first_record.identifiers == (
        Identifier("doi", "10.1101/2026.08.30.123456"),
        Identifier("biorxiv:version", "10.1101/2026.08.30.123456v1"),
    )
    assert {link.relation for link in first_record.links} == {"preprint", "doi", "full_text"}
    assert revised_record.source_record_id == "biorxiv:10.64898/2026.08.29.654321:v2"
    assert Identifier("doi", "10.1038/s41586-026-01234-5") not in revised_record.identifiers
    assert next(link for link in revised_record.links if link.relation == "published_as").url == (
        "https://doi.org/10.1038/s41586-026-01234-5"
    )
    assert revised_record.raw["license"] == "cc_by_nc_nd"
    assert revised_record.raw["category"] == "bioengineering"


def test_details_daily_window_replays_only_configured_closed_days() -> None:
    client = QueuedClient(
        json_response(
            {
                "messages": [
                    {"status": "no articles found for 2026-08-30 2026-08-31"}
                ],
                "collection": [],
            }
        )
    )
    adapter = BioRxivSourceAdapter(
        name="medrxiv",
        url="https://api.biorxiv.org/details",
        server="medrxiv",
        overlap_days=2,
        client=client,
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({"watermark": "2026-08-31"})

    assert page.complete is True
    assert page.next_state["watermark"] == "2026-08-31"
    assert client.calls[0][0] == (
        "https://api.biorxiv.org/details/medrxiv/2026-08-30/2026-08-31/0/json"
    )


def test_details_quarantines_malformed_version_and_holds_page_for_retry(tmp_path) -> None:
    malformed = {
        "messages": [{"status": "ok", "cursor": 0, "count": 1, "total": "1"}],
        "collection": [
            {
                "doi": "10.1101/2026.08.31.123456",
                "title": "Malformed version",
                "version": "not-an-integer",
                "date": "2026-08-31",
                "server": "bioRxiv",
            }
        ],
    }
    corrected = {
        "messages": [{"status": "ok", "cursor": 0, "count": 1, "total": "1"}],
        "collection": [
            {
                "doi": "10.1101/2026.08.31.123456",
                "title": "Corrected version",
                "version": "1",
                "date": "2026-08-31",
                "server": "bioRxiv",
            }
        ],
    }
    client = QueuedClient(json_response(malformed), json_response(corrected))
    current_time = [NOW]
    source = BioRxivSourceAdapter(
        name="biorxiv",
        url="https://api.biorxiv.org/details",
        server="biorxiv",
        initial_lookback_days=1,
        client=client,
        clock=lambda: current_time[0],
    )
    database = Database(tmp_path / "store")
    database.initialize()
    engine = SyncEngine(database, {source.name: source})

    failed = engine.sync()[0]
    held_state = database.get_source_state(source.name)
    current_time[0] = NOW + timedelta(days=1)
    retried = engine.sync()[0]

    assert failed.status == "failed"
    assert held_state["cursor"] == 0
    assert held_state["window_start"] == "2026-08-31"
    assert held_state["window_end"] == "2026-08-31"
    assert retried.status == "complete"
    assert client.calls[1][0] == client.calls[0][0]
    assert database.list_dead_letters("biorxiv")[0]["stage"] == "source_normalize"


def test_details_detects_count_mismatch_and_premature_empty_page() -> None:
    count_mismatch = {
        "messages": [{"status": "ok", "cursor": 0, "count": 30, "total": 2}],
        "collection": [
            {
                "doi": "10.1101/2026.08.31.1",
                "title": "One record",
                "version": 1,
                "date": "2026-08-31",
                "server": "biorxiv",
            }
        ],
    }
    premature = {
        "messages": [{"status": "ok", "cursor": 1, "count": 0, "total": 2}],
        "collection": [],
    }
    client = QueuedClient(json_response(count_mismatch), json_response(premature))
    adapter = BioRxivSourceAdapter(
        name="biorxiv",
        url="https://api.biorxiv.org/details",
        server="biorxiv",
        initial_lookback_days=1,
        client=client,
        clock=lambda: NOW,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.next_state["cursor"] == 1
    assert first.issues[0].stage == "source_pagination"
    assert "declared count 30" in first.issues[0].error
    assert second.complete is False
    assert second.next_state == second.retry_state
    assert any("no records before declared total" in issue.error for issue in second.issues)


def test_publication_source_preserves_both_dois_from_independent_stream() -> None:
    client = QueuedClient(fixture_response("source_medrxiv_publications.json"))
    adapter = BioRxivPublicationSourceAdapter(
        name="medrxiv-publications",
        url="https://api.biorxiv.org/pubs",
        server="medrxiv",
        initial_lookback_days=7,
        client=client,
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert client.calls[0][0] == (
        "https://api.biorxiv.org/pubs/medrxiv/2026-08-25/2026-08-31/0"
    )
    record = page.records[0]
    assert record.source_record_id == (
        "medrxiv:10.1101/2025.01.02.25320000:published:"
        "10.1016/j.example.2026.100001"
    )
    assert record.identifiers == (Identifier("doi", "10.1101/2025.01.02.25320000"),)
    assert record.published_at == "2025-01-03"
    assert record.modified_at == "2026-08-30"
    assert next(link for link in record.links if link.relation == "published_as").url == (
        "https://doi.org/10.1016/j.example.2026.100001"
    )


def test_publication_source_accepts_both_preprint_doi_field_names() -> None:
    for field in ("preprint_doi", "biorxiv_doi"):
        payload = {
            "messages": [{"status": "ok", "cursor": 0, "count": 1, "total": 1}],
            "collection": [
                {
                    field: "10.1101/2025.01.02.25320000",
                    "published_doi": "10.1016/j.example.2026.100001",
                    "preprint_platform": "medRxiv",
                }
            ],
        }
        adapter = BioRxivPublicationSourceAdapter(
            name=f"medrxiv-publications-{field}",
            url="https://api.biorxiv.org/pubs",
            server="medrxiv",
            initial_lookback_days=1,
            client=QueuedClient(json_response(payload)),
            clock=lambda: NOW,
        )

        page = adapter.fetch_page({})

        assert page.records[0].source_record_id.startswith(
            "medrxiv:10.1101/2025.01.02.25320000:published:"
        )


def test_catalog_factory_supports_both_biorxiv_stream_types() -> None:
    details = create_source(
        {
            "name": "medrxiv",
            "adapter": "biorxiv",
            "url": "https://api.biorxiv.org/details",
            "server": "medrxiv",
            "initial_lookback_days": 3,
            "overlap_days": 2,
        },
        client=QueuedClient(),
        clock=lambda: NOW,
    )
    publications = create_source(
        {
            "name": "biorxiv-publications",
            "adapter": "biorxiv_publications",
            "url": "https://api.biorxiv.org/pubs",
            "server": "biorxiv",
        },
        client=QueuedClient(),
        clock=lambda: NOW,
    )

    assert isinstance(details, BioRxivSourceAdapter)
    assert details.server == "medrxiv"
    assert details.initial_lookback_days == 3
    assert details.overlap_days == 2
    assert isinstance(publications, BioRxivPublicationSourceAdapter)
    assert publications.server == "biorxiv"
    assert publications.overlap_days == 90


def test_details_versions_are_distinct_artifacts_despite_shared_base_doi(tmp_path) -> None:
    doi = "10.1101/2026.08.30.123456"
    payload = {
        "messages": [{"status": "ok", "cursor": 0, "count": 2, "total": 2}],
        "collection": [
            {
                "doi": doi,
                "title": "First version",
                "version": 1,
                "date": "2026-08-30",
                "server": "bioRxiv",
            },
            {
                "doi": doi,
                "title": "Second version",
                "version": 2,
                "date": "2026-08-31",
                "server": "bioRxiv",
            },
        ],
    }
    source = BioRxivSourceAdapter(
        name="biorxiv",
        url="https://api.biorxiv.org/details",
        server="biorxiv",
        initial_lookback_days=2,
        client=QueuedClient(json_response(payload)),
        clock=lambda: NOW,
    )
    database = Database(tmp_path / "store")
    database.initialize()

    outcome = SyncEngine(database, {source.name: source}).sync()[0]

    assert outcome.status == "complete"
    assert outcome.stats["new_artifacts"] == 2
    assert database.stats()["artifacts"] == 2
