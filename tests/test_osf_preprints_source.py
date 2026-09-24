from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.pipeline import SyncEngine
from modelome.sources.osf_preprints import OsfPreprintSourceAdapter
from modelome.storage import Database

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


def response(
    *items: Mapping[str, Any],
    total: int,
    next_page: int | None = None,
) -> HttpResponse:
    next_url = (
        f"https://api.osf.io/v2/preprints/?page={next_page}&page%5Bsize%5D=2"
        if next_page is not None
        else None
    )
    payload = {
        "data": list(items),
        "links": {"next": next_url, "meta": {"total": total, "per_page": 2}},
        "meta": {"version": "2.0"},
    }
    return HttpResponse(
        status=200,
        headers={"content-type": "application/vnd.api+json"},
        body=json.dumps(payload).encode(),
        url="https://api.osf.io/v2/preprints/",
    )


def preprint(
    identifier: str,
    *,
    title: str = "A neural model for cell states",
    provider: str = "psyarxiv",
    modified: str = "2026-08-31T10:00:00.000000",
) -> dict[str, Any]:
    return {
        "id": identifier,
        "type": "preprints",
        "attributes": {
            "title": title,
            "description": "Code is available at https://github.com/example/cells.",
            "date_published": "2026-08-30T09:00:00.000000",
            "date_modified": modified,
            "doi": "10.1000/osf-test",
            "version": 1,
            "is_latest_version": True,
            "tags": ["machine learning", "biology"],
        },
        "relationships": {
            "provider": {"data": {"id": provider, "type": "preprint-providers"}}
        },
        "links": {
            "html": f"https://osf.io/preprints/{provider}/{identifier}/",
            "preprint_doi": f"https://doi.org/10.31234/osf.io/{identifier}",
            "iri": f"https://osf.io/{identifier.removesuffix('_v1')}",
        },
    }


def test_source_uses_a_closed_global_osf_window_and_preserves_preprint_identity() -> None:
    client = QueuedClient(
        response(
            preprint("a1b2c_v1"),
            preprint("d3e4f_v1", provider="socarxiv"),
            total=3,
            next_page=2,
        ),
        response(preprint("g5h6i_v2", provider="metaarxiv"), total=3),
    )
    adapter = OsfPreprintSourceAdapter(
        page_size=2,
        initial_lookback_days=7,
        overlap_days=2,
        client=client,
        clock=lambda: NOW,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.upstream_count == 3
    assert first.next_state["page"] == 2
    assert first.next_state["raw_items_seen"] == 2
    assert first.next_state["scan_total"] == 3
    assert first.next_state["window_start"] == "2026-08-25"
    assert first.next_state["window_end"] == "2026-08-31"
    assert client.calls[0][0] == "https://api.osf.io/v2/preprints"
    assert client.calls[0][1] == {
        "page": 1,
        "page[size]": 2,
        "sort": "date_modified",
        "filter[date_modified][gte]": "2026-08-25T00:00:00Z",
        "filter[date_modified][lte]": "2026-08-31T23:59:59.999999Z",
    }
    assert client.calls[1][1]["page"] == 2
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-08-31"

    record = first.records[0]
    assert record.source_record_id == "osf-preprints:a1b2c_v1"
    assert record.canonical_url == "https://osf.io/preprints/psyarxiv/a1b2c_v1"
    assert record.identifiers == (
        Identifier("osf:preprint", "a1b2c_v1"),
        Identifier("doi", "10.1000/osf-test"),
        Identifier("doi", "10.31234/osf.io/a1b2c_v1"),
    )
    assert record.raw["provider"] == "psyarxiv"
    assert {link.relation for link in record.links} == {
        "preprint",
        "doi",
        "related_project",
    }
    assert record.published_at == "2026-08-30T09:00:00Z"
    assert record.modified_at == "2026-08-31T10:00:00Z"


def test_source_retries_a_malformed_item_from_the_same_frozen_page(tmp_path) -> None:
    malformed = preprint("a1b2c_v1", title="")
    corrected = preprint("a1b2c_v1")
    client = QueuedClient(response(malformed, total=1), response(corrected, total=1))
    current_time = [NOW]
    source = OsfPreprintSourceAdapter(
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
    assert held_state["page"] == 1
    assert held_state["window_start"] == "2026-08-31"
    assert held_state["window_end"] == "2026-08-31"
    assert retried.status == "complete"
    assert client.calls[1][1] == client.calls[0][1]
    assert database.list_dead_letters("osf-preprints")[0]["stage"] == "source_normalize"


def test_source_restarts_a_frozen_window_when_the_total_changes() -> None:
    client = QueuedClient(response(preprint("d3e4f_v1"), total=3, next_page=3))
    adapter = OsfPreprintSourceAdapter(page_size=2, client=client, clock=lambda: NOW)

    page = adapter.fetch_page(
        {
            "page": 2,
            "window_start": "2026-08-30",
            "window_end": "2026-08-31",
            "raw_items_seen": 2,
            "scan_total": 2,
        }
    )

    assert page.complete is False
    assert any("window total changed from 2 to 3" in issue.error for issue in page.issues)
    assert page.retry_state is not None
    assert page.retry_state["page"] == 1
    assert page.retry_state["raw_items_seen"] == 0
