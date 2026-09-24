from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpFailure, HttpResponse
from modelome.models import Identifier
from modelome.normalize import content_hash
from modelome.sources.crossref import CrossrefSourceAdapter

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


class QueuedClient:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(self, url, *, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.payloads:
            raise AssertionError(f"unexpected GET {url}")
        import json

        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(self.payloads.pop(0)).encode(),
            url=url,
        )


class FailingClient:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(self, url, *, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        raise self.error


def response(items, *, total=None, cursor="next-cursor"):
    return {
        "status": "ok",
        "message-type": "work-list",
        "message-version": "1.0.0",
        "message": {
            "total-results": len(items) if total is None else total,
            "items": items,
            "next-cursor": cursor,
        },
    }


def work(doi="10.1038/example", title="A neural system in a broad journal"):
    return {
        "DOI": doi,
        "title": [title],
        "subtitle": ["A cross-disciplinary result"],
        "abstract": "<jats:p>We introduce <jats:bold>GeneralNet</jats:bold>.</jats:p>",
        "URL": f"https://doi.org/{doi}",
        "publisher": "Example Publisher",
        "container-title": ["An Arbitrary Journal"],
        "published-online": {"date-parts": [[2026, 8, 31]]},
        "indexed": {"date-time": "2026-09-01T11:22:33Z"},
        "link": [
            {
                "URL": "https://publisher.example/paper.pdf",
                "content-type": "application/pdf",
                "intended-application": "text-mining",
            }
        ],
        "relation": {
            "is-preprint-of": [
                {"id-type": "doi", "id": "10.1101/2026.01.01.123456"}
            ]
        },
        "reference": [
            {"DOI": "10.1126/cited-work.7"},
            {"DOI": "not-a-doi"},
            {"unstructured": "A cited work without a DOI"},
        ],
    }


def adapter(client, **kwargs):
    return CrossrefSourceAdapter(client=client, clock=lambda: NOW, **kwargs)


def test_enumerates_all_works_and_resumes_opaque_cursor() -> None:
    first_client = QueuedClient(
        response([work()], total=2, cursor="opaque=="),
        response([work("10.1126/second", "Second architecture")], total=2, cursor="unused"),
    )
    source = adapter(first_client, page_size=1, mailto="registry@example.test")

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state["cursor"] == "opaque=="
    assert first.next_state["scan_total"] == 2
    assert first.next_state["raw_items_seen"] == 1
    assert first_client.calls[0][1] == {
        "cursor": "*",
        "rows": 1,
        "filter": "from-index-date:2026-08-26,until-index-date:2026-09-01",
        "mailto": "registry@example.test",
    }
    assert first_client.calls[1][1]["cursor"] == "opaque=="
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-09-01"
    assert second.upstream_count == 2

    record = first.records[0]
    assert record.source_record_id == "10.1038/example"
    assert record.identifiers == (Identifier("doi", "10.1038/example"),)
    assert record.title == "A neural system in a broad journal"
    assert "We introduce GeneralNet ." in record.text
    assert record.published_at == "2026-08-31"
    assert record.modified_at == "2026-09-01T11:22:33Z"
    assert any(link.relation == "full_text" for link in record.links)
    assert any(link.relation == "is-preprint-of" for link in record.links)
    citation = next(link for link in record.links if link.relation == "cites")
    assert citation.url == "https://doi.org/10.1126/cited-work.7"
    assert citation.locator == "$.reference[0].DOI"
    assert citation.crawl is False
    assert "query" not in first_client.calls[0][1]


def test_daily_window_overlaps_closed_utc_days() -> None:
    client = QueuedClient(response([], total=0, cursor=""))

    page = adapter(client, overlap_days=2).fetch_page({"watermark": "2026-08-31"})

    assert page.complete is True
    assert client.calls[0][1]["filter"] == (
        "from-index-date:2026-08-30,until-index-date:2026-09-01"
    )
    assert page.next_state["watermark"] == "2026-09-01"


def test_fixed_historical_window_is_preserved_without_venue_filters() -> None:
    client = QueuedClient(response([], total=0, cursor=""))

    adapter(client).fetch_page(
        {"window_start": "1950-01-01", "window_end": "1950-12-31"}
    )

    params = client.calls[0][1]
    assert params["filter"] == (
        "from-index-date:1950-01-01,until-index-date:1950-12-31"
    )
    assert set(params) == {"cursor", "rows", "filter"}


def test_malformed_item_is_quarantined_at_same_frozen_boundary() -> None:
    client = QueuedClient(response([{"title": ["No DOI"]}], total=1, cursor=""))

    page = adapter(client).fetch_page({})

    assert page.complete is False
    assert page.records == ()
    assert page.issues[0].stage == "source_normalize"
    assert page.retry_state == page.next_state
    assert page.next_state["cursor"] == "*"
    assert page.next_state["window_start"] == "2026-08-26"


def test_premature_pagination_end_is_reported_and_retried() -> None:
    client = QueuedClient(response([work()], total=2, cursor=""))

    page = adapter(client, page_size=2).fetch_page({})

    assert page.complete is False
    assert page.issues[0].stage == "source_pagination"
    assert "before the declared total of 2" in page.issues[0].error
    assert page.next_state == page.retry_state
    assert page.retry_state["cursor"] == "*"
    assert page.retry_state["raw_items_seen"] == 0


def test_repeated_cursor_is_rejected() -> None:
    cursor = "same-cursor"
    state = {
        "cursor": cursor,
        "window_start": "2026-08-31",
        "window_end": "2026-09-01",
        "raw_items_seen": 1,
        "scan_total": 3,
        "seen_cursor_hashes": [content_hash(cursor)],
        "started_at": "2026-09-02T00:00:00Z",
    }
    client = QueuedClient(response([work()], total=3, cursor=cursor))

    page = adapter(client, page_size=1).fetch_page(state)

    assert page.complete is False
    assert any("cursor cycle" in issue.error for issue in page.issues)
    assert page.next_state == page.retry_state


def test_total_drift_does_not_advance_checkpoint() -> None:
    state = {
        "cursor": "page-2",
        "window_start": "2026-08-31",
        "window_end": "2026-09-01",
        "raw_items_seen": 1,
        "scan_total": 2,
        "started_at": "2026-09-02T00:00:00Z",
    }
    client = QueuedClient(response([work()], total=3, cursor="page-3"))

    page = adapter(client, page_size=1).fetch_page(state)

    assert page.complete is False
    assert any("total-results changed" in issue.error for issue in page.issues)
    assert page.next_state == page.retry_state
    assert page.retry_state["cursor"] == "*"
    assert page.retry_state["raw_items_seen"] == 0
    assert "scan_total" not in page.retry_state


def test_expired_cursor_restarts_the_same_frozen_window() -> None:
    client = FailingClient(
        HttpFailure(
            "GET https://api.crossref.org/works failed: HTTP Error 404: Not Found"
        )
    )
    state = {
        "cursor": "expired-cursor",
        "window_start": "2026-08-31",
        "window_end": "2026-09-01",
        "raw_items_seen": 1000,
        "scan_total": 1200,
        "seen_cursor_hashes": [content_hash("expired-cursor")],
        "started_at": "2026-09-02T00:00:00Z",
    }

    page = adapter(client).fetch_page(state)

    assert page.complete is False
    assert page.records == ()
    assert page.next_state == page.retry_state
    assert page.retry_state["cursor"] == "*"
    assert page.retry_state["window_start"] == "2026-08-31"
    assert page.retry_state["window_end"] == "2026-09-01"
    assert page.retry_state["raw_items_seen"] == 0
    assert "scan_total" not in page.retry_state
    assert "expired or was rejected" in page.issues[0].error


def test_invalid_optional_relation_does_not_quarantine_primary_work() -> None:
    item = work()
    item["relation"] = {
        "is-version-of": [{"id-type": "doi", "id": "not-a-doi"}]
    }
    client = QueuedClient(response([item], total=1, cursor=""))

    page = adapter(client).fetch_page({})

    assert page.complete is True
    assert len(page.records) == 1
    assert page.issues == ()
    assert page.records[0].raw["relation"] == item["relation"]


def test_uri_typed_supplement_relation_emits_exact_hosted_artifact_link() -> None:
    item = work()
    item["relation"] = {
        "is-supplemented-by": [
            {
                "id-type": "uri",
                "id": "https://zenodo.org/records/123/files/model.safetensors?download=1",
            },
            {"id-type": "uri", "id": "javascript:alert(1)"},
        ]
    }
    client = QueuedClient(response([item], total=1, cursor=""))

    page = adapter(client).fetch_page({})

    assert page.complete is True
    record = page.records[0]
    artifact_link = next(
        link for link in record.links if "model.safetensors" in link.url
    )
    assert artifact_link.url == (
        "https://zenodo.org/records/123/files/model.safetensors?download=1"
    )
    assert artifact_link.relation == "is-supplemented-by"
    assert artifact_link.locator == "$.relation.is-supplemented-by[0]"
    assert not any(link.url.startswith("javascript:") for link in record.links)
    assert record.models == ()


def test_constructor_rejects_non_web_api_url() -> None:
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        CrossrefSourceAdapter(url="file:///private/crossref.json")


@pytest.mark.parametrize(
    "state,error",
    [
        ({"window_start": "2026-01-01"}, "both boundaries"),
        (
            {"window_start": "2026-09-01", "window_end": "2026-09-02"},
            "closed UTC day",
        ),
        ({"cursor": "orphan"}, "missing its frozen window"),
    ],
)
def test_invalid_checkpoint_windows_fail_before_http(state, error) -> None:
    client = QueuedClient()

    with pytest.raises(ValueError, match=error):
        adapter(client).fetch_page(state)

    assert client.calls == []
