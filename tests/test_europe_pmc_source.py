from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.normalize import content_hash
from modelome.sources.europe_pmc import EuropePmcBootstrapSource, EuropePmcSourceAdapter

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


class QueuedClient:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(self, url, *, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.payloads:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(self.payloads.pop(0)).encode(),
            url=url,
        )


def result(record_id="12345"):
    return {
        "id": record_id,
        "source": "MED",
        "pmid": record_id,
        "pmcid": "PMC999",
        "doi": "10.1038/biomed.1",
        "title": "A neural architecture for clinical biology",
        "abstractText": "We introduce CellRisk-X, a deep neural network.",
        "firstPublicationDate": "2026-08-29",
        "firstIndexDate": "2026-08-31",
        "fullTextUrlList": {
            "fullTextUrl": [
                {
                    "availabilityCode": "OA",
                    "documentStyle": "html",
                    "url": "https://europepmc.org/articles/PMC999",
                }
            ]
        },
    }


def response(results, *, total=None, cursor="next=="):
    return {
        "version": "6.9",
        "hitCount": len(results) if total is None else total,
        "nextCursorMark": cursor,
        "resultList": {"result": results},
    }


def adapter(client, **kwargs):
    return EuropePmcSourceAdapter(client=client, clock=lambda: NOW, **kwargs)


def test_core_update_scan_resumes_and_preserves_biomedical_evidence() -> None:
    client = QueuedClient(
        response([result()], total=2, cursor="opaque=="),
        response([result("67890")], total=2, cursor="unused"),
    )
    source = adapter(client, page_size=1, email="registry@example.test")

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state["cursor_mark"] == "opaque=="
    assert client.calls[0][1] == {
        "query": "UPDATE_DATE:[2026-08-26 TO 2026-09-01]",
        "format": "json",
        "resultType": "core",
        "cursorMark": "*",
        "pageSize": 1,
        "synonym": "false",
        "email": "registry@example.test",
    }
    assert client.calls[1][1]["cursorMark"] == "opaque=="
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-09-01"

    record = first.records[0]
    assert record.source_record_id == "MED:12345"
    assert record.canonical_url == "https://europepmc.org/article/MED/12345"
    assert set(record.identifiers) == {
        Identifier("europepmc", "MED:12345"),
        Identifier("pmid", "12345"),
        Identifier("pmcid", "PMC999"),
        Identifier("doi", "10.1038/biomed.1"),
    }
    assert "CellRisk-X" in record.text
    assert any(link.relation == "open_full_text" for link in record.links)
    assert any(link.relation == "doi" for link in record.links)


def test_zero_hit_window_completes_without_result_list() -> None:
    client = QueuedClient({"version": "6.9", "hitCount": 0})

    page = adapter(client).fetch_page({"watermark": "2026-08-31"})

    assert page.complete is True
    assert page.records == ()
    assert page.upstream_count == 0
    assert client.calls[0][1]["query"] == (
        "UPDATE_DATE:[2026-08-30 TO 2026-09-01]"
    )


def test_fixed_history_has_no_subject_or_journal_query() -> None:
    client = QueuedClient(response([], total=0, cursor=""))

    adapter(client).fetch_page(
        {"window_start": "1990-01-01", "window_end": "1990-12-31"}
    )

    assert client.calls[0][1]["query"] == (
        "UPDATE_DATE:[1990-01-01 TO 1990-12-31]"
    )


def test_malformed_result_is_quarantined_without_advancing() -> None:
    client = QueuedClient(response([{"source": "MED"}], total=1, cursor=""))

    page = adapter(client).fetch_page({})

    assert page.complete is False
    assert page.records == ()
    assert page.issues[0].stage == "source_normalize"
    assert page.next_state == page.retry_state
    assert page.next_state["cursor_mark"] == "*"


def test_short_page_before_hit_count_is_a_pagination_failure() -> None:
    client = QueuedClient(response([result()], total=2, cursor=""))

    page = adapter(client, page_size=2).fetch_page({})

    assert page.complete is False
    assert "before the declared hitCount of 2" in page.issues[0].error
    assert page.next_state == page.retry_state
    assert page.retry_state["cursor_mark"] == "*"
    assert page.retry_state["raw_items_seen"] == 0
    assert "scan_total" not in page.retry_state


def test_cursor_cycle_is_quarantined() -> None:
    cursor = "same"
    state = {
        "cursor_mark": cursor,
        "window_start": "2026-08-31",
        "window_end": "2026-09-01",
        "raw_items_seen": 1,
        "scan_total": 3,
        "seen_cursor_hashes": [content_hash(cursor)],
        "started_at": "2026-09-02T00:00:00Z",
    }
    client = QueuedClient(response([result()], total=3, cursor=cursor))

    page = adapter(client, page_size=1).fetch_page(state)

    assert any("cursor cycle" in issue.error for issue in page.issues)
    assert page.next_state == page.retry_state
    assert page.retry_state["cursor_mark"] == "*"
    assert page.retry_state["raw_items_seen"] == 0
    assert "scan_total" not in page.retry_state


def test_hit_count_drift_restarts_the_same_frozen_window() -> None:
    state = {
        "cursor_mark": "page-2",
        "window_start": "2026-08-31",
        "window_end": "2026-09-01",
        "raw_items_seen": 1,
        "scan_total": 2,
        "started_at": "2026-09-02T00:00:00Z",
    }
    client = QueuedClient(response([result()], total=3, cursor="page-3"))

    page = adapter(client, page_size=1).fetch_page(state)

    assert page.complete is False
    assert any("hitCount changed" in issue.error for issue in page.issues)
    assert page.retry_state["cursor_mark"] == "*"
    assert page.retry_state["window_start"] == "2026-08-31"
    assert page.retry_state["window_end"] == "2026-09-01"
    assert page.retry_state["raw_items_seen"] == 0
    assert "scan_total" not in page.retry_state


def test_historical_bootstrap_freezes_month_chunks_and_resumes_cursors() -> None:
    client = QueuedClient(
        response([result("1")], total=2, cursor="aug-page-2"),
        response([result("2")], total=2, cursor="unused"),
        response([result("3")], total=1, cursor="unused"),
    )
    daily = adapter(client, page_size=1)
    bootstrap = EuropePmcBootstrapSource(
        daily, earliest_update_date=date(2026, 8, 31)
    )

    first = bootstrap.fetch_page({})
    second = bootstrap.fetch_page(first.next_state)
    third = bootstrap.fetch_page(second.next_state)

    assert not first.complete and first.next_state["bootstrap_complete"] is False
    assert first.next_state["bootstrap"]["adapter_state"]["cursor_mark"] == "aug-page-2"
    assert not second.complete
    assert second.next_state["bootstrap"]["chunk_start"] == "2026-09-01"
    assert second.next_state["bootstrap"]["chunk_end"] == "2026-09-01"
    assert third.complete and bootstrap.is_complete(third.next_state)
    assert [call[1]["query"] for call in client.calls] == [
        "UPDATE_DATE:[2026-08-31 TO 2026-08-31]",
        "UPDATE_DATE:[2026-08-31 TO 2026-08-31]",
        "UPDATE_DATE:[2026-09-01 TO 2026-09-01]",
    ]
    assert client.calls[1][1]["cursorMark"] == "aug-page-2"
    assert bootstrap.artifact_source == daily.name


def test_historical_bootstrap_rejects_changed_or_corrupt_checkpoint() -> None:
    bootstrap = EuropePmcBootstrapSource(
        adapter(QueuedClient()), earliest_update_date=date(2026, 8, 1)
    )
    descriptor = bootstrap._start_descriptor()
    descriptor["chunk_end"] = "2026-08-30"
    with pytest.raises(ValueError, match="month boundary"):
        bootstrap._descriptor({"bootstrap": descriptor})


def test_historical_bootstrap_replays_current_month_when_total_drifts() -> None:
    client = QueuedClient(
        response([result("1")], total=2, cursor="page-2"),
        response([result("2")], total=3, cursor="page-3"),
    )
    bootstrap = EuropePmcBootstrapSource(
        adapter(client, page_size=1), earliest_update_date=date(2026, 8, 1)
    )

    first = bootstrap.fetch_page({})
    second = bootstrap.fetch_page(first.next_state)

    assert any("hitCount changed" in issue.error for issue in second.issues)
    retry = second.retry_state["bootstrap"]
    assert retry["chunk_start"] == "2026-08-01"
    assert retry["chunk_end"] == "2026-08-31"
    assert retry["adapter_state"]["cursor_mark"] == "*"
    assert retry["adapter_state"]["raw_items_seen"] == 0


def test_historical_bootstrap_rejects_lower_bound_after_frozen_end() -> None:
    bootstrap = EuropePmcBootstrapSource(
        adapter(QueuedClient()), earliest_update_date=date(2026, 9, 2)
    )
    with pytest.raises(ValueError, match="later than yesterday"):
        bootstrap.fetch_page({})


def test_constructor_rejects_non_web_api_url() -> None:
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        EuropePmcSourceAdapter(url="file:///private/europe-pmc.json")


@pytest.mark.parametrize(
    "state,error",
    [
        ({"window_start": "2026-01-01"}, "both boundaries"),
        (
            {"window_start": "2026-09-01", "window_end": "2026-09-02"},
            "closed UTC day",
        ),
        ({"cursor_mark": "orphan"}, "missing its frozen window"),
        (
            {
                "cursor_mark": "page-2",
                "window_start": "2026-08-31",
                "window_end": "2026-09-01",
            },
            "requires raw_items_seen and scan_total",
        ),
    ],
)
def test_invalid_checkpoint_fails_before_http(state, error) -> None:
    client = QueuedClient()

    with pytest.raises(ValueError, match=error):
        adapter(client).fetch_page(state)

    assert client.calls == []
