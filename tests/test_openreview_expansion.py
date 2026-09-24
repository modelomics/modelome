from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.openreview import OpenReviewSourceAdapter

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
V2_URL = "https://api2.openreview.net/notes"


def _millis(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1_000)


def _note(note_id: str, tmdate: str) -> dict[str, Any]:
    return {
        "id": note_id,
        "forum": note_id,
        "invitations": ["Example.org/2026/Conference/-/Submission"],
        "tcdate": _millis("2026-01-01T00:00:00Z"),
        "tmdate": _millis(tmdate),
        "content": {
            "title": {"value": f"Model paper {note_id}"},
            "abstract": {"value": "A learned model with an empirical evaluation."},
            "authors": {"value": ["A. Researcher"]},
        },
    }


class _QueueClient:
    def __init__(self, *payloads: Any) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        body = json.dumps(self.payloads.pop(0)).encode()
        return HttpResponse(200, {"content-type": "application/json"}, body, url)


def test_v2_keyset_pages_keep_the_lower_bound_and_accept_equal_tmdate() -> None:
    """V2's id cursor must advance cleanly across notes sharing one tmdate."""
    same_time = "2026-09-02T10:00:00Z"
    client = _QueueClient(
        {"count": 2, "notes": [_note("v2-a", same_time), _note("v2-b", same_time)]},
        {"notes": [_note("v2-c", "2026-09-02T10:01:00Z"), _note("v2-d", "2026-09-02T10:01:00Z")]},
        {"notes": []},
    )
    source = OpenReviewSourceAdapter(
        client=client,
        clock=lambda: NOW,
        consistency_lag_seconds=300,
        page_size=2,
    )
    start = {
        "stage": "v2",
        "window_start": "2026-08-30T12:00:00Z",
        "window_end": "2026-09-03T11:55:00Z",
        "started_at": "2026-09-03T12:00:00Z",
    }

    first = source.fetch_page(start)
    second = source.fetch_page(first.next_state)
    terminal = source.fetch_page(second.next_state)

    assert first.complete is False
    assert first.next_state["after"] == "v2-b"
    assert second.next_state["after"] == "v2-d"
    assert [params.get("mintmdate") for _, params in client.calls] == [
        _millis("2026-08-30T12:00:00Z"),
        _millis("2026-08-30T12:00:00Z"),
        _millis("2026-08-30T12:00:00Z"),
    ]
    assert "after" not in client.calls[0][1]
    assert client.calls[1][1]["after"] == "v2-b"
    assert client.calls[2][1]["after"] == "v2-d"
    assert "count" not in client.calls[1][1]
    assert [r.source_record_id for r in (*first.records, *second.records)] == [
        "v2-a",
        "v2-b",
        "v2-c",
        "v2-d",
    ]
    assert all(r.raw["api_version"] == "v2" for r in (*first.records, *second.records))
    assert terminal.complete is True
    assert terminal.next_state["watermark"] == "2026-09-03T11:55:00Z"
