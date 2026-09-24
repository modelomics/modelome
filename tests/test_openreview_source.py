from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.normalize import content_hash
from modelome.sources.openreview import OpenReviewSourceAdapter

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
V1_URL = "https://api.openreview.net/notes"
V2_URL = "https://api2.openreview.net/notes"


def millis(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(parsed.timestamp() * 1_000)


class QueueClient:
    def __init__(self, *payloads: Any) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.payloads:
            raise AssertionError(f"unexpected GET {url}")
        body = json.dumps(self.payloads.pop(0)).encode()
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=body,
            url=url,
        )


def response(notes: list[Any], *, count: Any | None = None) -> dict[str, Any]:
    return {"notes": notes, "count": len(notes) if count is None else count}


def v1_note(
    note_id: str = "paper-one",
    *,
    modified: str = "2026-09-02T10:00:00Z",
    content: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    note: dict[str, Any] = {
        "id": note_id,
        "forum": note_id,
        "invitation": "Example.org/2026/Workshop/-/Submission",
        "number": 7,
        "tcdate": millis("2017-06-12T17:57:34Z"),
        "tmdate": millis(modified),
        "pdate": millis("2017-06-12T17:57:34Z"),
        "content": dict(
            content
            or {
                "title": "Attention Across Every Field",
                "abstract": "We introduce a deep neural architecture.",
                "authors": ["Ada Researcher", "Lin Scientist"],
                "authorids": ["~Ada_Researcher1", "~Lin_Scientist1"],
                "pdf": "/pdf/paper-one.pdf",
                "code": "Official code: https://github.com/example/all-fields",
                "doi": "10.5555/EXAMPLE.1",
                "submission_id": "workshop-paper-7",
            }
        ),
        "details": {
            "revisions": [
                {
                    "id": "edit-one",
                    "version": 1,
                    "tcdate": millis("2017-06-12T17:57:34Z"),
                    "tmdate": millis(modified),
                }
            ]
        },
        "externalId": "https://arxiv.org/abs/1706.03762v7",
    }
    note.update(overrides)
    return note


def v2_note(
    note_id: str = "v2-paper",
    *,
    modified: str = "2026-09-02T11:00:00Z",
    content: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    values = content or {
        "title": "A molecular neural system",
        "abstract": "We present a learned architecture for molecules.",
        "authors": ["Bio Author"],
        "pdf": "/pdf/v2-paper.pdf",
        "project_page": "https://research.example/projects/molecules",
    }
    note: dict[str, Any] = {
        "id": note_id,
        "forum": note_id,
        "invitations": ["Example.org/2026/Journal/-/Submission"],
        "version": 3,
        "tcdate": millis("2025-01-02T00:00:00Z"),
        "tmdate": millis(modified),
        "pdate": millis("2025-01-02T00:00:00Z"),
        "content": {key: {"value": value} for key, value in values.items()},
    }
    note.update(overrides)
    return note


def adapter(client: QueueClient, **overrides: Any) -> OpenReviewSourceAdapter:
    values: dict[str, Any] = {
        "client": client,
        "clock": lambda: NOW,
        "consistency_lag_seconds": 300,
    }
    values.update(overrides)
    return OpenReviewSourceAdapter(**values)


def v2_state(
    *,
    start: str = "1970-01-01T00:00:00Z",
    end: str = "2026-09-03T11:55:00Z",
    **overrides: Any,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "stage": "v2",
        "window_start": start,
        "window_end": end,
        "started_at": "2026-09-03T12:00:00Z",
    }
    state.update(overrides)
    return state


def test_scans_both_public_apis_without_venue_or_content_filters() -> None:
    review = {
        "id": "review-one",
        "forum": "paper-one",
        "replyto": "paper-one",
        "tmdate": millis("2026-09-02T10:01:00Z"),
        "content": {"title": "Official review", "rating": "8"},
    }
    client = QueueClient(response([v1_note(), review], count=2), response([], count=0))
    source = adapter(client, page_size=10)

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    assert client.calls[0] == (
        V1_URL,
        {
            "limit": 10,
            "sort": "tmdate:asc",
            "trash": "true",
            "count": "true",
            "details": "revisions",
        },
        {"Accept": "application/json"},
    )
    assert not any(
        key in client.calls[0][1]
        for key in ("venue", "venueid", "domain", "invitation", "content", "query")
    )
    assert first.complete is False
    assert first.next_state["stage"] == "v2"
    assert first.next_state["window_start"] == "1970-01-01T00:00:00Z"
    assert first.next_state["window_end"] == "2026-09-03T11:55:00Z"
    assert len(first.records) == 2

    record = first.records[0]
    assert record.kind is ArtifactKind.PAPER
    assert record.source_record_id == "paper-one"
    assert record.canonical_url == "https://openreview.net/forum?id=paper-one"
    assert record.title == "Attention Across Every Field"
    assert record.published_at == "2017-06-12T17:57:34Z"
    assert record.modified_at == "2026-09-02T10:00:00Z"
    assert Identifier("openreview", "paper-one") in record.identifiers
    assert Identifier("arxiv", "1706.03762") in record.identifiers
    assert Identifier("doi", "10.5555/example.1") in record.identifiers
    assert Identifier("github:repository", "example/all-fields") in record.identifiers
    assert "deep neural architecture" in record.text
    assert record.raw["identity_evidence"]["invitation_ids"] == [
        "Example.org/2026/Workshop/-/Submission"
    ]
    assert record.raw["identity_evidence"]["content_ids"] == [
        {"field": "submission_id", "value": "workshop-paper-7"}
    ]
    assert record.raw["identity_evidence"]["revisions"][0]["id"] == "edit-one"
    relations = {(link.relation, link.url, link.crawl) for link in record.links}
    assert (
        "implementation",
        "https://github.com/example/all-fields",
        True,
    ) in relations
    assert ("full_text", "https://openreview.net/pdf/paper-one.pdf", False) in relations
    review_record = first.records[1]
    assert review_record.kind is ArtifactKind.OTHER
    assert review_record.source_record_id == "review-one"
    assert review_record.canonical_url == (
        "https://openreview.net/forum?id=paper-one&noteId=review-one"
    )
    assert any(
        link.relation == "forum" and link.url == "https://openreview.net/forum?id=paper-one"
        for link in review_record.links
    )

    assert client.calls[1][0] == V2_URL
    assert client.calls[1][1]["mintmdate"] == 0
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-09-03T11:55:00Z"


def test_daily_overlap_rescans_v1_fully_and_bounds_v2_by_modification_time() -> None:
    client = QueueClient(response([], count=0), response([], count=0))
    source = adapter(client, overlap_days=2)

    first = source.fetch_page({"watermark": "2026-09-01T12:00:00Z"})
    second = source.fetch_page(first.next_state)

    assert "mintmdate" not in client.calls[0][1]
    assert client.calls[1][1]["mintmdate"] == millis("2026-08-30T12:00:00Z")
    assert first.next_state["watermark"] == "2026-09-01T12:00:00Z"
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-09-03T11:55:00Z"


def test_after_cursor_resumes_without_offset_and_requires_terminal_probe() -> None:
    client = QueueClient(
        response(
            [
                v1_note("paper-a", modified="2026-09-01T01:00:00Z"),
                v1_note("paper-b", modified="2026-09-01T02:00:00Z"),
            ],
            count=3,
        ),
        {"notes": [v1_note("paper-c", modified="2026-09-01T03:00:00Z")]},
    )
    source = adapter(client, page_size=2)

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state["after"] == "paper-b"
    assert first.next_state["raw_items_seen"] == 2
    assert first.next_state["last_tmdate_ms"] == millis("2026-09-01T02:00:00Z")
    assert first.next_state["seen_after_hashes"] == [content_hash("paper-b")]
    assert "offset" not in client.calls[1][1]
    assert client.calls[1][1]["after"] == "paper-b"
    assert "count" not in client.calls[1][1]
    assert second.complete is False
    assert second.next_state["stage"] == "v2"
    assert [record.source_record_id for record in (*first.records, *second.records)] == [
        "paper-a",
        "paper-b",
        "paper-c",
    ]


def test_client_side_frozen_upper_boundary_stops_an_append_only_scan() -> None:
    client = QueueClient(
        response(
            [
                v1_note("closed", modified="2026-09-03T11:54:00Z"),
                v1_note("future", modified="2026-09-03T11:56:00Z"),
            ],
            count=1_000_000,
        )
    )
    page = adapter(client, page_size=2).fetch_page({})

    assert page.complete is False
    assert page.next_state["stage"] == "v2"
    assert [record.source_record_id for record in page.records] == ["closed"]
    assert "after" not in page.next_state


def test_v2_unwraps_content_and_preserves_project_and_identity_evidence() -> None:
    note = v2_note(
        content={
            "paper_title": "Neural matter fields",
            "paper_abstract": "A deep network for materials.",
            "paper_authors": [{"name": "Mat Author", "id": "~Mat_Author1"}],
            "paper_pdf": "/attachment?id=v2-paper&name=paper_pdf",
            "project_website": "https://materials.example/model?utm_source=openreview",
            "external_ids": ["doi:10.7777/MATTER.2"],
            "arxiv_id": "2602.00002v3",
            "url": "https://github.com/example/material-network",
        },
        externalIds=["https://arxiv.org/abs/2601.00001"],
    )
    page = adapter(QueueClient(response([note], count=1)), page_size=10).fetch_page(v2_state())

    assert page.complete is True
    record = page.records[0]
    assert record.title == "Neural matter fields"
    assert record.raw["authors"] == [{"name": "Mat Author", "id": "~Mat_Author1"}]
    assert Identifier("doi", "10.7777/matter.2") in record.identifiers
    assert Identifier("arxiv", "2601.00001") in record.identifiers
    assert Identifier("arxiv", "2602.00002") in record.identifiers
    assert any(
        link.relation == "project_page"
        and link.url == "https://materials.example/model"
        and link.crawl
        for link in record.links
    )
    assert any(
        link.relation == "full_text"
        and link.url == "https://openreview.net/attachment?id=v2-paper&name=paper_pdf"
        and not link.crawl
        for link in record.links
    )
    assert any(
        link.relation == "implementation"
        and link.url == "https://github.com/example/material-network"
        and link.crawl
        for link in record.links
    )


def test_deleted_notes_become_tombstones_and_withdrawals_remain_evidence() -> None:
    deleted = {
        "id": "deleted-paper",
        "forum": "deleted-paper",
        "tmdate": millis("2026-09-02T02:00:00Z"),
        "ddate": millis("2026-09-02T02:00:00Z"),
        "content": {},
    }
    scheduled = v2_note(
        "withdrawn-paper",
        modified="2026-09-02T03:00:00Z",
        invitations=["Example.org/2026/Journal/-/Withdrawal"],
        ddate=millis("2026-09-04T00:00:00Z"),
        content={
            "title": "A withdrawn but documented system",
            "abstract": "The public record remains evidence.",
            "authors": ["Author"],
            "venue": "Withdrawn Submission",
        },
    )
    page = adapter(QueueClient(response([deleted, scheduled], count=2)), page_size=10).fetch_page(
        v2_state()
    )

    tombstone, withdrawn = page.records
    assert tombstone.deleted is True
    assert tombstone.title == "[deleted OpenReview note] deleted-paper"
    assert tombstone.raw["lifecycle"]["deleted_at_boundary"] is True
    assert withdrawn.deleted is False
    assert withdrawn.raw["lifecycle"]["deleted_at_boundary"] is False
    evidence = withdrawn.raw["lifecycle"]["withdrawal_evidence"]
    assert {item["locator"] for item in evidence} == {
        "$.invitations",
        "$.content.venue",
    }


def test_malformed_note_is_quarantined_at_the_same_page_boundary() -> None:
    malformed = v1_note()
    malformed.pop("id")
    client = QueueClient(response([malformed], count=1))
    page = adapter(client, page_size=10).fetch_page({})

    assert page.complete is False
    assert page.next_state == page.retry_state
    assert page.retry_state == {
        "stage": "v1",
        "window_start": "1970-01-01T00:00:00Z",
        "window_end": "2026-09-03T11:55:00Z",
        "started_at": "2026-09-03T12:00:00Z",
    }
    assert page.issues[0].stage == "source_normalize"
    assert "invalid note.id" in page.issues[0].error


def test_content_schema_drift_is_bounded_and_retried() -> None:
    note = v1_note(content={"title": "Paper", "abstract": "Text", "extra": "value"})
    page = adapter(
        QueueClient(response([note], count=1)),
        page_size=10,
        max_content_fields=2,
    ).fetch_page({})

    assert page.complete is False
    assert page.next_state == page.retry_state
    assert page.issues[0].stage == "source_normalize"
    assert "note.content exceeds 2 fields" in page.issues[0].error


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"notes": {}, "count": 0}, "response.notes must be an array"),
        ({"notes": [], "count": True}, "invalid response.count"),
        ({"notes": [v1_note()], "count": 0}, "count 0 is smaller"),
    ],
)
def test_response_shape_and_count_are_strict(payload: Any, message: str) -> None:
    source = adapter(QueueClient(payload), page_size=10)
    if message == "count 0 is smaller":
        page = source.fetch_page({})
        assert page.issues[0].stage == "source_pagination"
        assert message in page.issues[0].error
    else:
        with pytest.raises(ValueError, match=message):
            source.fetch_page({})


def test_page_size_overrun_and_after_cursor_cycles_restart_the_stream() -> None:
    overrun = adapter(
        QueueClient(
            response(
                [
                    v1_note("a", modified="2026-09-01T01:00:00Z"),
                    v1_note("b", modified="2026-09-01T02:00:00Z"),
                ],
                count=2,
            )
        ),
        page_size=1,
    ).fetch_page({})
    assert overrun.issues[0].stage == "source_pagination"
    assert "above configured page_size" in overrun.issues[0].error

    cycle_state = {
        "stage": "v1",
        "window_start": "1970-01-01T00:00:00Z",
        "window_end": "2026-09-03T11:55:00Z",
        "started_at": "2026-09-03T12:00:00Z",
        "after": "repeat",
        "raw_items_seen": 2,
        "scan_total": 4,
        "last_tmdate_ms": millis("2026-09-01T00:00:00Z"),
        "seen_after_hashes": [content_hash("repeat")],
    }
    cycle = adapter(
        QueueClient(
            response(
                [
                    v1_note("new", modified="2026-09-01T01:00:00Z"),
                    v1_note("repeat", modified="2026-09-01T02:00:00Z"),
                ],
                count=4,
            )
        ),
        page_size=2,
    ).fetch_page(cycle_state)
    assert cycle.issues[0].stage == "source_pagination"
    assert "cursor cycle" in cycle.issues[0].error
    assert cycle.retry_state == {
        "stage": "v1",
        "window_start": "1970-01-01T00:00:00Z",
        "window_end": "2026-09-03T11:55:00Z",
        "started_at": "2026-09-03T12:00:00Z",
    }


def test_timestamp_order_drift_and_v2_lower_bound_violations_restart() -> None:
    resumed = {
        "stage": "v1",
        "window_start": "1970-01-01T00:00:00Z",
        "window_end": "2026-09-03T11:55:00Z",
        "started_at": "2026-09-03T12:00:00Z",
        "after": "paper-b",
        "raw_items_seen": 2,
        "scan_total": 3,
        "last_tmdate_ms": millis("2026-09-02T12:00:00Z"),
        "seen_after_hashes": [content_hash("paper-b")],
    }
    drift = adapter(QueueClient(response([v1_note("older")], count=3)), page_size=10).fetch_page(
        resumed
    )
    assert drift.issues[0].stage == "source_pagination"
    assert "order moved backward" in drift.issues[0].error
    assert "after" not in drift.retry_state

    below = adapter(
        QueueClient(
            response(
                [v2_note(modified="2026-09-01T23:59:59Z")],
                count=1,
            )
        ),
        page_size=10,
    ).fetch_page(v2_state(start="2026-09-02T00:00:00Z"))
    assert below.issues[0].stage == "source_pagination"
    assert "below mintmdate" in below.issues[0].error


def test_checkpoint_and_constructor_limits_are_validated_and_signed() -> None:
    first = adapter(QueueClient(), page_size=10)
    second = adapter(QueueClient(), page_size=11)
    assert first.checkpoint_signature != second.checkpoint_signature
    assert "private/confidential" in first.coverage_limitation

    with pytest.raises(ValueError, match="must not exceed OpenReview's 1000 limit"):
        adapter(QueueClient(), page_size=1_001)
    with pytest.raises(ValueError, match="max_item_bytes must not exceed"):
        adapter(QueueClient(), max_response_bytes=100, max_item_bytes=101)
    with pytest.raises(ValueError, match="missing frozen boundaries"):
        first.fetch_page({"after": "orphan"})
    with pytest.raises(ValueError, match="absent from seen_after_hashes"):
        first.fetch_page(
            v2_state(
                after="cursor",
                raw_items_seen=10,
                scan_total=20,
                last_tmdate_ms=1,
                seen_after_hashes=[],
            )
        )


def test_v2_recovers_arxiv_ids_from_html_urls_in_unrecognized_fields() -> None:
    note = v2_note(
        content={
            "paper_title": "An evaluated language model",
            "paper_abstract": "We train and evaluate a language model.",
            "paper_authors": ["Model Author"],
            "proceedings_url": ("https://arxiv.org/html/2603.01234v2?source=workshop#section-1"),
            "published_doi_url": "https://doi.org/10.7777/MODEL.4?ref=proceedings",
        }
    )
    page = adapter(QueueClient(response([note], count=1))).fetch_page(v2_state())

    record = page.records[0]
    assert Identifier("arxiv", "2603.01234") in record.identifiers
    assert Identifier("doi", "10.7777/model.4") in record.identifiers
    assert any("arxiv.org/html/2603.01234v2" in link.url for link in record.links)


@pytest.mark.parametrize(
    ("stage", "make_note", "checkpoint_field"),
    [
        ("v1", v1_note, "model_checkpoint"),
        ("v2", v2_note, "checkpoint_file"),
    ],
)
def test_file_attachments_in_checkpoint_fields_are_preserved_as_weight_links(
    stage: str, make_note: Any, checkpoint_field: str
) -> None:
    note = make_note(
        content={
            "title": "A released model",
            "abstract": "The paper publishes a checkpoint.",
            "authors": ["Model Author"],
            checkpoint_field: "/attachment?id=model-paper&name=checkpoint",
        }
    )
    state = {} if stage == "v1" else v2_state()
    page = adapter(QueueClient(response([note], count=1))).fetch_page(state)

    checkpoint_links = [
        link
        for link in page.records[0].links
        if link.url == "https://openreview.net/attachment?id=model-paper&name=checkpoint"
    ]
    assert len(checkpoint_links) == 1
    assert checkpoint_links[0].relation == "weights"
    assert checkpoint_links[0].crawl is False


def test_generic_file_field_uses_openreview_attachment_name_for_checkpoint_relation() -> None:
    note = v2_note(
        content={
            "title": "An archived checkpoint",
            "abstract": "The invitation uses a generic file field.",
            "authors": ["Model Author"],
            "artifact": "/attachment?id=generic-paper&name=model_weights",
        }
    )
    page = adapter(QueueClient(response([note], count=1))).fetch_page(v2_state())

    assert any(
        link.url == "https://openreview.net/attachment?id=generic-paper&name=model_weights"
        and link.relation == "weights"
        and not link.crawl
        for link in page.records[0].links
    )


def test_absolute_generic_attachment_url_uses_name_for_checkpoint_relation() -> None:
    note = v2_note(
        content={
            "title": "A released checkpoint",
            "abstract": "The generic field contains an absolute attachment URL.",
            "authors": ["Model Author"],
            "artifact": "https://openreview.net/attachment?id=absolute-paper&name=checkpoint",
        }
    )
    page = adapter(QueueClient(response([note], count=1))).fetch_page(v2_state())

    assert any(
        link.url == "https://openreview.net/attachment?id=absolute-paper&name=checkpoint"
        and link.relation == "weights"
        and not link.crawl
        for link in page.records[0].links
    )


def test_historical_edit_attachment_url_uses_name_for_checkpoint_relation() -> None:
    note = v2_note(
        content={
            "title": "A paper with a versioned checkpoint",
            "abstract": "The invitation uses a generic file field.",
            "authors": ["Model Author"],
            "artifact": (
                "https://openreview.net/notes/edits/attachment"
                "?id=historical-edit&name=model_weights"
            ),
        }
    )
    page = adapter(QueueClient(response([note], count=1))).fetch_page(v2_state())

    assert any(
        link.url
        == "https://openreview.net/notes/edits/attachment?id=historical-edit&name=model_weights"
        and link.relation == "weights"
        and not link.crawl
        for link in page.records[0].links
    )
