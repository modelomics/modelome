from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.plos import PlosSourceAdapter

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


def response(*documents: Mapping[str, Any], total: int, start: int) -> HttpResponse:
    payload = {"response": {"numFound": total, "start": start, "docs": documents}}
    return HttpResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=json.dumps(payload).encode(),
        url="https://api.plos.org/search",
    )


def document(
    suffix: str,
    *,
    published_at: str = "2003-08-18T00:00:00Z",
) -> Mapping[str, Any]:
    return {
        "id": f"10.1371/journal.pone.{suffix}",
        "title_display": "A <i>reliable</i> neural article",
        "abstract": ["Code: https://github.com/example/reliable-neural-evidence"],
        "publication_date": published_at,
        "author": ["Example Researcher"],
        "article_type": "Research Article",
        "subject": ["Machine learning"],
        "journal": "PLOS ONE",
        "eissn": "1932-6203",
        "copyright": "CC-BY-4.0",
    }


def test_plos_scans_all_journals_in_a_frozen_date_window_and_resumes_offsets() -> None:
    client = QueuedClient(
        response(document("0000001"), document("0000002"), total=3, start=0),
        response(document("0000003"), total=3, start=2),
    )
    adapter = PlosSourceAdapter(client=client, clock=lambda: NOW, page_size=2)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert client.calls[0][0] == "https://api.plos.org/search"
    assert client.calls[0][1] == {
        "q": "publication_date:[2003-08-18T00:00:00Z TO 2026-08-31T23:59:59Z]",
        "fq": "doc_type:full",
        "fl": (
            "id,title_display,abstract,publication_date,author,article_type,subject,journal,"
            "eissn,pissn,copyright"
        ),
        "sort": "publication_date asc,id asc",
        "rows": "2",
        "start": "0",
        "wt": "json",
    }
    assert client.calls[1][1]["start"] == "2"
    assert first.complete is False
    assert first.next_state["offset"] == 2
    assert first.next_state["scan_total"] == 3
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-08-31"

    source_record = first.records[0]
    assert source_record.source_record_id == "plos:10.1371/journal.pone.0000001"
    assert source_record.title == "A reliable neural article"
    assert source_record.canonical_url == "https://doi.org/10.1371/journal.pone.0000001"
    assert source_record.identifiers == (
        Identifier("doi", "10.1371/journal.pone.0000001"),
        Identifier("plos:document", "10.1371/journal.pone.0000001"),
    )
    assert {link.relation for link in source_record.links} == {
        "doi",
        "publisher_article",
        "full_text",
    }
    assert all(link.crawl is False for link in source_record.links)


def test_plos_restarts_a_frozen_window_when_its_offset_total_changes() -> None:
    client = QueuedClient(response(document("0000003"), total=4, start=2))
    adapter = PlosSourceAdapter(client=client, clock=lambda: NOW, page_size=2)

    page = adapter.fetch_page(
        {
            "window_start": "2003-08-18",
            "window_end": "2026-08-31",
            "offset": 2,
            "scan_total": 3,
        }
    )

    assert page.records == ()
    assert page.complete is False
    assert page.next_state == page.retry_state
    assert page.next_state["offset"] == 0
    assert "scan_total" not in page.next_state
    assert page.issues[0].stage == "source_pagination"


def test_plos_retains_structural_issue_images_without_mislabeling_them_as_papers() -> None:
    image = dict(document("0000004"))
    image["id"] = "10.1371/image.pone.v01.i01"
    image["article_type"] = "Issue Image"
    client = QueuedClient(response(image, total=1, start=0))

    page = PlosSourceAdapter(client=client, clock=lambda: NOW).fetch_page(
        {"window_start": "2003-08-18", "window_end": "2003-08-18"}
    )

    assert page.complete is True
    assert page.records[0].kind is ArtifactKind.OTHER
    assert {link.relation for link in page.records[0].links} == {"doi"}
