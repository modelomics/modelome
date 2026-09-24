from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.normalize import content_hash
from modelome.pipeline import SyncEngine
from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.storage import Database

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
OAI_NAMESPACE = "http://www.openarchives.org/OAI/2.0/"
RAW_NAMESPACE = "http://arxiv.org/OAI/arXivRaw/"


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
        headers={"content-type": "application/xml"},
        body=(FIXTURES / name).read_bytes(),
        url=f"https://fixtures.test/{name}",
    )


def xml_response(value: str) -> HttpResponse:
    return HttpResponse(
        status=200,
        headers={"content-type": "application/xml"},
        body=value.encode(),
        url="https://oaipmh.arxiv.org/oai",
    )


def oai_response(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="{OAI_NAMESPACE}" xmlns:raw="{RAW_NAMESPACE}">
  <responseDate>2026-09-01T12:00:00Z</responseDate>
  <request verb="ListRecords">https://oaipmh.arxiv.org/oai</request>
  {body}
</OAI-PMH>"""


def raw_record(
    arxiv_id: str = "2608.00001",
    *,
    version: str = "v1",
    datestamp: str = "2026-08-31",
) -> str:
    return f"""
<record>
  <header>
    <identifier>oai:arXiv.org:{arxiv_id}</identifier>
    <datestamp>{datestamp}</datestamp>
    <setSpec>cs:cs.LG</setSpec>
  </header>
  <metadata>
    <raw:arXivRaw>
      <raw:id>{arxiv_id}</raw:id>
      <raw:submitter>Example Submitter</raw:submitter>
      <raw:version version="{version}">
        <raw:date>Sun, 30 Aug 2026 12:00:00 GMT</raw:date>
      </raw:version>
      <raw:title>Example paper</raw:title>
      <raw:authors>First Author and Second Author</raw:authors>
      <raw:categories>cs.LG</raw:categories>
      <raw:abstract>We introduce an example architecture.</raw:abstract>
    </raw:arXivRaw>
  </metadata>
</record>"""


def test_oai_source_resumes_token_and_preserves_complete_record_evidence() -> None:
    client = QueuedClient(
        fixture_response("arxiv_oai_page1.xml"),
        fixture_response("arxiv_oai_page2.xml"),
    )
    adapter = ArxivSourceAdapter(
        client=client,
        clock=lambda: NOW,
        initial_lookback_days=7,
        overlap_days=2,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.upstream_count == 3
    assert first.next_state["raw_items_seen"] == 2
    assert first.next_state["scan_total"] == 3
    assert first.next_state["window_start"] == "2026-08-25"
    assert first.next_state["window_end"] == "2026-08-31"
    assert first.next_state["token_expires_at"] == "2026-09-02T12:00:00Z"
    assert client.calls[0][1] == {
        "verb": "ListRecords",
        "metadataPrefix": "arXivRaw",
        "from": "2026-08-25",
        "until": "2026-08-31",
    }
    assert client.calls[1][1] == {
        "verb": "ListRecords",
        "resumptionToken": "opaque-token-page-2==",
    }
    assert client.calls[0][2]["Accept"].startswith("application/xml")

    record = first.records[0]
    assert record.kind is ArtifactKind.PAPER
    assert record.source_record_id == "1706.03762"
    assert record.canonical_url == "https://arxiv.org/abs/1706.03762"
    assert record.title == "Attention Is All You Need"
    assert record.text.startswith("Attention Is All You Need\n\nThe dominant")
    assert record.published_at == "2017-06-12T17:57:34Z"
    assert record.modified_at == "2026-08-30"
    assert record.identifiers == (
        Identifier("arxiv", "1706.03762"),
        Identifier("arxiv:version", "1706.03762v1"),
        Identifier("arxiv:version", "1706.03762v7"),
        Identifier("doi", "10.48550/arxiv.1706.03762"),
    )
    assert record.raw["arxiv_raw"]["authors"].startswith("Ashish Vaswani")
    assert record.raw["arxiv_raw"]["categories_list"] == ["cs.CL", "cs.LG"]
    assert record.raw["arxiv_raw"]["journal-ref"].startswith("Advances")
    assert [item["version"] for item in record.raw["arxiv_raw"]["versions"]] == [
        "v1",
        "v7",
    ]
    assert sum(link.relation == "full_text" for link in record.links) == 2
    assert sum(link.relation == "source_archive" for link in record.links) == 2
    assert any(link.relation == "license" for link in record.links)
    assert any(link.relation == "published_as" for link in record.links)

    old_style = first.records[1]
    assert old_style.source_record_id == "math.GT/0309136"
    assert Identifier("arxiv:version", "math.GT/0309136v1") in old_style.identifiers
    assert old_style.published_at == "2003-09-15T10:00:00Z"

    assert second.complete is True
    assert second.next_state["watermark"] == "2026-08-31"
    deleted = second.records[0]
    assert deleted.source_record_id == "2608.00003"
    assert deleted.raw["deleted"] is True
    assert deleted.raw["oai_header"]["status"] == "deleted"
    assert deleted.deleted is True
    assert deleted.modified_at == "2026-08-31"
    assert deleted.text == ""
    assert deleted.identifiers == (Identifier("arxiv", "2608.00003"),)


def test_oai_daily_window_replays_configured_closed_days() -> None:
    response = oai_response(
        '<error code="noRecordsMatch">No matching records in this date range</error>'
    )
    client = QueuedClient(xml_response(response))
    adapter = ArxivSourceAdapter(
        client=client,
        clock=lambda: NOW,
        overlap_days=2,
    )

    page = adapter.fetch_page({"watermark": "2026-08-31"})

    assert page.complete is True
    assert page.records == ()
    assert page.upstream_count == 0
    assert page.next_state["watermark"] == "2026-08-31"
    assert client.calls[0][1]["from"] == "2026-08-30"
    assert client.calls[0][1]["until"] == "2026-08-31"


def test_oai_accepts_a_fixed_historical_window_without_keyword_filters() -> None:
    response = oai_response(
        f"<ListRecords>{raw_record(datestamp='1995-01-02')}<resumptionToken /></ListRecords>"
    )
    client = QueuedClient(xml_response(response))
    adapter = ArxivSourceAdapter(client=client, clock=lambda: NOW)

    page = adapter.fetch_page(
        {"window_start": "1995-01-01", "window_end": "1995-01-31"}
    )

    assert page.complete is True
    assert client.calls[0][1] == {
        "verb": "ListRecords",
        "metadataPrefix": "arXivRaw",
        "from": "1995-01-01",
        "until": "1995-01-31",
    }
    assert "set" not in client.calls[0][1]


def test_oai_quarantines_malformed_record_and_replays_page(tmp_path: Path) -> None:
    malformed = oai_response(
        f"<ListRecords>{raw_record(version='v0')}"
        '<resumptionToken completeListSize="1" cursor="0" /></ListRecords>'
    )
    corrected = oai_response(
        f"<ListRecords>{raw_record()}"
        '<resumptionToken completeListSize="1" cursor="0" /></ListRecords>'
    )
    client = QueuedClient(xml_response(malformed), xml_response(corrected))
    current_time = [NOW]
    source = ArxivSourceAdapter(
        client=client,
        clock=lambda: current_time[0],
        initial_lookback_days=1,
    )
    database = Database(tmp_path / "store")
    database.initialize()
    engine = SyncEngine(database, {source.name: source})

    failed = engine.sync()[0]
    held_state = database.get_source_state(source.name)
    current_time[0] = NOW + timedelta(days=1)
    retried = engine.sync()[0]

    assert failed.status == "failed"
    assert held_state["window_start"] == "2026-08-31"
    assert held_state["window_end"] == "2026-08-31"
    assert held_state["raw_items_seen"] == 0
    assert retried.status == "complete"
    assert client.calls[1][1] == client.calls[0][1]
    dead_letter = database.list_dead_letters("arxiv")[0]
    assert dead_letter["stage"] == "source_normalize"
    assert "invalid arXiv version" in dead_letter["error"]


def test_oai_detects_truncation_before_complete_list_size() -> None:
    response = oai_response(
        f"<ListRecords>{raw_record()}"
        '<resumptionToken completeListSize="2" cursor="0" /></ListRecords>'
    )
    adapter = ArxivSourceAdapter(
        client=QueuedClient(xml_response(response)),
        clock=lambda: NOW,
        initial_lookback_days=1,
    )

    page = adapter.fetch_page({})

    assert page.complete is False
    assert page.next_state == page.retry_state
    assert any("before completeListSize 2" in issue.error for issue in page.issues)


def test_oai_detects_repeated_resumption_token() -> None:
    repeated_token = "same-token"
    response = oai_response(
        f"<ListRecords>{raw_record('2608.00002')}"
        '<resumptionToken completeListSize="4" cursor="2">same-token</resumptionToken>'
        "</ListRecords>"
    )
    adapter = ArxivSourceAdapter(
        client=QueuedClient(xml_response(response)),
        clock=lambda: NOW,
    )
    state = {
        "window_start": "2026-08-31",
        "window_end": "2026-08-31",
        "resumption_token": repeated_token,
        "seen_token_hashes": [content_hash(repeated_token)],
        "raw_items_seen": 2,
        "scan_total": 4,
    }

    page = adapter.fetch_page(state)

    assert page.complete is False
    assert page.next_state == page.retry_state
    assert any("resumption token repeated" in issue.error for issue in page.issues)


def test_oai_bad_resumption_token_restarts_same_frozen_window() -> None:
    response = oai_response(
        '<error code="badResumptionToken">The token has expired</error>'
    )
    adapter = ArxivSourceAdapter(
        client=QueuedClient(xml_response(response)),
        clock=lambda: NOW,
    )
    state = {
        "window_start": "2026-08-30",
        "window_end": "2026-08-31",
        "resumption_token": "expired-token",
        "raw_items_seen": 1000,
        "scan_total": 1200,
    }

    page = adapter.fetch_page(state)

    assert page.complete is False
    assert page.next_state == page.retry_state
    assert page.retry_state["window_start"] == "2026-08-30"
    assert page.retry_state["window_end"] == "2026-08-31"
    assert page.retry_state["raw_items_seen"] == 0
    assert "resumption_token" not in page.retry_state
    assert "scan_total" not in page.retry_state
    assert "restarting the frozen window" in page.issues[0].error


def test_oai_rejects_unsafe_or_malformed_xml() -> None:
    unsafe = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY y "boom">]><x>&y;</x>'
    client = QueuedClient(
        HttpResponse(200, {}, unsafe, "https://oaipmh.arxiv.org/oai")
    )
    adapter = ArxivSourceAdapter(client=client, clock=lambda: NOW)

    with pytest.raises(ValueError, match="unsafe XML"):
        adapter.fetch_page({})


def test_oai_requires_closed_complete_frozen_windows() -> None:
    adapter = ArxivSourceAdapter(client=QueuedClient(), clock=lambda: NOW)

    with pytest.raises(ValueError, match="requires both boundaries"):
        adapter.fetch_page({"window_start": "2026-08-30"})
    with pytest.raises(ValueError, match="closed UTC day"):
        adapter.fetch_page(
            {"window_start": "2026-08-31", "window_end": "2026-09-01"}
        )
    with pytest.raises(ValueError, match="missing its frozen window"):
        adapter.fetch_page({"resumption_token": "opaque"})


def test_oai_rejects_metadata_id_mismatch_as_record_issue() -> None:
    mismatched = raw_record().replace(
        "<raw:id>2608.00001</raw:id>",
        "<raw:id>2608.99999</raw:id>",
    )
    response = oai_response(
        f"<ListRecords>{mismatched}<resumptionToken /></ListRecords>"
    )
    adapter = ArxivSourceAdapter(
        client=QueuedClient(xml_response(response)),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.records == ()
    assert len(page.issues) == 1
    assert "does not match metadata ID" in page.issues[0].error
    assert page.retry_state["raw_items_seen"] == 0
