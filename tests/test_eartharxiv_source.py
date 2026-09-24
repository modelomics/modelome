from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.eartharxiv import EarthArxivSourceAdapter

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


def response(*records: str, token: str | None = None) -> HttpResponse:
    next_token = f"<resumptionToken>{token}</resumptionToken>" if token else ""
    body = f'''<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>{''.join(records)}{next_token}</ListRecords>
</OAI-PMH>'''
    return HttpResponse(
        status=200,
        headers={"content-type": "application/xml"},
        body=body.encode(),
        url="https://eartharxiv.org/api/oai/",
    )


def record(object_id: int, *, title: str = "Neural ocean dynamics") -> str:
    return f'''<record>
  <header>
    <identifier>oai:EA:id:{object_id}</identifier>
    <datestamp>2026-08-31T10:00:00Z</datestamp>
  </header>
  <metadata>
    <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>{title}</dc:title>
      <dc:creator>Example Researcher</dc:creator>
      <dc:description>Model code is available at https://github.com/example/ocean.</dc:description>
      <dc:date>2026-08-30T09:00:00Z</dc:date>
      <dc:identifier>10.31223/X58Z29</dc:identifier>
      <dc:identifier>https://eartharxiv.org/repository/object/{object_id}/download/1/</dc:identifier>
      <dc:subject>Machine learning</dc:subject>
      <dc:rights>https://creativecommons.org/licenses/by/4.0/</dc:rights>
    </oai_dc:dc>
  </metadata>
</record>'''


def test_source_pages_the_first_party_oai_feed_and_preserves_dc_evidence() -> None:
    client = QueuedClient(response(record(14734), token="token-two"), response(record(14736)))
    adapter = EarthArxivSourceAdapter(client=client, clock=lambda: NOW)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state["resumption_token"] == "token-two"
    assert first.next_state["window_start"] == "2026-08-25"
    assert first.next_state["window_end"] == "2026-08-31"
    assert client.calls[0][0] == "https://eartharxiv.org/api/oai"
    assert client.calls[0][1] == {
        "verb": "ListRecords",
        "metadataPrefix": "oai_dc",
        "from": "2026-08-25T00:00:00Z",
        "until": "2026-08-31T23:59:59.999999Z",
    }
    assert client.calls[1][1] == {"verb": "ListRecords", "resumptionToken": "token-two"}
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-08-31"

    source_record = first.records[0]
    assert source_record.source_record_id == "eartharxiv:oai:EA:id:14734"
    assert source_record.canonical_url == "https://eartharxiv.org/repository/object/14734"
    assert source_record.identifiers == (
        Identifier("oai:eartharxiv", "oai:EA:id:14734"),
        Identifier("eartharxiv:object", "14734"),
        Identifier("doi", "10.31223/x58z29"),
    )
    assert source_record.raw["subjects"] == ["Machine learning"]
    assert {link.relation for link in source_record.links} == {"preprint", "doi", "full_text"}


def test_source_treats_no_records_match_as_a_complete_empty_window() -> None:
    body = b'''<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <error code="noRecordsMatch">No records match</error>
</OAI-PMH>'''
    client = QueuedClient(
        HttpResponse(200, {"content-type": "application/xml"}, body, "https://eartharxiv.org/api/oai/")
    )
    adapter = EarthArxivSourceAdapter(client=client, clock=lambda: NOW)

    page = adapter.fetch_page({"watermark": "2026-08-31"})

    assert page.complete is True
    assert page.next_state["watermark"] == "2026-08-31"
