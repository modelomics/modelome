from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.hal import HalSourceAdapter

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
        url="https://api.archives-ouvertes.fr/oai/hal/",
    )


def record(document_id: str = "hal-01234567v2") -> str:
    return f'''<record>
  <header>
    <identifier>oai:HAL:{document_id}</identifier>
    <datestamp>2026-08-31</datestamp>
    <setSpec>type:ART</setSpec>
    <setSpec>subject:info</setSpec>
  </header>
  <metadata>
    <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>Reliable neural evidence</dc:title>
      <dc:creator>Example Researcher</dc:creator>
      <dc:description>Code: https://github.com/example/reliable-neural-evidence</dc:description>
      <dc:date>2026-08-30</dc:date>
      <dc:date>2024</dc:date>
      <dc:identifier>{document_id.removesuffix('v2')}</dc:identifier>
      <dc:identifier>https://hal.science/{document_id}</dc:identifier>
      <dc:identifier>10.1234/Example.HAL</dc:identifier>
      <dc:subject>Machine learning</dc:subject>
      <dc:type>Journal articles</dc:type>
      <dc:rights>CC-BY-4.0</dc:rights>
    </oai_dc:dc>
  </metadata>
</record>'''


def test_hal_harvests_full_history_with_date_granularity_and_resumption_tokens() -> None:
    client = QueuedClient(
        response(record(), token="opaque-page-two"),
        response(record("hal-07654321v1")),
    )
    adapter = HalSourceAdapter(client=client, clock=lambda: NOW)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert client.calls[0][0] == "https://api.archives-ouvertes.fr/oai/hal"
    assert client.calls[0][1] == {
        "verb": "ListRecords",
        "metadataPrefix": "oai_dc",
        "from": "2002-09-23",
        "until": "2026-08-31",
    }
    assert client.calls[1][1] == {"verb": "ListRecords", "resumptionToken": "opaque-page-two"}
    assert first.complete is False
    assert first.next_state["window_start"] == "2002-09-23"
    assert first.next_state["resumption_token"] == "opaque-page-two"
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-08-31"

    source_record = first.records[0]
    assert source_record.source_record_id == "hal:oai:HAL:hal-01234567v2"
    assert source_record.canonical_url == "https://hal.science/hal-01234567v2"
    assert source_record.published_at == "2026-08-30T00:00:00Z"
    assert source_record.identifiers == (
        Identifier("oai:hal", "oai:HAL:hal-01234567v2"),
        Identifier("hal:version", "hal-01234567v2"),
        Identifier("hal:document", "hal-01234567"),
        Identifier("doi", "10.1234/example.hal"),
    )
    assert source_record.raw["sets"] == ["type:ART", "subject:info"]
    assert {link.relation for link in source_record.links} == {
        "repository_record",
        "doi",
    }


def test_hal_rejects_invalid_identifiers_without_advancing_its_frozen_window() -> None:
    client = QueuedClient(response(record("unsafe/document")))
    adapter = HalSourceAdapter(client=client, clock=lambda: NOW)

    page = adapter.fetch_page({})

    assert page.records == ()
    assert page.complete is False
    assert page.next_state == page.retry_state
    assert page.issues[0].stage == "source_normalize"
    assert page.next_state["window_start"] == "2002-09-23"
