from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.eartharxiv import EarthArxivSourceAdapter


class Client:
    def __init__(self, body: str) -> None:
        self.body = body.encode()
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params: dict[str, str] | None = None, **_: Any) -> HttpResponse:
        self.calls.append(dict(params or {}))
        return HttpResponse(200, {"content-type": "application/xml"}, self.body, url)


def test_eartharxiv_preserves_dc_relation_targets_as_references() -> None:
    body = """<?xml version="1.0"?>
    <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
      <ListRecords><record>
        <header>
          <identifier>oai:EA:id:14734</identifier>
          <datestamp>2026-08-31T10:00:00Z</datestamp>
        </header>
        <metadata><oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
          xmlns:dc="http://purl.org/dc/elements/1.1/">
          <dc:title>Neural ocean dynamics</dc:title>
          <dc:description>We introduce OceanNet, a neural model for forecasts.</dc:description>
          <dc:relation>doi:10.1234/journal.paper</dc:relation>
          <dc:relation>https://github.com/example/oceannet</dc:relation>
          <dc:relation>unresolved related work description</dc:relation>
        </oai_dc:dc></metadata>
      </record></ListRecords>
    </OAI-PMH>"""
    client = Client(body)
    adapter = EarthArxivSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
    )

    page = adapter.fetch_page({"window_start": "2026-08-31", "window_end": "2026-08-31"})

    assert page.complete is True
    assert len(page.records) == 1
    record = page.records[0]
    assert record.raw["relations"] == [
        "doi:10.1234/journal.paper",
        "https://github.com/example/oceannet",
        "unresolved related work description",
    ]
    assert [
        (link.url, link.relation, link.locator)
        for link in record.links
        if link.locator and link.locator.startswith("dc.relation[")
    ] == [
        ("https://doi.org/10.1234/journal.paper", "references", "dc.relation[0]"),
        ("https://github.com/example/oceannet", "references", "dc.relation[1]"),
    ]
    assert not any(identifier.namespace == "doi" for identifier in record.identifiers)
