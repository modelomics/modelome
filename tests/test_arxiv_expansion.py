from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.arxiv import ArxivSourceAdapter

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
OAI = "http://www.openarchives.org/OAI/2.0/"
RAW = "http://arxiv.org/OAI/arXivRaw/"


class OnePageClient:
    def __init__(self, body: str) -> None:
        self.body = body.encode()
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def get(self, url: str, *, params: Mapping[str, Any] | None = None, **_: Any) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        return HttpResponse(
            status=200,
            headers={"content-type": "application/xml"},
            body=self.body,
            url=url,
        )


def _record(paper_id: str, category: str, title: str, abstract: str) -> str:
    return f"""
    <record>
      <header>
        <identifier>oai:arXiv.org:{paper_id}</identifier>
        <datestamp>2026-08-31</datestamp>
        <setSpec>{category}</setSpec>
      </header>
      <metadata><raw:arXivRaw>
        <raw:id>{paper_id}</raw:id>
        <raw:version version="v1"><raw:date>Mon, 31 Aug 2026 09:00:00 GMT</raw:date></raw:version>
        <raw:title>{title}</raw:title>
        <raw:authors>Example Author</raw:authors>
        <raw:categories>{category}</raw:categories>
        <raw:abstract>{abstract}</raw:abstract>
      </raw:arXivRaw></metadata>
    </record>"""


def test_oai_discovery_keeps_cross_category_papers_and_abstract_evidence() -> None:
    records = (
        _record(
            "2608.10001",
            "astro-ph.GA",
            "A neural model for galaxy morphology",
            "We introduce a transformer model to classify galaxy images.",
        )
        + _record(
            "2608.10002",
            "econ.GN",
            "Forecasting with a learned representation",
            "Our model estimates economic activity from sparse observations.",
        )
        + _record(
            "2608.10003",
            "cs.LG",
            "A new optimization method",
            "We prove convergence for a stochastic optimizer.",
        )
    )
    body = f"""<?xml version="1.0"?>
    <OAI-PMH xmlns="{OAI}" xmlns:raw="{RAW}">
      <ListRecords>{records}
        <resumptionToken completeListSize="3" cursor="0">next-page</resumptionToken>
      </ListRecords>
    </OAI-PMH>"""
    client = OnePageClient(body)
    adapter = ArxivSourceAdapter(client=client, clock=lambda: NOW)

    page = adapter.fetch_page({"window_start": "2026-08-31", "window_end": "2026-08-31"})

    assert page.complete is False
    assert page.upstream_count == 3
    assert [record.source_record_id for record in page.records] == [
        "2608.10001",
        "2608.10002",
        "2608.10003",
    ]
    assert client.calls[0][1] == {
        "verb": "ListRecords",
        "metadataPrefix": "arXivRaw",
        "from": "2026-08-31",
        "until": "2026-08-31",
    }
    assert all("set" not in params and "q" not in params for _, params in client.calls)

    astronomy, economics, machine_learning = page.records
    assert astronomy.raw["oai_header"]["set_specs"] == ["astro-ph.GA"]
    assert astronomy.raw["arxiv_raw"]["categories_list"] == ["astro-ph.GA"]
    assert astronomy.text == (
        "A neural model for galaxy morphology\n\n"
        "We introduce a transformer model to classify galaxy images."
    )
    assert economics.raw["oai_header"]["set_specs"] == ["econ.GN"]
    assert economics.raw["arxiv_raw"]["abstract"] == (
        "Our model estimates economic activity from sparse observations."
    )
    assert machine_learning.raw["arxiv_raw"]["abstract"] == (
        "We prove convergence for a stochastic optimizer."
    )
