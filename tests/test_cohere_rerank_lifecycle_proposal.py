from __future__ import annotations

import tomllib
from pathlib import Path

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter


class FixtureClient:
    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        body = b"""
        <html><body><h2>Rerank v2.0</h2><table>
          <thead><tr><th>Shutdown Date</th><th>Deprecated Model</th>
          <th>Deprecated Model Price</th><th>Recommended Replacement</th></tr></thead>
          <tbody>
            <tr><td>2025-04-30</td><td><code>rerank-english-v2.0</code></td>
              <td>$1.00 / 1K searches</td><td><code>rerank-v3.5</code></td></tr>
            <tr><td>2025-04-30</td><td><code>rerank-multilingual-v2.0</code></td>
              <td>$1.00 / 1K searches</td><td><code>rerank-v3.5</code></td></tr>
          </tbody></table></body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_cohere_rerank_deprecation_table_keeps_ids_dates_and_replacements() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/cohere_rerank_lifecycle.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    adapter = create_source(config, client=FixtureClient(), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert len(page.records) == 1
    catalog = page.records[0]
    assert {model.identifiers[0].value for model in catalog.models} == {
        "rerank-english-v2.0",
        "rerank-multilingual-v2.0",
    }
    entries = {entry["identity"]: entry for entry in catalog.raw["entries"]}
    assert entries["rerank-english-v2.0"]["occurrences"][0]["cells"] == [
        "2025-04-30",
        "rerank-english-v2.0",
        "$1.00 / 1K searches",
        "rerank-v3.5",
    ]
