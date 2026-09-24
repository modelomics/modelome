from __future__ import annotations

import tomllib
from pathlib import Path

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter


class FixtureClient:
    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        body = b"""
        <html><body><h2>IBM embedding models</h2><table>
          <thead><tr><th>Model name</th><th>API model ID</th>
          <th>Price (USD/1,000 tokens)</th><th>Maximum input tokens</th>
          <th>Number of dimensions</th><th>More information</th></tr></thead>
          <tbody>
            <tr><td>granite-embedding-278m-multilingual</td>
              <td><code>ibm/granite-embedding-278m-multilingual</code></td>
              <td>$0.0001</td><td>512</td><td>768</td><td>Model card</td></tr>
            <tr><td>slate-125m-english-rtrvr-v2 Deprecated</td>
              <td><code>ibm/slate-125m-english-rtrvr-v2</code></td>
              <td>$0.0001</td><td>512</td><td>768</td><td>Model card</td></tr>
            <tr><td>slate-30m-english-rtrvr-v2 Deprecated</td>
              <td><code>ibm/slate-30m-english-rtrvr-v2</code></td>
              <td>$0.0001</td><td>512</td><td>384</td><td>Model card</td></tr>
            <tr><td>all-minilm-l6-v2 Deprecated</td>
              <td><code>sentence-transformers/all-minilm-l6-v2</code></td>
              <td>$0.0001</td><td>128</td><td>384</td><td>Model card</td></tr>
          </tbody></table></body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_ibm_watsonx_embedding_table_keeps_exact_ids_and_deprecation_labels() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/ibm_watsonx_embedding_models.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    adapter = create_source(config, client=FixtureClient(), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert len(page.records) == 1
    catalog = page.records[0]
    assert {model.identifiers[0].value for model in catalog.models} == {
        "ibm/granite-embedding-278m-multilingual",
        "ibm/slate-125m-english-rtrvr-v2",
        "ibm/slate-30m-english-rtrvr-v2",
    }
    entries = {entry["identity"]: entry for entry in catalog.raw["entries"]}
    assert entries["ibm/slate-125m-english-rtrvr-v2"]["occurrences"][0][
        "cells"
    ][0] == "slate-125m-english-rtrvr-v2 Deprecated"
