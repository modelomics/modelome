from __future__ import annotations

import tomllib
from pathlib import Path

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter


class FixtureClient:
    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        body = b"""
        <html><body><h2>Deprecated &amp; retired models</h2>
          <table><thead><tr><th>Model</th><th>Version</th><th>API</th></tr></thead>
          <tbody>
            <tr><td><a href="/models/mistral-medium-3-1">Mistral Medium 3.1</a></td>
              <td><code>25.08</code></td><td><code>mistral-medium-2508</code></td></tr>
            <tr><td><a href="/models/mistral-small-3-2">Mistral Small 3.2</a></td>
              <td><code>25.06</code></td><td><code>mistral-small-2506</code></td></tr>
            <tr><td>Mathstral 7B</td><td><code>0.1</code></td><td></td></tr>
          </tbody></table>
        </body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_mistral_lifecycle_table_preserves_exact_api_ids_and_versions() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/mistral_model_lifecycle.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    adapter = create_source(config, client=FixtureClient(), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert len(page.records) == 1
    catalog = page.records[0]
    assert {model.identifiers[0].value for model in catalog.models} == {
        "Mistral Medium 3.1",
        "Mistral Small 3.2",
        "Mathstral 7B",
    }
    entries = {entry["identity"]: entry for entry in catalog.raw["entries"]}
    assert entries["Mistral Medium 3.1"]["occurrences"][0]["cells"] == [
        "Mistral Medium 3.1",
        "25.08",
        "mistral-medium-2508",
    ]
