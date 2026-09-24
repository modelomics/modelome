from __future__ import annotations

import tomllib
from pathlib import Path

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter


class FixtureClient:
    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        body = b"""
        <html><body><h2>Use Mistral embedding presets</h2>
          <table><thead><tr><th>Preset</th><th>Full model name</th>
          <th>Dimensions</th><th>Distance metric</th></tr></thead>
          <tbody>
            <tr><td><code>MISTRAL_EMBED_DIM_1024</code></td>
              <td><code>mistral-embed</code></td><td>1024</td><td>COSINE</td></tr>
            <tr><td><code>MISTRAL_EMBED_DIM_256</code></td>
              <td><code>mistral-embed-dim256-2510</code></td><td>256</td><td>COSINE</td></tr>
            <tr><td><code>MISTRAL_EMBED_DIM_128</code></td>
              <td><code>mistral-embed-dim128-2510</code></td><td>128</td><td>COSINE</td></tr>
          </tbody></table></body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_mistral_embedding_presets_capture_exact_ids_and_dimensions() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/mistral_embedding_model_presets.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    adapter = create_source(config, client=FixtureClient(), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert len(page.records) == 1
    catalog = page.records[0]
    assert {model.identifiers[0].value for model in catalog.models} == {
        "mistral-embed",
        "mistral-embed-dim256-2510",
        "mistral-embed-dim128-2510",
    }
    entries = {entry["identity"]: entry for entry in catalog.raw["entries"]}
    assert entries["mistral-embed-dim256-2510"]["occurrences"][0]["cells"] == [
        "MISTRAL_EMBED_DIM_256",
        "mistral-embed-dim256-2510",
        "256",
        "COSINE",
    ]
