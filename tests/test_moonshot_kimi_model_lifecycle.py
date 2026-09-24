from __future__ import annotations

import tomllib
from pathlib import Path

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter


class _FixtureClient:
    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        body = b"""
        <html><body>
          <h2>Multi-modal Model</h2>
          <table><thead><tr><th>Model Name</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><code>kimi-k3</code></td><td>Current flagship model</td></tr>
            <tr><td><code>kimi-k2.7-code</code></td><td>Coding model</td></tr>
            <tr><td><code>kimi-k2.7-code-highspeed</code></td><td>High-speed coding model</td></tr>
            <tr><td><code>kimi-k2.6</code></td><td>Multimodal model</td></tr>
          </tbody></table>
          <h2>Deprecated Models</h2>
          <table><thead><tr><th>Model Name</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><code>kimi-k2.5</code></td><td>Deprecated</td></tr>
            <tr><td><code>moonshot-v1-8k</code></td><td>Deprecated</td></tr>
            <tr><td><code>moonshot-v1-32k-vision-preview</code></td><td>Deprecated</td></tr>
            <tr><td><code>kimi-k2-0905-preview</code></td><td>Deprecated</td></tr>
          </tbody></table>
          <p><code>kimi-latest</code> was discontinued.</p>
          <p><code>kimi-thinking-preview</code> was discontinued.</p>
          <p><code>invented-model</code> is not a listed API model.</p>
        </body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_moonshot_model_list_captures_exact_current_and_deprecated_ids() -> None:
    proposal_path = (
        Path(__file__).parents[1] / "config/proposals/moonshot_kimi_model_lifecycle.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    adapter = create_source(config, client=_FixtureClient(), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})

    values = {
        identifier.value for model in page.records[0].models for identifier in model.identifiers
    }
    assert values == {
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
        "kimi-k2.6",
        "kimi-k2.5",
        "moonshot-v1-8k",
        "moonshot-v1-32k-vision-preview",
        "kimi-k2-0905-preview",
        "kimi-latest",
        "kimi-thinking-preview",
    }
    assert all(
        identifier.namespace == "moonshot:model"
        for model in page.records[0].models
        for identifier in model.identifiers
    )
