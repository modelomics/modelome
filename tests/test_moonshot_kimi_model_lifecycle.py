from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/moonshot_kimi_model_lifecycle.toml"


class _FixtureClient:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        body = b"""
        <html><body>
          <h2>Multi-modal Model</h2>
          <table><thead><tr><th>Model Name</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><code>kimi-k3</code></td><td>Currently available</td></tr>
            <tr><td><code>kimi-k2.7-code</code></td><td>Currently available</td></tr>
            <tr><td><code>kimi-k2.7-code-highspeed</code></td><td>Currently available</td></tr>
            <tr><td><code>kimi-k2.6</code></td><td>Currently available</td></tr>
          </tbody></table>
          <h2>Deprecated Models</h2>
          <table><thead><tr><th>Model Name</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><code>kimi-k2.5</code></td><td>Deprecated</td></tr>
            <tr><td><code>moonshot-v1-8k</code></td><td>Deprecated</td></tr>
            <tr><td><code>moonshot-v1-32k</code></td><td>Deprecated</td></tr>
            <tr><td><code>moonshot-v1-128k</code></td><td>Deprecated</td></tr>
            <tr><td><code>moonshot-v1-auto</code></td><td>Deprecated</td></tr>
            <tr><td><code>moonshot-v1-8k-vision-preview</code></td><td>Deprecated</td></tr>
            <tr><td><code>moonshot-v1-32k-vision-preview</code></td><td>Deprecated</td></tr>
            <tr><td><code>moonshot-v1-128k-vision-preview</code></td><td>Deprecated</td></tr>
            <tr><td><code>kimi-k2-0905-preview</code></td><td>Deprecated</td></tr>
            <tr><td><code>kimi-k2-0711-preview</code></td><td>Deprecated</td></tr>
            <tr><td><code>kimi-k2-turbo-preview</code></td><td>Deprecated</td></tr>
            <tr><td><code>kimi-k2-thinking</code></td><td>Deprecated</td></tr>
            <tr><td><code>kimi-k2-thinking-turbo</code></td><td>Deprecated</td></tr>
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


def test_kimi_lifecycle_cohorts_preserve_provider_ids_and_dates() -> None:
    with _PROPOSAL.open("rb") as handle:
        sources = tomllib.load(handle)["source"]

    expected = {
        "moonshot-kimi-current-model-documentation": {
            "kimi-k3",
            "kimi-k2.7-code",
            "kimi-k2.7-code-highspeed",
            "kimi-k2.6",
        },
        "moonshot-kimi-retired-2026-08-31": {
            "kimi-k2.5",
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
            "moonshot-v1-auto",
            "moonshot-v1-8k-vision-preview",
            "moonshot-v1-32k-vision-preview",
            "moonshot-v1-128k-vision-preview",
        },
        "moonshot-kimi-retired-2026-05-25": {
            "kimi-k2-0905-preview",
            "kimi-k2-0711-preview",
            "kimi-k2-turbo-preview",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
        },
        "moonshot-kimi-retired-2026-01-28": {"kimi-latest"},
        "moonshot-kimi-retired-2025-11-11": {"kimi-thinking-preview"},
    }
    assert {source["name"] for source in sources} == set(expected)

    for source in sources:
        adapter = create_source(source, client=_FixtureClient(), environ={})
        assert isinstance(adapter, HtmlCatalogSourceAdapter)
        page = adapter.fetch_page({})
        actual = {
            identifier.value
            for record in page.records
            for model in record.models
            for identifier in model.identifiers
        }
        assert actual == expected[source["name"]]
        assert all(
            identifier.namespace == "moonshot:model"
            for record in page.records
            for model in record.models
            for identifier in model.identifiers
        )

    for source in sources:
        dates = [
            tag.removeprefix("retired-on:")
            for tag in source["entry_tags"]
            if tag.startswith("retired-on:")
        ]
        if dates:
            assert dates == [source["name"].removeprefix("moonshot-kimi-retired-")]
