from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/llm_docs_wave2.toml"


class _FixtureClient:
    def __init__(self, url: str, body: str) -> None:
        self.url = url
        self.body = body

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert url == self.url
        assert params is None
        assert "text/html" in (headers or {}).get("Accept", "")
        return HttpResponse(200, {"Content-Type": "text/html"}, self.body.encode(), url)


def _adapter(source: dict[str, Any], body: str) -> HtmlCatalogSourceAdapter:
    return HtmlCatalogSourceAdapter(
        name=source["name"],
        url=source["url"],
        provider_namespace=source["provider_namespace"],
        rules=source["rules"],
        artifact_kind=source["artifact_kind"],
        model_status=source["model_status"],
        client=_FixtureClient(source["url"], body),
    )


def test_cerebras_public_catalog_uses_provider_declared_id_column() -> None:
    sources = tomllib.loads(_PROPOSAL.read_text())["source"]
    source = next(item for item in sources if item["name"].startswith("cerebras-"))
    body = """<table>
      <tr><th>Model Name</th><th>Model ID</th><th>Parameters</th></tr>
      <tr><td>OpenAI GPT OSS</td><td><code>gpt-oss-120b</code></td><td>120 billion</td></tr>
      <tr><td>Qwen 3.8 27B</td><td><code>qwen-3.8-27b</code></td><td>27 billion</td></tr>
    </table>
    <table><tr><th>Other</th><th>Model</th></tr><tr><td>example</td><td>ignored</td></tr></table>"""

    page = _adapter(source, body).fetch_page({})

    models = [model for record in page.records for model in record.models]
    assert [model.name for model in models] == ["gpt-oss-120b", "qwen-3.8-27b"]
    assert {identifier.value for model in models for identifier in model.identifiers} == {
        "gpt-oss-120b",
        "qwen-3.8-27b",
    }


def test_perplexity_hosted_catalog_filters_exact_hosted_router_ids() -> None:
    sources = tomllib.loads(_PROPOSAL.read_text())["source"]
    source = next(item for item in sources if item["name"].startswith("perplexity-"))
    body = """<table>
      <tr><th>Model</th><th>Input ($/1M)</th><th>Output ($/1M)</th>
      <th>Cache read ($/1M)</th><th>Docs</th></tr>
      <tr><td><code>perplexity/kimi-k3</code></td><td>3.00</td><td>15.00</td><td>0.30</td><td></td></tr>
      <tr><td><code>perplexity/glm-5.3</code></td><td>1.40</td><td>4.40</td><td>0.26</td><td></td></tr>
      <tr><td><code>openai/gpt-example</code></td><td>1</td><td>2</td><td>0</td><td></td></tr>
    </table>"""

    page = _adapter(source, body).fetch_page({})

    models = [model for record in page.records for model in record.models]
    assert [model.name for model in models] == ["perplexity/kimi-k3", "perplexity/glm-5.3"]
    assert all(
        identifier.namespace == "perplexity:hosted-model"
        for model in models
        for identifier in model.identifiers
    )
