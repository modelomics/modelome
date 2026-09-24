from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/voyage_model_documentation.toml"


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


def test_voyage_embedding_catalog_preserves_exact_api_model_names() -> None:
    sources = tomllib.loads(_PROPOSAL.read_text())["source"]
    source = next(item for item in sources if "embedding" in item["name"])
    body = """<table>
      <tr><th>Model</th><th>Context Length (tokens)</th>
      <th>Embedding Dimension</th><th>Description</th></tr>
      <tr><td><code>voyage-4-large</code></td><td>32,000</td><td>1024</td><td>Current</td></tr>
      <tr><td><code>voyage-3.5-lite</code></td><td>32,000</td><td>512</td><td>Earlier</td></tr>
      <tr><td><code>voyage-01</code></td><td>4,000</td><td>1024</td><td>Deprecated</td></tr>
    </table>
    <table><tr><th>Other</th><th>Column</th></tr>
      <tr><td>ignore</td><td>unrelated</td></tr></table>"""

    page = _adapter(source, body).fetch_page({})

    models = [model for record in page.records for model in record.models]
    assert [model.name for model in models] == [
        "voyage-4-large",
        "voyage-3.5-lite",
        "voyage-01",
    ]
    assert {identifier.value for model in models for identifier in model.identifiers} == {
        "voyage-4-large",
        "voyage-3.5-lite",
        "voyage-01",
    }


def test_voyage_reranker_catalog_strips_preview_label_from_exact_ids() -> None:
    sources = tomllib.loads(_PROPOSAL.read_text())["source"]
    source = next(item for item in sources if "reranker" in item["name"])
    body = """<table>
      <tr><th>Model</th><th>Context Length (tokens)</th><th>Description</th></tr>
      <tr><td>(In Preview) <code>rerank-3</code></td><td>32,000</td><td>Preview</td></tr>
      <tr><td><code>rerank-2.5-lite</code></td><td>32,000</td><td>Current</td></tr>
      <tr><td><code>rerank-lite-1</code></td><td>4,000</td><td>Earlier</td></tr>
    </table>"""

    page = _adapter(source, body).fetch_page({})

    models = [model for record in page.records for model in record.models]
    assert [model.name for model in models] == [
        "rerank-3",
        "rerank-2.5-lite",
        "rerank-lite-1",
    ]
    assert all(
        identifier.namespace == "voyage:model"
        for model in models
        for identifier in model.identifiers
    )
