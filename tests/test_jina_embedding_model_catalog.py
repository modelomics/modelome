from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/jina_embedding_model_catalog.toml"


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


def test_jina_embedding_catalog_extracts_exact_ids_from_official_model_page_urls() -> None:
    source = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    body = """<nav><a href="/models/">Models</a></nav>
    <main>
      <a href="/models/jina-embeddings-v3/">Model: jina-embeddings-v3</a>
      <a href="/models/jina-embeddings-v5-text-small/">v5 small</a>
      <a href="/models/jina-code-embeddings-0.5b/">Code embedding model</a>
      <a href="/models/jina-colbert-v2/">ColBERT</a>
      <a href="/models/jina-clip-v1/">CLIP</a>
      <a href="/models/jina-vlm/">Not an embedding catalog row</a>
      <a href="/models/reader-lm-v2/">Not an embedding catalog row</a>
    </main>"""
    adapter = HtmlCatalogSourceAdapter(
        name=source["name"],
        url=source["url"],
        provider_namespace=source["provider_namespace"],
        rules=source["rules"],
        artifact_kind=source["artifact_kind"],
        model_status=source["model_status"],
        client=_FixtureClient(source["url"].rstrip("/"), body),
    )

    page = adapter.fetch_page({})

    models = [model for record in page.records for model in record.models]
    assert [model.name for model in models] == [
        "jina-embeddings-v3",
        "jina-embeddings-v5-text-small",
        "jina-code-embeddings-0.5b",
        "jina-colbert-v2",
        "jina-clip-v1",
    ]
    assert {identifier.value for model in models for identifier in model.identifiers} == {
        "jina-embeddings-v3",
        "jina-embeddings-v5-text-small",
        "jina-code-embeddings-0.5b",
        "jina-colbert-v2",
        "jina-clip-v1",
    }
    assert all(
        identifier.namespace == "jina:model" for model in models for identifier in model.identifiers
    )
