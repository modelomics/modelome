from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/deepinfra_embedding_models.toml"


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


def test_deepinfra_embedding_page_one_uses_exact_serving_ids_from_model_urls() -> None:
    sources = tomllib.loads(_PROPOSAL.read_text())["source"]
    source = next(item for item in sources if item["name"].endswith("page-1"))
    body = """<main>
      <a href="/BAAI/bge-base-en-v1.5">embeddings BAAI/bge-base-en-v1.5</a>
      <a href="/Qwen/Qwen3-Embedding-8B">embeddings Qwen/Qwen3-Embedding-8B</a>
      <a href="/models/embeddings/2">2</a>
      <a href="/models/text-generation">Text Generation</a>
    </main>"""

    page = _adapter(source, body).fetch_page({})

    models = [model for record in page.records for model in record.models]
    assert [model.name for model in models] == [
        "BAAI/bge-base-en-v1.5",
        "Qwen/Qwen3-Embedding-8B",
    ]
    assert {identifier.value for model in models for identifier in model.identifiers} == {
        "BAAI/bge-base-en-v1.5",
        "Qwen/Qwen3-Embedding-8B",
    }


def test_deepinfra_embedding_page_two_captures_ids_beyond_first_page() -> None:
    sources = tomllib.loads(_PROPOSAL.read_text())["source"]
    source = next(item for item in sources if item["name"].endswith("page-2"))
    body = """<main>
      <a href="/sentence-transformers/paraphrase-MiniLM-L6-v2">
        embeddings sentence-transformers/paraphrase-MiniLM-L6-v2</a>
      <a href="/thenlper/gte-large">embeddings thenlper/gte-large</a>
      <a href="/models/embeddings">1</a>
    </main>"""

    page = _adapter(source, body).fetch_page({})

    models = [model for record in page.records for model in record.models]
    assert [model.name for model in models] == [
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "thenlper/gte-large",
    ]
    assert all(
        identifier.namespace == "deepinfra:model"
        for model in models
        for identifier in model.identifiers
    )
