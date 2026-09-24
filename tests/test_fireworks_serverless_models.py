from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/fireworks_serverless_models.toml"


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


def test_fireworks_library_extracts_model_card_slugs_and_excludes_non_cards() -> None:
    source = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    adapter = HtmlCatalogSourceAdapter(
        name=source["name"],
        url=source["url"],
        provider_namespace=source["provider_namespace"],
        rules=source["rules"],
        artifact_kind=source["artifact_kind"],
        model_status=source["model_status"],
        client=_FixtureClient(
            source["url"],
            """<main>
              <a href="/models/fireworks/glm-5p3">GLM-5.3</a>
              <a href="/models/fireworks/llama-v3p1-8b-instruct">Llama 3.1 8B</a>
              <a href="/models/deepseek-ai/deepseek-v4p1-flash">DeepSeek V4.1 Flash</a>
              <a href="/models/fireworks">Fireworks models</a>
              <a href="/models/zai/glm-5p3">Other provider route</a>
              <a href="/models/fireworks/docs/model-library">Nested docs route</a>
              <a href="/docs/model-library">Docs</a>
            </main>""",
        ),
    )

    page = adapter.fetch_page({})
    models = [model for record in page.records for model in record.models]

    assert [model.name for model in models] == [
        "glm-5p3",
        "llama-v3p1-8b-instruct",
        "deepseek-v4p1-flash",
    ]
    assert {
        (identifier.namespace, identifier.value)
        for model in models
        for identifier in model.identifiers
    } == {
        ("fireworks:model", "glm-5p3"),
        ("fireworks:model", "llama-v3p1-8b-instruct"),
        ("fireworks:model", "deepseek-v4p1-flash"),
    }
