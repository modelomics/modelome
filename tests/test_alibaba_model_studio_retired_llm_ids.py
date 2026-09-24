from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/alibaba_model_studio_retired_llm_ids.toml"


class _FixtureClient:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert url == "https://help.aliyun.com/zh/model-studio/rate-limit"
        assert params is None
        assert "text/html" in (headers or {}).get("Accept", "")
        body = b"""
        <html><body>
          <h2>Retired models</h2>
          <h3>July 30, 2025 retired</h3>
          <table>
            <tr><td>Qwen VL</td><td>qwen-vl-plus-2023-12-01</td></tr>
            <tr><td>01.AI</td><td>yi-large</td></tr>
            <tr><td>01.AI</td><td>yi-medium</td></tr>
            <tr><td>01.AI</td><td>yi-large-rag</td></tr>
            <tr><td>01.AI</td><td>yi-large-turbo</td></tr>
            <tr><td>Dolly</td><td>dolly-12b-v2</td></tr>
          </table>
          <h3>Active models</h3><p><code>qwen-vl-plus</code></p>
        </body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_alibaba_retirement_section_captures_only_exact_documented_ids() -> None:
    with _PROPOSAL.open("rb") as handle:
        source = tomllib.load(handle)["source"][0]
    adapter = create_source(source, client=_FixtureClient(), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})
    models = [model for record in page.records for model in record.models]

    assert {model.name for model in models} == {
        "qwen-vl-plus-2023-12-01",
        "yi-large",
        "yi-medium",
        "yi-large-rag",
        "yi-large-turbo",
        "dolly-12b-v2",
    }
    assert len(models) == 6
    assert {identifier.value for model in models for identifier in model.identifiers} == {
        model.name for model in models
    }
    assert "retired-on:2025-07-30" in source["entry_tags"]
