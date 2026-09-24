from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = (
    Path(__file__).parents[1] / "config/proposals/cloudflare_workers_ai_deprecated_2025_10.toml"
)

_DEPRECATED = {
    "@hf/thebloke/zephyr-7b-beta-awq",
    "@hf/thebloke/mistral-7b-instruct-v0.1-awq",
    "@hf/thebloke/llama-2-13b-chat-awq",
    "@hf/thebloke/openhermes-2.5-mistral-7b-awq",
    "@hf/thebloke/neural-chat-7b-v3-1-awq",
    "@hf/thebloke/llamaguard-7b-awq",
    "@hf/thebloke/deepseek-coder-6.7b-base-awq",
    "@hf/thebloke/deepseek-coder-6.7b-instruct-awq",
    "@cf/deepseek-ai/deepseek-math-7b-instruct",
    "@cf/openchat/openchat-3.5-0106",
    "@cf/tiiuae/falcon-7b-instruct",
    "@cf/thebloke/discolm-german-7b-v1-awq",
    "@cf/qwen/qwen1.5-0.5b-chat",
    "@cf/qwen/qwen1.5-7b-chat-awq",
    "@cf/qwen/qwen1.5-14b-chat-awq",
    "@cf/tinyllama/tinyllama-1.1b-chat-v1.0",
    "@cf/qwen/qwen1.5-1.8b-chat",
    "@hf/nexusflow/starling-lm-7b-beta",
    "@cf/fblgit/una-cybertron-7b-v2-bf16",
}


class _FixtureClient:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert url == "https://developers.cloudflare.com/workers-ai/changelog"
        assert params is None
        assert "text/html" in (headers or {}).get("Accept", "")
        body = b"""
        <html><body>
          <h2>2025-09-18 Model Catalog updates</h2>
          <p>Some older Workers AI models are being deprecated on October 1st, 2025:</p>
          <ul>
            <li>@hf/thebloke/zephyr-7b-beta-awq</li>
            <li>@hf/thebloke/mistral-7b-instruct-v0.1-awq</li>
            <li>@hf/thebloke/llama-2-13b-chat-awq</li>
            <li>@hf/thebloke/openhermes-2.5-mistral-7b-awq</li>
            <li>@hf/thebloke/neural-chat-7b-v3-1-awq</li>
            <li>@hf/thebloke/llamaguard-7b-awq</li>
            <li>@hf/thebloke/deepseek-coder-6.7b-base-awq</li>
            <li>@hf/thebloke/deepseek-coder-6.7b-instruct-awq</li>
            <li>@cf/deepseek-ai/deepseek-math-7b-instruct</li>
            <li>@cf/openchat/openchat-3.5-0106</li>
            <li>@cf/tiiuae/falcon-7b-instruct</li>
            <li>@cf/thebloke/discolm-german-7b-v1-awq</li>
            <li>@cf/qwen/qwen1.5-0.5b-chat</li>
            <li>@cf/qwen/qwen1.5-7b-chat-awq</li>
            <li>@cf/qwen/qwen1.5-14b-chat-awq</li>
            <li>@cf/tinyllama/tinyllama-1.1b-chat-v1.0</li>
            <li>@cf/qwen/qwen1.5-1.8b-chat</li>
            <li>@hf/nexusflow/starling-lm-7b-beta</li>
            <li>@cf/fblgit/una-cybertron-7b-v2-bf16</li>
          </ul>
          <p>Variants that remain active: <code>@cf/meta/llama-3.3-70b-instruct-fp8-fast</code></p>
        </body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_cloudflare_changelog_captures_only_the_exact_deprecation_ids() -> None:
    with _PROPOSAL.open("rb") as handle:
        source = tomllib.load(handle)["source"][0]
    adapter = HtmlCatalogSourceAdapter(
        name=source["name"],
        url=source["url"],
        provider_namespace=source["provider_namespace"],
        rules=source["rules"],
        artifact_kind=source["artifact_kind"],
        model_status=source["model_status"],
        max_response_bytes=source["max_response_bytes"],
        max_entries=source["max_entries"],
        client=_FixtureClient(),
    )
    page = adapter.fetch_page({})
    models = [model for record in page.records for model in record.models]

    assert {model.name for model in models} == _DEPRECATED
    assert len(models) == 19
    assert {identifier.value for model in models for identifier in model.identifiers} == _DEPRECATED
    assert "deprecated-on:2025-10-01" in source["entry_tags"]
