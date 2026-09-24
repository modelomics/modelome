from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/alibaba_modelstudio_qwen_api_catalog.toml"
_EXPECTED = {
    "qwen3.5-plus",
    "qwen3.5-plus-2026-02-15",
    "qwen3-coder-plus",
    "qwen3-vl-plus",
    "qwen3-omni-flash",
    "qwq-plus",
    "qvq-max",
    "qwen-audio-3.1-asr-flash",
    "qwen3-asr-flash",
    "qwen3-tts-flash",
    "qwen-image-2.0",
    "qwen3.5-ocr",
    "qwen-mt-flash",
    "qwen3-livetranslate-flash",
    "qwen3-vl-embedding",
    "qwen3-vl-rerank",
}
_EXPECTED_BY_TYPE = {
    "audio": {"qwen-audio-3.1-asr-flash"},
    "asr": {"qwen3-asr-flash"},
    "tts": {"qwen3-tts-flash"},
    "image": {"qwen-image-2.0"},
    "ocr": {"qwen3.5-ocr"},
    "translation": {"qwen-mt-flash", "qwen3-livetranslate-flash"},
    "embedding": {"qwen3-vl-embedding"},
    "rerank": {"qwen3-vl-rerank"},
}


class _FixtureClient:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert url == "https://help.aliyun.com/en/model-studio/rate-limit"
        assert params is None
        assert "text/html" in (headers or {}).get("Accept", "")
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=b"""
            <html><body>
              <h3>Qwen</h3>
              <table>
                <tr><th>Model name</th><th>Requests per minute</th></tr>
                <tr><td>qwen3.5-plus</td><td>30000</td></tr>
                <tr><td>qwen3.5-plus-2026-02-15</td><td>600</td></tr>
                <tr><td>qwen3-coder-plus</td><td>600</td></tr>
                <tr><td>qwen3-vl-plus</td><td>600</td></tr>
                <tr><td>qwen3-omni-flash</td><td>600</td></tr>
                <tr><td>qwq-plus</td><td>600</td></tr>
                <tr><td>qvq-max</td><td>600</td></tr>
                <tr><td>qwen-audio-3.1-asr-flash</td><td>600</td></tr>
                <tr><td>qwen3-asr-flash</td><td>600</td></tr>
                <tr><td>qwen3-tts-flash</td><td>600</td></tr>
                <tr><td>qwen-image-2.0</td><td>600</td></tr>
                <tr><td>qwen-audio-3.1-asr-flash</td><td>600</td></tr>
                <tr><td>qwen3.5-ocr</td><td>600</td></tr>
                <tr><td>qwen-mt-flash</td><td>600</td></tr>
                <tr><td>qwen3-livetranslate-flash</td><td>600</td></tr>
                <tr><td>qwen3-vl-embedding</td><td>600</td></tr>
                <tr><td>qwen3-vl-rerank</td><td>600</td></tr>
                <tr><td>qwen-language-model</td><td>600</td></tr>
                <tr><td>qwen-language-model-open-source-version</td><td>600</td></tr>
                <tr><td>qwen-open-source</td><td>600</td></tr>
                <tr><td>qwenwork.cn</td><td>600</td></tr>
                <tr><td>deepseek-v4-pro</td><td>600</td></tr>
              </table>
            </body></html>
            """,
            url=url,
        )


def test_alibaba_qwen_api_model_docs_extract_exact_qwen_llm_ids() -> None:
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

    assert {model.name for model in models} == _EXPECTED
    assert len(models) == len(_EXPECTED)
    assert {identifier.value for model in models for identifier in model.identifiers} == _EXPECTED
    names = {model.name for model in models}
    for model_type_ids in _EXPECTED_BY_TYPE.values():
        assert model_type_ids <= names
