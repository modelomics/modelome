from __future__ import annotations

from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.cloudflare_workers_ai_legacy_deprecations import (
    CloudflareWorkersAILegacyDeprecations,
)

URL = "https://developers.cloudflare.com/workers-ai/changelog"


class Client:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response

    def get(
        self, url: str, *, params: Any = None, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        assert url == URL
        assert params is None
        assert headers == {"Accept": "text/markdown"}
        return self.response


def response(body: str, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=body.encode(), url=URL)


def test_legacy_changelog_deprecation_ids_are_exact_and_bounded() -> None:
    body = "## 2025-09-18\nModel Catalog updates\n"
    body += "Some older Workers AI models are being deprecated on October 1st, 2025.\n"
    model_ids = [
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
    ]
    body += "".join(f"  * {model_id}\n" for model_id in model_ids)
    body += "## 2025-09-05\n`@cf/google/embeddinggemma-300m` available\n"
    page = CloudflareWorkersAILegacyDeprecations(
        client=Client(response(body))
    ).fetch_page({})

    assert page.complete and not page.authoritative_snapshot
    assert page.upstream_count == 19
    assert [record.raw["model_id"] for record in page.records] == model_ids
    first = page.records[0]
    assert first.source_record_id == f"deprecation:{model_ids[0]}"
    assert first.kind is ArtifactKind.CATALOG_RECORD
    assert first.identifiers == (Identifier("cloudflare:workers-ai", model_ids[0]),)
    assert first.models[0].status is ModelStatus.DOCUMENTED
    assert first.raw["provider_lifecycle_status"] == "deprecated"
    assert first.raw["deprecated_on"] == "October 1, 2025"
    assert first.raw["announcement_date"] == "2025-09-18"


@pytest.mark.parametrize(
    ("body", "status", "error"),
    [
        ("", 503, "HTTP 503"),
        ("No deprecation data", 200, "lacks the October 2025 deprecation notice"),
        (
            "Some older Workers AI models are being deprecated on October 1st, 2025.\n",
            200,
            "contains no recognized model IDs",
        ),
    ],
)
def test_legacy_changelog_rejects_unexpected_payloads(
    body: str, status: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        CloudflareWorkersAILegacyDeprecations(
            client=Client(response(body, status))
        ).fetch_page({})
