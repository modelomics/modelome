from __future__ import annotations

from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.cloudflare_workers_ai_deprecations import (
    CloudflareWorkersAIDeprecations,
)

URL = (
    "https://developers.cloudflare.com/changelog/post/"
    "2026-05-08-planned-model-deprecations"
)


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


def test_cloudflare_announcement_preserves_retired_ids_and_replacement() -> None:
    client = Client(response(
        "## Models deprecated on May 30, 2026\n"
        "- `@cf/moonshotai/kimi-k2.5` --> `@cf/moonshotai/kimi-k2.6`\n"
        "- `@hf/meta-llama/meta-llama-3-8b-instruct`\n"
        "- `@cf/meta/llama-3.1-8b-instruct`\n"
        "## Variants that remain active\n"
        "- `@cf/meta/llama-3.3-70b-instruct-fp8-fast`\n"
    ))
    page = CloudflareWorkersAIDeprecations(client=client).fetch_page({})

    assert page.complete and not page.authoritative_snapshot
    assert page.upstream_count == 3
    first, second, _ = page.records
    assert first.source_record_id == "deprecation:@cf/moonshotai/kimi-k2.5"
    assert first.kind is ArtifactKind.CATALOG_RECORD
    assert first.identifiers == (
        Identifier("cloudflare:workers-ai", "@cf/moonshotai/kimi-k2.5"),
    )
    assert first.models[0].status is ModelStatus.DOCUMENTED
    assert first.raw["provider_lifecycle_status"] == "deprecated"
    assert first.raw["deprecated_on"] == "May 30, 2026"
    assert first.raw["replacement_model_id"] == "@cf/moonshotai/kimi-k2.6"
    assert second.raw["model_id"] == "@hf/meta-llama/meta-llama-3-8b-instruct"
    assert "replacement_model_id" not in second.raw
    assert all(
        record.raw["model_id"] != "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
        for record in page.records
    )


@pytest.mark.parametrize(
    ("body", "status", "error"),
    [
        ("", 503, "HTTP 503"),
        ("No lifecycle details", 200, "dated deprecation section"),
        ("## Models deprecated on May 30, 2026\n- `not-an-id`", 200, "no recognized model IDs"),
    ],
)
def test_cloudflare_announcement_rejects_incomplete_or_unexpected_payloads(
    body: str, status: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        CloudflareWorkersAIDeprecations(
            client=Client(response(body, status))
        ).fetch_page({})


def test_cloudflare_announcement_rejects_oversized_response() -> None:
    with pytest.raises(ValueError, match="exceeds 1 bytes"):
        CloudflareWorkersAIDeprecations(
            client=Client(response("xx")), max_response_bytes=1
        ).fetch_page({})
