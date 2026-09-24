from __future__ import annotations

from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.cloud_catalogs import CloudflareWorkersAIModelCatalog

URL = "https://developers.cloudflare.com/workers-ai/models"


class Client:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[str] = []

    def get(
        self, url: str, *, params: Any = None, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        assert params is None
        self.calls.append(url)
        assert headers == {"Accept": "text/html"}
        return self.response


def response(body: str, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=body.encode(), url=URL)


def test_public_workers_ai_catalog_emits_exact_provider_ids() -> None:
    client = Client(response(
        '<tr><td>@cf/meta/llama-3.1-8b-instruct</td></tr>'
        '<tr><td>@cf/baai/bge-small-en-v1.5</td></tr>'
        '<tr><td>@cf/meta/llama-3.1-8b-instruct</td></tr>'
    ))
    page = CloudflareWorkersAIModelCatalog(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert client.calls == [URL]
    first = page.records[0]
    assert first.kind is ArtifactKind.MODEL_CARD
    assert first.source_record_id == "model:@cf/meta/llama-3.1-8b-instruct"
    assert first.models[0].name == "@cf/meta/llama-3.1-8b-instruct"
    assert first.models[0].identifiers == (
        Identifier("cloudflare:workers-ai", "@cf/meta/llama-3.1-8b-instruct"),
    )
    assert first.models[0].status is ModelStatus.RELEASED


def test_catalog_rejects_unexpected_empty_or_oversized_response() -> None:
    with pytest.raises(ValueError, match="no Workers AI model IDs"):
        CloudflareWorkersAIModelCatalog(client=Client(response("no model list"))).fetch_page({})
    with pytest.raises(ValueError, match="exceeds 1 model entries"):
        CloudflareWorkersAIModelCatalog(
            client=Client(response("@cf/meta/model-one @cf/meta/model-two")), max_entries=1
        ).fetch_page({})
    with pytest.raises(ValueError, match="HTTP 503"):
        CloudflareWorkersAIModelCatalog(client=Client(response("", 503))).fetch_page({})
