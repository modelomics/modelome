from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.markdown_model_card_list import MarkdownModelCardListSourceAdapter

_REVISION = "c" * 40


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: str | Mapping[str, Any]) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/model-cards.md")


_DOCUMENT = """\
# AudioCraft

## API

- `facebook/musicgen-small`: [Hub](https://huggingface.co/facebook/musicgen-small)
- `facebook/musicgen-large`: [Hub](https://huggingface.co/facebook/musicgen-large)
  and [other](https://example.test/other)

## Training

[do not admit](https://huggingface.co/facebook/musicgen-training)
"""


def _adapter(client: _QueuedClient) -> MarkdownModelCardListSourceAdapter:
    return MarkdownModelCardListSourceAdapter(
        name="fixture-model-card-list",
        repository="example-org/models",
        branch="main",
        document_path="docs/models.md",
        section_heading_pattern=r"^API$",
        model_url_pattern=(
            r"https://huggingface\.co/facebook/(?P<handle>musicgen-(?:small|large))"
        ),
        provider_namespace="fixture:model-card",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_markdown_model_card_list_reads_only_configured_section_and_url_shape() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_DOCUMENT))

    page = _adapter(client).fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    small, large = page.records
    assert small.source_record_id == "model:musicgen-small"
    assert small.models[0].identifiers == (
        Identifier("fixture:model-card", "musicgen-small"),
        Identifier("huggingface:model", "facebook/musicgen-small"),
    )
    assert small.releases[0].metadata["model_card"] == (
        "https://huggingface.co/facebook/musicgen-small"
    )
    assert ("https://huggingface.co/facebook/musicgen-large", "model_card") in {
        (link.url, link.relation) for link in large.links
    }


def test_markdown_model_card_list_skips_document_when_commit_is_unchanged() -> None:
    adapter = _adapter(_QueuedClient(_response({"sha": _REVISION}), _response(_DOCUMENT)))
    first = adapter.fetch_page({})
    adapter.client = _QueuedClient(_response({"sha": _REVISION}))

    second = adapter.fetch_page(first.next_state)

    assert second.records == ()
    assert second.upstream_count == 2


def test_markdown_model_card_list_requires_named_handle_capture() -> None:
    with pytest.raises(ValueError, match="named 'handle' group"):
        MarkdownModelCardListSourceAdapter(
            name="fixture-model-card-list",
            repository="example-org/models",
            branch="main",
            document_path="docs/models.md",
            section_heading_pattern=r"^API$",
            model_url_pattern=r"https://huggingface\.co/facebook/musicgen-small",
            provider_namespace="fixture:model-card",
        )
