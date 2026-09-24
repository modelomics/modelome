from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.markdown_checkpoint_list import MarkdownCheckpointListSourceAdapter

_REVISION = "d" * 40


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
    return HttpResponse(200, {}, body, "https://fixtures.test/models.md")


_DOCUMENT = """\
# Segment Anything

## Model Checkpoints

- **`default` or `vit_h`: [ViT-H SAM model.](https://weights.example.test/vit-h.pth)**
- `vit_l`: [ViT-L SAM model.](https://weights.example.test/vit-l.pth)
- `vit_b`: [ViT-B SAM model.](https://weights.example.test/vit-b.pth)

## Dataset

- `not_a_model`: [dataset](https://weights.example.test/data.zip)
"""


def _adapter(client: _QueuedClient) -> MarkdownCheckpointListSourceAdapter:
    return MarkdownCheckpointListSourceAdapter(
        name="fixture-checkpoint-list",
        repository="example-org/models",
        branch="main",
        document_path="README.md",
        section_heading_pattern=r"Model Checkpoints$",
        model_handle_pattern=r"`(?:default` or `)?(?P<handle>vit_[hlb])`",
        provider_namespace="fixture:list-model",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_markdown_checkpoint_list_reads_only_explicit_section_bullets() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_DOCUMENT))

    page = _adapter(client).fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 3
    assert client.calls[1][0].endswith(f"/{_REVISION}/README.md")
    h_model, l_model, b_model = page.records
    assert h_model.kind is ArtifactKind.MODEL_CARD
    assert h_model.models[0].identifiers == (Identifier("fixture:list-model", "vit_h"),)
    assert l_model.title == "vit_l"
    assert b_model.links[-1].url == "https://weights.example.test/vit-b.pth"


def test_markdown_checkpoint_list_skips_an_unchanged_document() -> None:
    adapter = _adapter(_QueuedClient(_response({"sha": _REVISION}), _response(_DOCUMENT)))
    first = adapter.fetch_page({})
    adapter.client = _QueuedClient(_response({"sha": _REVISION}))

    second = adapter.fetch_page(first.next_state)

    assert second.records == ()
    assert second.upstream_count == 3


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ("## Model Checkpoints\n- [model](https://example.test/model.pt)", "no configured"),
        (
            "## Model Checkpoints\n- `vit_h`: [model](https://example.test/model.pt) [other](https://example.test/other.pth)",
            "exactly one",
        ),
        (
            "## Model Checkpoints\n- `vit_h`: [paper](https://arxiv.org/abs/2401.12345)",
            "exactly one",
        ),
        (
            "## Model Checkpoints\n- `vit_h`: [model](https://example.test/model.pt)\n- `vit_h`: [model](https://example.test/other.pt)",
            "duplicate checkpoint handle",
        ),
    ],
)
def test_markdown_checkpoint_list_rejects_ambiguous_items(
    document: str,
    match: str,
) -> None:
    adapter = _adapter(_QueuedClient(_response({"sha": _REVISION}), _response(document)))

    with pytest.raises(ValueError, match=match):
        adapter.fetch_page({})


def test_markdown_checkpoint_list_requires_a_named_handle_capture() -> None:
    with pytest.raises(ValueError, match="named 'handle' group"):
        MarkdownCheckpointListSourceAdapter(
            name="fixture-checkpoint-list",
            repository="example-org/models",
            branch="main",
            document_path="README.md",
            section_heading_pattern=r"Model Checkpoints$",
            model_handle_pattern=r"vit_[hlb]",
            provider_namespace="fixture:list-model",
        )
