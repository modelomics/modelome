from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.openvla_checkpoints import (
    OpenVLACheckpointSourceAdapter,
    _parse_checkpoints,
)

_SHA = "d" * 40
_README = "\n".join(
    (
        "# OpenVLA",
        "## Pretrained VLAs",
        "[openvla-7b](https://huggingface.co/openvla/openvla-7b)",
        "[openvla-v01-7b](https://huggingface.co/openvla/openvla-v01-7b)",
        "## Fine-Tuning OpenVLA",
        "[not a model](https://huggingface.co/openvla/ignore-me)",
        "## Fully Fine-Tuning OpenVLA",
        "[Prismatic-compatible checkpoint]("
        "https://huggingface.co/openvla/openvla-7b-prismatic)",
        "#### Launching LIBERO Evaluations",
        "[openvla/openvla-7b-finetuned-libero-spatial]("
        "https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)",
        "[openvla/openvla-7b-finetuned-libero-object]("
        "https://huggingface.co/openvla/openvla-7b-finetuned-libero-object)",
        "[openvla/openvla-7b-finetuned-libero-goal]("
        "https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal)",
        "[openvla/openvla-7b-finetuned-libero-10]("
        "https://huggingface.co/openvla/openvla-7b-finetuned-libero-10)",
        "## VLA Performance Troubleshooting",
        "[unlisted](https://huggingface.co/openvla/unlisted)",
    )
)


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        body = (
            json.dumps({"sha": _SHA}).encode()
            if url.endswith("/commits/main")
            else _README.encode()
        )
        return HttpResponse(200, {}, body, url)


def test_parser_keeps_only_refs_in_checkpoint_sections() -> None:
    entries = _parse_checkpoints(_README, source="test", maximum=10)

    assert [entry[2] for entry in entries] == [
        "https://huggingface.co/openvla/openvla-7b",
        "https://huggingface.co/openvla/openvla-v01-7b",
        "https://huggingface.co/openvla/openvla-7b-prismatic",
        "https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial",
        "https://huggingface.co/openvla/openvla-7b-finetuned-libero-object",
        "https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal",
        "https://huggingface.co/openvla/openvla-7b-finetuned-libero-10",
    ]
    assert [entry[1] for entry in entries[:2]] == ["pretrained vlas"] * 2
    assert entries[2][1] == "fully fine-tuning openvla"
    assert [entry[1] for entry in entries[3:]] == ["launching libero evaluations"] * 4


def test_adapter_emits_exact_checkpoint_repo_refs() -> None:
    client = _Client()
    adapter = OpenVLACheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/openvla/openvla/commits/main",
        f"https://raw.githubusercontent.com/openvla/openvla/{_SHA}/README.md",
    ]
    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 7
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == "https://huggingface.co/openvla/openvla-7b"
    assert record.identifiers == (Identifier("huggingface:model", "openvla/openvla-7b"),)
    assert record.releases[0].metadata["revision"] == _SHA


@pytest.mark.parametrize(
    "document,maximum",
    [
        (
            "## Pretrained VLAs\n"
            "[wrong](https://huggingface.co/other/openvla-7b)",
            10,
        ),
        (_README, 5),
        (
            "## Pretrained VLAs\n"
            "[wrong](https://huggingface.co/openvla/blob/main/openvla-7b)",
            10,
        ),
    ],
)
def test_parser_rejects_nonmatching_refs_and_oversized_lists(
    document: str,
    maximum: int,
) -> None:
    if document == _README and maximum == 5:
        with pytest.raises(ValueError):
            _parse_checkpoints(document, source="test", maximum=maximum)
    else:
        assert _parse_checkpoints(document, source="test", maximum=maximum) == ()
