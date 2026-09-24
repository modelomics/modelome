from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.pelican_vla_checkpoint_registry import (
    PelicanVLACheckpointRegistrySourceAdapter,
    _parse_registry,
)

_SHA = "b" * 40
_README = "\n".join(
    (
        "# Pelican-VLA 0.5",
        "## Overview",
        "See [backbone](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).",
        "## Model Download",
        "| Model Name | Hugging Face | Description |",
        "| --- | --- | --- |",
        (
            "| Pelican-VLA 0.5 | [Pelican-VLA05]("
            "https://huggingface.co/X-Humanoid/Pelican-VLA05) | "
            "Cross-embodiment pre-trained model (~2,400 hours) |"
        ),
        (
            "| Pelican-VLA 0.5 RoboTwin | [Pelican-VLA05-Robotwin]("
            "https://huggingface.co/X-Humanoid/Pelican-VLA05-Robotwin) | "
            "Fine-tuned on RoboTwin 2.0 (clean + randomized) |"
        ),
        "To run the model, download `X-Humanoid/Pelican-VLA05`.",
        "## Performance",
        "| Model Name | Hugging Face | Description |",
        "| --- | --- | --- |",
        (
            "| Not a model | [Wrong]("
            "https://huggingface.co/X-Humanoid/Should-Not-Match) | "
            "Excluded outside section |"
        ),
    )
)

_NON_HF_DOC = "\n".join(
    (
        "## Model Download",
        "| A | B | C |",
        "|---|---|---|",
        "| bad | [Repo](http://huggingface.co/X-Humanoid/Bad) | bad |",
    )
)
_OTHER_OWNER_DOC = "\n".join(
    (
        "## Model Download",
        "| A | B | C |",
        "|---|---|---|",
        "| bad | [Repo](https://huggingface.co/Other/Bad) | bad |",
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


def test_parser_selects_only_first_party_checkpoint_rows_in_named_section() -> None:
    entries = _parse_registry(_README, source="test", maximum=10)

    assert entries == (
        (
            "Pelican-VLA 0.5",
            "Cross-embodiment pre-trained model (~2,400 hours)",
            "https://huggingface.co/X-Humanoid/Pelican-VLA05",
            7,
        ),
        (
            "Pelican-VLA 0.5 RoboTwin",
            "Fine-tuned on RoboTwin 2.0 (clean + randomized)",
            "https://huggingface.co/X-Humanoid/Pelican-VLA05-Robotwin",
            8,
        ),
    )


def test_adapter_preserves_exact_model_repository_urls_and_source_revision() -> None:
    client = _Client()
    adapter = PelicanVLACheckpointRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/Open-X-Humanoid/Pelican-VLA05/commits/main",
        f"https://raw.githubusercontent.com/Open-X-Humanoid/Pelican-VLA05/{_SHA}/readme.md",
    ]
    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert [record.canonical_url for record in page.records] == [
        "https://huggingface.co/X-Humanoid/Pelican-VLA05",
        "https://huggingface.co/X-Humanoid/Pelican-VLA05-Robotwin",
    ]
    record = page.records[0]
    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.identifiers == (
        Identifier("huggingface:model", "X-Humanoid/Pelican-VLA05"),
    )
    assert record.releases[0].metadata["revision"] == _SHA
    assert record.releases[0].metadata["description"] == (
        "Cross-embodiment pre-trained model (~2,400 hours)"
    )


@pytest.mark.parametrize(
    "document,maximum",
    [
        (_NON_HF_DOC, 10),
        (_README, 1),
        (_OTHER_OWNER_DOC, 10),
    ],
)
def test_parser_rejects_or_excludes_untrusted_rows(document: str, maximum: int) -> None:
    if document == _README and maximum == 1:
        with pytest.raises(ValueError):
            _parse_registry(document, source="test", maximum=maximum)
    else:
        assert _parse_registry(document, source="test", maximum=maximum) == ()
