from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.nvidia_groot_n17_checkpoints import (
    NvidiaGR00TN17CheckpointSourceAdapter,
    _parse_checkpoints,
)

_SHA = "a" * 40
_CHECKPOINTS = (
    (
        "GR00T-N1.7-3B",
        "Base",
        "See [pretrain tags](https://example.org/tags)",
        "Base model (3B params) — zero-shot inference",
    ),
    ("GR00T-N1.7-LIBERO", "Finetuned", "`LIBERO_PANDA`", "Finetuned on LIBERO benchmark"),
    (
        "GR00T-N1.7-DROID",
        "Finetuned",
        "`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`",
        "Finetuned on DROID dataset",
    ),
    (
        "GR00T-N1.7-SimplerEnv-Bridge",
        "Finetuned",
        "`SIMPLER_ENV_WIDOWX`",
        "Finetuned on SimplerEnv Bridge",
    ),
    (
        "GR00T-N1.7-SimplerEnv-Fractal",
        "Finetuned",
        "`SIMPLER_ENV_GOOGLE`",
        "Finetuned on SimplerEnv Fractal",
    ),
)
_README = "\n".join(
    (
        "# NVIDIA Isaac GR00T",
        "## Model Checkpoints & Embodiment Tags",
        "### Checkpoints",
        "| Checkpoint | Type | Embodiment Tag | Description |",
        "| --- | --- | --- | --- |",
        *(
            f"| [`nvidia/{slug}`](https://huggingface.co/nvidia/{slug}) | "
            f"{kind} | {tag} | {description} |"
            for slug, kind, tag, description in _CHECKPOINTS
        ),
        "### Embodiment Tags",
        "| ignored | Finetuned | `NOPE` | not a checkpoint list |",
        "## Inference",
        "| [`nvidia/GR00T-N1.7-DROID`](https://huggingface.co/nvidia/GR00T-N1.7-DROID) | "
        "Finetuned | `NOPE` | outside checkpoint section |",
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


def test_parser_keeps_exact_checkpoint_table_entries_and_tags() -> None:
    entries = _parse_checkpoints(_README, source="test", maximum=8)

    assert [entry[0] for entry in entries] == [item[0] for item in _CHECKPOINTS]
    assert [entry[2] for entry in entries] == [
        "See pretrain tags",
        *(item[2].strip(" `") for item in _CHECKPOINTS[1:]),
    ]
    assert all(entry[4] == index + 6 for index, entry in enumerate(entries))


def test_adapter_emits_exact_model_and_release_identities() -> None:
    client = _Client()
    adapter = NvidiaGR00TN17CheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/NVIDIA/Isaac-GR00T/commits/main",
        f"https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/{_SHA}/README.md",
    ]
    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 5
    assert [record.canonical_url for record in page.records] == [
        f"https://huggingface.co/nvidia/{slug}" for slug, *_ in _CHECKPOINTS
    ]
    for record, (slug, *_rest) in zip(page.records, _CHECKPOINTS, strict=True):
        assert record.kind is ArtifactKind.WEIGHTS
        assert record.identifiers == (Identifier("huggingface:model", f"nvidia/{slug}"),)
        assert record.releases[0].identifiers == (Identifier("nvidia:groot-checkpoint", slug),)
        assert record.releases[0].metadata["revision"] == _SHA
        expected_tag = "See pretrain tags" if slug.endswith("3B") else _rest[1].strip(" `")
        assert record.releases[0].metadata["embodiment_tag"] == expected_tag


@pytest.mark.parametrize(
    "document,maximum",
    [
        ("## Model Checkpoints & Embodiment Tags\n| other | Finetuned | tag | desc |", 8),
        (_README, 4),
        (
            "## Model Checkpoints & Embodiment Tags\n"
            "| [`nvidia/GR00T-N1.7-DROID`](https://huggingface.co/other/GR00T-N1.7-DROID) | "
            "Finetuned | tag | desc |",
            8,
        ),
    ],
)
def test_parser_rejects_wrong_owner_and_oversized_table(document: str, maximum: int) -> None:
    if maximum == 4:
        with pytest.raises(ValueError):
            _parse_checkpoints(document, source="test", maximum=maximum)
    else:
        assert _parse_checkpoints(document, source="test", maximum=maximum) == ()
