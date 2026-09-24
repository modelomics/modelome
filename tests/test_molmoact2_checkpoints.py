from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.molmoact2_checkpoints import (
    MolmoAct2CheckpointSourceAdapter,
    _parse_checkpoints,
)

_SHA = "b" * 40
_BASE = (
    ("MolmoAct2", "Fine-tuning", "Continuous action expert foundation."),
    ("MolmoAct2-Think", "Fine-tuning", "Depth-token reasoning foundation."),
    ("MolmoAct2-Pretrain", "Post-training", "Discrete VLA backbone for continued training."),
)
_FINETUNED = (
    ("MolmoAct2-DROID", "Inference / Fine-tuning", "DROID Franka policy."),
    ("MolmoAct2-BimanualYAM", "Inference / Fine-tuning", "Bimanual YAM policy."),
    ("MolmoAct2-SO100_101", "Inference / Fine-tuning", "SO-100/SO-101 policy."),
    ("MolmoAct2-LIBERO", "Inference / Fine-tuning", "LIBERO policy."),
    ("MolmoAct2-Think-LIBERO", "Inference / Fine-tuning", "Depth reasoning LIBERO policy."),
)


def _table(rows: tuple[tuple[str, str, str], ...]) -> tuple[str, ...]:
    return (
        "Model | Use Case | Description | Checkpoint Path",
        "--- | --- | --- | ---",
        *(
            f"| {name} | {use_case} | {description} | "
            f"[https://huggingface.co/allenai/{name}]"
            f"(https://huggingface.co/allenai/{name}) |"
            for name, use_case, description in rows
        ),
    )


_README = "\n".join(
    (
        "# MolmoAct2",
        "## 1. Models",
        "### Base Models",
        "Base checkpoints for training.",
        *_table(_BASE),
        "| Molmo2-ER | Pre-training | VLM backbone | "
        "[https://huggingface.co/allenai/Molmo2-ER](https://huggingface.co/allenai/Molmo2-ER) |",
        "### Finetuned Models",
        "These policies target common embodiments.",
        *_table(_FINETUNED),
        "## 2. Datasets",
        "| data | desc | path |",
        "| --- | --- | --- |",
        "| dataset | robotics data | https://huggingface.co/allenai/MolmoAct2-Dataset |",
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


def test_parser_keeps_policy_refs_in_base_and_finetuned_tables() -> None:
    entries = _parse_checkpoints(_README, source="test", maximum=12)

    assert [entry[0] for entry in entries] == [name for name, *_ in (*_BASE, *_FINETUNED)]
    assert [entry[1] for entry in entries] == ["base models"] * 3 + ["finetuned models"] * 5
    assert "Molmo2-ER" not in {entry[0] for entry in entries}
    assert "MolmoAct2-Dataset" not in {entry[0] for entry in entries}


def test_adapter_emits_exact_model_refs_and_release_ids() -> None:
    client = _Client()
    adapter = MolmoAct2CheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/allenai/molmoact2/commits/main",
        f"https://raw.githubusercontent.com/allenai/molmoact2/{_SHA}/README.md",
    ]
    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 8
    for record, (slug, *_rest) in zip(page.records, (*_BASE, *_FINETUNED), strict=True):
        assert record.kind is ArtifactKind.WEIGHTS
        assert record.canonical_url == f"https://huggingface.co/allenai/{slug}"
        assert record.identifiers == (Identifier("huggingface:model", f"allenai/{slug}"),)
        assert record.releases[0].identifiers == (Identifier("allenai:molmoact2-checkpoint", slug),)
        assert record.releases[0].metadata["revision"] == _SHA


@pytest.mark.parametrize(
    "document,maximum",
    [
        ("### Base Models\nnot a table", 12),
        (_README, 7),
        (
            "### Base Models\n| X | Fine-tuning | desc | "
            "[https://huggingface.co/other/X](https://huggingface.co/other/X) |",
            12,
        ),
    ],
)
def test_parser_ignores_noninventory_and_rejects_oversize(
    document: str,
    maximum: int,
) -> None:
    if maximum == 7:
        with pytest.raises(ValueError):
            _parse_checkpoints(document, source="test", maximum=maximum)
    else:
        assert _parse_checkpoints(document, source="test", maximum=maximum) == ()
