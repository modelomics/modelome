from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.galaxea_vla_checkpoints import (
    GalaxeaVLACheckpointSourceAdapter,
    _parse_checkpoints,
    _parse_legacy_checkpoints,
)

_SHA = "c" * 40
_LEGACY_SHA = "13a16a9049aee8f1d799b56fccc0c5832a75fc2f"
_README = "\n".join(
    (
        "# GalaxeaVLA",
        "## News",
        "[base](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-base)",
        "### Model Checkpoints",
        "| Model | Use Case | Local `--ckpt_path` |",
        "| --- | --- | --- |",
        (
            "| [G05-base](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-base) | "
            "Fine-tuning and R1 Lite/R1 Pro zero-shot deployment | "
            "`checkpoints/g05-base/checkpoints/model_state_dict.pt` |"
        ),
        (
            "| [G05-so101](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-so101) | "
            "SO-100/101 zero-shot deployment | "
            "`checkpoints/g05-so101/checkpoints/model_state_dict.pt` |"
        ),
        (
            "| [G05-droid](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-droid) | "
            "DROID zero-shot deployment | "
            "`checkpoints/g05-droid/checkpoints/model_state_dict.pt` |"
        ),
        (
            "| [G05-libero](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-libero) | "
            "LIBERO evaluation | `checkpoints/g05-libero/model.pt` |"
        ),
        (
            "| [G05-robotwin20](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-robotwin20) | "
            "RoboTwin 2.0 evaluation | "
            "`checkpoints/g05-robotwin20/checkpoints/model_state_dict.pt` |"
        ),
        "## Inference",
        "[not in registry](https://huggingface.co/OpenGalaxea/G05/tree/main/g05-other)",
    )
)
_LEGACY_README = "\n".join(
    (
        "# Galaxea VLA",
        "## Model Checkpoints",
        "Model | Use Case | Description | Checkpoint Path",
        "--- | --- | --- | ---",
        "G0_3B_base | Fine-Tuning | Base G0-VLA Model for fine-tuning | "
        "[link](https://huggingface.co/OpenGalaxea/G0-VLA/blob/main/G0_3B_base.pt)",
        "G0Plus_3B_base | Fine-Tuning | Base G0Plus-VLA Model for fine-tuning | "
        "[link](https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Plus_3B_base)",
        "G0Tiny_250M_base | Fine-Tuning | Lightweight G0Tiny model | "
        "[link](https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Tiny_260120)",
        "G0Plus_3B_base-pick_and_place | Deployment | Pick-and-Place Demo | "
        "[link](https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Plus_PP_CKPT)",
        "## Fine-Tuning",
        "[other](https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/ignored)",
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
        if url.endswith("/commits/main"):
            body = json.dumps({"sha": _SHA}).encode()
        elif _LEGACY_SHA in url:
            body = _LEGACY_README.encode()
        else:
            body = _README.encode()
        return HttpResponse(200, {}, body, url)


def test_parser_selects_exact_checkpoint_folders_from_named_table() -> None:
    entries = _parse_checkpoints(_README, source="test", maximum=10)

    assert [entry[0] for entry in entries] == [
        "G05-base",
        "G05-so101",
        "G05-droid",
        "G05-libero",
        "G05-robotwin20",
    ]
    assert [entry[3] for entry in entries] == [
        f"https://huggingface.co/OpenGalaxea/G05/tree/main/{slug}"
        for slug in ("g05-base", "g05-so101", "g05-droid", "g05-libero", "g05-robotwin20")
    ]
    assert entries[3][2] == "checkpoints/g05-libero/model.pt"


def test_legacy_parser_preserves_four_named_checkpoint_paths_and_forms() -> None:
    entries = _parse_legacy_checkpoints(_LEGACY_README, source="test", maximum=10)

    assert [entry[0] for entry in entries] == [
        "G0_3B_base",
        "G0Plus_3B_base",
        "G0Tiny_250M_base",
        "G0Plus_3B_base-pick_and_place",
    ]
    assert [entry[3] for entry in entries] == [
        "https://huggingface.co/OpenGalaxea/G0-VLA/blob/main/G0_3B_base.pt",
        "https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Plus_3B_base",
        "https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Tiny_260120",
        "https://huggingface.co/OpenGalaxea/G0-VLA/tree/main/G0Plus_PP_CKPT",
    ]


def test_adapter_emits_first_party_checkpoint_folder_links_and_paths() -> None:
    client = _Client()
    adapter = GalaxeaVLACheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/OpenGalaxea/GalaxeaVLA/commits/main",
        f"https://raw.githubusercontent.com/OpenGalaxea/GalaxeaVLA/{_SHA}/README.md",
        "https://raw.githubusercontent.com/OpenGalaxea/GalaxeaVLA/"
        f"{_LEGACY_SHA}/README.md",
    ]
    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 9
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == (
        "https://huggingface.co/OpenGalaxea/G05/tree/main/g05-base"
    )
    assert record.identifiers == (Identifier("galaxea:g05-checkpoint", "g05-base"),)
    assert record.releases[0].metadata["checkpoint_path"] == (
        "checkpoints/g05-base/checkpoints/model_state_dict.pt"
    )
    assert record.releases[0].metadata["requires_model_terms_acceptance"] is True
    assert record.releases[0].metadata["revision"] == _SHA
    assert record.models[0].identifiers == (
        Identifier("galaxea:g05-checkpoint", "g05-base"),
    )
    legacy_record = next(
        item for item in page.records if item.title == "Galaxea G0_3B_base"
    )
    assert legacy_record.canonical_url == (
        "https://huggingface.co/OpenGalaxea/G0-VLA/blob/main/G0_3B_base.pt"
    )
    assert legacy_record.releases[0].metadata["checkpoint_form"] == "file"
    assert legacy_record.releases[0].metadata["revision"] == _LEGACY_SHA
    assert legacy_record.models[0].identifiers == (
        Identifier("galaxea:g0-checkpoint", "G0_3B_base.pt"),
    )


def test_entry_build_keeps_all_current_and_legacy_variants_distinct() -> None:
    adapter = GalaxeaVLACheckpointSourceAdapter(
        client=_Client(),
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )
    page = adapter.fetch_page({})
    seeds = [
        source_record_to_entry_seed(record, source="galaxea-vla")
        for record in page.records
    ]

    result = build_entries(seeds)

    assert len(result.entries) == 9
    entries_by_name = {entry.canonical_name: entry for entry in result.entries}
    assert set(entries_by_name) == {
        "G05-base",
        "G05-so101",
        "G05-droid",
        "G05-libero",
        "G05-robotwin20",
        "G0_3B_base",
        "G0Plus_3B_base",
        "G0Tiny_250M_base",
        "G0Plus_3B_base-pick_and_place",
    }
    assert all(len(entry.releases) == 1 for entry in result.entries)
    assert {
        entry.releases[0].identifiers[0].value for entry in result.entries
    } == {
        "g05-base",
        "g05-so101",
        "g05-droid",
        "g05-libero",
        "g05-robotwin20",
        "G0_3B_base.pt",
        "G0Plus_3B_base",
        "G0Tiny_260120",
        "G0Plus_PP_CKPT",
    }
    assert all(
        not any(identifier.namespace == "huggingface:model" for identifier in entry.identifiers)
        for entry in result.entries
    )


@pytest.mark.parametrize(
    "document,maximum",
    [
        (
            "### Model Checkpoints\n| A | B | C |\n|---|---|---|\n"
            "[bad](https://huggingface.co/Other/G05/tree/main/g05-bad) | use | path",
            10,
        ),
        (_README, 4),
        (
            "### Model Checkpoints\n| A | B | C |\n|---|---|---|\n"
            "[bad](https://huggingface.co/OpenGalaxea/G05/blob/main/g05-bad) | use | path",
            10,
        ),
    ],
)
def test_parser_excludes_wrong_repositories_and_rejects_oversized_lists(
    document: str,
    maximum: int,
) -> None:
    if document == _README and maximum == 4:
        with pytest.raises(ValueError):
            _parse_checkpoints(document, source="test", maximum=maximum)
    else:
        assert _parse_checkpoints(document, source="test", maximum=maximum) == ()
