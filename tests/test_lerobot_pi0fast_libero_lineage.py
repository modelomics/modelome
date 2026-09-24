from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.lerobot_pi0fast_libero_lineage import (
    LeRobotPi0FastLiberoLineageSourceAdapter,
    _parse_lineage,
)

_SHA = "d" * 40
_BASE = "https://huggingface.co/lerobot/pi0fast-base"
_TUNED = "https://huggingface.co/lerobot/pi0fast-libero"
_DOC = "\n".join(
    (
        "# Pi0Fast",
        "## Reproducing π₀Fast results",
        f"We take the LeRobot PiFast base model [{_BASE}]({_BASE}) and finetune "
        "for an additional 40k steps in bfloat16 using the HuggingFace LIBERO dataset.",
        "The finetuned model can be found here:",
        f"* π₀Fast LIBERO: [{_TUNED}]({_TUNED})",
        "With the following training command:",
        "--dataset.repo_id=lerobot/libero \\",
        f"--policy.path={_BASE.removeprefix('https://huggingface.co/')} \\",
        "## Results",
    )
)


class _Client:
    def __init__(self, document: str = _DOC) -> None:
        self.document = document
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        del params, headers
        self.calls.append(url)
        body = (
            json.dumps({"sha": _SHA}).encode()
            if url.endswith("/commits/main")
            else self.document.encode()
        )
        return HttpResponse(200, {}, body, url)


def test_parser_requires_explicit_base_and_tuned_refs_in_reproduction_section() -> None:
    assert _parse_lineage(_DOC) == "docs/source/pi0fast.mdx:line:3-8"
    assert _parse_lineage(_DOC.replace(_TUNED, "https://huggingface.co/lerobot/other")) is None
    assert _parse_lineage(_DOC.replace("Reproducing π₀Fast results", "Inference")) is None


def test_adapter_emits_two_distinct_checkpoint_identities_and_lineage() -> None:
    client = _Client()
    adapter = LeRobotPi0FastLiberoLineageSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/huggingface/lerobot/commits/main",
        f"https://raw.githubusercontent.com/huggingface/lerobot/{_SHA}/docs/source/pi0fast.mdx",
    ]
    assert page.complete and page.authoritative_snapshot
    assert len(page.records) == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert {item.name for item in record.models} == {
        "LeRobot π₀-FAST base checkpoint",
        "LeRobot π₀-FAST LIBERO fine-tuned checkpoint",
    }
    assert {release.identifiers[0].value for release in record.releases} == {
        "lerobot/pi0fast-base",
        "lerobot/pi0fast-libero",
    }
    assert record.model_relations[0].predicate == "fine_tuned_from"
    assert record.model_relations[0].subject_local_id == "model:pi0fast-libero"
    assert record.model_relations[0].target.identifiers == (
        Identifier("huggingface:model", "lerobot/pi0fast-base"),
    )
    result = build_entries(
        [source_record_to_entry_seed(record, source="lerobot-pi0fast-libero-lineage")]
    )
    assert len(result.entries) == 2
    assert {identifier.key for entry in result.entries for identifier in entry.identifiers} == {
        "huggingface:model:lerobot/pi0fast-base",
        "huggingface:model:lerobot/pi0fast-libero",
    }


def test_adapter_fails_closed_when_either_exact_ref_is_missing() -> None:
    client = _Client(document=_DOC.replace("lerobot/pi0fast-base", "lerobot/other"))
    adapter = LeRobotPi0FastLiberoLineageSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    with pytest.raises(ValueError, match="no exact π₀-FAST checkpoint lineage"):
        adapter.fetch_page({})
