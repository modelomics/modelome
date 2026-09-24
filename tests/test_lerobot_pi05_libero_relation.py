from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.lerobot_pi05_libero_relation import (
    LeRobotPi05LiberoRelationSourceAdapter,
    _parse_relation,
)

_SHA = "e" * 40
_MODEL = "https://huggingface.co/lerobot/pi05_libero_finetuned_v044"
_DATASET = "https://huggingface.co/datasets/lerobot/libero"
_DOC = "\n".join(
    (
        "# Pi0.5",
        "## Training",
        "### Quickstart on LIBERO",
        f"Finetune the LIBERO base model on [{_DATASET}]({_DATASET}), "
        "the demonstrations behind the results below.",
        "Training command and parameters.",
        f"Matching [{_MODEL}]({_MODEL}), the checkpoint the results below were measured on.",
        "### Quantile statistics",
        "## Performance Results",
        "Other context.",
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
        self.calls.append(url)
        body = (
            json.dumps({"sha": _SHA}).encode()
            if url.endswith("/commits/main")
            else self.document.encode()
        )
        return HttpResponse(200, {}, body, url)


def test_parser_requires_exact_refs_and_lineage_text_in_quickstart_section() -> None:
    assert _parse_relation(_DOC) == "docs/source/pi05.mdx:line:4-6"
    wrong_model = _DOC.replace(_MODEL, "https://huggingface.co/lerobot/pi05_base")
    assert _parse_relation(wrong_model) is None
    wrong_dataset = _DOC.replace(_DATASET, "https://huggingface.co/datasets/other/libero")
    assert _parse_relation(wrong_dataset) is None


def test_adapter_emits_exact_checkpoint_to_dataset_relation() -> None:
    client = _Client()
    adapter = LeRobotPi05LiberoRelationSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/huggingface/lerobot/commits/main",
        f"https://raw.githubusercontent.com/huggingface/lerobot/{_SHA}/docs/source/pi05.mdx",
    ]
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == _MODEL
    assert record.identifiers == (
        Identifier("huggingface:model", "lerobot/pi05_libero_finetuned_v044"),
    )
    assert record.releases[0].metadata["training_dataset_repo"] == "lerobot/libero"
    assert [link.url for link in record.links if link.relation == "trained_on_dataset"] == [
        _DATASET
    ]


def test_adapter_fails_closed_if_exact_model_ref_is_removed() -> None:
    client = _Client(document=_DOC.replace(_MODEL, "https://huggingface.co/lerobot/pi05_base"))
    adapter = LeRobotPi05LiberoRelationSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="no exact Pi0.5 LIBERO pair"):
        adapter.fetch_page({})
