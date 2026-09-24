from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.lerobot_molmoact2_relation import (
    LeRobotMolmoAct2RelationSourceAdapter,
    _parse_relation,
)

_SHA = "c" * 40
_MODEL = "https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot"
_DATASET = "https://huggingface.co/allenai/MolmoAct2-LIBERO-Dataset"
_DOC = "\n".join(
    (
        "# MolmoAct2",
        "## Performance Results",
        "### LIBERO Benchmark Results",
        "We fine-tuned the LIBERO model.",
        "",
        f"The fine-tuned checkpoint reported here is available at [{_MODEL}]({_MODEL}) "
        f"and was trained on [{_DATASET}]({_DATASET}).",
        "",
        "### Hardware Deployment",
        f"Unrelated references: [{_MODEL}]({_MODEL}) and [{_DATASET}]({_DATASET}).",
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
            json.dumps({"sha": _SHA}).encode() if url.endswith("/commits/main") else _DOC.encode()
        )
        return HttpResponse(200, {}, body, url)


def test_parser_requires_exact_model_dataset_pair_in_results_section() -> None:
    assert _parse_relation(_DOC, source="test") == ("docs/source/molmoact2.mdx:line:6-6")
    assert (
        _parse_relation(
            f"## LIBERO Benchmark Results\nCheckpoint was available at [{_MODEL}]({_MODEL}).",
            source="test",
        )
        is None
    )
    assert (
        _parse_relation(
            "### LIBERO Benchmark Results\n"
            f"The fine-tuned checkpoint was trained on [{_DATASET}]({_DATASET}) "
            f"and available at [other](https://huggingface.co/other/model).",
            source="test",
        )
        is None
    )


def test_adapter_emits_one_exact_checkpoint_to_dataset_edge() -> None:
    client = _Client()
    adapter = LeRobotMolmoAct2RelationSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/huggingface/lerobot/commits/main",
        f"https://raw.githubusercontent.com/huggingface/lerobot/{_SHA}/docs/source/molmoact2.mdx",
    ]
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == _MODEL
    assert record.identifiers == (
        Identifier("huggingface:model", "allenai/MolmoAct2-LIBERO-LeRobot"),
    )
    assert record.releases[0].metadata["trained_on_dataset"] == ("allenai/MolmoAct2-LIBERO-Dataset")
    dataset_links = [link.url for link in record.links if link.relation == "trained_on_dataset"]
    assert dataset_links == ["https://huggingface.co/datasets/allenai/MolmoAct2-LIBERO-Dataset"]


def test_adapter_fails_closed_if_source_pair_disappears() -> None:
    adapter = LeRobotMolmoAct2RelationSourceAdapter(
        client=_ClientWithoutPair(),
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="no exact LIBERO model/dataset pair"):
        adapter.fetch_page({})


class _ClientWithoutPair(_Client):
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
            else b"## Performance Results\nNo exact pair."
        )
        return HttpResponse(200, {}, body, url)
