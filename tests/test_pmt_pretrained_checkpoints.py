from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.pmt_pretrained_checkpoints import (
    PMTPretrainedCheckpointSourceAdapter,
    _parse_inventory,
)

_GITHUB_SHA = "a" * 40
_ASSET_SHA = "b" * 40
_README = "\n".join(
    (
        "# PMT pretrained checkpoints",
        "File | Task / gym id | Network | iter | reward",
        "--- | --- | --- | --- | ---",
        "`multimotionv2_flat.pt` | `PMT-G1-MultiMotionV2-Flat-v0` | ActorCritic (MLP) | 20k | 40.9",
        "`sonic_onnx/` | `PMT-SONIC-G1-MultiMotionV2-Flat-v0` | SONIC (official ONNX) | — | —",
        "## Load and roll out",
    )
)
_ROOT_FILES = [
    {
        "type": "file",
        "path": "checkpoints/pretrained/multimotionv2_flat.pt",
        "oid": "git-oid-pt",
        "size": 7_543_199,
    },
    {"type": "directory", "path": "checkpoints/pretrained/sonic_onnx"},
]
_SONIC_FILES = [
    {
        "type": "file",
        "path": "checkpoints/pretrained/sonic_onnx/model_encoder.onnx",
        "oid": "git-oid-encoder",
        "size": 100,
    },
    {
        "type": "file",
        "path": "checkpoints/pretrained/sonic_onnx/model_decoder.onnx",
        "oid": "git-oid-decoder",
        "size": 200,
    },
]


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
        del headers
        self.calls.append(url)
        if url.endswith("/commits/main"):
            body: Any = {"sha": _GITHUB_SHA}
        elif url.endswith("/api/datasets/aCodeDog/PMT-assets"):
            body = {"sha": _ASSET_SHA}
        elif "/tree/" in url and url.endswith("/sonic_onnx"):
            body = _SONIC_FILES
        elif "/tree/" in url:
            body = _ROOT_FILES
        elif url.endswith("/README.md"):
            body = _README
        else:
            raise AssertionError(f"unexpected URL: {url}; params={params}")
        encoded = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        return HttpResponse(200, {}, encoded, url)


def test_parser_selects_exact_file_and_sonic_directory_rows() -> None:
    entries = _parse_inventory(_README, source="test", maximum=5)

    assert [(entry[0], entry[1]) for entry in entries] == [
        ("multimotionv2_flat.pt", "PMT-G1-MultiMotionV2-Flat-v0"),
        ("sonic_onnx/", "PMT-SONIC-G1-MultiMotionV2-Flat-v0"),
    ]
    assert entries[0][2:5] == ("ActorCritic (MLP)", "20k", "40.9")


def test_adapter_joins_first_party_rows_to_exact_published_assets() -> None:
    client = _Client()
    adapter = PMTPretrainedCheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/Mondo-Robotics/PMT/commits/main",
        f"https://raw.githubusercontent.com/Mondo-Robotics/PMT/{_GITHUB_SHA}/"
        "checkpoints/pretrained/README.md",
        "https://huggingface.co/api/datasets/aCodeDog/PMT-assets",
        "https://huggingface.co/api/datasets/aCodeDog/PMT-assets/tree/"
        f"{_ASSET_SHA}/checkpoints/pretrained",
        "https://huggingface.co/api/datasets/aCodeDog/PMT-assets/tree/"
        f"{_ASSET_SHA}/checkpoints/pretrained/sonic_onnx",
    ]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    weights = next(
        record for record in page.records if record.title == "PMT PMT-G1-MultiMotionV2-Flat-v0"
    )
    assert weights.kind is ArtifactKind.WEIGHTS
    assert weights.identifiers == (Identifier("pmt:policy", "PMT-G1-MultiMotionV2-Flat-v0"),)
    assert weights.links[0].url == (
        "https://huggingface.co/datasets/aCodeDog/PMT-assets/resolve/"
        f"{_ASSET_SHA}/checkpoints/pretrained/multimotionv2_flat.pt"
    )
    assert weights.releases[0].metadata["network"] == "ActorCritic (MLP)"
    sonic = next(record for record in page.records if "SONIC" in record.title)
    assert {
        link.url.rsplit("/", 1)[-1] for link in sonic.links if link.relation == "model_artifact"
    } == {
        "model_encoder.onnx",
        "model_decoder.onnx",
    }


@pytest.mark.parametrize(
    "document,maximum",
    [
        (_README, 1),
        (
            "File | Task / gym id | Network | iter | reward\n--- | --- | --- | --- | ---\n"
            "`../escape.pt` | task | net | 1 | 2",
            5,
        ),
    ],
)
def test_parser_enforces_bounds_and_safe_checkpoint_names(document: str, maximum: int) -> None:
    if maximum == 1:
        with pytest.raises(ValueError, match="exceeds 1"):
            _parse_inventory(document, source="test", maximum=maximum)
    else:
        assert _parse_inventory(document, source="test", maximum=maximum) == ()
