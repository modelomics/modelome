from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.nvidia_cosmos3_checkpoints import (
    NvidiaCosmos3CheckpointSourceAdapter,
    _parse_checkpoints,
)

_SHA = "c" * 40
_BASES = (
    ("Cosmos3-Super", "64B", "H200 / B200 / GB200", "Highest quality"),
    ("Cosmos3-Nano", "16B", "RTX Pro 6000 / H100 / B200", "Balanced speed"),
    ("Cosmos3-Edge", "4B", "Jetson AGX Orin / Thor", "Edge deployment"),
)
_EXAMPLES = (
    ("Cosmos3-Super-Text2Image", "Super", "Elite quality text-to-image"),
    ("Cosmos3-Super-Text2Image-4Step", "Super", "Text-to-image, 17-25x faster"),
    ("Cosmos3-Super-Image2Video", "Super", "Elite quality image-to-video"),
    ("Cosmos3-Super-Image2Video-4Step", "Super", "Image-to-video, 17-25x faster"),
    ("Cosmos3-Nano-Policy-DROID", "Nano", "Open SOTA DROID robot policy"),
    ("Cosmos3-Edge-Policy-DROID", "Edge", "DROID robot policy at edge scale"),
)
_README = "\n".join(
    [
        "# NVIDIA Cosmos",
        "## Models",
        "Base model | Size | Runs on | Best for",
        "--- | --- | --- | ---",
        *[
            f"[ {name} ](https://huggingface.co/nvidia/{name}) | {size} | {hardware} | {use}"
            for name, size, hardware, use in _BASES
        ],
        "Example checkpoint | Base | Demonstrates",
        "--- | --- | ---",
        *[
            f"[{name}](https://huggingface.co/nvidia/{name}) | {base} | {use}"
            for name, base, use in _EXAMPLES
        ],
        "## Other section",
        "[ignored](https://huggingface.co/nvidia/Cosmos3-Other) | Nano | ignored",
    ]
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
        del params, headers
        self.calls.append(url)
        body = (
            json.dumps({"sha": _SHA}).encode()
            if url.endswith("/commits/main")
            else _README.encode()
        )
        return HttpResponse(200, {}, body, url)


def test_parser_extracts_only_first_party_base_and_example_rows() -> None:
    entries = _parse_checkpoints(_README, source="test", maximum=12)

    assert [entry[1] for entry in entries] == [
        *(name for name, *_ in _BASES),
        *(name for name, *_ in _EXAMPLES),
    ]
    assert [entry[0] for entry in entries] == ["base"] * 3 + ["example"] * 6
    assert [entry[3] for entry in entries] == [
        "",
        "",
        "",
        "Cosmos3-Super",
        "Cosmos3-Super",
        "Cosmos3-Super",
        "Cosmos3-Super",
        "Cosmos3-Nano",
        "Cosmos3-Edge",
    ]


def test_adapter_preserves_model_refs_and_base_relations() -> None:
    client = _Client()
    adapter = NvidiaCosmos3CheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/NVIDIA/cosmos/commits/main",
        f"https://raw.githubusercontent.com/NVIDIA/cosmos/{_SHA}/README.md",
    ]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 9
    policy = next(item for item in page.records if item.title.endswith("Cosmos3-Nano-Policy-DROID"))
    assert policy.kind is ArtifactKind.WEIGHTS
    assert policy.identifiers == (
        Identifier("huggingface:model", "nvidia/Cosmos3-Nano-Policy-DROID"),
    )
    assert policy.releases[0].metadata["checkpoint_category"] == "example"
    assert policy.releases[0].metadata["base_model"] == "Cosmos3-Nano"
    assert policy.model_relations[0].predicate == "fine_tuned_from"
    assert policy.model_relations[0].target.identifiers == (
        Identifier("huggingface:model", "nvidia/Cosmos3-Nano"),
    )


@pytest.mark.parametrize(
    "document,maximum",
    [
        (_README, 8),
        (
            "## Models\nBase model | Size\n--- | ---\n"
            "[Cosmos3-Other](https://huggingface.co/other/Cosmos3-Other) | 1B",
            10,
        ),
    ],
)
def test_parser_rejects_unlisted_models_and_enforces_maximum(document: str, maximum: int) -> None:
    if maximum == 8:
        with pytest.raises(ValueError, match="exceeds 8"):
            _parse_checkpoints(document, source="test", maximum=maximum)
    else:
        assert _parse_checkpoints(document, source="test", maximum=maximum) == ()
