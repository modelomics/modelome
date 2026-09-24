from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier, ModelStatus
from modelome.sources.jax_extra_registry import JaxExtraRegistrySourceAdapter

_REVISION = "b" * 40


class _Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def _response(body: bytes | Mapping[str, Any]) -> HttpResponse:
    if isinstance(body, Mapping):
        body = json.dumps(body).encode()
    return HttpResponse(200, {}, body, "https://fixture.test")


def test_scenic_baseline_model_zoo_reads_exact_first_party_rows() -> None:
    document = (
        b"## Model Zoo\n"
        b"| Model | Dataset | Pretraining | ImageNet Accuracy | Checkpoint |\n"
        b"|-------|:-:|:-:|:-:|:-:|\n"
        b"| ViT-B/16 | ImageNet | - | 73.7* | "
        b"[Link](https://storage.googleapis.com/scenic-bucket/baselines/"
        b"ViT_B_16_ImageNet1k) |\n"
        b"| ViT-AugReg-B/16 | ImageNet | - | 79.7 | "
        b"[Link](https://storage.googleapis.com/scenic-bucket/baselines/"
        b"ViT-AugReg_B_16_ImageNet1k) |\n"
    )
    client = _Client(_response({"sha": _REVISION}), _response(document))

    page = JaxExtraRegistrySourceAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    first, second = page.records
    assert first.models[0].name == "ViT-B/16"
    assert first.models[0].status is ModelStatus.RELEASED
    assert first.identifiers == (
        Identifier(
            "scenic:model",
            "ViT-B/16|ImageNet|-|https://storage.googleapis.com/scenic-bucket/baselines/ViT_B_16_ImageNet1k",
        ),
    )
    assert first.raw["checkpoint_url"].endswith("ViT_B_16_ImageNet1k")
    assert first.raw["reported_accuracy"] == "73.7*"
    assert second.models[0].name == "ViT-AugReg-B/16"
    assert len(client.calls) == 2
    assert not any("storage.googleapis.com" in url for url in client.calls)


def test_scenic_model_zoo_rejects_untrusted_checkpoint_hosts() -> None:
    document = """| Model | Dataset | Pretraining | Accuracy | Checkpoint |
|-------|:-:|:-:|:-:|:-:|
| ViT-B/16 | ImageNet | - | 73.7 | [Link](https://evil.example/ViT_B_16) |
"""
    client = _Client(_response({"sha": _REVISION}), _response(document.encode()))

    with pytest.raises(ValueError, match="non-public Google Storage checkpoint"):
        JaxExtraRegistrySourceAdapter(client=client).fetch_page({})
