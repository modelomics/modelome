from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.satmae_checkpoint_registry import (
    SatMAECheckpointRegistrySourceAdapter,
    _parse_readme,
)

REV = "f" * 40
FILES = {
    "fmow_pretrain.pth": "https://zenodo.org/record/7369797/files/fmow_pretrain.pth",
    "fmow_finetune.pth": "https://zenodo.org/record/7369797/files/fmow_finetune.pth",
    "pretrain_fmow_temporal.pth": "https://zenodo.org/record/7369797/files/pretrain_fmow_temporal.pth",
    "finetune_fmow_temporal.pth": "https://zenodo.org/record/7369797/files/finetune_fmow_temporal.pth",
    "pretrain-vit-base-e199.pth": "https://zenodo.org/record/7338613/files/pretrain-vit-base-e199.pth",
    "finetune-vit-base-e7.pth": "https://zenodo.org/record/7338613/files/finetune-vit-base-e7.pth",
    "pretrain-vit-large-e199.pth": "https://zenodo.org/record/7338613/files/pretrain-vit-large-e199.pth",
    "finetune-vit-large-e7.pth": "https://zenodo.org/record/7338613/files/finetune-vit-large-e7.pth",
}


def _table(model: str, pretrain: str, finetune: str) -> str:
    return "\n".join(
        (
            f"| {model} | 80% | [download]({FILES[pretrain]}) | [download]({FILES[finetune]}) |",
        )
    )


README = "\n".join(
    (
        "# SatMAE",
        "## Temporal SatMAE",
        "#### fMoW Non-Temporal Checkpoints",
        "| Model | Top 1 Accuracy | Pretrain | Finetune |",
        _table("ViT-Large", "fmow_pretrain.pth", "fmow_finetune.pth"),
        "#### fMoW Temporal Checkpoints",
        "| Model | Top 1 Accuracy | Pretrain | Finetune |",
        _table("ViT-Large", "pretrain_fmow_temporal.pth", "finetune_fmow_temporal.pth"),
        "## Multi-Spectral SatMAE",
        "### Model Weights",
        "| Model | Top 1 Accuracy | Pretrain | Finetune |",
        _table("ViT-Base (200 epochs)", "pretrain-vit-base-e199.pth", "finetune-vit-base-e7.pth"),
        _table(
            "ViT-Large (200 epochs)",
            "pretrain-vit-large-e199.pth",
            "finetune-vit-large-e7.pth",
        ),
    )
)


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.urls.append(url)
        return self.responses.pop(0)


def response(value: Any) -> HttpResponse:
    body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
    return HttpResponse(200, {}, body, "https://fixture.test")


def test_satmae_readme_emits_eight_exact_file_links() -> None:
    client = Client(response({"sha": REV}), response(README))
    adapter = SatMAECheckpointRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot and page.upstream_count == 8
    found_urls = {
        link.url
        for record in page.records
        for link in record.links
        if link.relation == "weights"
    }
    assert found_urls == set(FILES.values())
    assert len({record.models[0].local_id for record in page.records}) == 8
    assert all(
        record.releases[0].metadata["checkpoint_file_url"] in FILES.values()
        and "weight_url" not in record.releases[0].metadata
        for record in page.records
    )
    assert client.urls[1].endswith(f"/{REV}/README.md")


@pytest.mark.parametrize(
    "document",
    [
        README.replace(FILES["fmow_pretrain.pth"], "https://example.org/weights.pth"),
        README.replace("#### fMoW Temporal Checkpoints", "#### Unknown temporal checkpoints"),
        README.replace(FILES["finetune-vit-large-e7.pth"], FILES["pretrain-vit-large-e199.pth"]),
        README.replace(
            "## Multi-Spectral SatMAE",
            "## Multi-Spectral SatMAE\n## Multi-Spectral SatMAE",
        ),
    ],
)
def test_satmae_parser_rejects_incomplete_or_ambiguous_inventory(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_readme(document, "test", "README.md")
