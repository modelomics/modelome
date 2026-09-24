from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.timm_legacy_vit import TimmLegacyViTSourceAdapter

_REVISION = "d" * 40
_CONFIGS = {
    "vit_tiny_patch16_224": "https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "vit_base_patch16_224": "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "vit_small_patch16_224_dino": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
}


def _source(*, wrong_url: bool = False, count: int = 32) -> bytes:
    configs = dict(_CONFIGS)
    configs.update(
        {
            f"vit_fixture_{index:02d}": (
                f"https://storage.googleapis.com/example/timm-fixture/vit-{index:02d}.npz"
            )
            for index in range(count - len(configs))
        }
    )
    if wrong_url:
        configs["vit_tiny_patch16_224"] = configs["vit_tiny_patch16_224"].replace(
            "https://", "http://"
        )
    entries = [f"    {model}=_cfg(url={url!r})," for model, url in configs.items()]
    return ("default_cfgs = dict(\n" + "\n".join(entries) + "\n)\n").encode()


class _Client:
    def __init__(self, source: bytes) -> None:
        self.responses = [
            HttpResponse(200, {}, json.dumps({"sha": _REVISION}).encode(), "https://test"),
            HttpResponse(200, {}, source, "https://test"),
        ]

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        return self.responses.pop(0)


def _adapter(source: bytes) -> TimmLegacyViTSourceAdapter:
    return TimmLegacyViTSourceAdapter(
        client=_Client(source),
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_archived_vit_config_emits_all_thirty_two_direct_urls() -> None:
    page = _adapter(_source()).fetch_page({})

    assert len(page.records) == page.upstream_count == 32
    found = {
        record.models[0].name: next(
            link.url for link in record.links if link.relation == "weights"
        )
        for record in page.records
    }
    assert {key: found[key] for key in _CONFIGS} == _CONFIGS
    assert all(record.releases[0].version == "v0.6.13" for record in page.records)


def test_archived_vit_config_rejects_unsupported_urls() -> None:
    with pytest.raises(ValueError, match="expected 32 literal ViT URLs"):
        _adapter(_source(wrong_url=True)).fetch_page({})


def test_archived_vit_config_requires_expected_count() -> None:
    with pytest.raises(ValueError, match="expected 32 literal ViT URLs"):
        _adapter(_source(count=31)).fetch_page({})
