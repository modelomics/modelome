from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.timm_legacy_mobilevit import TimmLegacyMobileViTSourceAdapter

_REVISION = "e" * 40
_CONFIGS = {
    "mobilevit_xxs": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevit_xxs-ad385b40.pth",
    "mobilevit_xs": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevit_xs-8fbd6366.pth",
    "mobilevitv2_050": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-mvit-weights/mobilevitv2_050-49951ee2.pth",
}


def _source(*, wrong_url: bool = False, count: int = 16) -> bytes:
    configs = dict(_CONFIGS)
    configs.update(
        {
            f"mobilevit_fixture_{index:02d}": (
                f"https://github.com/example/timm-fixture/mobilevit-{index:02d}.pth"
            )
            for index in range(count - len(configs))
        }
    )
    if wrong_url:
        configs["mobilevit_xxs"] = configs["mobilevit_xxs"].replace("https://", "http://")
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


def _adapter(source: bytes) -> TimmLegacyMobileViTSourceAdapter:
    return TimmLegacyMobileViTSourceAdapter(
        client=_Client(source),
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_archived_mobilevit_config_emits_all_sixteen_direct_urls() -> None:
    page = _adapter(_source()).fetch_page({})

    assert len(page.records) == page.upstream_count == 16
    found = {
        record.models[0].name: next(
            link.url for link in record.links if link.relation == "weights"
        )
        for record in page.records
    }
    assert {key: found[key] for key in _CONFIGS} == _CONFIGS
    assert all(record.releases[0].version == "v0.6.13" for record in page.records)


def test_archived_mobilevit_config_rejects_unsupported_urls() -> None:
    with pytest.raises(ValueError, match="expected 16 literal MobileViT URLs"):
        _adapter(_source(wrong_url=True)).fetch_page({})


def test_archived_mobilevit_config_requires_expected_count() -> None:
    with pytest.raises(ValueError, match="expected 16 literal MobileViT URLs"):
        _adapter(_source(count=15)).fetch_page({})
