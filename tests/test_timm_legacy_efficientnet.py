from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.timm_legacy_efficientnet import (
    TimmLegacyEfficientNetSourceAdapter,
)

_REVISION = "f" * 40
_CONFIGS = {
    "mnasnet_100": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth",
    "semnasnet_075": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/semnasnet_075-18710866.pth",
    "semnasnet_100": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth",
}


def _source(*, wrong_url: bool = False, count: int = 95) -> bytes:
    configs = dict(_CONFIGS)
    configs.update(
        {
            f"efficientnet_fixture_{index:02d}": (
                f"https://github.com/example/timm-fixture/weights-{index:02d}.pth"
            )
            for index in range(count - len(configs))
        }
    )
    if wrong_url:
        configs["mnasnet_100"] = configs["mnasnet_100"].replace("https://", "http://")
    entries = [
        f"    {model!r}: _cfg(url={url!r})," for model, url in configs.items()
    ]
    return ("default_cfgs = {\n" + "\n".join(entries) + "\n}\n").encode()


class _Client:
    def __init__(self, source: bytes) -> None:
        self.responses = [
            HttpResponse(200, {}, json.dumps({"sha": _REVISION}).encode(), "https://test"),
            HttpResponse(200, {}, source, "https://test"),
        ]

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        return self.responses.pop(0)


def _adapter(source: bytes) -> TimmLegacyEfficientNetSourceAdapter:
    return TimmLegacyEfficientNetSourceAdapter(
        client=_Client(source),
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_archived_efficientnet_config_emits_ninety_five_direct_urls() -> None:
    page = _adapter(_source()).fetch_page({})

    assert len(page.records) == page.upstream_count == 95
    found = {
        record.models[0].name: next(
            link.url for link in record.links if link.relation == "weights"
        )
        for record in page.records
    }
    assert {key: found[key] for key in _CONFIGS} == _CONFIGS
    assert all(record.releases[0].version == "v0.6.13" for record in page.records)


def test_archived_efficientnet_config_rejects_unsupported_urls() -> None:
    with pytest.raises(ValueError, match="expected 95 literal EfficientNet URLs"):
        _adapter(_source(wrong_url=True)).fetch_page({})


def test_archived_efficientnet_config_requires_expected_count() -> None:
    with pytest.raises(ValueError, match="expected 95 literal EfficientNet URLs"):
        _adapter(_source(count=94)).fetch_page({})
