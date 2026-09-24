from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.timm_legacy_byobnet import TimmLegacyByobNetSourceAdapter

_REVISION = "a" * 40
_CONFIGS = {
    "gernet_s": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-ger-weights/gernet_s-756b4751.pth",
    "gernet_m": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-ger-weights/gernet_m-0873c53a.pth",
    "gernet_l": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-ger-weights/gernet_l-f31e2e8d.pth",
}


def _source(*, wrong_url: bool = False, count: int = 32) -> bytes:
    configs = dict(_CONFIGS)
    configs.update(
        {
            f"byobnet_fixture_{index:02d}": (
                f"https://github.com/example/timm-fixture/byobnet-{index:02d}.pth"
            )
            for index in range(count - len(configs))
        }
    )
    if wrong_url:
        configs["gernet_s"] = configs["gernet_s"].replace("https://", "http://")
    entries = [f"    {model!r}: _cfg(url={url!r})," for model, url in configs.items()]
    return ("default_cfgs = {\n" + "\n".join(entries) + "\n}\n").encode()


class _Client:
    def __init__(self, source: bytes) -> None:
        self.responses = [
            HttpResponse(200, {}, json.dumps({"sha": _REVISION}).encode(), "https://test"),
            HttpResponse(200, {}, source, "https://test"),
        ]

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        return self.responses.pop(0)


def _adapter(source: bytes) -> TimmLegacyByobNetSourceAdapter:
    return TimmLegacyByobNetSourceAdapter(
        client=_Client(source),
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_archived_byobnet_config_emits_all_thirty_two_urls() -> None:
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


def test_archived_byobnet_config_rejects_unsupported_urls() -> None:
    with pytest.raises(ValueError, match="expected 32 literal BYOBNet URLs"):
        _adapter(_source(wrong_url=True)).fetch_page({})


def test_archived_byobnet_config_requires_expected_count() -> None:
    with pytest.raises(ValueError, match="expected 32 literal BYOBNet URLs"):
        _adapter(_source(count=31)).fetch_page({})
