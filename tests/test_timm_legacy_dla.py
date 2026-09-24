from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.timm_legacy_dla import TimmLegacyDLASourceAdapter

_REVISION = "f" * 40
_CONFIGS = {
    "dla34": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla34-2b83ff04.pth",
    "dla46_c": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/dla46_c-9b68d685.pth",
    "dla60_res2net": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net_dla60_4s-d88db7f9.pth",
}


def _source(*, wrong_url: bool = False, count: int = 12) -> bytes:
    configs = dict(_CONFIGS)
    configs.update(
        {
            f"dla_fixture_{index:02d}": (
                f"https://github.com/example/timm-fixture/dla-{index:02d}.pth"
            )
            for index in range(count - len(configs))
        }
    )
    if wrong_url:
        configs["dla34"] = configs["dla34"].replace("https://", "http://")
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


def _adapter(source: bytes) -> TimmLegacyDLASourceAdapter:
    return TimmLegacyDLASourceAdapter(
        client=_Client(source),
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_archived_dla_config_emits_all_twelve_direct_urls() -> None:
    page = _adapter(_source()).fetch_page({})

    assert len(page.records) == page.upstream_count == 12
    found = {
        record.models[0].name: next(
            link.url for link in record.links if link.relation == "weights"
        )
        for record in page.records
    }
    assert {key: found[key] for key in _CONFIGS} == _CONFIGS
    assert all(record.releases[0].version == "v0.6.13" for record in page.records)


def test_archived_dla_config_rejects_unsupported_urls() -> None:
    with pytest.raises(ValueError, match="expected 12 literal DLA URLs"):
        _adapter(_source(wrong_url=True)).fetch_page({})


def test_archived_dla_config_requires_expected_count() -> None:
    with pytest.raises(ValueError, match="expected 12 literal DLA URLs"):
        _adapter(_source(count=11)).fetch_page({})
