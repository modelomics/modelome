from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.timm_legacy_resnetv2 import (
    TimmLegacyResNetV2SourceAdapter,
)

_REVISION = "e" * 40
_URLS = {
    "resnetv2_50x1_bitm": "https://storage.googleapis.com/bit_models/BiT-M-R50x1-ILSVRC2012.npz",
    "resnetv2_50x3_bitm": "https://storage.googleapis.com/bit_models/BiT-M-R50x3-ILSVRC2012.npz",
    "resnetv2_101x1_bitm": "https://storage.googleapis.com/bit_models/BiT-M-R101x1-ILSVRC2012.npz",
    "resnetv2_101x3_bitm": "https://storage.googleapis.com/bit_models/BiT-M-R101x3-ILSVRC2012.npz",
    "resnetv2_152x2_bitm": "https://storage.googleapis.com/bit_models/BiT-M-R152x2-ILSVRC2012.npz",
    "resnetv2_152x4_bitm": "https://storage.googleapis.com/bit_models/BiT-M-R152x4-ILSVRC2012.npz",
    "resnetv2_50x1_bitm_in21k": "https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz",
    "resnetv2_50x3_bitm_in21k": "https://storage.googleapis.com/bit_models/BiT-M-R50x3.npz",
    "resnetv2_101x1_bitm_in21k": "https://storage.googleapis.com/bit_models/BiT-M-R101x1.npz",
    "resnetv2_101x3_bitm_in21k": "https://storage.googleapis.com/bit_models/BiT-M-R101x3.npz",
    "resnetv2_152x2_bitm_in21k": "https://storage.googleapis.com/bit_models/BiT-M-R152x2.npz",
    "resnetv2_152x4_bitm_in21k": "https://storage.googleapis.com/bit_models/BiT-M-R152x4.npz",
    "resnetv2_50x1_bit_distilled": "https://storage.googleapis.com/bit_models/distill/R50x1_224.npz",
    "resnetv2_152x2_bit_teacher": "https://storage.googleapis.com/bit_models/distill/R152x2_T_224.npz",
    "resnetv2_152x2_bit_teacher_384": "https://storage.googleapis.com/bit_models/distill/R152x2_T_384.npz",
    "resnetv2_50": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnetv2_50_a1h-000cdf49.pth",
    "resnetv2_101": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnetv2_101_a1h-5d01f016.pth",
    "resnetv2_50d_gn": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnetv2_50d_gn_ah-c415c11a.pth",
    "resnetv2_50d_evos": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnetv2_50d_evos_ah-7c4dd548.pth",
}


def _source(*, wrong_url: bool = False) -> bytes:
    entries = []
    for model, url in _URLS.items():
        if wrong_url and model == "resnetv2_50d_evos":
            url = url.replace("https://", "http://")
        entries.append(f"    {model!r}: _cfg(url={url!r}),")
    return ("default_cfgs = {\n" + "\n".join(entries) + "\n}\n").encode()


class _Client:
    def __init__(self, source: bytes) -> None:
        self.responses = [
            HttpResponse(200, {}, json.dumps({"sha": _REVISION}).encode(), "https://test"),
            HttpResponse(200, {}, source, "https://test"),
        ]

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        return self.responses.pop(0)


def _adapter(source: bytes) -> TimmLegacyResNetV2SourceAdapter:
    return TimmLegacyResNetV2SourceAdapter(
        client=_Client(source),
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_archived_resnetv2_config_emits_all_nineteen_checkpoint_urls() -> None:
    page = _adapter(_source()).fetch_page({})

    assert page.upstream_count == len(_URLS) == 19
    assert len(page.records) == 19
    found = {
        record.models[0].name: next(
            link.url for link in record.links if link.relation == "weights"
        )
        for record in page.records
    }
    assert found == _URLS
    assert all(record.releases[0].version == "v0.6.13" for record in page.records)


def test_archived_resnetv2_config_rejects_url_changes() -> None:
    with pytest.raises(ValueError, match="unsupported checkpoint URL"):
        _adapter(_source(wrong_url=True)).fetch_page({})


def test_archived_resnetv2_config_requires_all_expected_entries() -> None:
    source = _source().replace(b"resnetv2_50x1_bitm': _cfg", b"other_model': _cfg")

    with pytest.raises(ValueError, match="missing checkpoint URLs"):
        _adapter(source).fetch_page({})
