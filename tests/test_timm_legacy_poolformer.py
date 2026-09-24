from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.timm_legacy_poolformer import (
    TimmLegacyPoolFormerSourceAdapter,
)

_REVISION = "d" * 40
_MODELS = ("poolformer_s12", "poolformer_s24", "poolformer_s36", "poolformer_m36", "poolformer_m48")


def _source(*, wrong_url: bool = False) -> bytes:
    entries = []
    for model in _MODELS:
        url = f"https://github.com/sail-sg/poolformer/releases/download/v1.0/{model}.pth.tar"
        if wrong_url and model == "poolformer_m48":
            url = url.replace("v1.0", "v2.0")
        entries.append(f"    {model}=_cfg(url={url!r}),")
    return ("default_cfgs = dict(\n" + "\n".join(entries) + "\n)\n").encode()


class _Client:
    def __init__(self, source: bytes) -> None:
        self.responses = [
            HttpResponse(200, {}, json.dumps({"sha": _REVISION}).encode(), "https://test"),
            HttpResponse(200, {}, source, "https://test"),
        ]
        self.urls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.urls.append(url)
        return self.responses.pop(0)


def _adapter(source: bytes) -> tuple[TimmLegacyPoolFormerSourceAdapter, _Client]:
    client = _Client(source)
    return (
        TimmLegacyPoolFormerSourceAdapter(
            client=client,
            clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
        ),
        client,
    )


def test_archived_poolformer_configs_emit_all_five_exact_checkpoint_urls() -> None:
    adapter, client = _adapter(_source())

    page = adapter.fetch_page({})

    assert page.upstream_count == 5
    assert len(page.records) == 5
    assert client.urls[-1] == (
        f"https://raw.githubusercontent.com/huggingface/pytorch-image-models/"
        f"{_REVISION}/timm/models/poolformer.py"
    )
    for model, record in zip(_MODELS, page.records, strict=True):
        expected = (
            "https://github.com/sail-sg/poolformer/releases/download/v1.0/"
            f"{model}.pth.tar"
        )
        assert record.models[0].name == model
        assert record.releases[0].version == "v0.6.13"
        assert record.releases[0].metadata["weights_url"] == expected
        assert any(link.url == expected and link.relation == "weights" for link in record.links)


def test_archived_poolformer_config_rejects_unexpected_checkpoint_url() -> None:
    adapter, _ = _adapter(_source(wrong_url=True))

    with pytest.raises(ValueError, match="unexpected checkpoint URL"):
        adapter.fetch_page({})


def test_archived_poolformer_config_requires_every_declared_model() -> None:
    source = _source().replace(b"poolformer_s24=_cfg", b"unused_s24=_cfg")
    adapter, _ = _adapter(source)

    with pytest.raises(ValueError, match="missing checkpoint URLs"):
        adapter.fetch_page({})
