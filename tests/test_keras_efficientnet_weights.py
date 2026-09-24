from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.keras_efficientnet_weights import (
    KerasEfficientNetWeightsSourceAdapter,
    _parse_manifest,
)

REVISION = "e" * 40
_SOURCE = """\
BASE_WEIGHTS_PATH = "https://weights.example/"
WEIGHTS_HASHES = {
    "b0": ("0123456789abcdef0123456789abcdef", "abcdef0123456789abcdef0123456789"),
    "b1": ("1123456789abcdef0123456789abcdef", "bcdef0123456789abcdef0123456789a"),
}
"""


class _QueuedClient:
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
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: bytes | Mapping[str, Any]) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixture.test/keras")


def test_efficientnet_manifest_emits_exact_declared_top_and_no_top_assets() -> None:
    source_url = (
        "https://raw.githubusercontent.com/keras-team/keras/"
        f"{REVISION}/keras/src/applications/efficientnet.py"
    )
    client = _QueuedClient(_response({"sha": REVISION}), _response(_SOURCE.encode()))
    page = KerasEfficientNetWeightsSourceAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    assert client.calls == [
        "https://api.github.com/repos/keras-team/keras/commits/master",
        source_url,
    ]
    top = next(
        record
        for record in page.records
        if record.raw["variant"] == "efficientnetb0" and record.raw["include_top"]
    )
    no_top = next(
        record
        for record in page.records
        if record.raw["variant"] == "efficientnetb0" and not record.raw["include_top"]
    )
    assert top.kind is ArtifactKind.WEIGHTS
    assert top.canonical_url == "https://weights.example/efficientnetb0.h5"
    assert no_top.canonical_url == "https://weights.example/efficientnetb0_notop.h5"
    assert top.releases[0].metadata["checksum"] == "0123456789abcdef0123456789abcdef"
    assert top.releases[0].metadata["checksum_algorithm"] == "md5"
    assert not any(link.crawl for record in page.records for link in record.links)


def test_efficientnet_adapter_skips_fetch_for_same_source_revision() -> None:
    adapter = KerasEfficientNetWeightsSourceAdapter(
        client=_QueuedClient(_response({"sha": REVISION}), _response(_SOURCE.encode()))
    )
    initial = adapter.fetch_page({})
    adapter.client = _QueuedClient(_response({"sha": REVISION}))

    page = adapter.fetch_page(initial.next_state)

    assert page.records == ()
    assert page.upstream_count == 4


def test_efficientnet_parser_rejects_unknown_variants_and_invalid_hashes() -> None:
    with pytest.raises(ValueError, match="unexpected EfficientNet variant"):
        _parse_manifest(
            'BASE_WEIGHTS_PATH = "https://weights.example/"\n'
            'WEIGHTS_HASHES = {"b8": ("00000000000000000000000000000000", '
            '"11111111111111111111111111111111")}\n',
            "https://source.example/efficientnet.py",
            REVISION,
            "fixture",
        )
    with pytest.raises(ValueError, match="MD5 hashes"):
        _parse_manifest(
            'BASE_WEIGHTS_PATH = "https://weights.example/"\n'
            'WEIGHTS_HASHES = {"b0": ("short", "short")}\n',
            "https://source.example/efficientnet.py",
            REVISION,
            "fixture",
        )
