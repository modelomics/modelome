from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.keras_convnext_weights import (
    KerasConvNeXtWeightsSourceAdapter,
    _parse_manifest,
)

REVISION = "c" * 40
_SOURCE = """\
BASE_WEIGHTS_PATH = ("https://weights.example/convnext/")
WEIGHTS_HASHES = {
    "convnext_tiny": (
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
    "convnext_small": (
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "bcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789a",
    ),
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


def test_convnext_manifest_emits_exact_top_and_no_top_links_without_fetching_weights() -> None:
    source_url = (
        "https://raw.githubusercontent.com/keras-team/keras/"
        f"{REVISION}/keras/src/applications/convnext.py"
    )
    client = _QueuedClient(_response({"sha": REVISION}), _response(_SOURCE.encode()))
    adapter = KerasConvNeXtWeightsSourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    assert page.next_state["model_count"] == 4
    assert client.calls == [
        "https://api.github.com/repos/keras-team/keras/commits/master",
        source_url,
    ]
    top = next(record for record in page.records if record.raw["include_top"])
    no_top = next(record for record in page.records if not record.raw["include_top"])
    assert top.kind is ArtifactKind.WEIGHTS
    assert top.canonical_url == ("https://weights.example/convnext/convnext_small.h5")
    assert no_top.canonical_url == ("https://weights.example/convnext/convnext_small_notop.h5")
    assert top.releases[0].metadata["sha256"] == (
        "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )
    assert no_top.releases[0].metadata["sha256"] == (
        "bcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789a"
    )
    assert top.identifiers == (Identifier("keras-applications:checkpoint", top.canonical_url),)
    assert all(not link.crawl for record in page.records for link in record.links)


def test_convnext_manifest_skips_unchanged_revision() -> None:
    adapter = KerasConvNeXtWeightsSourceAdapter(
        client=_QueuedClient(_response({"sha": REVISION}), _response(_SOURCE.encode()))
    )
    initial = adapter.fetch_page({})
    adapter.client = _QueuedClient(_response({"sha": REVISION}))

    page = adapter.fetch_page(initial.next_state)

    assert page.records == ()
    assert page.upstream_count == 4


def test_convnext_parser_rejects_unrecognized_variants_and_incomplete_hashes() -> None:
    with pytest.raises(ValueError, match="unexpected ConvNeXt variant"):
        _parse_manifest(
            'BASE_WEIGHTS_PATH = "https://weights.example/"\n'
            'WEIGHTS_HASHES = {"other_model": ("a", "b")}\n',
            "https://source.example/convnext.py",
            REVISION,
            "fixture",
        )
    with pytest.raises(ValueError, match="SHA-256 hashes"):
        _parse_manifest(
            'BASE_WEIGHTS_PATH = "https://weights.example/"\n'
            'WEIGHTS_HASHES = {"convnext_tiny": ("short", "short")}\n',
            "https://source.example/convnext.py",
            REVISION,
            "fixture",
        )
