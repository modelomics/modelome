from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.keras_resnet_weights import KerasResNetWeightsSourceAdapter, _parse_manifest

REVISION = "d" * 40
_SOURCE = """\
BASE_WEIGHTS_PATH = "https://weights.example/resnet/"
WEIGHTS_HASHES = {
    "resnet50": ("0123456789abcdef0123456789abcdef", "abcdef0123456789abcdef0123456789"),
    "resnet101v2": ("1123456789abcdef0123456789abcdef", "bcdef0123456789abcdef0123456789a"),
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


def test_resnet_manifest_emits_declared_top_and_no_top_links() -> None:
    source_url = (
        "https://raw.githubusercontent.com/keras-team/keras/"
        f"{REVISION}/keras/src/applications/resnet.py"
    )
    client = _QueuedClient(_response({"sha": REVISION}), _response(_SOURCE.encode()))
    page = KerasResNetWeightsSourceAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    assert client.calls == [
        "https://api.github.com/repos/keras-team/keras/commits/master",
        source_url,
    ]
    top = next(
        record
        for record in page.records
        if record.raw["variant"] == "resnet50" and record.raw["include_top"]
    )
    no_top = next(
        record
        for record in page.records
        if record.raw["variant"] == "resnet50" and not record.raw["include_top"]
    )
    assert top.kind is ArtifactKind.WEIGHTS
    assert top.canonical_url == (
        "https://weights.example/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
    )
    assert no_top.canonical_url == (
        "https://weights.example/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    assert top.releases[0].metadata["checksum"] == "0123456789abcdef0123456789abcdef"
    assert top.releases[0].metadata["checksum_algorithm"] == "md5"
    assert not any(link.crawl for record in page.records for link in record.links)


def test_resnet_adapter_skips_source_fetch_for_known_revision() -> None:
    adapter = KerasResNetWeightsSourceAdapter(
        client=_QueuedClient(_response({"sha": REVISION}), _response(_SOURCE.encode()))
    )
    initial = adapter.fetch_page({})
    adapter.client = _QueuedClient(_response({"sha": REVISION}))

    page = adapter.fetch_page(initial.next_state)

    assert page.records == ()
    assert page.upstream_count == 4


def test_resnet_parser_rejects_unrecognized_variants_and_bad_checksums() -> None:
    with pytest.raises(ValueError, match="unexpected ResNet variant"):
        _parse_manifest(
            'BASE_WEIGHTS_PATH = "https://weights.example/"\n'
            'WEIGHTS_HASHES = {"other_model": ("00000000000000000000000000000000", '
            '"11111111111111111111111111111111")}\n',
            "https://source.example/resnet.py",
            REVISION,
            "fixture",
        )
    with pytest.raises(ValueError, match="MD5 hashes"):
        _parse_manifest(
            'BASE_WEIGHTS_PATH = "https://weights.example/"\n'
            'WEIGHTS_HASHES = {"resnet50": ("short", "short")}\n',
            "https://source.example/resnet.py",
            REVISION,
            "fixture",
        )
