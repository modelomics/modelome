from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.sherpa_source_separation import SherpaSourceSeparationSourceAdapter

_REVISION = "d" * 40
_PREFIX = "https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/"


def _model_link(filename: str) -> str:
    return f"* - `{filename} <{_PREFIX}{filename}>`_"


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


def _response(body: str | Mapping[str, Any]) -> HttpResponse:
    raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    return HttpResponse(200, {}, raw, "https://fixtures.test/sherpa")


def test_sherpa_source_separation_index_extracts_only_declared_model_files() -> None:
    document = "\n".join(
        (
            _model_link("sherpa-onnx-spleeter-2stems.tar.bz2"),
            _model_link("sherpa-onnx-spleeter-2stems-int8.tar.bz2"),
            _model_link("sherpa-onnx-spleeter-2stems-fp16.tar.bz2"),
            _model_link("UVR_MDXNET_9482.onnx"),
            _model_link("UVR_MDXNET_9482.onnx"),
            "wget " + _PREFIX + "audio_example.wav",
        )
    )
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(document),
    )
    adapter = SherpaSourceSeparationSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 1, 2, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    assert [record.releases[0].version for record in page.records] == [
        "UVR_MDXNET_9482.onnx",
        "sherpa-onnx-spleeter-2stems-fp16.tar.bz2",
        "sherpa-onnx-spleeter-2stems-int8.tar.bz2",
        "sherpa-onnx-spleeter-2stems.tar.bz2",
    ]
    assert page.records[0].links[-1].url == _PREFIX + "UVR_MDXNET_9482.onnx"
    assert page.records[0].releases[0].revision == _REVISION
    assert client.calls == [
        "https://api.github.com/repos/k2-fsa/sherpa/commits/master",
        f"https://raw.githubusercontent.com/k2-fsa/sherpa/{_REVISION}/"
        "docs/source/onnx/source-separation/models.rst",
    ]


def test_sherpa_source_separation_skips_unchanged_pinned_document() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}))
    adapter = SherpaSourceSeparationSourceAdapter(client=client)

    page = adapter.fetch_page({"completed_revision": _REVISION, "model_count": 21})

    assert page.records == ()
    assert page.upstream_count == 21
    assert len(client.calls) == 1


def test_sherpa_model_link_must_match_exact_release_filename() -> None:
    bad = (
        "`UVR_MDXNET_9482.onnx <"
        + _PREFIX
        + "UVR_MDXNET_9482.onnx?mirror=1>`_"
    )
    client = _QueuedClient(_response({"sha": _REVISION}), _response(bad))
    adapter = SherpaSourceSeparationSourceAdapter(client=client)

    with pytest.raises(ValueError, match="unexpected filename"):
        adapter.fetch_page({})


def test_sherpa_source_adapter_is_bound_to_its_exact_document() -> None:
    with pytest.raises(ValueError, match="document_path"):
        SherpaSourceSeparationSourceAdapter(document_path="README.md")
