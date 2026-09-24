from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.sherpa_audio_tagging import SherpaAudioTaggingSourceAdapter

_REVISION = "e" * 40
_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/"
    "sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2"
)


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


def test_sherpa_audio_tagging_adapter_keeps_exact_model_archive() -> None:
    document = "\n".join(
        (
            "This section lists pre-trained models for audio tagging.",
            "wget " + _MODEL_URL,
            "tar xvf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2",
            "wget https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "audio-tagging-models/test.wav",
            "wget https://example.org/audio-tagging-models/not-a-model.tar.bz2",
        )
    )
    client = _QueuedClient(_response({"sha": _REVISION}), _response(document))
    adapter = SherpaAudioTaggingSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 1, 2, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.title == "sherpa-onnx-zipformer-small-audio-tagging-2024-04-15"
    assert record.links[-1].url == _MODEL_URL
    assert record.releases[0].revision == _REVISION
    assert client.calls == [
        "https://api.github.com/repos/k2-fsa/sherpa/commits/master",
        f"https://raw.githubusercontent.com/k2-fsa/sherpa/{_REVISION}/"
        "docs/source/onnx/audio-tagging/pretrained_models.rst",
    ]


def test_sherpa_audio_tagging_unchanged_document_needs_no_refetch() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}))
    adapter = SherpaAudioTaggingSourceAdapter(client=client)

    page = adapter.fetch_page({"completed_revision": _REVISION, "model_count": 1})

    assert page.records == ()
    assert page.upstream_count == 1
    assert len(client.calls) == 1


def test_sherpa_audio_tagging_rejects_url_suffixes() -> None:
    malformed = _MODEL_URL + "?mirror=1"
    client = _QueuedClient(_response({"sha": _REVISION}), _response(malformed))
    adapter = SherpaAudioTaggingSourceAdapter(client=client)

    with pytest.raises(ValueError, match="no checkpoint links"):
        adapter.fetch_page({})


def test_sherpa_audio_tagging_adapter_is_bound_to_its_document() -> None:
    with pytest.raises(ValueError, match="document_path"):
        SherpaAudioTaggingSourceAdapter(document_path="README.md")
