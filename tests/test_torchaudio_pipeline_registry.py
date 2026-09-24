from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.torchaudio_pipeline_registry import (
    TorchaudioPipelineRegistrySourceAdapter,
)

_REVISION = "e" * 40


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: bytes | Mapping[str, Any]) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/torchaudio")


def _archive(files: Mapping[str, str]) -> bytes:
    output = io.BytesIO()
    root = f"audio-{_REVISION}"
    with zipfile.ZipFile(output, "w") as package:
        for path, text in files.items():
            package.writestr(f"{root}/{path}", text)
    return output.getvalue()


_INITIALIZER = '''\
from ._source_separation_pipeline import BAR
from ._tts import TTS
from ._wav2vec2.impl import FOO
'''

_WAV2VEC = '''\
FOO = Wav2Vec2Bundle(_path="foo.pth")
FOO.__doc__ = """Training code is https://github.com/pytorch/audio/tree/main/examples/foo
and a demo is https://download.pytorch.org/torchaudio/doc-assets/foo.wav."""
'''

_SEPARATION = '''\
BAR = SourceSeparationBundle(_model_path="models/bar.pt")
BAR.__doc__ = "License: https://example.test/LICENSE"
'''

_TTS = '''\
_BASE_URL = "https://download.pytorch.org/torchaudio/models"

TTS = TacotronBundle(
    _tacotron2_path="tacotron.pth",
    _wavernn_path="wavernn.pth",
)
'''

_DOWNLOAD_HELPER = '''\
def _download(key, path, progress):
    url = f"https://download.pytorch.org/torchaudio/{key}"
'''

_WAV2VEC_HELPER = '''\
def _get_state_dict(url, dl_kwargs):
    if not url.startswith("https"):
        url = f"https://download.pytorch.org/torchaudio/models/{url}"
'''


def _catalog_archive() -> bytes:
    return _archive(
        {
            "src/torchaudio/pipelines/__init__.py": _INITIALIZER,
            "src/torchaudio/pipelines/_wav2vec2/impl.py": _WAV2VEC,
            "src/torchaudio/pipelines/_source_separation_pipeline.py": _SEPARATION,
            "src/torchaudio/pipelines/_tts/impl.py": _TTS,
            "src/torchaudio/pipelines/_wav2vec2/utils.py": _WAV2VEC_HELPER,
            "src/torchaudio/utils/download.py": _DOWNLOAD_HELPER,
        }
    )


def test_pipeline_registry_enumerates_static_public_bundles() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_catalog_archive()))
    adapter = TorchaudioPipelineRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 3
    assert page.next_state["completed_revision"] == _REVISION
    assert page.next_state["module_count"] == 5
    assert page.next_state["pipeline_count"] == 3
    assert page.next_state["weight_reference_count"] == 4
    assert len(client.calls) == 2
    assert client.calls[1][0].endswith(f"/archive/{_REVISION}.zip")

    bar, foo, tts = page.records
    assert bar.kind is ArtifactKind.MODEL_CARD
    assert bar.identifiers == (Identifier("torchaudio:pipeline", "BAR"),)
    assert bar.models[0].status is ModelStatus.RELEASED
    assert bar.raw["pipeline_source_sha256"]
    assert foo.models[0].name == "FOO"
    assert foo.models[0].identifiers == (Identifier("torchaudio:pipeline", "FOO"),)
    assert foo.releases[0].identifiers[0].namespace == "torchaudio:pipeline-asset"
    assert {
        (link.url, link.relation, link.crawl, link.model_local_ids)
        for link in foo.links
    } == {
        ("https://github.com/pytorch/audio", "source_repository", False, ()),
        (
            "https://github.com/pytorch/audio/blob/"
            f"{_REVISION}/src/torchaudio/pipelines/_wav2vec2/impl.py",
            "pipeline_definition",
            False,
            ("model:FOO",),
        ),
        (
            "https://download.pytorch.org/torchaudio/models/foo.pth",
            "weights",
            False,
            ("model:FOO",),
        ),
        (
            "https://github.com/pytorch/audio/tree/main/examples/foo",
            "official_implementation",
            False,
            ("model:FOO",),
        ),
        (
            "https://download.pytorch.org/torchaudio/doc-assets/foo.wav",
            "demo",
            False,
            ("model:FOO",),
        ),
    }
    assert {release.metadata["url"] for release in tts.releases} == {
        "https://download.pytorch.org/torchaudio/models/tacotron.pth",
        "https://download.pytorch.org/torchaudio/models/wavernn.pth",
    }
    assert any(link.relation == "license" for link in bar.links)


def test_pipeline_registry_skips_archive_when_commit_is_unchanged() -> None:
    first_client = _QueuedClient(_response({"sha": _REVISION}), _response(_catalog_archive()))
    adapter = TorchaudioPipelineRegistrySourceAdapter(client=first_client)
    first = adapter.fetch_page({})

    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client
    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.records == ()
    assert second.next_state["checked_at"]
    assert len(second_client.calls) == 1


def test_pipeline_registry_fails_closed_for_dynamic_public_bundle() -> None:
    archive = _archive(
        {
            "src/torchaudio/pipelines/__init__.py": "from ._foo import FOO\n",
            "src/torchaudio/pipelines/_foo.py": "FOO = build_bundle()\n",
            "src/torchaudio/pipelines/_wav2vec2/utils.py": _WAV2VEC_HELPER,
            "src/torchaudio/pipelines/_tts/impl.py": (
                '_BASE_URL = "https://download.pytorch.org/torchaudio/models"\n'
            ),
            "src/torchaudio/utils/download.py": _DOWNLOAD_HELPER,
        }
    )
    client = _QueuedClient(_response({"sha": _REVISION}), _response(archive))

    with pytest.raises(ValueError, match="unsupported constructor"):
        TorchaudioPipelineRegistrySourceAdapter(client=client).fetch_page({})
