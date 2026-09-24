from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.torchaudio_pipeline_registry import (
    TorchaudioPipelineRegistrySourceAdapter,
)

_REVISION = "f" * 40


class _QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: bytes | Mapping[str, Any]) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/audio")


def _rnnt_archive() -> bytes:
    files = {
        "src/torchaudio/pipelines/__init__.py": (
            "from .rnnt_pipeline import EMFORMER_RNNT_BASE_LIBRISPEECH\n"
        ),
        "src/torchaudio/pipelines/rnnt_pipeline.py": (
            'EMFORMER_RNNT_BASE_LIBRISPEECH = RNNTBundle(\n'
            '    _rnnt_path="models/emformer.pt",\n'
            '    _global_stats_path="pipeline-assets/global_stats.json",\n'
            '    _sp_model_path="pipeline-assets/spm.model",\n'
            ')\n'
        ),
        "src/torchaudio/pipelines/_wav2vec2/utils.py": (
            'def _get_state_dict(url, kwargs):\n'
            '    url = f"https://download.pytorch.org/torchaudio/models/{url}"\n'
        ),
        "src/torchaudio/pipelines/_tts/impl.py": (
            '_BASE_URL = "https://download.pytorch.org/torchaudio/models"\n'
        ),
        "src/torchaudio/utils/download.py": (
            'def _download(key, path, progress):\n'
            '    url = f"https://download.pytorch.org/torchaudio/{key}"\n'
        ),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path, source in files.items():
            archive.writestr(f"audio-{_REVISION}/{path}", source)
    return output.getvalue()


def test_rnnt_pipeline_includes_declared_auxiliary_checkpoint_assets() -> None:
    adapter = TorchaudioPipelineRegistrySourceAdapter(
        client=_QueueClient(_response({"sha": _REVISION}), _response(_rnnt_archive()))
    )

    page = adapter.fetch_page({})

    record = page.records[0]
    assert record.title == "EMFORMER_RNNT_BASE_LIBRISPEECH"
    assert {
        (item["field"], item["declared_path"]) for item in record.raw["weights"]
    } == {
        ("_global_stats_path", "pipeline-assets/global_stats.json"),
        ("_rnnt_path", "models/emformer.pt"),
        ("_sp_model_path", "pipeline-assets/spm.model"),
    }
    assert {release.metadata["url"] for release in record.releases} == {
        "https://download.pytorch.org/torchaudio/pipeline-assets/global_stats.json",
        "https://download.pytorch.org/torchaudio/models/emformer.pt",
        "https://download.pytorch.org/torchaudio/pipeline-assets/spm.model",
    }
    assert page.next_state["weight_reference_count"] == 3
