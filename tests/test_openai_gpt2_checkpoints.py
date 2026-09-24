from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.openai_gpt2_checkpoints import OpenAIGPT2CheckpointSourceAdapter

_REVISION = "b" * 40
_ROOT = "https://openaipublic.blob.core.windows.net/gpt-2"


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if "/commits/" in url:
            return HttpResponse(200, {}, f'{{"sha":"{_REVISION}"}}'.encode(), url)
        if url.endswith("/DEVELOPERS.md"):
            body = b"""python3 download_model.py 124M
python3 download_model.py 355M
python3 download_model.py 774M
python3 download_model.py 1558M
"""
            return HttpResponse(200, {}, body, url)
        if url.endswith("/download_model.py"):
            body = b'''for filename in ['checkpoint','encoder.json','hparams.json',
    'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:
    r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/" + subdir + "/" + filename,
        stream=True)
'''
            return HttpResponse(200, {}, body, url)
        raise AssertionError(f"unexpected URL {url}")


def test_official_gpt2_declarations_emit_only_exact_tensorflow_weight_shards() -> None:
    client = FakeClient()
    adapter = OpenAIGPT2CheckpointSourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    assert {record.raw["checkpoint_handle"] for record in page.records} == {
        "124M",
        "355M",
        "774M",
        "1558M",
    }
    assert {
        record.links[-1].url for record in page.records
    } == {
        f"{_ROOT}/models/{model}/model.ckpt.data-00000-of-00001"
        for model in ("124M", "355M", "774M", "1558M")
    }
    assert all(record.links[-1].relation == "weights" for record in page.records)
    assert not any("/models/" in url for url in client.calls)


def test_adapter_is_only_in_a_disabled_proposal() -> None:
    path = Path(__file__).parents[1] / "config/proposals/openai_gpt2_checkpoints.toml"
    proposal = tomllib.loads(path.read_text())["source"][0]

    assert proposal["adapter"] == "openai_gpt2_checkpoints"
    assert proposal["enabled"] is False
    assert proposal["repository"] == "openai/gpt-2"
