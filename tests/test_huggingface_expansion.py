from __future__ import annotations

import json
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.huggingface import HuggingFaceSourceAdapter


class _Client:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append(dict(params or {}))
        return HttpResponse(
            status=200,
            headers={},
            body=json.dumps(self.payload).encode(),
            url=url,
        )


def test_huggingface_records_additional_checkpoint_formats_and_shard_indexes() -> None:
    filenames = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.msgpack",
        "model.flax",
        "model.ggml",
        "model.tflite",
        "model.mlpackage",
        "bert_model.ckpt.index",
        "bert_model.ckpt.data-00000-of-00001",
        "variables.index",
        "variables.data-00000-of-00001",
        "unpaired.index",
        "unpaired-shard.data-00000-of-00001",
        "metadata.json",
    ]
    client = _Client(
        [
            {
                "id": "lab/multiformat-model",
                "sha": "abc123",
                "config": {"source": "https://github.com/lab/architecture"},
                "siblings": [{"rfilename": name} for name in filenames],
            }
        ]
    )
    page = HuggingFaceSourceAdapter(client=client).fetch_page({})

    record = page.records[0]
    weights = {
        link.url.split("/resolve/abc123/", maxsplit=1)[-1]
        for link in record.links
        if link.relation == "weights"
    }
    assert weights == set(filenames) - {
        "metadata.json",
        "unpaired.index",
        "unpaired-shard.data-00000-of-00001",
    }
    assert record.releases[0].metadata["weight_files"] == sorted(weights)
    assert all(link.crawl is False for link in record.links if link.relation == "weights")
    assert (
        "https://github.com/lab/architecture",
        "code_reference",
        "$.config",
    ) in {(link.url, link.relation, link.locator) for link in record.links}
    assert client.calls[0]["config"] == "true"
