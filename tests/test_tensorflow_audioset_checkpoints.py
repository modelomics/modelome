from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.tensorflow_audioset_checkpoints import (
    TensorFlowAudioSetCheckpointSourceAdapter,
)

REVISION = "a" * 40
YAMNET = "https://storage.googleapis.com/audioset/yamnet.h5"
VGGISH = "https://storage.googleapis.com/audioset/vggish_model.ckpt"
VGGISH_PCA = "https://storage.googleapis.com/audioset/vggish_pca_params.npz"


class Client:
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
        assert params is None
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: str | dict[str, str]) -> HttpResponse:
    raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    return HttpResponse(200, {}, raw, "https://github.com/tensorflow/models")


def test_yamnet_checkpoint_map_keeps_exact_first_party_weight_url() -> None:
    client = Client(
        response({"sha": REVISION}),
        response(f"[YAMNet model weights]({YAMNET})\n`curl -O {YAMNET}`"),
    )
    adapter = TensorFlowAudioSetCheckpointSourceAdapter(
        document_path="research/audioset/yamnet/README.md", client=client
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.title == "YAMNet"
    assert record.releases[0].metadata["checkpoint_urls"] == [YAMNET]
    assert record.links[-1].url == YAMNET
    assert record.models[0].identifiers == (
        Identifier("tensorflow-audioset:model", "yamnet"),
    )
    assert client.calls == [
        adapter.commit_url,
        adapter.raw_url(REVISION),
    ]


def test_vggish_model_map_keeps_checkpoint_and_required_pca_asset() -> None:
    client = Client(
        response({"sha": REVISION}),
        response(f"VGGish model checkpoint: {VGGISH}\nEmbedding PCA parameters: {VGGISH_PCA}"),
    )
    adapter = TensorFlowAudioSetCheckpointSourceAdapter(
        document_path="research/audioset/vggish/README.md", client=client
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert page.records[0].releases[0].metadata["checkpoint_urls"] == [
        VGGISH,
        VGGISH_PCA,
    ]
    assert [link.relation for link in page.records[0].links] == [
        "model_card",
        "weights",
        "weights",
    ]


def test_adapter_rejects_unapproved_document_path() -> None:
    with pytest.raises(ValueError, match="YAMNet or VGGish README"):
        TensorFlowAudioSetCheckpointSourceAdapter(document_path="research/other/README.md")
