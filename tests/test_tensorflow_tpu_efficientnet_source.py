from __future__ import annotations

import json
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.tensorflow_tpu_efficientnet import (
    TensorFlowTPUEfficientNetSourceAdapter,
    _parse_checkpoint_matrix,
)

REVISION = "b" * 40
BASE_B0 = (
    "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/"
    "ckpts/efficientnet-b0.tar.gz"
)
BASE_B1 = (
    "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/"
    "ckpts/efficientnet-b1.tar.gz"
)
AA_B0 = (
    "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/"
    "ckptsaug/efficientnet-b0.tar.gz"
)
NS_B0 = (
    "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/"
    "noisystudent/noisy_student_efficientnet-b0.tar.gz"
)


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self, url: str, *, params: Any = None, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        assert params is None
        self.calls.append(url)
        return self.responses.pop(0)


def response(value: str | dict[str, Any]) -> HttpResponse:
    body = json.dumps(value).encode() if isinstance(value, dict) else value.encode()
    return HttpResponse(status=200, headers={}, body=body, url="https://example.test")


def test_tpu_efficientnet_matrix_preserves_variant_and_method_columns() -> None:
    markdown = "\n".join(
        [
            "# EfficientNets",
            "",
            "## 2. Using Pretrained EfficientNet Checkpoints",
            "|               | B0 | B1 | B2 |",
            "|----------     | --- | --- | --- |",
            "| Baseline preprocessing | 76.7% ([ckpt](" + BASE_B0 + ")) | "
            "78.7% ([ckpt](" + BASE_B1 + ")) | |",
            "| AutoAugment (AA) | 77.1% ([ckpt](" + AA_B0 + ")) | | |",
            "| NoisyStudent + RA | 78.8% ([ckpt](" + NS_B0 + ")) | | |",
            "## 3. Using EfficientNet as Feature Extractor",
            "| Model | Checkpoint |",
            "| --- | --- |",
            "| unrelated | [ckpt](https://storage.googleapis.com/cloud-tpu-"
            "checkpoints/efficientnet/eval_data/labels_map.json) |",
        ]
    )

    entries = _parse_checkpoint_matrix(markdown)

    assert entries == (
        (
            "Baseline preprocessing",
            "B0",
            BASE_B0,
        ),
        (
            "Baseline preprocessing",
            "B1",
            BASE_B1,
        ),
        (
            "AutoAugment (AA)",
            "B0",
            AA_B0,
        ),
        ("NoisyStudent + RA", "B0", NS_B0),
    )


def test_tpu_efficientnet_adapter_pins_revision_and_emits_exact_checkpoint_refs() -> None:
    markdown = "\n".join(
        [
            "## 2. Using Pretrained EfficientNet Checkpoints",
            "| | B0 |",
            "|---|---|",
            "| Baseline preprocessing | 76.7% ([ckpt](" + BASE_B0 + ")) |",
        ]
    )
    client = Client(response({"sha": REVISION}), response(markdown))
    adapter = TensorFlowTPUEfficientNetSourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.title == "Baseline preprocessing / EfficientNet-B0"
    assert record.releases[0].metadata["training_method"] == "Baseline preprocessing"
    assert record.releases[0].metadata["variant"] == "B0"
    assert record.links[1].url == (
        BASE_B0
    )
    assert REVISION in record.raw["revision"]
    assert client.calls == [
        adapter.commit_url,
        f"https://raw.githubusercontent.com/tensorflow/tpu/{REVISION}/models/official/efficientnet/README.md",
    ]
