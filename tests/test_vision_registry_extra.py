from __future__ import annotations

import json

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.sources.vision_registry_extra import OnnxModelZooHubSourceAdapter


class Client:
    def __init__(self, *, total: int = 2326) -> None:
        self.total = total
        self.calls: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        body = json.dumps(
            [
                {
                    "id": "onnxmodelzoo/resnet18_Opset18_timm",
                    "sha": "a" * 40,
                    "pipeline_tag": "image-classification",
                    "siblings": [
                        {"rfilename": "README.md"},
                        {"rfilename": "model.onnx"},
                        {"rfilename": "config.json"},
                    ],
                }
            ]
        ).encode()
        return HttpResponse(
            status=200,
            headers={
                "X-Total-Count": str(self.total),
                "Link": (
                    '<https://huggingface.co/api/models?author=onnxmodelzoo&cursor=next>'
                    '; rel="next"'
                ),
            },
            body=body,
            url=url,
        )


def test_onnx_hub_adapter_uses_owner_scope_and_retains_model_file_links() -> None:
    client = Client()
    page = OnnxModelZooHubSourceAdapter(client=client).fetch_page({})

    assert client.calls[0][0] == "https://huggingface.co/api/models?author=onnxmodelzoo"
    assert client.calls[0][1]["limit"] == 100
    assert page.complete is False
    assert page.upstream_count == 2326
    record = page.records[0]
    assert record.source_record_id == "onnxmodelzoo/resnet18_Opset18_timm"
    assert any(
        link.url.endswith("/resolve/" + "a" * 40 + "/model.onnx") and link.relation == "weights"
        for link in record.links
    )


def test_onnx_hub_exact_huggingface_id_joins_global_hub_record() -> None:
    client = Client(total=1)
    record = OnnxModelZooHubSourceAdapter(client=client).fetch_page({}).records[0]
    onnx_seed = source_record_to_entry_seed(record, source="onnx-model-zoo-hub")
    hub_id = "onnxmodelzoo/resnet18_Opset18_timm"
    huggingface_seed = {
        "source": "huggingface",
        "source_record_id": hub_id,
        "canonical_url": f"https://huggingface.co/{hub_id}",
        "title": "ResNet18 Opset18 timm",
        "kind": "model_card",
        "identifiers": [
            {"namespace": "huggingface:model", "value": hub_id},
        ],
        "models": [
            {
                "local_id": "model",
                "name": "ResNet18 Opset18 timm",
                "identifiers": [
                    {"namespace": "huggingface:model", "value": hub_id},
                ],
            }
        ],
    }

    entries = build_entries([onnx_seed, huggingface_seed]).entries

    assert len(entries) == 1
    assert {(member.source, member.source_record_id) for member in entries[0].members} == {
        ("onnx-model-zoo-hub", hub_id),
        ("huggingface", hub_id),
    }
    assert [(item.namespace, item.value) for item in entries[0].identifiers] == [
        ("huggingface:model", hub_id)
    ]
    assert len(client.calls) == 1  # Listing metadata only; the ONNX link is not downloaded.
    assert any(
        link.url == f"https://huggingface.co/{hub_id}/resolve/{'a' * 40}/model.onnx"
        and link.relation == "weights"
        for link in record.links
    )


def test_onnx_hub_adapter_rejects_inventory_above_configured_bound() -> None:
    with pytest.raises(ValueError, match="exceeds 100 model entries"):
        OnnxModelZooHubSourceAdapter(client=Client(total=101), max_entries=100).fetch_page({})
