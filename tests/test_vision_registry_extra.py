from __future__ import annotations

import json

import pytest

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


def test_onnx_hub_adapter_rejects_inventory_above_configured_bound() -> None:
    with pytest.raises(ValueError, match="exceeds 100 model entries"):
        OnnxModelZooHubSourceAdapter(client=Client(total=101), max_entries=100).fetch_page({})
