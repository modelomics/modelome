"""Focused coverage for OpenCSG public model catalog normalization."""

import json

from modelome.http import HttpResponse
from modelome.sources.opencsg import OpenCsgModelsSourceAdapter


class _Client:
    def get(self, url, *, params, headers):
        payload = {
            "data": [
                {
                    "path": "example/model",
                    "name": "model",
                    "license": "Apache-2.0",
                    "tags": [{"name": "text-generation"}],
                    "revision": "abc123",
                    "metadata": {
                        "model_params": 7.5,
                        "architecture": "LlamaForCausalLM",
                        "tensor_type": "BF16",
                        "model_type": "text-generation",
                        "mini_gpu_memory_gb": 16,
                        "mini_gpu_finetune_gb": 48,
                    },
                }
            ],
            "total": 1,
        }
        return HttpResponse(
            status=200,
            headers={},
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_model_license_is_included_in_normalized_search_text() -> None:
    page = OpenCsgModelsSourceAdapter(client=_Client()).fetch_page({})

    assert page.complete is True
    assert page.records[0].text == (
        "tags: text-generation\n"
        "license: Apache-2.0\n"
        "model metadata: parameters: 7.5; architecture: LlamaForCausalLM; "
        "tensor type: BF16; model type: text-generation; "
        "minimum GPU memory GB: 16; minimum GPU finetune memory GB: 48"
    )
    release = page.records[0].releases[0]
    assert release.model_local_id == "example/model#model"
    assert release.revision == "abc123"
    assert release.identifiers[0].namespace == "opencsg:revision"
    assert release.identifiers[0].value == "example/model@abc123"
    assert release.locator == "$.revision"
