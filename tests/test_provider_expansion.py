from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.openai_models import OpenAIModelsSourceAdapter
from modelome.sources.openrouter import OpenRouterModelsSourceAdapter


class _Client:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        return HttpResponse(200, {}, json.dumps(self.payload).encode(), url)


def _payload() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "example/image-embedder",
                "name": "Image Embedder",
                "description": "Maps images to vectors.",
                "created": 1_700_000_000,
                "context_length": 8192,
                "pricing": {"prompt": "0.00003", "completion": "0.00006"},
                "architecture": {
                    "modality": "image->embedding",
                    "input_modalities": ["image"],
                    "output_modalities": ["embeddings"],
                    "tokenizer": "ExampleTokenizer",
                    "instruct_type": "example-format",
                },
            }
        ],
        "links": {"next": None},
        "total_count": 1,
    }


def test_openrouter_can_request_all_modalities_and_records_full_catalog_fields() -> None:
    client = _Client(_payload())
    adapter = OpenRouterModelsSourceAdapter(client=client, output_modalities="all")

    page = adapter.fetch_page({})

    assert client.calls == [
        ("https://openrouter.ai/api/v1/models", {"output_modalities": "all"})
    ]
    assert page.authoritative_snapshot is True
    text = page.records[0].text
    assert "output modalities: embeddings" in text
    assert "context length: 8192" in text
    assert "pricing: prompt: 0.00003, completion: 0.00006" in text
    assert "tokenizer: ExampleTokenizer" in text
    assert "instruction format: example-format" in text


def test_openrouter_modality_scope_changes_checkpoint_and_snapshot_semantics() -> None:
    all_modalities = OpenRouterModelsSourceAdapter(output_modalities="all")
    client = _Client(_payload())
    image_only = OpenRouterModelsSourceAdapter(output_modalities="image", client=client)
    provider_default = OpenRouterModelsSourceAdapter(client=client)

    assert all_modalities.checkpoint_signature != image_only.checkpoint_signature
    assert provider_default.checkpoint_signature != all_modalities.checkpoint_signature
    page = provider_default.fetch_page({})
    assert client.calls == [("https://openrouter.ai/api/v1/models", {})]
    assert page.authoritative_snapshot is True
    image_page = image_only.fetch_page({})
    assert image_page.authoritative_snapshot is False


@pytest.mark.parametrize("value", ["", "video", "all,image", "text,unknown"])
def test_openrouter_rejects_undocumented_modality_filter(value: str) -> None:
    with pytest.raises(ValueError, match="output_modalities"):
        OpenRouterModelsSourceAdapter(output_modalities=value)


def test_openai_accepts_documented_string_model_ids_without_exposing_other_owners() -> None:
    payload = {
        "data": [
            {"id": "provider/model-preview", "owned_by": "openai"},
            {"id": "ft:private-model:org:run", "owned_by": "customer-org"},
        ]
    }
    client = _Client(payload)

    page = OpenAIModelsSourceAdapter(token="test-key", client=client).fetch_page({})

    assert [record.source_record_id for record in page.records] == [
        "model:provider/model-preview"
    ]
    assert page.records[0].canonical_url.endswith("provider%2Fmodel-preview")
    assert page.next_state["skipped_nonpublic_model_count"] == 1
