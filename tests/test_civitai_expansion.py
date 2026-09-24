from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.civitai import CivitaiModelsSourceAdapter


class _QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def _response(payload: object, url: str) -> HttpResponse:
    return HttpResponse(200, {}, json.dumps(payload).encode(), url)


def test_model_filters_and_version_file_evidence_are_retained() -> None:
    start_url = "https://civitai.com/api/v1/models"
    next_url = start_url + "?cursor=after"
    client = _QueueClient(
        _response(
            {
                "items": [
                    {
                        "id": 43,
                        "name": "Expanded LoRA",
                        "modelVersions": [
                            {
                                "id": 44,
                                "name": "release one",
                                "baseModel": "Flux.1 D",
                                "air": "urn:air:flux:lora:civitai:43@44",
                                "status": "Published",
                                "uploadType": "Created",
                                "usageControl": "Download",
                                "createdAt": "2026-01-01T00:00:00Z",
                                "updatedAt": "2026-01-02T00:00:00Z",
                                "supportsGeneration": True,
                                "stats": {"downloadCount": 8, "thumbsUpCount": 2},
                                "files": [
                                    {
                                        "id": 45,
                                        "name": "weights.safetensors",
                                        "type": "Model",
                                        "sizeKB": 123.5,
                                        "primary": True,
                                        "hashes": {"AutoV2": "ABCD", "SHA256": "CD" * 32},
                                        "metadata": {"format": "SafeTensor", "fp": "fp16"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "metadata": {"nextCursor": "after"},
            },
            start_url,
        ),
        _response({"items": [], "metadata": {}}, next_url),
    )
    source = CivitaiModelsSourceAdapter(
        client=client,
        query="portrait",
        tag="illustration",
        username="creator",
        model_types=("LORA", "Checkpoint"),
        base_models=("Flux.1 D",),
        period="Month",
    )

    page = source.fetch_page({})
    resumed = source.fetch_page(page.next_state)

    query = client.calls[0][1]
    assert query == {
        "limit": 100,
        "sort": "Newest",
        "nsfw": "true",
        "period": "Month",
        "query": "portrait",
        "tag": "illustration",
        "username": "creator",
        "types": ["LORA", "Checkpoint"],
        "baseModels": ["Flux.1 D"],
    }
    next_query = parse_qs(urlsplit(page.next_state["next_url"]).query)
    assert next_query["query"] == ["portrait"]
    assert set(next_query["types"]) == {"LORA", "Checkpoint"}
    assert next_query["baseModels"] == ["Flux.1 D"]
    assert next_query["period"] == ["Month"]
    assert resumed.complete

    release = page.records[0].releases[0]
    assert release.identifiers[:2] == (
        Identifier("civitai:model-version", "44"),
        Identifier("civitai:model-file", "45"),
    )
    assert release.metadata["air"] == "urn:air:flux:lora:civitai:43@44"
    assert release.metadata["stats"] == {"downloadCount": 8, "thumbsUpCount": 2}
    assert release.metadata["files"][0] == {
        "id": 45,
        "name": "weights.safetensors",
        "type": "Model",
        "sizeKB": 123.5,
        "primary": True,
        "hashes": {"AutoV2": "ABCD", "SHA256": "CD" * 32},
        "metadata": {"format": "SafeTensor", "fp": "fp16"},
    }


def test_civitai_rejects_invalid_filter_period_and_scalar_filter_list() -> None:
    import pytest

    with pytest.raises(ValueError, match="unsupported CivitAI sort period"):
        CivitaiModelsSourceAdapter(period="Forever")
    with pytest.raises(ValueError, match="model_types must be a sequence"):
        CivitaiModelsSourceAdapter(model_types="LORA")


def test_next_page_keeps_unfiltered_public_scan_scope_when_provider_omits_params() -> None:
    endpoint = "https://civitai.com/api/v1/models"
    next_page = endpoint + "?limit=100&cursor=opaque"
    client = _QueueClient(
        _response({"items": [{"id": 1}], "metadata": {"nextPage": next_page}}, endpoint),
        _response({"items": [], "metadata": {}}, next_page),
    )
    source = CivitaiModelsSourceAdapter(client=client)

    first = source.fetch_page({})
    source.fetch_page(first.next_state)

    continued = client.calls[1][0]
    params = parse_qs(urlsplit(continued).query)
    assert params["cursor"] == ["opaque"]
    assert params["limit"] == ["100"]
    assert params["sort"] == ["Newest"]
    assert params["nsfw"] == ["true"]
