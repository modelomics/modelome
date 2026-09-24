from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.openrouter import OpenRouterModelsSourceAdapter


class QueuedClient:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.responses = [
            HttpResponse(
                status=200,
                headers={"content-type": "application/json"},
                body=json.dumps(payload).encode(),
                url="https://openrouter.ai/api/v1/models",
            )
            for payload in payloads
        ]
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError("unexpected catalog request")
        return self.responses.pop(0)


def _payload() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "example/vision-v1",
                "canonical_slug": "example/vision-v1-20260901",
                "hugging_face_id": "example-lab/vision-v1",
                "name": "Example: Vision V1",
                "created": 1700000000,
                "description": (
                    "A vision-language model. Read the paper at "
                    "https://example.org/paper."
                ),
                "architecture": {
                    "modality": "text+image->text",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["temperature", "tools"],
                "links": {"details": "/api/v1/models/example/vision-v1/endpoints"},
            },
            {
                "id": "~example/vision-latest",
                "canonical_slug": "~example/vision-latest",
                "hugging_face_id": None,
                "name": "Example: Vision Latest",
                "created": 1700000001,
                "description": "This alias routes to the current Vision release.",
                "architecture": {"modality": "text->text"},
                "links": {"details": "/api/v1/models/example/vision-v1/endpoints"},
                "alias_target": {
                    "name": "Example: Vision V1",
                    "slug": "example/vision-v1",
                },
            },
            {
                "id": "example/vision-v1:free",
                "canonical_slug": "example/vision-v1-20260901",
                "hugging_face_id": "example-lab/vision-v1",
                "name": "Example: Vision V1 (free)",
                "created": 1700000000,
                "description": "A free routed variant.",
                "architecture": {"modality": "text->text"},
                "links": {"details": "/api/v1/models/example/vision-v1/endpoints"},
            },
        ],
        "links": {"next": None},
        "total_count": 3,
    }


def test_complete_catalog_preserves_hosted_models_and_exact_artifact_links() -> None:
    client = QueuedClient(_payload())
    adapter = OpenRouterModelsSourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 3
    assert page.next_state["model_count"] == 3
    assert page.next_state["catalog_sha256"]
    assert client.calls == [
        (
            "https://openrouter.ai/api/v1/models",
            {},
            {"Accept": "application/json"},
        )
    ]

    first, alias, free_variant = page.records
    assert first.kind is ArtifactKind.PROVIDER_PAGE
    assert first.source_record_id == "example/vision-v1"
    assert first.canonical_url == "https://openrouter.ai/example/vision-v1"
    assert first.published_at == "2023-11-14T22:13:20Z"
    assert first.identifiers == (Identifier("openrouter:model", "example/vision-v1"),)
    assert first.models[0].identifiers == first.identifiers
    assert first.models[0].aliases == (
        "example/vision-v1-20260901",
        "example/vision-v1",
    )
    assert "input modalities: text, image" in first.text
    assert "supported parameters: temperature, tools" in first.text
    assert {(link.url, link.relation, link.crawl) for link in first.links} == {
        ("https://openrouter.ai/example/vision-v1", "model_page", False),
        (
            "https://openrouter.ai/api/v1/models/example/vision-v1/endpoints",
            "provider_endpoints",
            False,
        ),
        ("https://example.org/paper", "documentation_reference", True),
        ("https://huggingface.co/example-lab/vision-v1", "linked_model_artifact", False),
    }
    assert first.model_relations[0].predicate == "hosted_huggingface_model"
    assert first.model_relations[0].target.identifiers == (
        Identifier("huggingface:model", "example-lab/vision-v1"),
    )

    assert alias.model_relations[0].predicate == "alias_of"
    assert alias.model_relations[0].target.identifiers == (
        Identifier("openrouter:model", "example/vision-v1"),
    )
    assert free_variant.source_record_id == "example/vision-v1:free"
    assert free_variant.models[0].identifiers == (
        Identifier("openrouter:model", "example/vision-v1:free"),
    )
    assert free_variant.model_relations[0].predicate == "hosted_huggingface_model"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value.update(total_count=4), "but reported 4"),
        (lambda value: value["links"].update(next="/api/v1/models?offset=3"), "next page"),
        (
            lambda value: (
                value["data"].append(dict(value["data"][0])),
                value.update(total_count=4),
            ),
            "duplicate model IDs",
        ),
    ],
)
def test_catalog_fails_closed_when_response_is_not_a_complete_unique_snapshot(
    change, message
) -> None:
    payload = _payload()
    change(payload)

    with pytest.raises(ValueError, match=message):
        OpenRouterModelsSourceAdapter(client=QueuedClient(payload)).fetch_page({})


def test_catalog_rejects_invalid_declared_hugging_face_identity() -> None:
    payload = _payload()
    payload["data"][0]["hugging_face_id"] = "not-a-repository"

    with pytest.raises(ValueError, match="owner/model identifier"):
        OpenRouterModelsSourceAdapter(client=QueuedClient(payload)).fetch_page({})
