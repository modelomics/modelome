from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.openrouter import OpenRouterVideoModelsSourceAdapter


class QueuedClient:
    def __init__(self, payload: Mapping[str, Any], *, status: int = 200) -> None:
        self.response = HttpResponse(
            status=status,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
            url="https://openrouter.ai/api/v1/videos/models",
        )
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return self.response


def _payload() -> dict[str, Any]:
    return {
        "data": [
            {
                "id": "google/veo-3.1",
                "canonical_slug": "google/veo-3.1",
                "name": "Veo 3.1",
                "created": 1700000000,
                "description": "Google video generation model",
                "generate_audio": True,
                "supported_durations": [5, 8],
                "supported_resolutions": ["720p"],
                "supported_aspect_ratios": ["16:9"],
            },
            {
                "id": "openai/sora-2",
                "name": "Sora 2",
                "description": "OpenAI video generation model",
            },
        ]
    }


def test_video_catalog_is_single_complete_authenticated_snapshot() -> None:
    client = QueuedClient(_payload())
    adapter = OpenRouterVideoModelsSourceAdapter(
        token="test-token",
        client=client,
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["model_count"] == 2
    assert page.next_state["catalog_sha256"]
    assert client.calls == [
        (
            "https://openrouter.ai/api/v1/videos/models",
            {},
            {
                "Accept": "application/json",
                "Authorization": "Bearer test-token",
            },
        )
    ]
    veo, sora = page.records
    assert veo.kind is ArtifactKind.PROVIDER_PAGE
    assert veo.identifiers == (Identifier("openrouter:model", "google/veo-3.1"),)
    assert veo.canonical_url == "https://openrouter.ai/google/veo-3.1"
    assert veo.links[0].crawl is False
    assert veo.raw["supported_durations"] == [5, 8]
    assert "generates audio: yes" in veo.text
    assert sora.source_record_id == "openai/sora-2"


def test_video_catalog_uses_environment_key_and_requires_one(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterVideoModelsSourceAdapter(client=QueuedClient(_payload()))

    monkeypatch.setenv("OPENROUTER_API_KEY", "env-token")
    client = QueuedClient(_payload())
    OpenRouterVideoModelsSourceAdapter(client=client).fetch_page({})
    assert client.calls[0][2]["Authorization"] == "Bearer env-token"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"data": "not-a-list"}, "data is not a JSON list"),
        ({"data": [{"name": "No ID"}]}, "ID is required"),
        (
            {"data": [{"id": "google/veo", "name": "Veo"}], "links": {"next": "/more"}},
            "unexpectedly returned a next page",
        ),
        (
            {"data": [{"id": "google/veo", "name": "Veo"}], "links": []},
            "pagination links are malformed",
        ),
    ],
)
def test_video_catalog_fails_closed_on_invalid_or_paged_responses(payload, message) -> None:
    adapter = OpenRouterVideoModelsSourceAdapter(
        token="test-token",
        client=QueuedClient(payload),
    )

    with pytest.raises(ValueError, match=message):
        adapter.fetch_page({})


def test_video_catalog_rejects_duplicate_model_ids() -> None:
    payload = _payload()
    payload["data"].append(dict(payload["data"][0]))
    adapter = OpenRouterVideoModelsSourceAdapter(
        token="test-token",
        client=QueuedClient(payload),
    )

    with pytest.raises(ValueError, match="duplicate model IDs"):
        adapter.fetch_page({})


def test_video_catalog_rejects_pagination_state() -> None:
    adapter = OpenRouterVideoModelsSourceAdapter(
        token="test-token",
        client=QueuedClient(_payload()),
    )

    with pytest.raises(ValueError, match="does not accept pagination state"):
        adapter.fetch_page({"next_url": "https://openrouter.ai/api/v1/videos/models?page=2"})
