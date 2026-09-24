from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.replicate import ReplicateModelsSourceAdapter

SECRET = "replicate-secret-value"


class QueuedClient:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.payloads:
            raise AssertionError("unexpected catalog request")
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(self.payloads.pop(0)).encode(),
            url=url,
        )


def _model(owner: str, name: str) -> dict[str, Any]:
    return {
        "owner": owner,
        "name": name,
        "url": f"https://replicate.com/{owner}/{name}",
        "visibility": "public",
        "created_at": "2025-01-02T03:04:05Z",
        "description": "Read the paper at https://example.org/paper.",
        "readme": "Source code: https://github.com/example/vision.",
        "github_url": "https://github.com/example/vision",
        "paper_url": "https://example.org/paper",
        "weights_url": "https://weights.example.org/vision.safetensors",
        "license_url": "https://example.org/license",
        "latest_version": {
            "id": "a" * 64,
            "created_at": "2025-01-03T04:05:06Z",
            "cog_version": "0.15.0",
        },
        "echoed_request": f"Bearer {SECRET}",
    }


def test_cursor_catalog_preserves_direct_resource_and_latest_release_evidence() -> None:
    first_payload = {
        "results": [_model("example", "vision")],
        "next": "https://api.replicate.com/v1/models?cursor=opaque-page-two",
    }
    second_payload = {"results": [_model("example", "audio")], "next": None}
    client = QueuedClient(
        first_payload,
        {"results": [{"id": "old-vision-version", "cog_version": "0.14.0"}], "next": None},
        second_payload,
        {"results": [{"id": "old-audio-version"}], "next": None},
    )
    adapter = ReplicateModelsSourceAdapter(token=SECRET, client=client)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    third = adapter.fetch_page(second.next_state)
    fourth = adapter.fetch_page(third.next_state)

    assert first.complete is False
    assert first.next_state == {
        "next_url": "https://api.replicate.com/v1/models?cursor=opaque-page-two",
        "raw_items_seen": 1,
        "version_queue": [{"model_id": "example/vision"}],
    }
    assert not second.complete
    assert second.records[0].source_record_id == "example/vision@old-vision-version"
    assert not third.complete
    assert third.records[0].source_record_id == "example/audio"
    assert fourth.complete is True
    assert fourth.next_state == {}
    assert client.calls[0][0] == "https://api.replicate.com/v1/models"
    assert client.calls[1][0] == "https://api.replicate.com/v1/models/example/vision/versions"
    assert client.calls[2][0] == "https://api.replicate.com/v1/models?cursor=opaque-page-two"
    assert client.calls[3][0] == "https://api.replicate.com/v1/models/example/audio/versions"
    assert all(call[2]["Authorization"] == f"Bearer {SECRET}" for call in client.calls)

    record = first.records[0]
    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.source_record_id == "example/vision"
    assert record.canonical_url == "https://replicate.com/example/vision"
    assert record.identifiers == (Identifier("replicate:model", "example/vision"),)
    assert record.models[0].identifiers == record.identifiers
    assert record.models[0].aliases == ("vision",)
    assert record.modified_at == "2025-01-03T04:05:06Z"
    assert "[REDACTED]" in record.raw["echoed_request"]
    assert SECRET not in json.dumps(record.raw)
    assert {(link.url, link.relation, link.crawl) for link in record.links} == {
        ("https://replicate.com/example/vision", "model_page", False),
        ("https://example.org/paper", "documentation_reference", True),
        ("https://github.com/example/vision", "documentation_reference", True),
        ("https://github.com/example/vision", "source_repository", True),
        ("https://example.org/paper", "paper", True),
        ("https://weights.example.org/vision.safetensors", "weights", False),
        ("https://example.org/license", "license", False),
    }
    assert record.releases[0].version == "a" * 64
    assert record.releases[0].identifiers == (
        Identifier("replicate:model-version", "a" * 64),
    )
    assert record.releases[0].metadata == {"cog_version": "0.15.0"}


@pytest.mark.parametrize(
    ("next_url", "message"),
    [
        ("https://attacker.example.test/models?cursor=bad", "catalog origin"),
        ("https://api.replicate.com/v1/models?token=bad", "sensitive query"),
    ],
)
def test_cursor_catalog_rejects_unsafe_pagination_urls(next_url: str, message: str) -> None:
    payload = {"results": [_model("example", "vision")], "next": next_url}

    with pytest.raises(ValueError, match=message):
        ReplicateModelsSourceAdapter(token=SECRET, client=QueuedClient(payload)).fetch_page({})


def test_catalog_rejects_nonpublic_model() -> None:
    payload = {"results": [_model("example", "vision")], "next": None}
    payload["results"][0]["visibility"] = "private"

    with pytest.raises(ValueError, match="non-public"):
        ReplicateModelsSourceAdapter(token=SECRET, client=QueuedClient(payload)).fetch_page({})
