from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.openai_models import OpenAIModelsSourceAdapter

SECRET = "openai-secret-value"
NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


class QueuedClient:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.responses = [
            HttpResponse(
                status=200,
                headers={"content-type": "application/json", "etag": '"catalog-v1"'},
                body=json.dumps(payload).encode(),
                url="https://api.openai.com/v1/models",
            )
            for payload in payloads
        ]
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError("unexpected model-list request")
        return self.responses.pop(0)


def _payload() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-example-1",
                "object": "model",
                "created": 1735689600,
                "owned_by": "openai",
                "shutdown_date": None,
                "unexpected_echo": f"Bearer {SECRET}",
            },
            {
                "id": "text-example-1",
                "object": "model",
                "created": 1735776000,
                "owned_by": "system",
            },
            {
                "id": "ft:gpt-example-1:my-org:private:abc123",
                "object": "model",
                "created": 1735862400,
                "owned_by": "my-org",
            },
        ],
    }


def test_provider_catalog_records_public_owners_without_persisting_private_rows() -> None:
    client = QueuedClient(_payload())
    adapter = OpenAIModelsSourceAdapter(token=SECRET, client=client, clock=lambda: NOW)

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is False
    assert page.upstream_count == 2
    assert page.next_state["returned_model_count"] == 3
    assert page.next_state["public_model_count"] == 2
    assert page.next_state["skipped_nonpublic_model_count"] == 1
    assert page.next_state["catalog_etag"] == '"catalog-v1"'
    assert client.calls == [
        (
            "https://api.openai.com/v1/models",
            {},
            {"Accept": "application/json", "Authorization": f"Bearer {SECRET}"},
        )
    ]

    first = page.records[0]
    assert first.kind is ArtifactKind.PROVIDER_PAGE
    assert first.source_record_id == "model:gpt-example-1"
    assert first.canonical_url == "https://api.openai.com/v1/models/gpt-example-1"
    assert first.published_at == "2025-01-01T00:00:00Z"
    assert first.modified_at == "2026-09-21T12:00:00Z"
    assert first.identifiers == (Identifier("openai:model", "gpt-example-1"),)
    assert first.models[0].identifiers == first.identifiers
    assert {(link.url, link.relation, link.crawl) for link in first.links} == {
        ("https://api.openai.com/v1/models/gpt-example-1", "provider_api_resource", False),
        (
            "https://developers.openai.com/api/docs/models",
            "provider_documentation",
            False,
        ),
    }
    rendered = json.dumps([record.raw for record in page.records])
    assert SECRET not in rendered
    assert "my-org" not in rendered
    assert "ft:gpt-example" not in rendered


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda payload: payload["data"].append(dict(payload["data"][0])),
            "duplicate model IDs",
        ),
        (
            lambda payload: payload["data"][0].update(id="bad model id"),
            "invalid model ID",
        ),
        (
            lambda payload: payload["data"][0].update(created="not-a-timestamp"),
            "invalid created timestamp",
        ),
    ],
)
def test_provider_catalog_fails_closed_on_invalid_public_rows(change, message) -> None:
    payload = _payload()
    change(payload)

    with pytest.raises(ValueError, match=message):
        OpenAIModelsSourceAdapter(token=SECRET, client=QueuedClient(payload)).fetch_page({})


def test_provider_catalog_rejects_empty_public_owner_policy() -> None:
    with pytest.raises(ValueError, match="public_owners"):
        OpenAIModelsSourceAdapter(token=SECRET, public_owners=(), client=QueuedClient())
