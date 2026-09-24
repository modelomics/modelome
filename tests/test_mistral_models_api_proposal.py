from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.json_catalog import JsonCatalogSourceAdapter


class FixtureClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        payload = {
            "data": [
                {
                    "id": "open-mistral-7b",
                    "capabilities": {"completion_chat": True, "vision": False},
                    "root": "open-mistral-7b",
                    "object": "model",
                    "created": 1756746619,
                    "owned_by": "mistralai",
                    "aliases": ["mistral-7b-latest"],
                    "TYPE": "base",
                    "archived": False,
                },
                {
                    "id": "ft:open-mistral-7b:abc123",
                    "capabilities": {"completion_chat": True},
                    "root": "open-mistral-7b",
                    "object": "model",
                    "created": 1756746620,
                    "owned_by": "workspace-123",
                    "aliases": [],
                    "TYPE": "fine-tuned",
                    "archived": True,
                },
            ],
            "object": "list",
        }
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_mistral_models_api_retains_account_scoped_provider_fields() -> None:
    proposal_path = (
        Path(__file__).parents[1] / "config/proposals/mistral_models_api.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    client = FixtureClient()
    adapter = create_source(
        config, client=client, environ={"MISTRAL_API_KEY": "fixture-key"}
    )

    assert isinstance(adapter, JsonCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert page.complete
    assert [record.models[0].identifiers[0].value for record in page.records] == [
        "open-mistral-7b",
        "ft:open-mistral-7b:abc123",
    ]
    assert [record.raw["TYPE"] for record in page.records] == ["base", "fine-tuned"]
    assert page.records[1].raw["archived"] is True
    assert client.calls[0][2]["Authorization"] == "Bearer fixture-key"
