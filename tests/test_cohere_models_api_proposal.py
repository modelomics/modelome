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
        self.responses = [
            {
                "models": [
                    {"name": "command-a-03-2025", "is_deprecated": False, "endpoints": ["chat"]},
                    {"name": "command-r-03-2024", "is_deprecated": True, "endpoints": ["chat"]},
                ],
                "next_page_token": "page-two",
            },
            {
                "models": [
                    {"name": "embed-english-v2.0", "is_deprecated": True, "endpoints": ["embed"]}
                ],
                "next_page_token": None,
            },
        ]
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        payload = self.responses.pop(0)
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_cohere_models_api_paginates_and_preserves_provider_deprecation_flags() -> None:
    proposal_path = (
        Path(__file__).parents[1] / "config/proposals/cohere_models_api.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    client = FixtureClient()
    adapter = create_source(
        config, client=client, environ={"COHERE_API_KEY": "fixture-key"}
    )

    assert isinstance(adapter, JsonCatalogSourceAdapter)
    first = adapter.fetch_page({})
    assert not first.complete
    assert first.next_state == {"cursor": "page-two", "raw_items_seen": 2}
    second = adapter.fetch_page(first.next_state)
    assert second.complete
    records = (*first.records, *second.records)
    assert [record.models[0].identifiers[0].value for record in records] == [
        "command-a-03-2025",
        "command-r-03-2024",
        "embed-english-v2.0",
    ]
    assert [record.raw["is_deprecated"] for record in records] == [False, True, True]
    assert client.calls[0][1] == {"page_size": 1000}
    assert client.calls[1][1] == {"page_size": 1000, "page_token": "page-two"}
    assert client.calls[0][2]["Authorization"] == "Bearer fixture-key"
