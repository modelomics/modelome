from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.json_catalog import JsonCatalogSourceAdapter

TOKEN = "fixture-vertex-access-token"
PROJECT = "fixture-project"


class FixtureClient:
    def __init__(self) -> None:
        self.responses = [
            {
                "publisherModels": [
                    {
                        "name": "publishers/xai/models/grok-4.6",
                        "versionId": "grok-4.6",
                        "launchStage": "GA",
                    }
                ],
                "nextPageToken": "xai-page-two",
            },
            {
                "publisherModels": [
                    {
                        "name": "publishers/xai/models/grok-4.20-reasoning",
                        "versionId": "grok-4.20-reasoning",
                        "launchStage": "PUBLIC_PREVIEW",
                    }
                ]
            },
        ]
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(self.responses.pop(0)).encode(),
            url=url,
        )


def test_vertex_xai_publisher_keeps_exact_versioned_grok_ids_and_pagination() -> None:
    proposal = (
        Path(__file__).parents[1]
        / "config/proposals/vertex_model_garden_xai_publisher.toml"
    )
    with proposal.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    client = FixtureClient()
    adapter = create_source(
        config,
        client=client,
        environ={
            "VERTEX_AI_ACCESS_TOKEN": TOKEN,
            "GOOGLE_CLOUD_PROJECT": PROJECT,
        },
    )

    assert isinstance(adapter, JsonCatalogSourceAdapter)
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    records = (*first.records, *second.records)

    assert [record.source_record_id for record in records] == [
        "publishers/xai/models/grok-4.6@grok-4.6",
        "publishers/xai/models/grok-4.20-reasoning@grok-4.20-reasoning",
    ]
    assert [record.releases[0].version for record in records] == [
        "grok-4.6",
        "grok-4.20-reasoning",
    ]
    assert client.calls[0][0].endswith("/publishers/xai/models?listAllVersions=true")
    assert client.calls[1][0] == client.calls[0][0]
    assert client.calls[1][1] == {
        "pageSize": 100,
        "pageToken": "xai-page-two",
    }
    assert client.calls[0][2]["Authorization"] == f"Bearer {TOKEN}"
