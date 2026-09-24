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
                        "name": "publishers/mistral-ai/models/mistral",
                        "versionId": "mistral-7b-instruct-v0.2",
                        "launchStage": "GA",
                    }
                ],
                "nextPageToken": "mistral-page-two",
            },
            {
                "publisherModels": [
                    {
                        "name": "publishers/mistral-ai/models/mistral",
                        "versionId": "mistral-7b-instruct-v0.3",
                        "launchStage": "GA",
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


def test_vertex_mistral_publisher_preserves_exact_model_versions_and_pages() -> None:
    proposal = (
        Path(__file__).parents[1]
        / "config/proposals/vertex_model_garden_mistral_publisher.toml"
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
        "publishers/mistral-ai/models/mistral@mistral-7b-instruct-v0.2",
        "publishers/mistral-ai/models/mistral@mistral-7b-instruct-v0.3",
    ]
    assert [record.releases[0].version for record in records] == [
        "mistral-7b-instruct-v0.2",
        "mistral-7b-instruct-v0.3",
    ]
    assert all(
        record.models[0].identifiers[0].value == record.raw["name"]
        for record in records
    )
    assert records[0].canonical_url == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/"
        "publishers/mistral-ai/models/mistral"
    )
    assert client.calls[0][0].endswith(
        "/publishers/mistral-ai/models?listAllVersions=true"
    )
    assert client.calls[1][0] == client.calls[0][0]
    assert client.calls[1][1] == {
        "pageSize": 100,
        "pageToken": "mistral-page-two",
    }
    assert client.calls[0][2]["Authorization"] == f"Bearer {TOKEN}"
