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
                        "name": "publishers/anthropic/models/claude-sonnet-4-5",
                        "versionId": "1",
                        "launchStage": "GA",
                        "versionState": "VERSION_STATE_STABLE",
                    }
                ],
                "nextPageToken": "anthropic-page-two",
            },
            {
                "publisherModels": [
                    {
                        "name": "publishers/anthropic/models/claude-sonnet-4-5",
                        "versionId": "2",
                        "launchStage": "GA",
                        "versionState": "VERSION_STATE_STABLE",
                    },
                    {
                        "name": "publishers/anthropic/models/claude-opus-4-5",
                        "versionId": "1",
                        "launchStage": "GA",
                        "versionState": "VERSION_STATE_STABLE",
                    },
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


def test_vertex_anthropic_publisher_lists_versioned_model_ids_across_pages() -> None:
    proposal = (
        Path(__file__).parents[1]
        / "config/proposals/vertex_model_garden_anthropic_publisher.toml"
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
    assert first.next_state == {
        "cursor": "anthropic-page-two",
        "raw_items_seen": 1,
    }
    second = adapter.fetch_page(first.next_state)
    records = (*first.records, *second.records)

    assert [record.source_record_id for record in records] == [
        "publishers/anthropic/models/claude-sonnet-4-5@1",
        "publishers/anthropic/models/claude-sonnet-4-5@2",
        "publishers/anthropic/models/claude-opus-4-5@1",
    ]
    assert [record.releases[0].version for record in records] == ["1", "2", "1"]
    assert [record.releases[0].identifiers[0].value for record in records] == [
        record.source_record_id for record in records
    ]
    assert all(
        record.models[0].identifiers[0].value == record.raw["name"]
        for record in records
    )
    assert records[0].canonical_url == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/"
        "publishers/anthropic/models/claude-sonnet-4-5"
    )
    assert client.calls[0] == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/"
        "publishers/anthropic/models?listAllVersions=true",
        {"pageSize": 100},
        {
            "Accept": "application/json",
            "Authorization": f"Bearer {TOKEN}",
            "x-goog-user-project": PROJECT,
        },
    )
    assert client.calls[1][0].endswith("?listAllVersions=true")
    assert client.calls[1][1] == {
        "pageSize": 100,
        "pageToken": "anthropic-page-two",
    }
