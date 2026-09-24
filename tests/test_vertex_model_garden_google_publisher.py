from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.json_catalog import JsonCatalogSourceAdapter

TOKEN = "fixture-oauth-access-token"
PROJECT = "fixture-quota-project"


class FixtureClient:
    def __init__(self) -> None:
        self.responses = [
            {
                "publisherModels": [
                    {
                        "name": "publishers/google/models/gemini-2.5-pro",
                        "versionId": "6",
                        "launchStage": "GA",
                        "versionState": "VERSION_STATE_READY",
                    },
                    {
                        "name": "publishers/google/models/gemini-2.5-pro",
                        "versionId": "7",
                        "launchStage": "GA",
                        "versionState": "VERSION_STATE_READY",
                    },
                    {
                        "name": "publishers/google/models/gemma-3-27b-it",
                        "versionId": "2",
                        "launchStage": "GA",
                        "versionState": "VERSION_STATE_READY",
                    },
                ],
                "nextPageToken": "next-page-token",
            },
            {
                "publisherModels": [
                    {
                        "name": "publishers/google/models/gemma-3-12b-it",
                        "versionId": "1",
                        "launchStage": "GA",
                        "versionState": "VERSION_STATE_READY",
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


def test_vertex_google_publisher_list_paginates_with_project_oauth_and_exact_names() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/vertex_model_garden_google_publisher.toml"
    )
    with proposal_path.open("rb") as handle:
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
    assert not first.complete
    assert first.next_state == {"cursor": "next-page-token", "raw_items_seen": 3}
    second = adapter.fetch_page(first.next_state)
    assert second.complete
    records = (*first.records, *second.records)
    assert [record.models[0].identifiers[0].value for record in records] == [
        "publishers/google/models/gemini-2.5-pro",
        "publishers/google/models/gemini-2.5-pro",
        "publishers/google/models/gemma-3-27b-it",
        "publishers/google/models/gemma-3-12b-it",
    ]
    assert [record.source_record_id for record in records[:2]] == [
        "publishers/google/models/gemini-2.5-pro@6",
        "publishers/google/models/gemini-2.5-pro@7",
    ]
    assert [record.releases[0].version for record in records[:2]] == ["6", "7"]
    assert [record.releases[0].identifiers[0].value for record in records[:2]] == [
        "publishers/google/models/gemini-2.5-pro@6",
        "publishers/google/models/gemini-2.5-pro@7",
    ]
    assert records[0].raw["versionId"] == "6"
    assert records[0].canonical_url == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/"
        "publishers/google/models/gemini-2.5-pro"
    )
    assert client.calls[0] == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/publishers/google/models?listAllVersions=true",
        {"pageSize": 100},
        {
            "Accept": "application/json",
            "Authorization": f"Bearer {TOKEN}",
            "x-goog-user-project": PROJECT,
        },
    )
    assert client.calls[1][1] == {"pageSize": 100, "pageToken": "next-page-token"}
    assert client.calls[1][2]["Authorization"] == f"Bearer {TOKEN}"
    assert client.calls[1][0].endswith("?listAllVersions=true")


def test_entry_assembly_groups_vertex_versions_but_keeps_different_models_separate() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/vertex_model_garden_google_publisher.toml"
    )
    with proposal_path.open("rb") as handle:
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
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    records = (*first.records, *second.records)

    result = build_entries(
        source_record_to_entry_seed(record, source="vertex-model-garden-google-publisher")
        for record in records
    )

    assert len(result.entries) == 3
    gemini = next(
        entry
        for entry in result.entries
        if entry.canonical_name == "publishers/google/models/gemini-2.5-pro"
    )
    assert len(gemini.members) == 2
    assert {release.version for release in gemini.releases} == {"6", "7"}
    assert all(len(entry.releases) == 1 for entry in result.entries if entry is not gemini)


def test_vertex_google_publisher_source_requires_caller_oauth_token() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/vertex_model_garden_google_publisher.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]

    try:
        create_source(config, client=FixtureClient(), environ={})
    except ValueError as error:
        assert "credential environment variable is unset" in str(error)
    else:
        raise AssertionError("missing OAuth token should prevent source construction")
