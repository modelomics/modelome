from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.json_catalog import JsonCatalogSourceAdapter

TOKEN = "fixture-vertex-access-token"
PROJECT = "fixture-project"
PROPOSALS = Path(__file__).parents[1] / "config/proposals"


class FixtureClient:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps({"publisherModels": self.rows}).encode(),
            url=url,
        )


@pytest.mark.parametrize(
    ("proposal_name", "rows", "expected_release_ids"),
    [
        (
            "vertex_model_garden_meta_publisher.toml",
            [
                {
                    "name": "publishers/meta/models/llama3_1",
                    "versionId": "llama-3.1-8b-instruct",
                },
                {
                    "name": "publishers/meta/models/llama3_1",
                    "versionId": "llama-3.1-70b-instruct",
                },
                {
                    "name": "publishers/meta/models/llama3-2",
                    "versionId": "llama-3.2-1b-instruct",
                },
            ],
            [
                "publishers/meta/models/llama3_1@llama-3.1-8b-instruct",
                "publishers/meta/models/llama3_1@llama-3.1-70b-instruct",
                "publishers/meta/models/llama3-2@llama-3.2-1b-instruct",
            ],
        ),
        (
            "vertex_model_garden_deepseek_publisher.toml",
            [
                {
                    "name": "publishers/deepseek-ai/models/deepseek-r1",
                    "versionId": "deepseek-r1-distill-llama-8b",
                }
            ],
            [
                "publishers/deepseek-ai/models/deepseek-r1@"
                "deepseek-r1-distill-llama-8b"
            ],
        ),
        (
            "vertex_model_garden_openai_publisher.toml",
            [
                {
                    "name": "publishers/openai/models/gpt-oss",
                    "versionId": "gpt-oss-20b",
                },
                {
                    "name": "publishers/openai/models/gpt-oss",
                    "versionId": "gpt-oss-120b",
                },
            ],
            [
                "publishers/openai/models/gpt-oss@gpt-oss-20b",
                "publishers/openai/models/gpt-oss@gpt-oss-120b",
            ],
        ),
    ],
)
def test_additional_vertex_publishers_preserve_documented_version_ids(
    proposal_name: str,
    rows: list[dict[str, str]],
    expected_release_ids: list[str],
) -> None:
    with (PROPOSALS / proposal_name).open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    client = FixtureClient(rows)
    adapter = create_source(
        config,
        client=client,
        environ={
            "VERTEX_AI_ACCESS_TOKEN": TOKEN,
            "GOOGLE_CLOUD_PROJECT": PROJECT,
        },
    )

    assert isinstance(adapter, JsonCatalogSourceAdapter)
    page = adapter.fetch_page({})
    assert page.complete
    assert [record.source_record_id for record in page.records] == expected_release_ids
    assert [record.releases[0].identifiers[0].value for record in page.records] == (
        expected_release_ids
    )
    assert [record.releases[0].version for record in page.records] == [
        row["versionId"] for row in rows
    ]
    assert client.calls[0][0].endswith("?listAllVersions=true")
    assert client.calls[0][2]["Authorization"] == f"Bearer {TOKEN}"
