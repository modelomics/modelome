from __future__ import annotations

import json
import tomllib
from pathlib import Path

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.json_catalog import JsonCatalogSourceAdapter


class FixtureClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(headers or {})))
        payload = {
            "object": "list",
            "data": [
                {
                    "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "digitalocean",
                },
                {
                    "id": "openai-gpt-oss-20b",
                    "object": "model",
                    "created": 1777593600,
                    "owned_by": "digitalocean",
                },
            ],
        }
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_digitalocean_serverless_list_records_exact_hosted_model_ids() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/digitalocean_serverless_models.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    client = FixtureClient()
    adapter = create_source(
        config, client=client, environ={"DIGITALOCEAN_TOKEN": "fixture-token"}
    )

    assert isinstance(adapter, JsonCatalogSourceAdapter)
    page = adapter.fetch_page({})
    assert page.complete
    assert [record.models[0].identifiers[0].value for record in page.records] == [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "openai-gpt-oss-20b",
    ]
    assert [record.raw["owned_by"] for record in page.records] == [
        "digitalocean",
        "digitalocean",
    ]
    assert client.calls == [
        (
            "https://inference.do-ai.run/v1/models",
            {"Accept": "application/json", "Authorization": "Bearer fixture-token"},
        )
    ]
