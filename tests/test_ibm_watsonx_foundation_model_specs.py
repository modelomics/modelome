from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.json_catalog import JsonCatalogSourceAdapter


class FixtureClient:
    def __init__(self) -> None:
        self.responses = [
            {
                "total_count": 3,
                "limit": 2,
                "first": {"href": "https://us-south.ml.cloud.ibm.com/ml/v1/foundation_model_specs?version=2024-05-01&limit=2"},
                "next": {"href": "https://us-south.ml.cloud.ibm.com/ml/v1/foundation_model_specs?version=2024-05-01&limit=2&start=opaque-page-two"},
                "resources": [
                    {
                        "model_id": "ibm/granite-4-h-small",
                        "label": "granite-4-h-small",
                        "provider": "IBM",
                        "lifecycle": [{"id": "available"}],
                    },
                    {
                        "model_id": "meta-llama/llama-3-3-70b-instruct",
                        "label": "llama-3-3-70b-instruct",
                        "provider": "Meta",
                        "lifecycle": [{"id": "available"}],
                    },
                ],
            },
            {
                "total_count": 3,
                "limit": 2,
                "resources": [
                    {
                        "model_id": "mistral-large-2512",
                        "label": "mistral-large-2512",
                        "provider": "Mistral AI",
                        "lifecycle": [{"id": "available"}],
                    }
                ],
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


def test_watsonx_foundation_model_specs_follows_provider_next_href() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/ibm_watsonx_foundation_model_specs.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    client = FixtureClient()
    adapter = create_source(
        config,
        client=client,
        environ={"IBM_WATSONX_BEARER_TOKEN": "fixture-token"},
    )

    assert isinstance(adapter, JsonCatalogSourceAdapter)
    first = adapter.fetch_page({})
    assert not first.complete
    assert parse_qs(urlsplit(first.next_state["next_url"]).query)["start"] == [
        "opaque-page-two"
    ]
    second = adapter.fetch_page(first.next_state)
    assert second.complete
    records = (*first.records, *second.records)
    assert [record.models[0].identifiers[0].value for record in records] == [
        "ibm/granite-4-h-small",
        "meta-llama/llama-3-3-70b-instruct",
        "mistral-large-2512",
    ]
    assert records[0].raw["lifecycle"] == [{"id": "available"}]
    assert urlsplit(client.calls[0][0]).path == "/ml/v1/foundation_model_specs"
    assert parse_qs(urlsplit(client.calls[0][0]).query) == {"version": ["2024-05-01"]}
    assert client.calls[0][1] == {"limit": 200}
    assert client.calls[0][2]["Authorization"] == "Bearer fixture-token"
    assert parse_qs(urlsplit(client.calls[1][0]).query)["start"] == [
        "opaque-page-two"
    ]
    assert client.calls[1][1] == {}
