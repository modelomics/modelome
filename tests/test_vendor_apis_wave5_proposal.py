from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.json_catalog import JsonCatalogSourceAdapter


class FixtureClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(headers or {})))
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(self.payload).encode(),
            url=url,
        )


def test_sambanova_cloud_model_list_proposal_uses_documented_shared_catalog_shape() -> None:
    proposal_path = (
        Path(__file__).parents[1] / "config/proposals/vendor_apis_wave5.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]

    payload = {
        "data": [
            {
                "id": "DeepSeek-R1",
                "object": "model",
                "owned_by": "SambaNova",
                "context_length": 16384,
                "max_completion_tokens": 16384,
                "pricing": {"prompt": "0.00000500", "completion": "0.00000700"},
            }
        ],
        "object": "list",
    }
    client = FixtureClient(payload)

    with pytest.raises(ValueError, match="credential environment variable is unset"):
        create_source(config, client=client, environ={})

    adapter = create_source(
        config, client=client, environ={"SAMBANOVA_API_KEY": "fixture-secret"}
    )
    assert isinstance(adapter, JsonCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.records[0].identifiers[0].value == "DeepSeek-R1"
    assert page.records[0].links[0].crawl is False
    assert client.calls == [
        (
            "https://api.sambanova.ai/v1/models",
            {"Accept": "application/json", "Authorization": "Bearer fixture-secret"},
        )
    ]
    assert "fixture-secret" not in json.dumps(
        {"state": page.next_state, "record": page.records[0].raw}
    )
