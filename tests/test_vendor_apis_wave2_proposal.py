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
    def __init__(self, payload: Any, url: str) -> None:
        self.payload = payload
        self.url = url
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(headers or {})))
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(self.payload).encode(),
            url=url,
        )


_PROPOSAL = Path(__file__).parents[1] / "config/proposals/vendor_apis_wave2.toml"


@pytest.mark.parametrize(
    ("name", "environment_name", "payload", "expected_id"),
    [
        (
            "together-models",
            "TOGETHER_API_KEY",
            [{"id": "meta-llama/Llama-3-8b", "display_name": "Llama 3 8B", "created": 1700000000}],
            "meta-llama/Llama-3-8b",
        ),
        (
            "cerebras-models",
            "CEREBRAS_API_KEY",
            {
                "object": "list",
                "data": [{"id": "gpt-oss-120b", "created": 0, "owned_by": "Cerebras"}],
            },
            "gpt-oss-120b",
        ),
        (
            "deepseek-models",
            "DEEPSEEK_API_KEY",
            {
                "object": "list",
                "data": [{"id": "deepseek-flash", "object": "model", "owned_by": "deepseek"}],
            },
            "deepseek-flash",
        ),
    ],
)
def test_proposed_vendor_model_lists_are_credential_gated_availability_evidence(
    name: str, environment_name: str, payload: Any, expected_id: str
) -> None:
    with _PROPOSAL.open("rb") as handle:
        sources = tomllib.load(handle)["source"]
    config = next(source for source in sources if source["name"] == name)
    client = FixtureClient(payload, config["url"])

    with pytest.raises(ValueError, match="credential environment variable is unset"):
        create_source(config, client=client, environ={})

    adapter = create_source(config, client=client, environ={environment_name: "fixture-secret"})
    assert isinstance(adapter, JsonCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.records[0].identifiers[0].value == expected_id
    assert page.records[0].links[0].crawl is False
    assert client.calls == [
        (config["url"], {"Accept": "application/json", "Authorization": "Bearer fixture-secret"})
    ]
    assert "fixture-secret" not in json.dumps(
        {"state": page.next_state, "record": page.records[0].raw}
    )
