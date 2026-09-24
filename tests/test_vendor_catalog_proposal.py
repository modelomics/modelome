from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source, load_source_configs
from modelome.sources.json_catalog import JsonCatalogSourceAdapter


class QueuedClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {
                    "object": "list",
                    "data": [{"id": "grok-fixture", "created": 1776556800}],
                }
            ).encode(),
            url=url,
        )


def test_xai_catalog_uses_documented_authenticated_model_list() -> None:
    config_path = Path("config/sources.toml")
    config = next(
        item for item in load_source_configs(config_path) if item["name"] == "xai-models"
    )

    client = QueuedClient()
    with pytest.raises(ValueError, match="credential environment variable is unset"):
        create_source(config, client=client, environ={})

    adapter = create_source(
        config, client=client, environ={"XAI_API_KEY": "xai-test-secret"}
    )
    assert isinstance(adapter, JsonCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert page.records[0].identifiers[0].value == "grok-fixture"
    assert page.records[0].canonical_url == "https://api.x.ai/v1/models/grok-fixture"
    assert page.records[0].links[0].crawl is False
    assert client.calls[0][2]["Authorization"] == "Bearer xai-test-secret"
    assert "xai-test-secret" not in json.dumps(
        {"state": page.next_state, "raw": page.records[0].raw}
    )
