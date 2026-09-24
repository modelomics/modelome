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
                    "id": "Qwen/Qwen3.8-27B",
                    "display_name": "Qwen3.8-27B",
                    "hf_repo": "Qwen/Qwen3.8-27B",
                    "hf_revision": "1111111111111111",
                    "status": "ready",
                },
                {
                    "id": "openai/gpt-oss-20b",
                    "display_name": "gpt-oss-20b",
                    "hf_repo": "openai/gpt-oss-20b",
                    "hf_revision": "2222222222222222",
                    "status": "ready",
                },
                {
                    "id": "Qwen/Qwen3.5-4B",
                    "display_name": "Qwen3.5-4B",
                    "hf_repo": "Qwen/Qwen3.5-4B",
                    "hf_revision": "3333333333333333",
                    "status": "ready",
                },
                {
                    "id": "Qwen/Qwen3-8B",
                    "display_name": "Qwen3-8B",
                    "hf_repo": "Qwen/Qwen3-8B",
                    "hf_revision": "4444444444444444",
                    "status": "ready",
                },
                {
                    "id": "Qwen/Qwen3-0.6B",
                    "display_name": "Qwen3-0.6B",
                    "hf_repo": "Qwen/Qwen3-0.6B",
                    "hf_revision": "5555555555555555",
                    "status": "ready",
                },
            ],
        }
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_boardwalk_catalog_records_public_inference_ids_without_auth() -> None:
    proposal_path = (
        Path(__file__).parents[1] / "config/proposals/boardwalk_public_models.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    client = FixtureClient()
    adapter = create_source(config, client=client, environ={})

    assert isinstance(adapter, JsonCatalogSourceAdapter)
    page = adapter.fetch_page({})
    assert page.complete
    assert [record.models[0].identifiers[0].value for record in page.records] == [
        "Qwen/Qwen3.8-27B",
        "openai/gpt-oss-20b",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-0.6B",
    ]
    assert [record.raw["hf_revision"] for record in page.records] == [
        "1111111111111111",
        "2222222222222222",
        "3333333333333333",
        "4444444444444444",
        "5555555555555555",
    ]
    assert [record.canonical_url for record in page.records] == [
        "https://www.boardwalk.cloud/docs/models/Qwen/Qwen3.8-27B",
        "https://www.boardwalk.cloud/docs/models/openai/gpt-oss-20b",
        "https://www.boardwalk.cloud/docs/models/Qwen/Qwen3.5-4B",
        "https://www.boardwalk.cloud/docs/models/Qwen/Qwen3-8B",
        "https://www.boardwalk.cloud/docs/models/Qwen/Qwen3-0.6B",
    ]
    assert len({record.canonical_url for record in page.records}) == 5
    assert client.calls == [
        ("https://api.boardwalk.cloud/v1/models", {"Accept": "application/json"})
    ]
