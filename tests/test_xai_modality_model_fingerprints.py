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


_PROPOSAL = (
    Path(__file__).parents[1]
    / "config/proposals/xai_modality_model_fingerprints.toml"
)


@pytest.mark.parametrize(
    ("source_index", "model_id", "fingerprint", "version", "modalities"),
    [
        (
            0,
            "grok-imagine-image",
            "fp-image-fixture",
            "1.0.0",
            {"input_modalities": ["text", "image"], "output_modalities": ["image"]},
        ),
        (
            1,
            "grok-imagine-video",
            "fp-video-fixture",
            "1.0.0",
            {"input_modalities": ["text", "image"], "output_modalities": ["video"]},
        ),
    ],
)
def test_xai_modality_lists_preserve_documented_provider_versions_and_fingerprints(
    source_index: int,
    model_id: str,
    fingerprint: str,
    version: str,
    modalities: dict[str, Any],
) -> None:
    with _PROPOSAL.open("rb") as handle:
        config = tomllib.load(handle)["source"][source_index]
    payload = {
        "models": [
            {
                "id": model_id,
                "fingerprint": fingerprint,
                "created": 1743724800,
                "object": "model",
                "owned_by": "xai",
                "version": version,
                "aliases": [],
                **modalities,
            }
        ]
    }
    client = FixtureClient(payload)

    with pytest.raises(ValueError, match="credential environment variable is unset"):
        create_source(config, client=client, environ={})

    adapter = create_source(
        config, client=client, environ={"XAI_API_KEY": "fixture-secret"}
    )
    assert isinstance(adapter, JsonCatalogSourceAdapter)
    page = adapter.fetch_page({})

    record = page.records[0]
    assert record.identifiers[0].value == model_id
    assert record.raw["fingerprint"] == fingerprint
    assert record.raw["version"] == version
    assert record.raw["aliases"] == []
    assert record.links[0].crawl is False
    assert client.calls == [
        (config["url"], {"Accept": "application/json", "Authorization": "Bearer fixture-secret"})
    ]
    assert "fixture-secret" not in json.dumps(
        {"state": page.next_state, "raw": record.raw}
    )
