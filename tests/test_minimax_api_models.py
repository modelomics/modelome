from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import load_sources
from modelome.sources.json_catalog import JsonCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/minimax_api_models.toml"
_MODELS = {"MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.5"}


class _FixtureClient:
    def __init__(self) -> None:
        self.headers: Mapping[str, str] | None = None

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert url == "https://api.minimax.io/v1/models"
        assert params is None
        self.headers = headers
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=(
                b'{"object":"list","data":['
                b'{"id":"MiniMax-M3","object":"model","created":1780272000,"owned_by":"minimax"},'
                b'{"id":"MiniMax-M2.7","object":"model","created":1773799200,"owned_by":"minimax"},'
                b'{"id":"MiniMax-M2.5","object":"model","created":1770948000,"owned_by":"minimax"}'
                b"]}"
            ),
            url=url,
        )


def test_minimax_openai_models_match_first_party_list_response_contract() -> None:
    with _PROPOSAL.open("rb") as handle:
        source = tomllib.load(handle)["source"][0]
    client = _FixtureClient()
    adapter = JsonCatalogSourceAdapter(
        name=source["name"],
        url=source["url"],
        provider_namespace=source["provider_namespace"],
        model_card_url_template=source["model_card_url_template"],
        mapping=source["mapping"],
        auth_header_name=source["auth_header"],
        auth_token="fixture-api-key",
        auth_scheme=source["auth_scheme"],
        model_page_crawl=source["model_page_crawl"],
        max_response_bytes=source["max_response_bytes"],
        client=client,
    )

    page = adapter.fetch_page({})
    models = [model for record in page.records for model in record.models]

    assert {model.name for model in models} == _MODELS
    assert len(models) == len(_MODELS)
    assert {identifier.value for model in models for identifier in model.identifiers} == _MODELS
    assert client.headers == {
        "Accept": "application/json",
        "Authorization": "Bearer fixture-api-key",
    }


def test_enabled_minimax_source_loads_with_key_and_keeps_key_out_of_evidence() -> None:
    class AccountFixtureClient:
        headers: Mapping[str, str] | None = None

        def get(
            self,
            url: str,
            *,
            params: Mapping[str, Any] | None = None,
            headers: Mapping[str, str] | None = None,
        ) -> HttpResponse:
            assert url == "https://api.minimax.io/v1/models"
            assert params is None
            self.headers = headers
            return HttpResponse(
                status=200,
                headers={"content-type": "application/json"},
                body=(
                    b'{"object":"list","data":['
                    b'{"id":"MiniMax-M3","object":"model","created":1780272000,"owned_by":"minimax"},'
                    b'{"id":"MiniMax-M2.7","object":"model","created":1773799200,"owned_by":"minimax"},'
                    b'{"id":"MiniMax-M2.5","object":"model","created":1770948000,"owned_by":"minimax"}'
                    b"]}"
                ),
                url=url,
            )

    secret = "test-minimax-api-key"
    client = AccountFixtureClient()
    sources = load_sources(client=client, environ={"MINIMAX_API_KEY": secret})

    assert "minimax-api-models" in sources
    source = sources["minimax-api-models"]
    page = source.fetch_page({})
    models = [model for record in page.records for model in record.models]

    assert {model.name for model in models} == _MODELS
    assert client.headers == {
        "Accept": "application/json",
        "Authorization": f"Bearer {secret}",
    }
    evidence = {
        "state": page.next_state,
        "records": [
            {
                "canonical_url": record.canonical_url,
                "raw": record.raw,
                "models": [model.name for model in record.models],
            }
            for record in page.records
        ],
    }
    assert secret not in str(evidence)
