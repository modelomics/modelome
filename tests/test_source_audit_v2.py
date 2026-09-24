from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.catalog import create_source


class FixtureClient:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        self.calls.append(url)
        return HttpResponse(
            status=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            body=self.body,
            url=url,
        )


def test_azure_foundry_proposal_extracts_documented_model_ids() -> None:
    root = Path(__file__).parents[1]
    with (root / "config/proposals/source_audit_v2.toml").open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    fixture = (root / "tests/fixtures/source_azure_foundry_models.html").read_bytes()
    client = FixtureClient(fixture)
    source = create_source(config, client=client)

    page = source.fetch_page({})

    assert client.calls == [config["url"]]
    assert page.complete is True
    assert page.upstream_count == 2
    assert [model.name for model in page.records[0].models] == [
        "gpt-6-astra (2026-09-03)",
        "gpt-6-luna (2026-09-22)",
    ]
    assert [model.identifiers for model in page.records[0].models] == [
        (Identifier("azure:foundry-model-offering", "gpt-6-astra (2026-09-03)"),),
        (Identifier("azure:foundry-model-offering", "gpt-6-luna (2026-09-22)"),),
    ]
