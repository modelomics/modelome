from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.paddlenlp_t5_registry import PaddleNlpT5RegistrySourceAdapter

_REVISION = "d" * 40
_URL = "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-small/model_state.pdparams"
_DOCUMENT = f'''\
T5_PRETRAINED_INIT_CONFIGURATION = {{"t5-small": {{"d_model": 512}}, "without-checkpoint": {{}}}}
T5_PRETRAINED_RESOURCE_FILES_MAP = {{"model_state": {{
    "t5-small": "{_URL}",
    "orphan-checkpoint": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/orphan/model_state.pdparams",
}}}}
'''


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if "/commits/" in url:
            body = json.dumps({"sha": _REVISION}).encode()
        elif url.endswith("/paddlenlp/transformers/t5/configuration.py"):
            body = _DOCUMENT.encode()
        else:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(200, {}, body, url)


def test_t5_registry_extracts_exact_id_and_checkpoint() -> None:
    client = _Client()
    adapter = PaddleNlpT5RegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.identifiers == (Identifier("paddlenlp:transformer-model", "t5-small"),)
    assert record.canonical_url == _URL
    assert client.calls[1].endswith(f"/{_REVISION}/paddlenlp/transformers/t5/configuration.py")


def test_t5_disabled_proposal_matches_adapter() -> None:
    proposal_path = Path(__file__).parents[1] / "config/proposals/paddlenlp_t5_registry.toml"
    source = tomllib.loads(proposal_path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = PaddleNlpT5RegistrySourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        source_path=source["source_path"],
        provider_namespace=source["provider_namespace"],
        max_response_bytes=source["max_response_bytes"],
    )
    assert adapter.name == source["name"]
    assert adapter.repository == source["repository"]
    assert adapter.branch == source["branch"]
    assert adapter.source_path == source["source_path"]
