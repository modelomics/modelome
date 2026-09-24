from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.paddlenlp_roformer_registry import PaddleNlpRoformerRegistrySourceAdapter

_REVISION = "a" * 40
_URL = "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/roformer-chinese-base/model_state.pdparams"
_DOCUMENT = f"""\
ROFORMER_PRETRAINED_INIT_CONFIGURATION = {{
    "roformer-chinese-base": {{"hidden_size": 768}},
    "no-resource": {{"hidden_size": 768}},
}}
ROFORMER_PRETRAINED_RESOURCE_FILES_MAP = {{
    "model_state": {{
        "roformer-chinese-base": "{_URL}",
        "unregistered-resource": "https://bj.bcebos.com/paddlenlp/models/transformers/roformer/missing/model_state.pdparams",
    }},
}}
"""


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
        elif url.endswith("/paddlenlp/transformers/roformer/configuration.py"):
            body = _DOCUMENT.encode()
        else:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(200, {}, body, url)


def test_roformer_registry_extracts_declared_model_state_url() -> None:
    client = _Client()
    adapter = PaddleNlpRoformerRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.identifiers == (
        Identifier("paddlenlp:transformer-model", "roformer-chinese-base"),
    )
    assert record.canonical_url == _URL
    assert client.calls[1].endswith(
        f"/{_REVISION}/paddlenlp/transformers/roformer/configuration.py"
    )


def test_roformer_disabled_proposal_matches_adapter() -> None:
    proposal_path = Path(__file__).parents[1] / "config/proposals/paddlenlp_roformer_registry.toml"
    source = tomllib.loads(proposal_path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = PaddleNlpRoformerRegistrySourceAdapter(
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
