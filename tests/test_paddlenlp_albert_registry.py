from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.paddlenlp_albert_registry import PaddleNlpAlbertRegistrySourceAdapter

_REVISION = "e" * 40
_URL = "https://bj.bcebos.com/paddlenlp/models/transformers/albert/albert-base-v1.pdparams"
_DOCUMENT = f"""\
ALBERT_PRETRAINED_INIT_CONFIGURATION = {{
    "albert-base-v1": {{"hidden_size": 768}},
    "missing-resource": {{"hidden_size": 768}},
}}
ALBERT_PRETRAINED_RESOURCE_FILES_MAP = {{
    "model_state": {{
        "albert-base-v1": "{_URL}",
        "orphan-resource": "https://bj.bcebos.com/paddlenlp/models/transformers/albert/orphan.pdparams",
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
        elif url.endswith("/paddlenlp/transformers/albert/configuration.py"):
            body = _DOCUMENT.encode()
        else:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(200, {}, body, url)


def test_albert_registry_extracts_exact_model_id_and_checkpoint_pair() -> None:
    client = _Client()
    adapter = PaddleNlpAlbertRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.identifiers == (Identifier("paddlenlp:transformer-model", "albert-base-v1"),)
    assert record.canonical_url == _URL
    assert client.calls[1].endswith(f"/{_REVISION}/paddlenlp/transformers/albert/configuration.py")


def test_albert_disabled_proposal_matches_adapter() -> None:
    proposal_path = Path(__file__).parents[1] / "config/proposals/paddlenlp_albert_registry.toml"
    source = tomllib.loads(proposal_path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = PaddleNlpAlbertRegistrySourceAdapter(
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
