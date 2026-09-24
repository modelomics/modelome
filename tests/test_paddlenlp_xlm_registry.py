from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.paddlenlp_xlm_registry import PaddleNlpXlmRegistrySourceAdapter

_REVISION = "f" * 40
_URL = (
    "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/xlm-mlm-en-2048/model_state.pdparams"
)
_DOCUMENT = f"""\
XLM_PRETRAINED_INIT_CONFIGURATION = {{
    "xlm-mlm-en-2048": {{"emb_dim": 1024}},
    "model-without-checkpoint": {{"emb_dim": 1024}},
}}
XLM_PRETRAINED_RESOURCE_FILES_MAP = {{
    "model_state": {{
        "xlm-mlm-en-2048": "{_URL}",
        "orphan-checkpoint": "https://bj.bcebos.com/paddlenlp/models/transformers/xlm/orphan/model_state.pdparams",
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
        elif url.endswith("/paddlenlp/transformers/xlm/configuration.py"):
            body = _DOCUMENT.encode()
        else:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(200, {}, body, url)


def test_xlm_registry_extracts_only_declared_model_state_checkpoint_pairs() -> None:
    client = _Client()
    adapter = PaddleNlpXlmRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.identifiers == (Identifier("paddlenlp:transformer-model", "xlm-mlm-en-2048"),)
    assert record.canonical_url == _URL
    assert client.calls[1].endswith(f"/{_REVISION}/paddlenlp/transformers/xlm/configuration.py")


def test_xlm_disabled_proposal_matches_adapter() -> None:
    proposal_path = Path(__file__).parents[1] / "config/proposals/paddlenlp_xlm_registry.toml"
    source = tomllib.loads(proposal_path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = PaddleNlpXlmRegistrySourceAdapter(
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
