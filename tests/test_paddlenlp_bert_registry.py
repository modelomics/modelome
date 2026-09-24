from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.paddlenlp_bert_registry import PaddleNlpBertRegistrySourceAdapter

_REVISION = "b" * 40
_URL = "https://bj.bcebos.com/paddlenlp/models/transformers/uer/chinese_roberta_base.pdparams"
_DOCUMENT = f"""\
BERT_PRETRAINED_INIT_CONFIGURATION = {{
    "uer/chinese-roberta-base": {{"hidden_size": 768}},
    "without-resource": {{"hidden_size": 768}},
}}
BERT_PRETRAINED_RESOURCE_FILES_MAP = {{
    "model_state": {{
        "uer/chinese-roberta-base": "{_URL}",
        "unregistered-resource": "https://bj.bcebos.com/paddlenlp/models/transformers/uer/missing.pdparams",
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
        elif url.endswith("/paddlenlp/transformers/bert/configuration.py"):
            body = _DOCUMENT.encode()
        else:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(200, {}, body, url)


def test_bert_registry_supports_source_declared_namespaced_model_ids() -> None:
    client = _Client()
    adapter = PaddleNlpBertRegistrySourceAdapter(
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
        Identifier("paddlenlp:transformer-model", "uer/chinese-roberta-base"),
    )
    assert record.canonical_url == _URL
    assert client.calls[1].endswith(f"/{_REVISION}/paddlenlp/transformers/bert/configuration.py")


def test_bert_disabled_proposal_matches_adapter() -> None:
    proposal_path = Path(__file__).parents[1] / "config/proposals/paddlenlp_bert_registry.toml"
    source = tomllib.loads(proposal_path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = PaddleNlpBertRegistrySourceAdapter(
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
