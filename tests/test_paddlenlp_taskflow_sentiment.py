from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.paddlenlp_taskflow_sentiment import (
    PaddleNlpTaskflowSentimentSourceAdapter,
    _parse_resource_maps,
)

_REVISION = "d" * 40
_MD5 = "0123456789abcdef0123456789abcdef"
_BILSTM = "https://bj.bcebos.com/paddlenlp/taskflow/sentiment_analysis/bilstm/model_state.pdparams"
_SKEP = "https://bj.bcebos.com/paddlenlp/taskflow/sentiment_analysis/skep_ernie_1.0_large_ch/model_state.pdparams"
_SOURCE = f'''\
class SentaTask:
    resource_files_urls = {{"bilstm": {{"model_state": ["{_BILSTM}", "{_MD5}"]}}}}
class SkepTask:
    resource_files_urls = {{
        "skep_ernie_1.0_large_ch": {{"model_state": [
            "{_SKEP}", "fedcba9876543210fedcba9876543210"
        ]}},
        "__internal_testing__/tiny-random-skep": {{"model_state": ["{_SKEP}", "{_MD5}"]}},
    }}
class UIESentaTask:
    resource_files_urls = {{"uie-senta-base": {{"model_state": ["{_SKEP}", "{_MD5}"]}}}}
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
        body = json.dumps({"sha": _REVISION}).encode() if "/commits/" in url else _SOURCE.encode()
        return HttpResponse(200, {}, body, url)


def _adapter(client: _Client, **kwargs: Any) -> PaddleNlpTaskflowSentimentSourceAdapter:
    return PaddleNlpTaskflowSentimentSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC), **kwargs
    )


def test_sentiment_parser_extracts_only_public_literal_direct_weights() -> None:
    assert _parse_resource_maps(_SOURCE, max_entries=10) == (
        ("bilstm", _BILSTM, _MD5),
        ("skep_ernie_1.0_large_ch", _SKEP, "fedcba9876543210fedcba9876543210"),
    )
    page = _adapter(_Client()).fetch_page({})
    assert page.complete is True and page.authoritative_snapshot is True
    assert page.upstream_count == 2
    row = next(record for record in page.records if record.identifiers[0].value == "bilstm")
    assert row.kind is ArtifactKind.WEIGHTS
    assert row.identifiers == (Identifier("paddlenlp:taskflow-sentiment", "bilstm"),)
    assert row.canonical_url == _BILSTM
    assert row.releases[0].metadata["md5"] == _MD5
    assert row.raw["revision"] == _REVISION


def test_parser_rejects_nonliteral_and_unknown_host_urls() -> None:
    with pytest.raises(ValueError, match="not literal"):
        _parse_resource_maps(
            "class SentaTask:\n resource_files_urls = get_map()\n"
            "class SkepTask:\n resource_files_urls = {}",
            max_entries=10,
        )
    unsafe = _SOURCE.replace(_BILSTM, "https://example.test/model.pdparams")
    assert all(name != "bilstm" for name, *_ in _parse_resource_maps(unsafe, max_entries=10))


def test_disabled_proposal_matches_adapter() -> None:
    path = Path(__file__).parents[1] / "config/proposals/paddlenlp_taskflow_sentiment.toml"
    source = tomllib.loads(path.read_text())["source"][0]
    assert source["enabled"] is False
    adapter = PaddleNlpTaskflowSentimentSourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        source_path=source["source_path"],
        max_response_bytes=source["max_response_bytes"],
        max_entries=source["max_entries"],
    )
    assert adapter.name == source["name"]
    assert adapter.source_path == source["source_path"]
