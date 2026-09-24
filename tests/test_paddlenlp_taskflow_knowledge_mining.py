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
from modelome.sources.paddlenlp_taskflow_knowledge_mining import (
    PaddleNlpTaskflowKnowledgeMiningSourceAdapter,
    _parse_resource_maps,
)

_REVISION = "e" * 40
_WORDTAG_URL = (
    "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/wordtag_v1.5/model_state.pdparams"
)
_NPTAG_URL = (
    "https://bj.bcebos.com/paddlenlp/taskflow/knowledge_mining/nptag_v1.2/model_state.pdparams"
)
_WORDTAG_MD5 = "c7c9cef72f73ee22c70c26ef11393025"
_NPTAG_MD5 = "34923c4d06acf936f52e1fa376b13748"
_SOURCE = f'''\
class WordTagTask:
    resource_files_urls = {{"wordtag": {{"model_state": ["{_WORDTAG_URL}", "{_WORDTAG_MD5}"]}}}}
class NPTagTask:
    resource_files_urls = {{"nptag": {{"model_state": ["{_NPTAG_URL}", "{_NPTAG_MD5}"]}}}}
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


def _adapter(client: _Client, **kwargs: Any) -> PaddleNlpTaskflowKnowledgeMiningSourceAdapter:
    return PaddleNlpTaskflowKnowledgeMiningSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
        **kwargs,
    )


def test_knowledge_mining_parser_extracts_official_checkpoint_maps() -> None:
    assert _parse_resource_maps(_SOURCE, max_entries=10) == (
        ("nptag", _NPTAG_URL, _NPTAG_MD5),
        ("wordtag", _WORDTAG_URL, _WORDTAG_MD5),
    )
    page = _adapter(_Client()).fetch_page({})
    assert page.complete is True and page.authoritative_snapshot is True
    assert page.upstream_count == 2
    wordtag = next(row for row in page.records if row.identifiers[0].value == "wordtag")
    assert wordtag.kind is ArtifactKind.WEIGHTS
    assert wordtag.identifiers == (Identifier("paddlenlp:taskflow-knowledge-mining", "wordtag"),)
    assert wordtag.canonical_url == _WORDTAG_URL
    assert wordtag.releases[0].metadata["md5"] == _WORDTAG_MD5
    assert wordtag.raw["revision"] == _REVISION


def test_parser_rejects_nonliteral_maps_and_untrusted_urls() -> None:
    with pytest.raises(ValueError, match="not literal"):
        _parse_resource_maps(
            "class WordTagTask:\n resource_files_urls = get_map()\n"
            "class NPTagTask:\n resource_files_urls = {}",
            max_entries=10,
        )
    unsafe = _SOURCE.replace(_NPTAG_URL, "https://example.test/model.pdparams")
    assert all(name != "nptag" for name, *_ in _parse_resource_maps(unsafe, max_entries=10))


def test_disabled_proposal_matches_adapter() -> None:
    path = Path(__file__).parents[1] / "config/proposals/paddlenlp_taskflow_knowledge_mining.toml"
    source = tomllib.loads(path.read_text())["source"][0]
    assert source["enabled"] is False
    adapter = PaddleNlpTaskflowKnowledgeMiningSourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        source_path=source["source_path"],
        max_response_bytes=source["max_response_bytes"],
        max_entries=source["max_entries"],
    )
    assert adapter.name == source["name"]
    assert adapter.source_path == source["source_path"]
