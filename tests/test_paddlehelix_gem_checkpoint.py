from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.paddlehelix_gem_checkpoint import (
    PaddleHelixGemCheckpointSourceAdapter,
    _parse_readme,
)

_REVISION = "c" * 40
_URL = "https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/compound/pretrain_models-chemrl_gem.tgz"
_README = f"""
We also provide our pretrained model here for reproducing downstream results.
`wget {_URL}`
`wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/compound_datasets/chemrl_downstream_datasets.tgz`
"""


class _Client:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        body = json.dumps({"sha": _REVISION}).encode() if "/commits/" in url else _README.encode()
        return HttpResponse(200, {}, body, url)


def test_exact_model_archive_is_parsed_and_emitted() -> None:
    assert _parse_readme(_README) == _URL
    adapter = PaddleHelixGemCheckpointSourceAdapter(
        client=_Client(), clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )
    page = adapter.fetch_page({})
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == _URL
    assert record.identifiers[0].value == "chemrl-gem"


def test_parser_rejects_missing_or_changed_direct_model_url() -> None:
    with pytest.raises(ValueError, match="expected one exact"):
        _parse_readme("wget https://example.test/pretrain_models-chemrl_gem.tgz")
    with pytest.raises(ValueError, match="expected one exact"):
        _parse_readme(_README + f"\nwget {_URL}\n")


def test_disabled_proposal_matches_adapter() -> None:
    path = Path(__file__).parents[1] / "config/proposals/paddlehelix_chemrl_gem.toml"
    source = tomllib.loads(path.read_text())["source"][0]
    assert source["enabled"] is False
    adapter = PaddleHelixGemCheckpointSourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        source_path=source["source_path"],
        max_response_bytes=source["max_response_bytes"],
    )
    assert adapter.name == source["name"]
