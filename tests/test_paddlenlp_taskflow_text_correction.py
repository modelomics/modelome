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
from modelome.sources.paddlenlp_taskflow_text_correction import (
    PaddleNlpTaskflowTextCorrectionSourceAdapter,
    _parse_resource_map,
)

_REVISION = "e" * 40
_URL = "https://bj.bcebos.com/paddlenlp/taskflow/text_correction/ernie-csc/model_state.pdparams"
_MD5 = "cdc53e7e3985ffc78fedcdf8e6dca6d2"
_VOCAB_URL = "https://bj.bcebos.com/paddlenlp/taskflow/text_correction/ernie-csc/pinyin_vocab.txt"
_SOURCE = f'''\
class CSCTask:
    resource_files_urls = {{
        "ernie-csc": {{
            "model_state": ["{_URL}", "{_MD5}"],
            "pinyin_vocab": ["{_VOCAB_URL}", "5599a8116b6016af573d08f8e686b4b2"],
        }},
    }}
'''


class _Client:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        body = json.dumps({"sha": _REVISION}).encode() if "/commits/" in url else _SOURCE.encode()
        return HttpResponse(200, {}, body, url)


def test_parser_and_adapter_emit_exact_ernie_csc_weights() -> None:
    assert _parse_resource_map(_SOURCE) == (("ernie-csc", _URL, _MD5),)
    adapter = PaddleNlpTaskflowTextCorrectionSourceAdapter(
        client=_Client(), clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )
    page = adapter.fetch_page({})
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == _URL
    assert record.releases[0].metadata["md5"] == _MD5


def test_parser_rejects_computed_or_untrusted_checkpoint() -> None:
    with pytest.raises(ValueError, match="not literal"):
        _parse_resource_map("class CSCTask:\n resource_files_urls = load_map()")
    bad = _SOURCE.replace(_URL, "https://example.test/model_state.pdparams")
    with pytest.raises(ValueError, match="invalid ernie-csc"):
        _parse_resource_map(bad)


def test_disabled_proposal_matches_adapter() -> None:
    path = Path(__file__).parents[1] / "config/proposals/paddlenlp_taskflow_text_correction.toml"
    source = tomllib.loads(path.read_text())["source"][0]
    assert source["enabled"] is False
    adapter = PaddleNlpTaskflowTextCorrectionSourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        source_path=source["source_path"],
        max_response_bytes=source["max_response_bytes"],
    )
    assert adapter.name == source["name"]
