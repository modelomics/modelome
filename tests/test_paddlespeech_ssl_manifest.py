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
from modelome.sources.paddlespeech_ssl_manifest import (
    PaddleSpeechSslManifestSourceAdapter,
    _parse_manifest,
)

_REVISION = "d" * 40
_URL = "https://paddlespeech.cdn.bcebos.com/wavlm/wavlm_baseplus_libriclean_100h.tar.gz"
_MD5 = "f2238e982bb8bcf046e536201f5ea629"
_SOURCE = f'''\
ssl_dynamic_pretrained_models = {{
    "wavlmASR_librispeech-en-16k": {{
        "1.0": {{"url": "{_URL}", "md5": "{_MD5}", "cfg_path": "model.yaml"}}
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


def test_official_wavlm_manifest_row_has_exact_url_and_checksum() -> None:
    assert _parse_manifest(_SOURCE) == ("wavlmASR_librispeech-en-16k", _URL, _MD5)
    adapter = PaddleSpeechSslManifestSourceAdapter(
        client=_Client(), clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )
    page = adapter.fetch_page({})
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == _URL
    assert record.releases[0].metadata["md5"] == _MD5


def test_parser_rejects_computed_and_changed_entries() -> None:
    with pytest.raises(ValueError, match="not literal"):
        _parse_manifest("ssl_dynamic_pretrained_models = load_models()")
    bad = _SOURCE.replace(_URL, "https://example.org/model.tar.gz")
    with pytest.raises(ValueError, match="invalid"):
        _parse_manifest(bad)


def test_disabled_proposal_matches_adapter() -> None:
    path = Path(__file__).parents[1] / "config/proposals/paddlespeech_wavlm_ssl_manifest.toml"
    source = tomllib.loads(path.read_text())["source"][0]
    assert source["enabled"] is False
    adapter = PaddleSpeechSslManifestSourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        source_path=source["source_path"],
        max_response_bytes=source["max_response_bytes"],
    )
    assert adapter.name == source["name"]
