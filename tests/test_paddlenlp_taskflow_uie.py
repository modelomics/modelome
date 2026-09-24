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
from modelome.sources.paddlenlp_taskflow_uie import (
    PaddleNlpTaskflowUieSourceAdapter,
    _parse_uie_resource_map,
)

_REVISION = "c" * 40
_MODEL_URL = (
    "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/"
    "uie_base_v1.1/model_state.pdparams"
)
_MED_URL = (
    "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/"
    "uie_medium_v1.1/model_state.pdparams"
)
_MD5 = "0123456789abcdef0123456789abcdef"
_DOCUMENT = f'''\
class InformationExtractionTask:
    resource_files_urls = {{
        "uie-base": {{"model_state": ["{_MODEL_URL}", "{_MD5}"]}},
        "uie-tiny": {{"model_state": ["{_MED_URL}", "fedcba9876543210fedcba9876543210"]}},
        "__internal_testing__/tiny-random-uie": {{
            "model_state": [
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/test.pdparams",
                "{_MD5}",
            ]
        }},
        "uie-bad-origin": {{"model_state": ["https://example.test/model.pdparams", "{_MD5}"]}},
    }}
'''


class _Client:
    def __init__(self, *, revision: str = _REVISION, source: str = _DOCUMENT) -> None:
        self.revision = revision
        self.source = source
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
            body = json.dumps({"sha": self.revision}).encode()
        elif url.endswith("/paddlenlp/taskflow/information_extraction.py"):
            body = self.source.encode()
        else:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(200, {}, body, url)


def _adapter(client: _Client, **kwargs: Any) -> PaddleNlpTaskflowUieSourceAdapter:
    return PaddleNlpTaskflowUieSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
        **kwargs,
    )


def test_uie_resource_map_emits_only_public_exact_weight_urls_and_checksums() -> None:
    entries = _parse_uie_resource_map(_DOCUMENT, max_entries=10)

    assert entries == (
        ("uie-base", _MODEL_URL, _MD5),
        ("uie-tiny", _MED_URL, "fedcba9876543210fedcba9876543210"),
    )

    client = _Client()
    page = _adapter(client).fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    record = next(row for row in page.records if row.identifiers[0].value == "uie-base")
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.identifiers == (Identifier("paddlenlp:taskflow-uie", "uie-base"),)
    assert record.canonical_url == _MODEL_URL
    assert record.releases[0].metadata["md5"] == _MD5
    assert record.raw["revision"] == _REVISION
    assert client.calls[1].endswith(f"/{_REVISION}/paddlenlp/taskflow/information_extraction.py")


def test_uie_registry_honors_revision_skip_and_rejects_nonliteral_maps() -> None:
    client = _Client()
    adapter = _adapter(client)
    skipped = adapter.fetch_page({"completed_revision": _REVISION, "checkpoint_count": 8})
    assert skipped.records == ()
    assert skipped.upstream_count == 8
    assert len(client.calls) == 1

    with pytest.raises(ValueError, match="not a literal mapping"):
        _parse_uie_resource_map(
            'resource_files_urls = load_resources()\n',
            max_entries=10,
        )


def test_uie_disabled_proposal_matches_adapter() -> None:
    proposal_path = Path(__file__).parents[1] / "config/proposals/paddlenlp_taskflow_uie.toml"
    source = tomllib.loads(proposal_path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = PaddleNlpTaskflowUieSourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        source_path=source["source_path"],
        max_response_bytes=source["max_response_bytes"],
        max_entries=source["max_entries"],
    )
    assert adapter.name == source["name"]
    assert adapter.repository == source["repository"]
    assert adapter.branch == source["branch"]
    assert adapter.source_path == source["source_path"]
