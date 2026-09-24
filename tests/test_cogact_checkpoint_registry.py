from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.cogact_checkpoint_registry import (
    CogACTCheckpointRegistryAdapter,
    _declares_all_variants,
)

_SHA = {
    "Small": "a" * 40,
    "Base": "b" * 40,
    "Large": "c" * 40,
}
_MODEL_SIZES = {"Small": 30_215_225_166, "Base": 30_521_280_578, "Large": 31_396_425_310}
_OIDS = {"Small": "small-oid", "Base": "base-oid", "Large": "large-oid"}
_README = """
# CogACT
We release three CogACT models with different model sizes, including Small, Base and Large.
`CogACT/CogACT-Base`
`CogACT-Small`, `CogACT-Base`, `CogACT-Large`
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
        del params, headers
        self.calls.append(url)
        if url.endswith("/README.md"):
            body: Any = _README
        elif "/api/models/CogACT/CogACT-" in url and "/tree/" not in url:
            size = url.rsplit("-", 1)[-1]
            body = {"sha": _SHA[size]}
        elif "/tree/" in url:
            size = url.split("/CogACT-", 1)[1].split("/", 1)[0]
            body = [
                {
                    "type": "file",
                    "path": f"checkpoints/CogACT-{size}.pt",
                    "oid": _OIDS[size],
                    "size": _MODEL_SIZES[size],
                    "lfs": {"oid": f"lfs-{size}", "size": _MODEL_SIZES[size]},
                }
            ]
        else:
            raise AssertionError(f"unexpected URL: {url}")
        encoded = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        return HttpResponse(200, {}, encoded, url)


def test_readme_gate_requires_all_official_sizes() -> None:
    assert _declares_all_variants(_README)
    assert not _declares_all_variants("We release three CogACT models: Small and Base")


def test_adapter_pins_each_official_checkpoint_with_hub_metadata() -> None:
    client = _Client()
    adapter = CogACTCheckpointRegistryAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )
    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert [record.source_record_id for record in page.records] == [
        "cogact:small",
        "cogact:base",
        "cogact:large",
    ]
    for record in page.records:
        size = record.title.removeprefix("CogACT ")
        repo_id = f"CogACT/CogACT-{size}"
        path = f"checkpoints/CogACT-{size}.pt"
        file_url = f"https://huggingface.co/{repo_id}/resolve/{_SHA[size]}/{path}"
        assert record.kind is ArtifactKind.WEIGHTS
        assert record.canonical_url == file_url
        assert record.identifiers == (Identifier("cogact:model", f"CogACT-{size}"),)
        assert (
            next(link for link in record.links if link.relation == "model_artifact").url == file_url
        )
        assert record.releases[0].metadata["checkpoint_oid"] == _OIDS[size]
        assert record.releases[0].metadata["checkpoint_size_bytes"] == _MODEL_SIZES[size]
        assert record.releases[0].metadata["lfs_oid"] == f"lfs-{size}"


def test_adapter_fails_closed_when_declared_file_is_missing() -> None:
    class MissingFileClient(_Client):
        def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
            if "/tree/" in url:
                body = []
                return HttpResponse(200, {}, json.dumps(body).encode(), url)
            return super().get(url, params=params, headers=headers)

    with pytest.raises(ValueError, match="expected exactly one published file"):
        CogACTCheckpointRegistryAdapter(client=MissingFileClient()).fetch_page({})
