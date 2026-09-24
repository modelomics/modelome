from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.sony_woosh_release_assets import (
    _MODEL_ASSETS,
    _NON_MODEL_ASSETS,
    _RELEASE_DOWNLOAD_PREFIX,
    SonyWooshReleaseAssetsSourceAdapter,
    _parse_assets,
)

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/sony_woosh_release_assets.toml"
_NAMES = (*_MODEL_ASSETS, *_NON_MODEL_ASSETS)
_SHA256_BY_NAME = {name: f"{index:064x}" for index, name in enumerate(_NAMES, start=1)}


def _row(name: str) -> str:
    return (
        f'<li><a href="{_RELEASE_DOWNLOAD_PREFIX}{name}">{name}</a>'
        f'<clipboard-copy value="sha256:{_SHA256_BY_NAME[name]}"></clipboard-copy></li>'
    )


_HTML = "<html><body>" + "".join(_row(name) for name in _NAMES) + "</body></html>"


class _Client:
    def __init__(self, body: bytes = _HTML.encode()) -> None:
        self.body = body
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(200, {"content-type": "text/html"}, self.body, url)


def test_parser_keeps_exactly_eight_release_model_assets() -> None:
    assert _parse_assets(_HTML, maximum=12) == tuple(
        (name, _SHA256_BY_NAME[name]) for name in _MODEL_ASSETS
    )


def test_parser_rejects_missing_duplicate_unexpected_and_oversized_assets() -> None:
    absent = _HTML.replace(_row("Woosh-Flow.zip"), "")
    with pytest.raises(ValueError, match="missing model assets"):
        _parse_assets(absent, maximum=12)
    with pytest.raises(ValueError, match="duplicate"):
        _parse_assets(_HTML + _row("Woosh-Flow.zip"), maximum=12)
    with pytest.raises(ValueError, match="unexpected Sony Woosh"):
        unknown = _HTML + (
            f'<li><a href="{_RELEASE_DOWNLOAD_PREFIX}future-model.zip">future</a></li>'
        )
        _parse_assets(unknown, maximum=12)
    with pytest.raises(ValueError, match="count exceeds"):
        _parse_assets(_HTML, maximum=9)


def test_adapter_emits_exact_non_crawlable_metadata_records() -> None:
    client = _Client()
    adapter = SonyWooshReleaseAssetsSourceAdapter(client=client)
    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 8
    assert client.calls == [adapter.release_page_url]
    assert {record.raw["asset_filename"] for record in page.records} == set(_MODEL_ASSETS)
    for record in page.records:
        filename = record.raw["asset_filename"]
        expected_url = f"https://github.com{_RELEASE_DOWNLOAD_PREFIX}{filename}"
        assert record.canonical_url == expected_url
        assert record.raw["asset_url"] == expected_url
        assert record.raw["asset_sha256"] == _SHA256_BY_NAME[filename]
        assert record.raw["checkpoint_bytes_fetched"] is False
        assert any(
            link.url == expected_url and link.relation == "weights" and not link.crawl
            for link in record.links
        )
        assert record.releases[0].version == "v1.0.0"


def test_adapter_rejects_mutable_or_unrelated_source_parameters() -> None:
    with pytest.raises(ValueError, match="first Woosh release"):
        SonyWooshReleaseAssetsSourceAdapter(release_tag="latest")


def test_disabled_proposal_matches_adapter_constructor() -> None:
    source = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "sony_woosh_release_assets"
    adapter = SonyWooshReleaseAssetsSourceAdapter(
        **{
            key: value
            for key, value in source.items()
            if key not in {"adapter", "enabled", "entry_tags", "schedule"}
        }
    )
    assert adapter.repository == source["repository"]
