from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.nvlabs_edm_checkpoints import NVlabsEDMCheckpointSourceAdapter, _parse_urls

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/nvlabs_edm_checkpoints.toml"
_SHA = "e" * 40
_BASE = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/"
_README = f"""## Pre-trained models
{_BASE}edm-cifar10-32x32-cond-vp.pkl
{_BASE}edm-ffhq-64x64-uncond-vp.pkl
{_BASE}edm-imagenet-64x64-cond-adm.pkl
{_BASE}baseline/baseline-cifar10-32x32-uncond-vp.pkl
{_BASE}edm-cifar10-32x32-cond-vp.pkl
## Calculating FID
--ref={_BASE}../fid-refs/cifar10-32x32.npz
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
        body = json.dumps({"sha": _SHA}).encode() if "/commits/" in url else _README.encode()
        return HttpResponse(200, {}, body, url)


def test_parser_captures_exact_unique_pkl_links_only_in_model_section() -> None:
    rows = _parse_urls(_README, maximum=10)
    assert len(rows) == 4
    assert rows[0] == _BASE + "edm-cifar10-32x32-cond-vp.pkl"
    assert any("baseline/baseline-cifar10" in row for row in rows)
    assert all(row.endswith(".pkl") for row in rows)
    assert not any("fid-refs" in row for row in rows)


def test_adapter_emits_distinct_metadata_only_records() -> None:
    client = _Client()
    page = NVlabsEDMCheckpointSourceAdapter(client=client).fetch_page({})
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 4
    records = {record.raw["checkpoint_name"]: record for record in page.records}
    assert len(records) == 4
    assert (
        records["edm-imagenet-64x64-cond-adm.pkl"].raw["category"]
        == "class-conditional-image-generation"
    )
    assert (
        records["edm-ffhq-64x64-uncond-vp.pkl"].raw["category"] == "unconditional-image-generation"
    )
    assert all(record.raw["checkpoint_bytes_fetched"] is False for record in page.records)
    assert len(client.calls) == 2


def test_parser_requires_expected_host_and_section() -> None:
    altered = _README.replace(_BASE, "https://evil.example/edm/pretrained/", 1)
    assert len(_parse_urls(altered, maximum=10)) == 4
    assert all("evil.example" not in row for row in _parse_urls(altered, maximum=10))
    assert (
        _parse_urls(
            "## Missing section\n" + _README.split("## Pre-trained models", 1)[1], maximum=10
        )
        == ()
    )


def test_proposal_is_disabled_and_matches_constructor() -> None:
    source = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "nvlabs_edm_checkpoints"
    adapter = NVlabsEDMCheckpointSourceAdapter(
        **{
            key: value
            for key, value in source.items()
            if key not in {"adapter", "enabled", "entry_tags"}
        }
    )
    assert adapter.repository == source["repository"]
