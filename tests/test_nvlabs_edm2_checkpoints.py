from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.nvlabs_edm2_checkpoints import NVlabsEDM2CheckpointSourceAdapter, _parse_urls

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/nvlabs_edm2_checkpoints.toml"
_SHA = "d" * 40
_PREFIX = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/"
_FILES = (
    "edm2-img512-xxl-0939524-0.015.pkl",
    "edm2-img512-xs-uncond-2147483-0.015.pkl",
    "edm2-img512-s-2147483-0.130.pkl",
)
_README = f"""## Using pre-trained models
--net={_PREFIX}{_FILES[0]}
--gnet={_PREFIX}{_FILES[1]}
## Calculating FLOPs and metrics
python count_flops.py \\
    {_PREFIX}{_FILES[2]}
--ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl
## Preparing datasets
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


def test_parser_finds_exact_model_urls_in_bounded_readme_sections() -> None:
    rows = _parse_urls(_README, maximum=10)
    assert len(rows) == 3
    assert {url.rsplit("/", 1)[-1] for url in rows} == set(_FILES)
    assert all(url.startswith(_PREFIX) for url in rows)


def test_parser_excludes_other_hosts_and_dataset_reference_stats() -> None:
    modified = _README.replace(_PREFIX + _FILES[0], "https://evil.example/model.pkl")
    rows = _parse_urls(modified, maximum=10)
    assert len(rows) == 2
    assert all("evil.example" not in url for url in rows)
    assert _parse_urls("## Nothing\n" + _PREFIX + _FILES[0], maximum=10) == ()


def test_adapter_emits_three_authoritative_metadata_only_records() -> None:
    client = _Client()
    page = NVlabsEDM2CheckpointSourceAdapter(client=client).fetch_page({})
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 3
    records = {record.raw["checkpoint_name"]: record for record in page.records}
    assert set(records) == set(_FILES)
    assert records[_FILES[1]].raw["role"] == "guidance"
    assert records[_FILES[1]].raw["category"] == "unconditional-image-generation"
    assert all(record.raw["checkpoint_bytes_fetched"] is False for record in page.records)
    assert len(client.calls) == 2


def test_proposal_is_disabled_and_matches_constructor() -> None:
    source = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "nvlabs_edm2_checkpoints"
    adapter = NVlabsEDM2CheckpointSourceAdapter(
        **{
            key: value
            for key, value in source.items()
            if key not in {"adapter", "enabled", "entry_tags"}
        }
    )
    assert adapter.repository == source["repository"]
