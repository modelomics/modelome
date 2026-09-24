from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.openai_consistency_checkpoints import (
    OpenAIConsistencyCheckpointSourceAdapter,
    _category,
    _parse_rows,
)

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/openai_consistency_checkpoints.toml"
_SHA = "f" * 40
_FILES = (
    ("EDM on ImageNet-64", "edm_imagenet64_ema.pt"),
    ("CD on ImageNet-64 with l2 metric", "cd_imagenet64_l2.pt"),
    ("CD on ImageNet-64 with LPIPS metric", "cd_imagenet64_lpips.pt"),
    ("CT on ImageNet-64", "ct_imagenet64.pt"),
    ("EDM on LSUN Bedroom-256", "edm_bedroom256_ema.pt"),
    ("CD on LSUN Bedroom-256 with l2 metric", "cd_bedroom256_l2.pt"),
    ("CD on LSUN Bedroom-256 with LPIPS metric", "cd_bedroom256_lpips.pt"),
    ("CT on LSUN Bedroom-256", "ct_bedroom256.pt"),
    ("EDM on LSUN Cat-256", "edm_cat256_ema.pt"),
    ("CD on LSUN Cat-256 with l2 metric", "cd_cat256_l2.pt"),
    ("CD on LSUN Cat-256 with LPIPS metric", "cd_cat256_lpips.pt"),
    ("CT on LSUN Cat-256", "ct_cat256.pt"),
)
_PREFIX = "https://openaipublic.blob.core.windows.net/consistency/"
_README = (
    "# Consistency Models\n# Pre-trained models\n\n"
    + "\n".join(f" * {label}: [{filename}]({_PREFIX}{filename})" for label, filename in _FILES)
    + "\n# Dependencies\nNot in inventory.\n"
)


class _Client:
    def __init__(self, readme: str = _README) -> None:
        self.readme = readme
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        body = json.dumps({"sha": _SHA}).encode() if "/commits/" in url else self.readme.encode()
        return HttpResponse(200, {}, body, url)


def test_parser_indexes_the_exact_finite_checkpoint_list() -> None:
    rows = _parse_rows(_README, maximum=20)
    assert len(rows) == 12
    assert {row["filename"] for row in rows} == {filename for _, filename in _FILES}
    assert rows[0]["url"] == _PREFIX + "edm_imagenet64_ema.pt"
    assert _category("EDM on ImageNet-64") == "class-conditional-image-generation"
    assert _category("CD on LSUN Cat-256 with l2 metric") == "unconditional-image-generation"


def test_parser_excludes_untrusted_host_and_rejects_duplicate_assets() -> None:
    modified = _README.replace(_PREFIX, "https://example.org/consistency/", 1)
    assert len(_parse_rows(modified, maximum=20)) == 11
    duplicate = _README.replace(
        "# Dependencies",
        " * EDM on ImageNet-64: [edm_imagenet64_ema.pt]("
        + _PREFIX
        + "edm_imagenet64_ema.pt)\n# Dependencies",
    )
    try:
        _parse_rows(duplicate, maximum=20)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate checkpoint was accepted")


def test_adapter_emits_twelve_authoritative_metadata_only_records() -> None:
    client = _Client()
    page = OpenAIConsistencyCheckpointSourceAdapter(client=client).fetch_page({})
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 12
    records = {record.raw["checkpoint_name"]: record for record in page.records}
    assert set(records) == {filename for _, filename in _FILES}
    assert records["edm_imagenet64_ema.pt"].raw["category"] == "class-conditional-image-generation"
    assert records["ct_bedroom256.pt"].raw["category"] == "unconditional-image-generation"
    assert records["ct_bedroom256.pt"].raw["checkpoint_bytes_fetched"] is False
    assert all(record.canonical_url.startswith(_PREFIX) for record in page.records)
    assert len(client.calls) == 2


def test_proposal_is_disabled_and_matches_adapter_constructor() -> None:
    source = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "openai_consistency_checkpoints"
    adapter = OpenAIConsistencyCheckpointSourceAdapter(
        **{
            key: value
            for key, value in source.items()
            if key not in {"adapter", "enabled", "entry_tags"}
        }
    )
    assert adapter.repository == source["repository"]
