from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.openai_consistency_cifar10_checkpoints import (
    OpenAIConsistencyCIFAR10CheckpointSourceAdapter,
    _parse_rows,
)

_PROPOSAL = (
    Path(__file__).parents[1] / "config/proposals/openai_consistency_cifar10_checkpoints.toml"
)
_SHA = "a" * 40
_PREFIX = "https://openaipublic.blob.core.windows.net/consistency/jcm_checkpoints/"
_ASSETS = (
    ("EDM on CIFAR-10", "edm_cifar10_ema", "edm_cifar10_ema"),
    ("CD on CIFAR-10 with l1 metric", "cd-l1", "cd-l1/checkpoints/checkpoint_80"),
    ("CD on CIFAR-10 with l2 metric", "cd-l2", "cd-l2/checkpoints/checkpoint_80"),
    ("CD on CIFAR-10 with LPIPS metric", "cd-lpips", "cd-lpips/checkpoints/checkpoint_80"),
    (
        "CT on CIFAR-10 with adaptive schedules and LPIPS metric",
        "ct-lpips",
        "ct-lpips/checkpoints/checkpoint_74",
    ),
    (
        "Continuous-time CD on CIFAR-10 with l2 metric",
        "cifar10-continuous-cd-l2",
        "cifar10-continuous-cd-l2/checkpoints/checkpoint_40",
    ),
    (
        "Continuous-time CD on CIFAR-10 with l2 metric and stopgrad",
        "cifar10-continuous-cd-l2-stopgrad",
        "cifar10-continuous-cd-l2-stopgrad/checkpoints/checkpoint_40",
    ),
    (
        "Continuous-time CD on CIFAR-10 with LPIPS metric and stopgrad",
        "cifar10-continuous-cd-lpips-stopgrad",
        "cifar10-continuous-cd-lpips-stopgrad/checkpoints/checkpoint_40",
    ),
    (
        "Continuous-time CT on CIFAR-10 with l2 metric",
        "continuous-ct-l2",
        "continuous-ct-l2/checkpoints/checkpoint_80",
    ),
    (
        "Continuous-time CT on CIFAR-10 with LPIPS metric",
        "continuous-ct-lpips",
        "continuous-ct-lpips/checkpoints/checkpoint_40",
    ),
)
_README = (
    "# Pre-trained models\n"
    + "\n".join(f" * {label}: [{name}]({_PREFIX}{path})" for label, name, path in _ASSETS)
    + "\n# Dependencies\nnot inventory\n"
)


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


def test_parser_extracts_ten_exact_first_party_model_links() -> None:
    rows = _parse_rows(_README, maximum=20)
    assert len(rows) == 10
    assert rows[0]["url"] == _PREFIX + "edm_cifar10_ema"
    assert rows[-1]["url"] == _PREFIX + "continuous-ct-lpips/checkpoints/checkpoint_40"
    assert len({row["url"] for row in rows}) == 10


def test_parser_rejects_unapproved_hosts_and_detects_duplicate_urls() -> None:
    altered = _README.replace(_PREFIX, "https://example.org/consistency/jcm_checkpoints/", 1)
    assert len(_parse_rows(altered, maximum=20)) == 9
    duplicate_line = _README.splitlines()[1]
    duplicated = _README.replace("# Dependencies", duplicate_line + "\n# Dependencies")
    try:
        _parse_rows(duplicated, maximum=20)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate URL was accepted")


def test_adapter_emits_ten_authoritative_metadata_only_records() -> None:
    client = _Client()
    page = OpenAIConsistencyCIFAR10CheckpointSourceAdapter(client=client).fetch_page({})
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 10
    assert {record.raw["checkpoint_url"] for record in page.records} == {
        row["url"] for row in _parse_rows(_README, maximum=20)
    }
    assert all(
        record.raw["category"] == "unconditional-image-generation" for record in page.records
    )
    assert all(record.raw["checkpoint_bytes_fetched"] is False for record in page.records)
    assert len(client.calls) == 2


def test_proposal_is_disabled_and_matches_constructor() -> None:
    source = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "openai_consistency_cifar10_checkpoints"
    adapter = OpenAIConsistencyCIFAR10CheckpointSourceAdapter(
        **{
            key: value
            for key, value in source.items()
            if key not in {"adapter", "enabled", "entry_tags"}
        }
    )
    assert adapter.repository == source["repository"]
