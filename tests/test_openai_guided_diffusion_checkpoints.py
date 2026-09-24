from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.openai_guided_diffusion_checkpoints import (
    OpenAIGuidedDiffusionCheckpointSourceAdapter,
    _category,
    _parse_download_rows,
)

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/openai_guided_diffusion_checkpoints.toml"
_SHA = "c" * 40
_ROWS = (
    ("64x64 classifier", "64x64_classifier.pt"),
    ("64x64 diffusion", "64x64_diffusion.pt"),
    ("128x128 classifier", "128x128_classifier.pt"),
    ("128x128 diffusion", "128x128_diffusion.pt"),
    ("256x256 classifier", "256x256_classifier.pt"),
    ("256x256 diffusion", "256x256_diffusion.pt"),
    ("256x256 diffusion (not class conditional)", "256x256_diffusion_uncond.pt"),
    ("512x512 classifier", "512x512_classifier.pt"),
    ("512x512 diffusion", "512x512_diffusion.pt"),
    ("64x64 -&gt; 256x256 upsampler", "64_256_upsampler.pt"),
    ("128x128 -&gt; 512x512 upsampler", "128_512_upsampler.pt"),
    ("LSUN bedroom", "lsun_bedroom.pt"),
    ("LSUN cat", "lsun_cat.pt"),
    ("LSUN horse", "lsun_horse.pt"),
    ("LSUN horse (no dropout)", "lsun_horse_nodropout.pt"),
)
_README = "\n".join(
    ["# guided-diffusion", "# Download pre-trained models", "Here are download links:"]
    + [
        f" * {label}: [{filename}](https://openaipublic.blob.core.windows.net/"
        f"diffusion/jul-2021/{filename})"
        for label, filename in _ROWS
    ]
    + ["# Sampling from pre-trained models", "The rest of the README is not inventory."]
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


def test_parser_indexes_the_complete_official_checkpoint_table() -> None:
    rows = _parse_download_rows(_README, maximum=20)

    assert len(rows) == 15
    assert {row["filename"] for row in rows} == {filename for _, filename in _ROWS}
    assert rows[0]["url"] == (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt"
    )
    assert _category("256x256 diffusion (not class conditional)") == (
        "unconditional-image-generation"
    )
    assert _category("64x64 classifier") == "image-classifier"
    assert _category("64x64 -&gt; 256x256 upsampler") == "super-resolution-diffusion"
    assert _category("LSUN cat") == "class-unconditional-image-generation"


def test_parser_rejects_rows_outside_official_host_and_manifest_section() -> None:
    malicious = _README.replace(
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt",
        "https://evil.example/diffusion/jul-2021/64x64_classifier.pt",
    )
    rows = _parse_download_rows(malicious, maximum=20)
    assert len(rows) == 14
    assert all("evil.example" not in row["url"] for row in rows)


def test_adapter_preserves_named_artifact_and_task_evidence() -> None:
    client = _Client()
    page = OpenAIGuidedDiffusionCheckpointSourceAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 15
    records = {record.raw["checkpoint_name"]: record for record in page.records}
    assert records["256x256_diffusion_uncond.pt"].raw["category"] == (
        "unconditional-image-generation"
    )
    assert records["64_256_upsampler.pt"].releases[0].metadata["category"] == (
        "super-resolution-diffusion"
    )
    assert records["64_256_upsampler.pt"].models[0].name == "64x64 -> 256x256 upsampler"
    assert records["lsun_horse_nodropout.pt"].models[0].name == "LSUN horse (no dropout)"
    assert records["64x64_diffusion.pt"].canonical_url.endswith("/64x64_diffusion.pt")
    assert records["64x64_diffusion.pt"].raw["checkpoint_bytes_fetched"] is False
    assert len(client.calls) == 2


def test_proposal_is_disabled_and_matches_adapter_constructor() -> None:
    config = tomllib.loads(_PROPOSAL.read_text())
    source = config["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "openai_guided_diffusion_checkpoints"
    adapter = OpenAIGuidedDiffusionCheckpointSourceAdapter(
        **{
            key: value
            for key, value in source.items()
            if key not in {"adapter", "enabled", "entry_tags"}
        }
    )
    assert adapter.repository == source["repository"]
    assert adapter.document_path == source["document_path"]
