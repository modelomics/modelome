from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.stability_sdxl_checkpoints import (
    StabilitySDXLCheckpointSourceAdapter,
    _parse_sdxl_sources,
)

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/stability_sdxl_checkpoints.toml"
_SHA = "b" * 40
_README = """\
- [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SDXL-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
- [SDXL-base-0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9)
- [SDXL-refiner-0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9)
The 0.9 models require application under a research license.
"""
_MANIFEST = """\
VERSION2SPECS = {
"SDXL-base-1.0": {
    "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
},
"SDXL-base-0.9": {
    "ckpt": "checkpoints/sd_xl_base_0.9.safetensors",
},
"SDXL-refiner-0.9": {
    "ckpt": "checkpoints/sd_xl_refiner_0.9.safetensors",
},
"SDXL-refiner-1.0": {
    "ckpt": "checkpoints/sd_xl_refiner_1.0.safetensors",
},
}
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
        if "/commits/" in url:
            body = json.dumps({"sha": _SHA}).encode()
        elif url.endswith("/README.md"):
            body = _README.encode()
        elif url.endswith("/scripts/demo/sampling.py"):
            body = _MANIFEST.encode()
        else:
            raise AssertionError(f"unexpected URL {url}")
        return HttpResponse(200, {}, body, url)


def test_parse_sdxl_manifest_joins_exact_variants_and_hf_repositories() -> None:
    rows = _parse_sdxl_sources(_README, _MANIFEST)

    assert {row["variant"]: (row["filename"], row["repo"]) for row in rows} == {
        "SDXL-base-1.0": (
            "sd_xl_base_1.0.safetensors",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ),
        "SDXL-base-0.9": (
            "sd_xl_base_0.9.safetensors",
            "stabilityai/stable-diffusion-xl-base-0.9",
        ),
        "SDXL-refiner-0.9": (
            "sd_xl_refiner_0.9.safetensors",
            "stabilityai/stable-diffusion-xl-refiner-0.9",
        ),
        "SDXL-refiner-1.0": (
            "sd_xl_refiner_1.0.safetensors",
            "stabilityai/stable-diffusion-xl-refiner-1.0",
        ),
    }


def test_adapter_emits_exact_checkpoint_files_and_preserves_09_gate() -> None:
    client = _Client()
    page = StabilitySDXLCheckpointSourceAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    by_variant = {record.raw["variant"]: record for record in page.records}
    assert set(by_variant) == {
        "SDXL-base-1.0",
        "SDXL-refiner-1.0",
        "SDXL-base-0.9",
        "SDXL-refiner-0.9",
    }
    base = by_variant["SDXL-base-1.0"]
    assert base.canonical_url == (
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/"
        "sd_xl_base_1.0.safetensors"
    )
    assert base.models[0].identifiers[0].value == "stabilityai/stable-diffusion-xl-base-1.0"
    assert base.releases[0].metadata["access_status"] == "public-open-rail"
    gated = by_variant["SDXL-base-0.9"]
    assert gated.releases[0].metadata["access_status"] == "research-gated"
    assert gated.raw["checkpoint_bytes_fetched"] is False
    assert len(client.calls) == 3


def test_parser_does_not_infer_a_missing_hugging_face_mapping() -> None:
    incomplete_readme = _README.replace(
        "- [SDXL-base-0.9](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9)\n",
        "",
    )
    rows = _parse_sdxl_sources(incomplete_readme, _MANIFEST)
    assert {row["variant"] for row in rows} == {
        "SDXL-base-1.0",
        "SDXL-refiner-1.0",
        "SDXL-refiner-0.9",
    }


def test_proposal_is_disabled_and_matches_adapter_constructor() -> None:
    config = tomllib.loads(_PROPOSAL.read_text())
    source = config["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "stability_sdxl_checkpoints"
    adapter = StabilitySDXLCheckpointSourceAdapter(
        **{
            key: value
            for key, value in source.items()
            if key not in {"adapter", "enabled", "entry_tags"}
        }
    )
    assert adapter.repository == source["repository"]
    assert adapter.manifest_path == source["manifest_path"]
