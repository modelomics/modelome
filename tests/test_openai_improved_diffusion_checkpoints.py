from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.openai_improved_diffusion_checkpoints import (
    OpenAIImprovedDiffusionCheckpointSourceAdapter,
    _category,
    _parse_rows,
)

_PROPOSAL = (
    Path(__file__).parents[1] / "config/proposals/openai_improved_diffusion_checkpoints.toml"
)
_SHA = "d" * 40
_ASSETS = (
    (
        "Unconditional ImageNet-64 with our `L_hybrid` objective and cosine noise schedule",
        "imagenet64_uncond_100M_1500K.pt",
    ),
    (
        "Unconditional CIFAR-10 with our `L_hybrid` objective and cosine noise schedule",
        "cifar10_uncond_50M_500K.pt",
    ),
    (
        "Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations)",
        "imagenet64_cond_270M_250K.pt",
    ),
    (
        "Upsampling 256x256 model (280M parameters, trained for 500K iterations)",
        "upsample_cond_500K.pt",
    ),
    ("LSUN bedroom model (lr=1e-4)", "lsun_uncond_100M_1200K_bs128.pt"),
    ("LSUN bedroom model (lr=2e-5)", "lsun_uncond_100M_2400K_bs64.pt"),
    (
        "Unconditional ImageNet-64 with the `L_vlb` objective and cosine noise schedule",
        "imagenet64_uncond_vlb_100M_1500K.pt",
    ),
    (
        "Unconditional CIFAR-10 with the `L_vlb` objective and cosine noise schedule",
        "cifar10_uncond_vlb_50M_500K.pt",
    ),
)
_PREFIX = "https://openaipublic.blob.core.windows.net/diffusion/march-2021/"
_README = "# improved-diffusion\n## Models and Hyperparameters\n" + "\n\n".join(
    f"{label} [[checkpoint]({_PREFIX}{filename})]:\n\n```bash\nflags\n```"
    for label, filename in _ASSETS
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


def test_parser_extracts_only_exact_checkpoint_links_from_section() -> None:
    rows = _parse_rows(_README + "\nUnlisted [checkpoint](https://example.org/no.pt)", maximum=10)
    assert len(rows) == 8
    assert {row["filename"] for row in rows} == {filename for _, filename in _ASSETS}
    assert rows[0]["url"] == _PREFIX + "imagenet64_uncond_100M_1500K.pt"
    assert _category(_ASSETS[2][0]) == "class-conditional-image-generation"
    assert _category(_ASSETS[3][0]) == "super-resolution-diffusion"
    assert _category(_ASSETS[4][0]) == "unconditional-image-generation"


def test_parser_excludes_foreign_host_and_rejects_duplicate_files() -> None:
    evil = _README.replace(_PREFIX, "https://evil.example/diffusion/march-2021/", 1)
    assert len(_parse_rows(evil, maximum=10)) == 7
    duplicate = _README + "\n" + _README.splitlines()[2]
    try:
        _parse_rows(duplicate, maximum=20)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate checkpoint was accepted")


def test_adapter_emits_eight_authoritative_metadata_only_records() -> None:
    client = _Client()
    page = OpenAIImprovedDiffusionCheckpointSourceAdapter(client=client).fetch_page({})
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 8
    records = {record.raw["checkpoint_name"]: record for record in page.records}
    assert set(records) == {filename for _, filename in _ASSETS}
    assert records["upsample_cond_500K.pt"].raw["category"] == "super-resolution-diffusion"
    assert records["imagenet64_cond_270M_250K.pt"].models[0].name.startswith("Class-conditional")
    assert records["cifar10_uncond_vlb_50M_500K.pt"].raw["checkpoint_bytes_fetched"] is False
    assert all(record.canonical_url.startswith(_PREFIX) for record in page.records)
    assert len(client.calls) == 2


def test_proposal_is_disabled_and_constructor_matches() -> None:
    source = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "openai_improved_diffusion_checkpoints"
    adapter = OpenAIImprovedDiffusionCheckpointSourceAdapter(
        **{
            key: value
            for key, value in source.items()
            if key not in {"adapter", "enabled", "entry_tags"}
        }
    )
    assert adapter.repository == source["repository"]
    assert adapter.max_checkpoints == source["max_checkpoints"]
