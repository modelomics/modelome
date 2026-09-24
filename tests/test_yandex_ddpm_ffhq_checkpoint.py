from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.yandex_ddpm_ffhq_checkpoint import (
    YandexDDPMFFHQCheckpointSourceAdapter,
    _parse_rows,
)

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/yandex_ddpm_ffhq_checkpoint.toml"
_SHA = "b" * 40
_URL = "https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/models/ddpm_checkpoints/ffhq.pt"
_README = f"""## DDPM
### Pretrained DDPMs
The LSUN models are adopted from guided-diffusion.
*LSUN-Bedroom:* [lsun_bedroom.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt)\\
*FFHQ-256:* [ffhq.pt]({_URL}) (Updated 3/8/2022)\\
*LSUN-Cat:* [lsun_cat.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_cat.pt)\\
### Run
Available checkpoint names: lsun_bedroom, ffhq
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


def test_parser_only_admits_the_separately_trained_ffhq_artifact() -> None:
    rows = _parse_rows(_README, maximum=5)
    assert rows == ({"label": "FFHQ-256", "filename": "ffhq.pt", "url": _URL},)


def test_parser_rejects_unexpected_host_or_filename() -> None:
    changed = _README.replace(_URL, "https://evil.example/models/ffhq.pt")
    assert _parse_rows(changed, maximum=5) == ()
    changed = _README.replace("ffhq.pt", "other.pt")
    assert _parse_rows(changed, maximum=5) == ()


def test_adapter_emits_exact_authoritative_metadata_only_ffhq_record() -> None:
    client = _Client()
    page = YandexDDPMFFHQCheckpointSourceAdapter(client=client).fetch_page({})
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 1
    record = page.records[0]
    assert record.canonical_url == _URL
    assert record.raw["dataset"] == "FFHQ-256"
    assert record.raw["checkpoint_bytes_fetched"] is False
    assert record.models[0].name == "Yandex DDPM FFHQ-256"
    assert len(client.calls) == 2


def test_proposal_is_disabled_and_constructor_matches() -> None:
    source = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "yandex_ddpm_ffhq_checkpoint"
    adapter = YandexDDPMFFHQCheckpointSourceAdapter(
        **{
            key: value
            for key, value in source.items()
            if key not in {"adapter", "enabled", "entry_tags"}
        }
    )
    assert adapter.repository == source["repository"]
