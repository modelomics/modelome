from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.sources.meta_sam3_checkpoints import MetaSAM3CheckpointSourceAdapter

_REVISION = "c" * 40
_BUILDER = '''
def download_ckpt_from_hf(version="sam3"):
    if version == "sam3.1":
        repo_id = "facebook/sam3.1"
        ckpt_name = "sam3.1_multiplex.pt"
        cfg_name = "config.json"
    else:
        repo_id = "facebook/sam3"
        ckpt_name = "sam3.pt"
        cfg_name = "config.json"
    _ = hf_hub_download(repo_id=repo_id, filename=cfg_name)
    return hf_hub_download(repo_id=repo_id, filename=ckpt_name)
'''


class FakeClient:
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
            body = f'{{"sha":"{_REVISION}"}}'.encode()
        elif url.endswith("/sam3/model_builder.py"):
            body = _BUILDER.encode()
        elif url.endswith("/README.md"):
            body = b"Request access to the checkpoints on Hugging Face before downloading."
        else:
            raise AssertionError(f"unexpected URL {url}")
        return HttpResponse(200, {}, body, url)


def test_meta_sam3_maps_both_gated_checkpoint_releases_without_fetching_weights() -> None:
    client = FakeClient()
    page = MetaSAM3CheckpointSourceAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    records = {record.raw["checkpoint_handle"]: record for record in page.records}
    assert set(records) == {"sam3", "sam3.1"}
    assert records["sam3"].links[-1].url == (
        "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"
    )
    assert records["sam3.1"].links[-1].url == (
        "https://huggingface.co/facebook/sam3.1/resolve/main/sam3.1_multiplex.pt"
    )
    assert all(record.releases[0].metadata["access_restricted"] for record in records.values())
    assert all(record.links[-1].relation == "weights" for record in records.values())
    assert not any("huggingface.co" in url for url in client.calls)

    entries = build_entries(
        source_record_to_entry_seed(record, source="meta-sam3-checkpoints")
        for record in page.records
    )
    assert len(entries.entries) == 2
    assert {entry.canonical_name for entry in entries.entries} == {
        "sam3",
        "sam3.1",
    }
    assert all(len(entry.releases) == 1 for entry in entries.entries)


def test_adapter_rejects_readme_without_gated_access_notice() -> None:
    client = FakeClient()
    adapter = MetaSAM3CheckpointSourceAdapter(client=client)

    def no_access_notice(url: str, **kwargs: Any) -> HttpResponse:
        if "/commits/" in url:
            return HttpResponse(200, {}, f'{{"sha":"{_REVISION}"}}'.encode(), url)
        return HttpResponse(200, {}, b"no access info", url)

    client.get = no_access_notice  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="no longer documents gated"):
        adapter.fetch_page({})


def test_adapter_is_only_in_a_disabled_proposal() -> None:
    path = Path(__file__).parents[1] / "config/proposals/meta_sam3_checkpoints.toml"
    proposal = tomllib.loads(path.read_text())["source"][0]

    assert proposal["adapter"] == "meta_sam3_checkpoints"
    assert proposal["enabled"] is False
    assert proposal["repository"] == "facebookresearch/sam3"
