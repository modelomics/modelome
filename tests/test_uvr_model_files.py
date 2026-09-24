from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.uvr_model_files import UvrModelFilesAdapter

_MANIFEST = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"
)


class _Client:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.body = json.dumps(payload).encode()
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(self.status, {"content-type": "application/json"}, self.body, url)


def _payload() -> dict[str, Any]:
    return {
        "current_version": "UVR_Patch_10_6_23_4_27",
        "vr_download_list": {
            "VR Arch Single Model v5: 1_HP-UVR": "1_HP-UVR.pth",
            "VR Arch Single Model v5: 2_HP-UVR": "2_HP-UVR.pth",
        },
        "mdx_download_list": {
            "MDX-Net Model: UVR-MDX-NET Karaoke 2": "UVR_MDXNET_KARA_2.onnx",
        },
        "mdx_download_vip_list": {"private": "private.onnx"},
        "demucs_download_list": {"Demucs v4: htdemucs": {"htdemucs.yaml": "url"}},
        "roformer_download_list": {"Roformer": {"model.ckpt": "url"}},
    }


def test_indexes_only_public_vr_and_mdx_models_with_exact_release_urls() -> None:
    client = _Client(_payload())
    adapter = UvrModelFilesAdapter(
        client=client,
        min_models=1,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [_MANIFEST]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert page.next_state["groups"] == {"vr_download_list": 2, "mdx_download_list": 1}
    records = {record.raw["filename"]: record for record in page.records}
    assert set(records) == {"1_HP-UVR.pth", "2_HP-UVR.pth", "UVR_MDXNET_KARA_2.onnx"}
    assert records["UVR_MDXNET_KARA_2.onnx"].canonical_url == (
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx"
    )
    assert records["1_HP-UVR.pth"].releases[0].metadata["architecture"] == "VR"
    assert records["UVR_MDXNET_KARA_2.onnx"].releases[0].metadata["architecture"] == "MDX"
    assert all(record.links[-1].crawl is False for record in page.records)


def test_unchanged_manifest_returns_empty_page_with_cached_count() -> None:
    adapter = UvrModelFilesAdapter(client=_Client(_payload()), min_models=1)
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    assert second.records == ()
    assert second.upstream_count == 3
    assert second.next_state["manifest_sha256"] == first.next_state["manifest_sha256"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(vr_download_list=[]), "group vr_download_list is missing"),
        (
            lambda p: p["mdx_download_list"].update(bad="../escape.onnx"),
            "invalid UVR filename",
        ),
        (
            lambda p: p["vr_download_list"].update(duplicate="1_HP-UVR.pth"),
            "duplicate UVR checkpoint",
        ),
    ],
)
def test_fails_closed_on_bad_manifest_groups_or_filenames(mutate, message: str) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        UvrModelFilesAdapter(client=_Client(payload), min_models=1).fetch_page({})


def test_enforces_inventory_bounds_and_http_status() -> None:
    with pytest.raises(ValueError, match="only 3 models"):
        UvrModelFilesAdapter(client=_Client(_payload()), min_models=4).fetch_page({})
    with pytest.raises(ValueError, match="HTTP 503"):
        UvrModelFilesAdapter(client=_Client(_payload(), status=503), min_models=1).fetch_page({})
