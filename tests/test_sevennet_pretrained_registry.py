from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.sevennet_pretrained_registry import (
    SevenNetPretrainedRegistryAdapter,
    _parse_checkpoint_map,
)

REVISION = "f" * 40
COMMIT_URL = "https://api.github.com/repos/MDIL-SNU/SevenNet/commits/main"
CONST_URL = (
    f"https://raw.githubusercontent.com/MDIL-SNU/SevenNet/{REVISION}/sevenn/_const.py"
)
UTIL_URL = f"https://raw.githubusercontent.com/MDIL-SNU/SevenNet/{REVISION}/sevenn/util.py"
CONST_SOURCE = '''\
_git_prefix = 'https://github.com/MDIL-SNU/SevenNet/releases/download'
SEVENNET_omni = f'{_prefix}/SevenNet_omni/checkpoint_sevennet_omni.pth'
CHECKPOINT_DOWNLOAD_LINKS = {
    SEVENNET_omni: f'{_git_prefix}/v0.12.0.cp/checkpoint_sevennet_omni.pth',
    SEVENNET_omni_i8: f'{_git_prefix}/v0.12.1.cp/checkpoint_sevennet_omni_i8.pth',
}
'''
UTIL_SOURCE = '''\
def get_available_pretrained_models():
    checkpoint_to_name = {
        'SEVENNET_omni': '7net-omni',
        'SEVENNET_omni_i8': '7net-omni-i8',
    }
    return checkpoint_to_name
'''


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(url: str, body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, url)


def release_response(tag: str, filename: str, asset_id: int) -> HttpResponse:
    url = f"https://api.github.com/repos/MDIL-SNU/SevenNet/releases/tags/{tag}"
    payload = {
        "assets": [
            {
                "id": asset_id,
                "name": filename,
                "size": asset_id * 100,
                "browser_download_url": (
                    f"https://github.com/MDIL-SNU/SevenNet/releases/download/{tag}/{filename}"
                ),
            }
        ]
    }
    return response(url, json.dumps(payload).encode())


def test_parses_official_model_name_to_checkpoint_release_map() -> None:
    rows = _parse_checkpoint_map(CONST_SOURCE, UTIL_SOURCE, "test")

    assert rows == (
        (
            "7net-omni",
            "SEVENNET_omni",
            "https://github.com/MDIL-SNU/SevenNet/releases/download/"
            "v0.12.0.cp/checkpoint_sevennet_omni.pth",
        ),
        (
            "7net-omni-i8",
            "SEVENNET_omni_i8",
            "https://github.com/MDIL-SNU/SevenNet/releases/download/"
            "v0.12.1.cp/checkpoint_sevennet_omni_i8.pth",
        ),
    )


def test_fetches_only_source_and_release_metadata_and_records_assets() -> None:
    client = QueueClient(
        response(COMMIT_URL, json.dumps({"sha": REVISION}).encode()),
        response(CONST_URL, CONST_SOURCE.encode()),
        response(UTIL_URL, UTIL_SOURCE.encode()),
        release_response("v0.12.0.cp", "checkpoint_sevennet_omni.pth", 1),
        release_response("v0.12.1.cp", "checkpoint_sevennet_omni_i8.pth", 2),
    )

    page = SevenNetPretrainedRegistryAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert len(client.calls) == 5
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.raw["checkpoint_filename"] == "checkpoint_sevennet_omni.pth"
    assert record.raw["size_bytes"] == 100
    assert record.models[0].identifiers == (Identifier("sevennet:checkpoint", "7net-omni"),)
    assert record.raw["binary_reachability_checked"] is False


@pytest.mark.parametrize(
    ("const_text", "util_text"),
    [
        (
            CONST_SOURCE.replace(
                "_git_prefix}/v0.12.0.cp", "https://evil.test/releases/v0.12.0.cp"
            ),
            UTIL_SOURCE,
        ),
        (
            CONST_SOURCE.replace(
                "checkpoint_sevennet_omni.pth", "checkpoint_sevennet_omni.zip"
            ),
            UTIL_SOURCE,
        ),
        (CONST_SOURCE, UTIL_SOURCE.replace("'7net-omni'", "None")),
    ],
)
def test_rejects_changed_or_unmapped_checkpoint_sources(const_text: str, util_text: str) -> None:
    with pytest.raises(ValueError):
        _parse_checkpoint_map(const_text, util_text, "test")


def test_live_sevennet_source_and_release_metadata_smoke() -> None:
    """Read first-party source and release metadata; never download checkpoint bytes."""
    try:
        page = SevenNetPretrainedRegistryAdapter().fetch_page({})
    except Exception as error:  # pragma: no cover - network-dependent smoke
        pytest.skip(f"live SevenNet metadata unavailable: {error}")
    assert page.authoritative_snapshot
    assert page.upstream_count == 9
    assert all(record.raw["github_asset_id"] for record in page.records)
