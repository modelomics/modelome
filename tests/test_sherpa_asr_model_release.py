from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.normalize import content_hash
from modelome.sources.sherpa_asr_model_release import SherpaAsrModelReleaseSourceAdapter

_API = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/asr-models"
_ASSETS_API = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/123/assets"
_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"


class _Client:
    def __init__(self, payload: dict, paginated_assets: list | None = None, status: int = 200):
        self.payload = payload
        self.paginated_assets = (
            payload["assets"] if paginated_assets is None else paginated_assets
        )
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        if url == _API:
            body = json.dumps(self.payload).encode()
        elif url.startswith(_ASSETS_API):
            page = int(url.rsplit("=", 1)[-1])
            start = (page - 1) * 100
            body = json.dumps(self.paginated_assets[start : start + 100]).encode()
        else:
            raise AssertionError(f"unexpected URL: {url}")
        return HttpResponse(self.status, {"etag": '"release-etag"'}, body, url)


def _asset(asset_id: int, name: str) -> dict:
    return {
        "id": asset_id,
        "name": name,
        "size": 12345,
        "content_type": "application/x-bzip2",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-02T00:00:00Z",
        "browser_download_url": f"{_RELEASE}{name}",
    }


def _payload() -> dict:
    return {
        "id": 123,
        "tag_name": "asr-models",
        "updated_at": "2025-01-02T00:00:00Z",
        "assets": [
            _asset(1, "sherpa-onnx-streaming-zipformer-en.tar.bz2"),
            _asset(2, "spoken-language-identification-test-wavs.tar.bz2"),
            _asset(3, "librknnrt-android.tar.bz2"),
            _asset(4, "silero_vad_v5.onnx"),
            _asset(5, "sherpa-onnx-whisper-large-v3.tar.bz2"),
        ],
    }


def test_enumerates_only_model_archives_from_complete_release_inventory() -> None:
    client = _Client(_payload())
    page = SherpaAsrModelReleaseSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    ).fetch_page({})

    assert client.calls == [_API, f"{_ASSETS_API}?per_page=100&page=1"]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert [record.title for record in page.records] == [
        "sherpa-onnx-streaming-zipformer-en",
        "sherpa-onnx-whisper-large-v3",
    ]
    assert page.next_state["verified_release_asset_count"] == 5
    assert page.next_state["embedded_inventory_matches"] is True
    assert page.records[0].releases[0].metadata["asset_id"] == 1
    assert page.records[1].links[-1].url == f"{_RELEASE}sherpa-onnx-whisper-large-v3.tar.bz2"


def test_uses_paginated_assets_as_authority_and_records_embedded_mismatch() -> None:
    payload = _payload()
    assets = payload["assets"]
    payload["assets"] = assets[:-1]
    page = SherpaAsrModelReleaseSourceAdapter(
        client=_Client(payload, paginated_assets=assets)
    ).fetch_page({})

    assert page.upstream_count == 2
    assert page.next_state["embedded_asset_count"] == 4
    assert page.next_state["verified_release_asset_count"] == 5
    assert page.next_state["embedded_inventory_matches"] is False
    assert page.next_state["paginated_only_asset_count"] == 1


def test_unchanged_inventory_skips_records_and_walks_terminal_page() -> None:
    payload = _payload()
    signature_rows = [
        [asset["id"], asset["name"], asset["size"], asset["browser_download_url"]]
        for asset in sorted(payload["assets"], key=lambda value: value["id"])
    ]
    client = _Client(payload)
    page = SherpaAsrModelReleaseSourceAdapter(client=client).fetch_page(
        {
            "completed_release_id": "123",
            "asset_inventory_sha256": content_hash(signature_rows),
            "model_count": 2,
        }
    )

    assert page.records == ()
    assert page.upstream_count == 2
    assert client.calls[-1].endswith("page=1")


def test_rejects_bad_release_metadata_urls_and_limits() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        SherpaAsrModelReleaseSourceAdapter(client=_Client(_payload(), status=503)).fetch_page({})

    payload = _payload()
    payload["assets"][0]["browser_download_url"] = "https://example.com/model.tar.bz2"
    with pytest.raises(ValueError, match="invalid ASR release asset URL"):
        SherpaAsrModelReleaseSourceAdapter(client=_Client(payload)).fetch_page({})

    with pytest.raises(ValueError, match="asset limit"):
        SherpaAsrModelReleaseSourceAdapter(client=_Client(_payload()), max_assets=2).fetch_page({})


def test_paginates_more_than_one_page_and_checks_cross_page_ids() -> None:
    payload = _payload()
    assets = [_asset(index + 1, f"sherpa-onnx-model-{index}.tar.bz2") for index in range(205)]
    client = _Client({**payload, "assets": assets})
    page = SherpaAsrModelReleaseSourceAdapter(client=client).fetch_page({})

    assert page.upstream_count == 205
    assert [url.rsplit("=", 1)[-1] for url in client.calls[1:]] == ["1", "2", "3"]

    distinct_assets = list(assets)
    assets[150] = dict(assets[150], id=assets[50]["id"])
    with pytest.raises(ValueError, match="duplicate release asset IDs"):
        SherpaAsrModelReleaseSourceAdapter(
            client=_Client({**payload, "assets": distinct_assets}, paginated_assets=assets)
        ).fetch_page({})
