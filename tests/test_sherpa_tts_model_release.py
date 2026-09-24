from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.normalize import content_hash
from modelome.sources.sherpa_tts_model_release import SherpaTtsModelReleaseSourceAdapter

_API = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/tts-models"
_ASSETS_API = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/123/assets"
_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"


class _Client:
    def __init__(
        self, payload: dict, status: int = 200, paginated_assets: list | None = None
    ) -> None:
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
        "tag_name": "tts-models",
        "html_url": "https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models",
        "updated_at": "2025-01-02T00:00:00Z",
        "assets": [
            _asset(1, "vits-piper-en_US-amy-low.tar.bz2"),
            _asset(2, "kokoro-multi-lang-v1_0.tar.bz2"),
            _asset(3, "espeak-ng-data.tar.bz2"),
            _asset(4, "checksums.txt"),
        ],
    }


def test_enumerates_exact_tts_release_archives_and_skips_auxiliary_assets() -> None:
    client = _Client(_payload())
    adapter = SherpaTtsModelReleaseSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [_API, f"{_ASSETS_API}?per_page=100&page=1"]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert [record.title for record in page.records] == [
        "kokoro-multi-lang-v1_0",
        "vits-piper-en_US-amy-low",
    ]
    assert page.records[0].releases[0].metadata["release_id"] == "123"
    assert page.records[1].links[-1].url == f"{_RELEASE}vits-piper-en_US-amy-low.tar.bz2"
    assert page.next_state["verified_release_asset_count"] == 4
    assert all(link.crawl is False for record in page.records for link in record.links)


def test_skips_unchanged_release_and_rejects_unexpected_urls_or_tag() -> None:
    payload = _payload()
    client = _Client(payload)
    page = SherpaTtsModelReleaseSourceAdapter(client=client).fetch_page(
        {
            "completed_release_id": "123",
            "asset_inventory_sha256": content_hash(
                [
                    [asset["id"], asset["name"], asset["size"], asset["browser_download_url"]]
                    for asset in sorted(payload["assets"], key=lambda value: value["id"])
                ]
            ),
            "model_count": 2,
        }
    )
    assert page.records == ()
    assert page.upstream_count == 2

    payload = _payload()
    payload["assets"][0]["browser_download_url"] = "https://example.com/model.tar.bz2"
    with pytest.raises(ValueError, match="invalid TTS release asset URL"):
        SherpaTtsModelReleaseSourceAdapter(client=_Client(payload)).fetch_page({})

    payload = _payload()
    payload["tag_name"] = "asr-models"
    with pytest.raises(ValueError, match="not the expected TTS release"):
        SherpaTtsModelReleaseSourceAdapter(client=_Client(payload)).fetch_page({})


def test_rejects_api_errors_incomplete_metadata_and_asset_limit() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        SherpaTtsModelReleaseSourceAdapter(client=_Client(_payload(), status=503)).fetch_page({})

    payload = _payload()
    payload["assets"][0]["size"] = True
    with pytest.raises(ValueError, match="invalid size"):
        SherpaTtsModelReleaseSourceAdapter(client=_Client(payload)).fetch_page({})

    with pytest.raises(ValueError, match="asset limit"):
        SherpaTtsModelReleaseSourceAdapter(client=_Client(_payload()), max_assets=2).fetch_page({})


def test_uses_paginated_inventory_and_reports_embedded_discrepancy() -> None:
    payload = _payload()
    paginated = list(payload["assets"])
    payload["assets"] = payload["assets"][:-1]
    page = SherpaTtsModelReleaseSourceAdapter(
        client=_Client(payload, paginated_assets=paginated)
    ).fetch_page({})

    assert page.next_state["embedded_asset_count"] == 3
    assert page.next_state["verified_release_asset_count"] == 4
    assert page.next_state["embedded_inventory_matches"] is False
    assert page.next_state["embedded_only_asset_count"] == 0
    assert page.next_state["paginated_only_asset_count"] == 1


def test_walks_full_pages_until_short_terminal_page() -> None:
    payload = _payload()
    assets = [_asset(index + 1, f"vits-model-{index}.tar.bz2") for index in range(205)]
    payload["assets"] = assets
    client = _Client(payload)

    page = SherpaTtsModelReleaseSourceAdapter(client=client).fetch_page({})

    assert page.next_state["verified_release_asset_count"] == 205
    assert client.calls == [
        _API,
        f"{_ASSETS_API}?per_page=100&page=1",
        f"{_ASSETS_API}?per_page=100&page=2",
        f"{_ASSETS_API}?per_page=100&page=3",
    ]
