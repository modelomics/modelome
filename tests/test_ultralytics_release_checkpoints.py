from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.ultralytics_release_checkpoints import (
    UltralyticsReleaseCheckpointSourceAdapter,
    _parse_docs,
)

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/ultralytics_release_checkpoints.toml"
_SHA = "a" * 40
_YOLOV8_DOC = """\
| Model | Filenames | Task |
| --- | --- | --- |
| YOLOv8 | `yolov8n.pt` `yolov8s.pt` | Detection |
| YOLOv8-seg | `yolov8n-seg.pt` | Segmentation |
| YOLOv8-cls | `yolov8n-cls.pt` | Classification |
| YOLOv8-pose | `yolov8n-pose.pt` | Pose |
| YOLOv8-obb | `yolov8n-obb.pt` | OBB |
| YOLOv8-world | `yolov8-world.pt` | Open vocabulary |
"""
_YOLO11_DOC = """\
| Model | Filenames | Task |
| --- | --- | --- |
| YOLO11 | `yolo11n.pt` `yolo11s.pt` | Detection |
| YOLO11-seg | `yolo11n-seg.pt` | Segmentation |
"""


class _Client:
    def __init__(self, releases: list[dict[str, Any]]) -> None:
        self.releases = releases
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
        elif "/releases/" in url and "/assets?" in url:
            release_id = int(url.split("/releases/", 1)[1].split("/assets", 1)[0])
            page = int(url.rsplit("page=", 1)[1])
            per_page = int(url.split("per_page=", 1)[1].split("&", 1)[0])
            release = next(item for item in self.releases if item["id"] == release_id)
            assets = release["all_assets"]
            start = (page - 1) * per_page
            body = json.dumps(assets[start : start + per_page]).encode()
        elif "/releases?" in url:
            page = int(url.rsplit("page=", 1)[1])
            per_page = int(url.split("per_page=", 1)[1].split("&", 1)[0])
            start = (page - 1) * per_page
            body = json.dumps(self.releases[start : start + per_page]).encode()
        elif "yolov8.md" in url:
            body = _YOLOV8_DOC.encode()
        elif "yolo11.md" in url:
            body = _YOLO11_DOC.encode()
        else:
            raise AssertionError(f"unexpected URL: {url}")
        return HttpResponse(200, {}, body, url)


def _asset(asset_id: int, name: str, url: str | None = None) -> dict[str, Any]:
    return {
        "id": asset_id,
        "name": name,
        "browser_download_url": url
        or (f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"),
    }


def _release(tag: str, assets: list[dict[str, Any]], *, release_id: int = 1) -> dict[str, Any]:
    return {
        "id": release_id,
        "tag_name": tag,
        "html_url": f"https://github.com/ultralytics/assets/releases/tag/{tag}",
        "assets_url": (
            f"https://api.github.com/repos/ultralytics/assets/releases/{release_id}/assets"
        ),
        # GitHub embeds a truncated first batch in release-list responses.
        "assets": assets[:30],
        "all_assets": assets,
        "draft": False,
    }


def test_parse_docs_pairs_only_supported_filenames_to_tasks() -> None:
    assert _parse_docs(_YOLOV8_DOC) == {
        "yolov8n.pt": "object-detection",
        "yolov8s.pt": "object-detection",
        "yolov8n-seg.pt": "instance-segmentation",
        "yolov8n-cls.pt": "image-classification",
        "yolov8n-pose.pt": "pose-estimation",
        "yolov8n-obb.pt": "oriented-object-detection",
    }


def test_release_asset_join_is_exact_and_keeps_release_evidence() -> None:
    client = _Client(
        [
            _release(
                "v8.3.0",
                [
                    _asset(101, "yolov8n.pt"),
                    _asset(102, "yolov8n-seg.pt"),
                    _asset(103, "yolo11n.pt"),
                    _asset(104, "yolo26n.pt"),
                    _asset(105, "yolov8n.onnx"),
                ],
            ),
        ]
    )
    adapter = UltralyticsReleaseCheckpointSourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    records = {record.title: record for record in page.records}
    assert set(records) == {
        "YOLOv8n checkpoint (v8.3.0)",
        "YOLOv8n-seg checkpoint (v8.3.0)",
        "YOLO11n checkpoint (v8.3.0)",
    }
    assert records["YOLOv8n checkpoint (v8.3.0)"].canonical_url.endswith("/yolov8n.pt")
    assert records["YOLOv8n checkpoint (v8.3.0)"].releases[0].metadata["task"] == "object-detection"
    assert (
        records["YOLOv8n-seg checkpoint (v8.3.0)"].releases[0].metadata["task"]
        == "instance-segmentation"
    )
    assert records["YOLO11n checkpoint (v8.3.0)"].models[0].name == "YOLO11n"
    assert records["YOLOv8n checkpoint (v8.3.0)"].raw["asset_metadata_only"] is True
    assert len(client.calls) == 5


def test_uses_paginated_assets_endpoint_past_embedded_release_asset_truncation() -> None:
    assets = [_asset(1000 + index, f"readme-{index}.txt") for index in range(100)]
    assets.append(_asset(1100, "yolo11n.pt"))
    client = _Client([_release("v8.3.0", assets)])
    adapter = UltralyticsReleaseCheckpointSourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert page.records[0].title == "YOLO11n checkpoint (v8.3.0)"
    asset_calls = [url for url in client.calls if "/assets?" in url]
    assert len(asset_calls) == 2
    assert any("page=2" in url for url in asset_calls)
    assert page.records[0].canonical_url.endswith("/yolo11n.pt")


def test_rejects_download_urls_outside_official_assets_repository() -> None:
    client = _Client(
        [
            _release(
                "v8.3.0",
                [
                    _asset(101, "yolov8n.pt", "https://evil.example/yolov8n.pt"),
                ],
            ),
        ]
    )
    with pytest.raises(ValueError, match="invalid asset"):
        UltralyticsReleaseCheckpointSourceAdapter(client=client).fetch_page({})


def test_release_pagination_requires_complete_inventory() -> None:
    releases = [_release(f"v8.{index}.0", [], release_id=index + 1) for index in range(101)]
    client = _Client(releases)
    adapter = UltralyticsReleaseCheckpointSourceAdapter(max_releases=1000, client=client)

    inventory = adapter._release_inventory()

    assert len(inventory) == 101
    assert any("page=2" in url for url in client.calls)


def test_proposal_is_disabled_and_matches_constructor_contract() -> None:
    config = tomllib.loads(_PROPOSAL.read_text())
    assert len(config["source"]) == 1
    source = config["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "ultralytics_release_checkpoints"
    assert set(source) == {
        "name",
        "adapter",
        "enabled",
        "entry_tags",
        "repository",
        "assets_repository",
        "branch",
        "model_docs",
        "max_response_bytes",
        "max_releases",
        "max_assets_per_release",
    }
    adapter = UltralyticsReleaseCheckpointSourceAdapter(
        **{
            key: value
            for key, value in source.items()
            if key not in {"adapter", "enabled", "entry_tags"}
        }
    )
    assert adapter.repository == source["repository"]
    assert adapter.assets_repository == source["assets_repository"]
