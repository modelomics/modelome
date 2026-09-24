from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.markdown_model_table import MarkdownModelTableSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/yolov5_pretrained_checkpoints.toml"
_REVISION = "e" * 40
_README = "\n".join(
    (
        "# YOLOv5",
        "### Pretrained Checkpoints",
        "| Model | Size | Weights |",
        "| --- | --- | --- |",
        "| [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/"
        "v7.0/yolov5n6.pt) | 1280 | COCO |",
        "| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/"
        "v7.0/yolov5s6.pt) | 1280 | COCO |",
        "| [YOLOv5n](https://platform.ultralytics.com/ultralytics/yolov5/yolov5nu)"
        " | 640 | successor page |",
        "### Segmentation Checkpoints",
        "| Model | Size | mAP | Weights |",
        "| --- | --- | --- | --- |",
        "| [YOLOv5n-seg](https://github.com/ultralytics/yolov5/releases/download/"
        "v7.0/yolov5n-seg.pt) | 640 | 23.4 | COCO |",
        "| [YOLOv5s-seg](https://github.com/ultralytics/yolov5/releases/download/"
        "v7.0/yolov5s-seg.pt) | 640 | 31.7 | COCO |",
        "### Classification Checkpoints",
        "| Model | Size | Top-1 | Weights |",
        "| --- | --- | --- | --- |",
        "| [YOLOv5s-cls](https://github.com/ultralytics/yolov5/releases/download/"
        "v7.0/yolov5s-cls.pt) | 224 | 71.5 | ImageNet |",
        "| [ResNet18](https://github.com/ultralytics/yolov5/releases/download/"
        "v7.0/resnet18.pt) | 224 | 70.3 | ImageNet |",
    )
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
        body = (
            f'{{"sha":"{_REVISION}"}}'.encode()
            if "/commits/" in url
            else _README.encode()
        )
        return HttpResponse(200, {}, body, url)


def _source_config() -> dict[str, Any]:
    config = tomllib.loads(_PROPOSAL.read_text())
    assert len(config["source"]) == 1
    source = config["source"][0]
    assert source["enabled"] is False
    return source


def test_yolov5_proposal_indexes_only_exact_first_party_checkpoint_rows() -> None:
    source = _source_config()
    client = _Client()
    adapter = MarkdownModelTableSourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        document_path=source["document_path"],
        provider_namespace=source["provider_namespace"],
        model_column=source["model_column"],
        model_header_pattern=source["model_header_pattern"],
        model_name_pattern=source["model_name_pattern"],
        identity_include_heading=source["identity_include_heading"],
        max_response_bytes=source["max_response_bytes"],
        max_rows=source["max_rows"],
        client=client,
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 5
    assert len(client.calls) == 2
    records = {record.models[0].name: record for record in page.records}
    assert set(records) == {
        "YOLOv5n6",
        "YOLOv5s6",
        "YOLOv5n-seg",
        "YOLOv5s-seg",
        "YOLOv5s-cls",
    }
    assert records["YOLOv5n6"].releases[0].metadata["artifacts"] == [
        {
            "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt",
            "relation": "weights",
        }
    ]
    assert records["YOLOv5n-seg"].releases[0].metadata["artifacts"][0]["url"].endswith(
        "yolov5n-seg.pt"
    )
    assert all(record.models[0].identifiers[0].namespace == "ultralytics:yolov5-checkpoint"
               for record in page.records)
