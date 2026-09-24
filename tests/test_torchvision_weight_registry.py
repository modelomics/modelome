from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.torchvision_weight_registry import (
    TorchvisionWeightRegistrySourceAdapter,
)

_REVISION = "c" * 40


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: bytes | Mapping[str, Any]) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/torchvision")


def _archive(files: Mapping[str, str]) -> bytes:
    output = io.BytesIO()
    root = f"vision-{_REVISION}"
    with zipfile.ZipFile(output, "w") as package:
        for path, text in files.items():
            package.writestr(f"{root}/{path}", text)
    return output.getvalue()


_RESNET = """\
from ._api import Weights, WeightsEnum

class ResNet50_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url="https://download.example.test/resnet50-v1.pth")
    IMAGENET1K_V2 = Weights(url="https://download.example.test/resnet50-v2.pth")
    DEFAULT = IMAGENET1K_V2
"""

_DETECTOR = """\
from ._api import Weights, WeightsEnum

class FasterRCNN_ResNet50_FPN_Weights(WeightsEnum):
    COCO_V1 = Weights(url="https://download.example.test/fasterrcnn-coco.pth")
"""


def _catalog_archive() -> bytes:
    return _archive(
        {
            "torchvision/models/resnet.py": _RESNET,
            "torchvision/models/detection/faster_rcnn.py": _DETECTOR,
            "torchvision/models/_api.py": "class WeightsEnum: pass\n",
        }
    )


def test_weight_registry_enumerates_literal_weight_enums_and_releases() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_catalog_archive()))
    adapter = TorchvisionWeightRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["completed_revision"] == _REVISION
    assert page.next_state["module_count"] == 3
    assert page.next_state["model_count"] == 2
    assert page.next_state["release_count"] == 3
    assert len(client.calls) == 2
    assert client.calls[1][0].endswith(f"/archive/{_REVISION}.zip")

    detector, resnet = page.records
    assert detector.kind is ArtifactKind.MODEL_CARD
    assert detector.source_record_id == "weight-enum:FasterRCNN_ResNet50_FPN_Weights"
    assert detector.identifiers == (
        Identifier("torchvision:weight-enum", "FasterRCNN_ResNet50_FPN_Weights"),
    )
    assert detector.models[0].name == "FasterRCNN_ResNet50_FPN"
    assert detector.models[0].status is ModelStatus.RELEASED
    assert detector.releases[0].identifiers == (
        Identifier(
            "torchvision:weight-enum-member",
            "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
        ),
    )
    assert resnet.source_record_id == "weight-enum:ResNet50_Weights"
    assert [release.version for release in resnet.releases] == [
        "IMAGENET1K_V1",
        "IMAGENET1K_V2",
    ]
    assert resnet.releases[0].metadata["aliases"] == ()
    assert resnet.releases[1].metadata["aliases"] == ("DEFAULT",)
    assert {
        (link.url, link.relation, link.crawl, link.model_local_ids)
        for link in resnet.links
    } == {
        ("https://github.com/pytorch/vision", "source_repository", False, ()),
        (
            f"https://github.com/pytorch/vision/blob/{_REVISION}/torchvision/models/resnet.py",
            "model_definition",
            False,
            ("model:ResNet50_Weights",),
        ),
        (
            "https://download.example.test/resnet50-v1.pth",
            "weights",
            False,
            ("model:ResNet50_Weights",),
        ),
        (
            "https://download.example.test/resnet50-v2.pth",
            "weights",
            False,
            ("model:ResNet50_Weights",),
        ),
    }


def test_weight_registry_skips_archive_when_commit_is_unchanged() -> None:
    first_client = _QueuedClient(_response({"sha": _REVISION}), _response(_catalog_archive()))
    adapter = TorchvisionWeightRegistrySourceAdapter(client=first_client)
    first = adapter.fetch_page({})

    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client
    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.records == ()
    assert second.next_state["checked_at"]
    assert len(second_client.calls) == 1


def test_weight_registry_fails_closed_for_a_nonliteral_weight_url() -> None:
    nonliteral = """\
class Broken_Weights(WeightsEnum):
    V1 = Weights(url=CHECKPOINT_URL)
"""
    archive = _archive({"torchvision/models/broken.py": nonliteral})
    client = _QueuedClient(_response({"sha": _REVISION}), _response(archive))

    with pytest.raises(ValueError, match="no literal web weight URL"):
        TorchvisionWeightRegistrySourceAdapter(client=client).fetch_page({})
