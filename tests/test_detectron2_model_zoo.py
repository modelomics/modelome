from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.detectron2_model_zoo import Detectron2ModelZooSourceAdapter

_REVISION = "f" * 40


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


def _response(payload: str | Mapping[str, Any]) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/detectron2")


_SOURCE = """\
class _ModelZooUrls:
    S3_PREFIX = "https://dl.example.test/detectron2/"
    CONFIG_PATH_TO_URL_SUFFIX = {
        "COCO-Detection/faster_rcnn_R_50_FPN_1x": "137257794/model_final_b275ba.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x": "137260431/model_final_a54504.pkl",
    }
"""


def test_detectron2_model_zoo_enumerates_literal_config_checkpoint_pairs() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_SOURCE))
    adapter = Detectron2ModelZooSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["model_count"] == 2
    assert client.calls[1][0].endswith(f"/{_REVISION}/detectron2/model_zoo/model_zoo.py")
    detection, segmentation = page.records
    assert detection.kind is ArtifactKind.MODEL_CARD
    assert detection.source_record_id == "model:COCO-Detection/faster_rcnn_R_50_FPN_1x"
    assert detection.models[0].identifiers == (
        Identifier("detectron2:model", "COCO-Detection/faster_rcnn_R_50_FPN_1x"),
    )
    assert detection.releases[0].identifiers == (
        Identifier("detectron2:model-zoo-config", "COCO-Detection/faster_rcnn_R_50_FPN_1x"),
    )
    assert (
        "https://dl.example.test/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/"
        "137257794/model_final_b275ba.pkl",
        "weights",
        False,
    ) in {(link.url, link.relation, link.crawl) for link in detection.links}
    assert (
        f"https://github.com/facebookresearch/detectron2/blob/{_REVISION}/configs/"
        "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
        "model_config",
        False,
    ) in {(link.url, link.relation, link.crawl) for link in detection.links}
    assert segmentation.title == "mask_rcnn_R_50_FPN_1x"


def test_detectron2_model_zoo_skips_source_file_when_commit_is_unchanged() -> None:
    first_client = _QueuedClient(_response({"sha": _REVISION}), _response(_SOURCE))
    adapter = Detectron2ModelZooSourceAdapter(client=first_client)
    first = adapter.fetch_page({})
    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client

    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.records == ()
    assert len(second_client.calls) == 1


def test_detectron2_model_zoo_rejects_nonliteral_static_mapping() -> None:
    source = """\
class _ModelZooUrls:
    S3_PREFIX = "https://dl.example.test/detectron2/"
    CONFIG_PATH_TO_URL_SUFFIX = make_map()
"""
    client = _QueuedClient(_response({"sha": _REVISION}), _response(source))

    with pytest.raises(ValueError, match="absent or not a literal dictionary"):
        Detectron2ModelZooSourceAdapter(client=client).fetch_page({})
