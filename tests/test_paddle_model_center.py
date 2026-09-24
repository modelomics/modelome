from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.paddle_model_center import PaddleModelCenterSourceAdapter

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


def _response(payload: str | Mapping[str, Any]) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/paddle")


def _tree(*, truncated: bool = False) -> dict[str, Any]:
    return {
        "truncated": truncated,
        "tree": [
            {"path": "modelcenter/ERNIE-3.0/info.yaml", "type": "blob"},
            {"path": "modelcenter/PP-HGNet/info.yaml", "type": "blob"},
            {"path": "modelcenter/PP-HGNet/download_en.md", "type": "blob"},
            {"path": "modelcenter/PP-HGNet/download_cn.md", "type": "blob"},
            {"path": "README.md", "type": "blob"},
        ],
    }


_ERNIE = """\
Model_Info:
  name: "ERNIE 3.0"
  description_en: "A lightweight language model"
  from_repo: "PaddleNLP"
Task:
  - tag_en: "Natural Language Processing"
    sub_tag_en: "Pretrained Model"
Datasets: ""
Publisher: "Baidu"
License: "apache.2.0"
Paper:
  - title: "ERNIE Paper"
    url: "https://arxiv.org/abs/2106.02241"
"""

_PPHGNET = """\
Model_Info:
  name: "PP-HGNet"
  description_en: "A high performance convolutional network"
  from_repo: "PaddleClas"
Task:
  - tag_en: "Computer Vision"
    sub_tag_en: "Image Classification"
Datasets: "ImageNet1k"
Publisher: "Baidu"
License: "apache.2.0"
Paper: ""
"""

_DOWNLOAD_EN = (
    "| Model | Introduction | Download link |\n"
    "| --- | --- | --- |\n"
    "| PPHGNet_tiny | Image classification | "
    "[Inference](https://weights.example.test/infer.tar)/"
    "[Pretrained](https://weights.example.test/pretrained.pdparams) |\n"
)

_DOWNLOAD_CN = """\
| 模型名称 | 下载链接 |
| --- | --- |
| PPHGNet_tiny | [推理模型](https://weights.example.test/infer-cn.tar) |
"""


def test_model_center_enumerates_every_manifest_and_source_declared_download_row() -> None:
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(_tree()),
        _response(_ERNIE),
        _response(_PPHGNET),
        _response(_DOWNLOAD_CN),
        _response(_DOWNLOAD_EN),
    )
    adapter = PaddleModelCenterSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["completed_revision"] == _REVISION
    assert page.next_state["family_count"] == 2
    assert page.next_state["download_file_count"] == 2
    assert page.next_state["model_count"] == 2
    assert client.calls[1][0].endswith(f"/{_REVISION}?recursive=1")

    ernie, pphgnet = page.records
    assert ernie.kind is ArtifactKind.MODEL_CARD
    assert ernie.source_record_id == "model:ERNIE-3.0/ERNIE 3.0"
    assert ernie.models[0].status is ModelStatus.DOCUMENTED
    assert ernie.models[0].identifiers == (Identifier("paddle:model", "ERNIE-3.0/ERNIE 3.0"),)
    assert ernie.releases == ()
    assert {(link.url, link.relation) for link in ernie.links} >= {
        ("https://github.com/PaddlePaddle/PaddleNLP", "source_implementation"),
        ("https://arxiv.org/abs/2106.02241", "paper"),
    }

    assert pphgnet.source_record_id == "model:PP-HGNet/PPHGNet_tiny"
    assert pphgnet.models[0].status is ModelStatus.RELEASED
    assert pphgnet.models[0].aliases == ("PP-HGNet",)
    assert pphgnet.releases[0].identifiers == (
        Identifier("paddle:model-release", "PP-HGNet/PPHGNet_tiny"),
    )
    assert set(pphgnet.releases[0].metadata["artifacts"]) == {
        "https://weights.example.test/infer.tar",
        "https://weights.example.test/infer-cn.tar",
        "https://weights.example.test/pretrained.pdparams",
    }
    assert {(link.url, link.relation, link.crawl) for link in pphgnet.links} >= {
        ("https://weights.example.test/infer.tar", "inference_artifact", False),
        ("https://weights.example.test/infer-cn.tar", "inference_artifact", False),
        ("https://weights.example.test/pretrained.pdparams", "weights", False),
    }
    assert len(
        [link for link in pphgnet.links if link.relation == "documentation"]
    ) == 2


def test_model_center_skips_tree_and_files_when_the_commit_is_unchanged() -> None:
    first_client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(_tree()),
        _response(_ERNIE),
        _response(_PPHGNET),
        _response(_DOWNLOAD_CN),
        _response(_DOWNLOAD_EN),
    )
    adapter = PaddleModelCenterSourceAdapter(client=first_client)
    first = adapter.fetch_page({})

    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client
    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.records == ()
    assert second.next_state["checked_at"]
    assert len(second_client.calls) == 1


def test_model_center_rejects_a_truncated_recursive_tree() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_tree(truncated=True)))

    with pytest.raises(ValueError, match="recursive repository tree is truncated"):
        PaddleModelCenterSourceAdapter(client=client).fetch_page({})
