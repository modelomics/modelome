from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.openmmlab import OpenMMLabModelIndexSourceAdapter

NOW = datetime(2026, 9, 21, 20, 0, tzinfo=UTC)
REPOSITORY = "open-mmlab/example"
REVISION = "a" * 40


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Any = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        self.calls.append((url, dict(headers or {})))
        return self.responses.pop(0)


def response(
    body: str | dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> HttpResponse:
    payload = json.dumps(body) if isinstance(body, dict) else body
    return HttpResponse(status=200, headers=headers or {}, body=payload.encode(), url="https://example.test")


def test_openmmlab_model_index_binds_models_to_configs_weights_papers_and_code() -> None:
    index = """Import:
- configs/atss/metafile.yml
- configs/pose/metafile.yml
"""
    detection = """Collections:
  - Name: ATSS
    Paper:
      URL: https://arxiv.org/abs/1912.02424
    README: configs/atss/README.md
    Code:
      URL: https://github.com/open-mmlab/example/blob/v1/model.py#L6
Models:
  - Name: atss_r50_fpn_1x_coco
    In Collection: ATSS
    Config: configs/atss/atss_r50_fpn_1x_coco.py
    Weights: https://download.example.test/atss.pth
"""
    pose = """Models:
- Config: configs/pose/rtmpose.py
  In Collection: RTMPose
  Alias: animal
  Metadata:
    Architecture:
    - RTMPose
  Name: rtmpose-m_ap10k
  Results:
  - Dataset: AP-10K
    Task: Animal 2D Keypoint
  Weights: https://download.example.test/rtmpose.pth
"""
    client = QueuedClient(
        response({"sha": REVISION}, headers={"ETag": '"commit-v1"'}),
        response(index),
        response(detection),
        response(pose),
    )
    adapter = OpenMMLabModelIndexSourceAdapter(
        name="openmmlab-example", repository=REPOSITORY, client=client, clock=lambda: NOW
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["completed_revision"] == REVISION
    assert page.next_state["index_import_count"] == 2
    assert page.next_state["model_count"] == 2

    atss, pose_record = page.records
    assert atss.kind is ArtifactKind.MODEL_CARD
    assert atss.models[0].name == "atss_r50_fpn_1x_coco"
    assert atss.models[0].status is ModelStatus.RELEASED
    assert atss.models[0].identifiers == (
        Identifier("openmmlab:model", "open-mmlab/example:configs/atss/atss_r50_fpn_1x_coco.py"),
    )
    assert atss.releases[0].identifiers == (
        Identifier("openmmlab:weights", "https://download.example.test/atss.pth"),
    )
    atss_links = {(item.url, item.relation) for item in atss.links}
    assert ("https://arxiv.org/abs/1912.02424", "paper_reference") in atss_links
    assert (
        "https://github.com/open-mmlab/example/blob/v1/model.py",
        "code_reference",
    ) in atss_links
    assert (
        f"https://github.com/{REPOSITORY}/blob/{REVISION}/configs/atss/atss_r50_fpn_1x_coco.py",
        "model_config",
    ) in atss_links
    assert ("https://download.example.test/atss.pth", "weights") in atss_links

    assert pose_record.models[0].name == "rtmpose-m_ap10k"
    assert pose_record.models[0].aliases == ("animal",)
    assert pose_record.models[0].status is ModelStatus.RELEASED
    assert pose_record.releases[0].metadata["config_path"] == "configs/pose/rtmpose.py"
    assert pose_record.releases[0].metadata["weights_url"] == (
        "https://download.example.test/rtmpose.pth"
    )


def test_openmmlab_model_index_skips_fetching_a_completed_unchanged_revision() -> None:
    client = QueuedClient(response({"sha": REVISION}, headers={"ETag": '"commit-v1"'}))
    adapter = OpenMMLabModelIndexSourceAdapter(
        name="openmmlab-example", repository=REPOSITORY, client=client, clock=lambda: NOW
    )

    page = adapter.fetch_page({"completed_revision": REVISION, "model_count": 7})

    assert page.records == ()
    assert page.complete is True
    assert page.upstream_count == 7
    assert page.next_state["checked_at"] == "2026-09-21T20:00:00Z"
    assert page.next_state["commit_etag"] == '"commit-v1"'
    assert client.calls == [
        (
            f"https://api.github.com/repos/{REPOSITORY}/commits/main",
            {"Accept": "application/vnd.github+json"},
        )
    ]
