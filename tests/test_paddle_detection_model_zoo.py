from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.paddle_detection_model_zoo import (
    PaddleDetectionModelZooSourceAdapter,
    _table_rows,
)

_REVISION = "3" * 40
_DOCUMENT = (
    b"# Faster R-CNN\n\n"
    b"## Model Zoo\n\n"
    b"| Backbone | Download | Config |\n"
    b"| --- | --- | --- |\n"
    b"| ResNet50 | [download](https://paddledet.example.test/models/faster_"
    b"r50_coco.pdparams) | [config](./faster_r50_coco.yml) |\n"
    b"| ResNet101 | [download](https://paddledet.example.test/models/faster_"
    b"r101_coco.pdparams) | [config](../faster_r101_coco.yml) |\n\n"
    b"## Metrics\n\n"
    b"| Backbone | AP |\n"
    b"| --- | --- |\n"
    b"| ResNet50 | 42.0 |\n"
)
_DUPLICATE_DOCUMENT = b"""# Inference models

| Model | Download |
| --- | --- |
| Faster R50 | [download](https://paddledet.example.test/models/faster_r50_coco.pdparams) |
"""
_PADDLE3D_DOCUMENT = (
    b"# PointPillars\n\n"
    b"## Model Zoo\n\n"
    b"| Model | Car AP | Model Download | Config |\n"
    b"| --- | --- | --- | --- |\n"
    b"| PointPillars | 86.90 | "
    b"[model](https://paddle3d.example.test/models/pointpillars/model.pdparams) | "
    b"[config](../../../configs/pointpillars/pointpillars_kitti.yml) |\n"
)
_MMHUMAN3D_DOCUMENT = (
    b"# HMR\n\n"
    b"| Config | 3DPW | Download |\n"
    b"| :---: | :---: | :---: |\n"
    b"| [resnet50_hmr_pw3d.py](resnet50_hmr_pw3d.py) | 112.34 | "
    b"[model](https://openmmlab.example.test/mmhuman3d/hmr/resnet50_hmr_pw3d.pth) |\n"
)
_MMACTION_DOCUMENT = (
    b"# Model Zoo\n\n"
    b"### TSN\n\n"
    b"| Modality | Backbone | Download |\n"
    b"| --- | --- | --- |\n"
    b"| RGB | ResNet50 | "
    b"[model](https://download.openmmlab.com/mmaction/tsn_r50.pth) |\n"
)


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append((url, headers))
        return self.responses.pop(0)


def _response(body: bytes, *, url: str = "https://fixtures.test/paddledetection") -> HttpResponse:
    return HttpResponse(200, {"etag": '"fixture"'}, body, url)


def _commit() -> HttpResponse:
    return _response(("{\"sha\": \"" + _REVISION + "\"}").encode())


def _archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        for path, body in files.items():
            package.writestr(f"PaddleDetection-{_REVISION}/{path}", body)
    return buffer.getvalue()


def test_paddledetection_archive_enumerates_direct_checkpoint_rows() -> None:
    archive = _archive(
        {
            "configs/faster_rcnn/README.md": _DOCUMENT,
            "configs/inference/README.md": _DUPLICATE_DOCUMENT,
            "docs/README.md": _DOCUMENT,
        }
    )
    client = _QueuedClient(_commit(), _response(archive))
    adapter = PaddleDetectionModelZooSourceAdapter(
        repository="example/PaddleDetection",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot
    assert page.upstream_count == 2
    assert page.next_state["document_count"] == 2
    assert page.next_state["checkpoint_row_count"] == 3
    assert client.calls[1][0].endswith(f"/archive/{_REVISION}.zip")
    r50 = next(record for record in page.records if record.title == "faster_r50_coco")
    assert r50.identifiers == (
        Identifier(
            "paddledetection:checkpoint-model",
            "https://paddledet.example.test/models/faster_r50_coco.pdparams",
        ),
    )
    assert r50.models[0].status.value == "released"
    assert {"ResNet50", "Faster R50"} <= set(r50.models[0].aliases)
    assert len(r50.releases[0].metadata["source_rows"]) == 2
    assert (
        "https://github.com/example/PaddleDetection/blob/"
        f"{_REVISION}/configs/faster_rcnn/faster_r50_coco.yml",
        "model_config",
        False,
    ) in {(link.url, link.relation, link.crawl) for link in r50.links}
    assert all(link.model_local_ids == (r50.models[0].local_id,) for link in r50.links)


def test_paddledetection_archive_skips_when_commit_is_unchanged() -> None:
    client = _QueuedClient(_commit())
    adapter = PaddleDetectionModelZooSourceAdapter(client=client)

    page = adapter.fetch_page({"completed_revision": _REVISION, "model_count": 77})

    assert page.records == ()
    assert page.complete
    assert page.upstream_count == 77
    assert len(client.calls) == 1


def test_compact_markdown_separator_and_escaped_cell_pipes_keep_checkpoint_link() -> None:
    document = (
        "| Model | Links |\n"
        "|:-:|:-:|\n"
        "| U-Net | [model](https://paddleseg.example.test/u-net/"
        "model.pdparams) \\| [log](https://example.test/log) |\n"
    )
    rows = _table_rows(
        document,
        path="configs/unet/README.md",
        source="fixture",
        revision=_REVISION,
        blob_url=lambda revision, path: f"https://github.com/example/repo/blob/{revision}/{path}",
    )

    assert len(rows) == 1
    assert rows[0].weight_url == "https://paddleseg.example.test/u-net/model.pdparams"


def test_project_specific_markdown_suffix_and_namespace_are_preserved() -> None:
    archive = _archive({"docs/en/model_zoo/recognition/slowfast.md": _DOCUMENT})
    client = _QueuedClient(_commit(), _response(archive))
    adapter = PaddleDetectionModelZooSourceAdapter(
        name="paddlevideo-model-zoo",
        repository="example/PaddleVideo",
        project_name="PaddleVideo",
        provider_namespace="paddlevideo",
        document_prefix="docs/en/model_zoo/",
        document_suffix=".md",
        client=client,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    assert page.records[0].identifiers[0].namespace == "paddlevideo:checkpoint-model"
    assert page.records[0].text.startswith("PaddleVideo checkpoint:")


def test_paddle3d_architecture_document_checkpoint_rows_remain_scoped() -> None:
    archive = _archive({"docs/models/pointpillars/README.md": _PADDLE3D_DOCUMENT})
    client = _QueuedClient(_commit(), _response(archive))
    adapter = PaddleDetectionModelZooSourceAdapter(
        name="paddle3d-model-zoo",
        repository="example/Paddle3D",
        branch="develop",
        project_name="Paddle3D",
        provider_namespace="paddle3d",
        document_prefix="docs/models/",
        client=client,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.title == "pointpillars"
    assert record.identifiers == (
        Identifier(
            "paddle3d:checkpoint-model",
            "https://paddle3d.example.test/models/pointpillars/model.pdparams",
        ),
    )
    assert record.models[0].aliases == ("PointPillars",)
    assert {
        (link.url, link.relation, link.model_local_ids)
        for link in record.links
    } >= {
        (
            "https://paddle3d.example.test/models/pointpillars/model.pdparams",
            "weights",
            (record.models[0].local_id,),
        ),
        (
            "https://github.com/example/Paddle3D/blob/"
            f"{_REVISION}/docs/models/pointpillars/README.md",
            "model_card",
            (record.models[0].local_id,),
        ),
        (
            "https://github.com/example/Paddle3D/blob/"
            f"{_REVISION}/configs/pointpillars/pointpillars_kitti.yml",
            "model_config",
            (record.models[0].local_id,),
        ),
    }


def test_generic_archive_reader_keeps_python_model_configs_with_pytorch_weights() -> None:
    archive = _archive({"configs/hmr/README.md": _MMHUMAN3D_DOCUMENT})
    client = _QueuedClient(_commit(), _response(archive))
    adapter = PaddleDetectionModelZooSourceAdapter(
        name="mmhuman3d-model-zoo",
        repository="example/mmhuman3d",
        project_name="MMHuman3D",
        provider_namespace="mmhuman3d",
        document_prefix="configs/",
        client=client,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.title == "resnet50_hmr_pw3d"
    assert record.identifiers[0] == Identifier(
        "mmhuman3d:checkpoint-model",
        "https://openmmlab.example.test/mmhuman3d/hmr/resnet50_hmr_pw3d.pth",
    )
    assert (
        "https://github.com/example/mmhuman3d/blob/"
        f"{_REVISION}/configs/hmr/resnet50_hmr_pw3d.py",
        "model_config",
        False,
    ) in {(link.url, link.relation, link.crawl) for link in record.links}


def test_generic_archive_reader_can_select_one_root_model_zoo_document() -> None:
    archive = _archive(
        {
            "MODEL_ZOO.md": _MMACTION_DOCUMENT,
            "configs/ignored/README.md": _DOCUMENT,
        }
    )
    client = _QueuedClient(_commit(), _response(archive))
    adapter = PaddleDetectionModelZooSourceAdapter(
        name="mmaction-legacy-model-zoo",
        repository="example/mmaction",
        project_name="MMAction",
        provider_namespace="mmaction-legacy",
        document_paths=("MODEL_ZOO.md",),
        client=client,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert page.next_state["document_count"] == 1
    record = page.records[0]
    assert record.title == "tsn_r50"
    assert record.models[0].aliases == ("RGB",)
    assert (
        "https://github.com/example/mmaction/blob/"
        f"{_REVISION}/MODEL_ZOO.md",
        "model_card",
        False,
    ) in {(link.url, link.relation, link.crawl) for link in record.links}


@pytest.mark.parametrize(
    "archive, message",
    [
        (
            b"not-a-zip",
            "not a ZIP",
        ),
        (
            _archive(
                {
                    "configs/faster_rcnn/README.md": b"""| Model | Download |
| --- | --- |
| ResNet | [source](https://example.test/readme.html) |
"""
                }
            ),
            "no direct checkpoint rows",
        ),
    ],
)
def test_paddledetection_archive_fails_closed_on_invalid_or_noncheckpoint_input(
    archive: bytes, message: str
) -> None:
    client = _QueuedClient(_commit(), _response(archive))

    with pytest.raises(ValueError, match=message):
        PaddleDetectionModelZooSourceAdapter(client=client).fetch_page({})
