from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.qualcomm_ai_hub_models import QualcommAIHubModelsSourceAdapter

_REVISION = "a" * 40
_README = (
    b"# Qualcomm AI Hub Models\n\n"
    b"| Model | README |\n| -- | -- |\n"
    b"| [MobileNet-v2](https://aihub.qualcomm.com/models/mobilenet_v2) | "
    b"[qai_hub_models.models.mobilenet_v2]"
    b"(src/qai_hub_models/models/mobilenet_v2/README.md) |\n"
    b"| [YOLOv8](https://aihub.qualcomm.com/models/yolov8_det) | "
    b"[qai_hub_models.models.yolov8_det]"
    b"(src/qai_hub_models/models/yolov8_det/README.md) |\n"
    b"| [non-model](https://example.org/) | other |\n"
)


class _Client:
    def __init__(self, readme: bytes = _README) -> None:
        self.readme = readme
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
            body = f'{{"sha":"{_REVISION}"}}'.encode()
        elif url.endswith("/README.md"):
            body = self.readme
        else:
            raise AssertionError(f"unexpected URL {url}")
        return HttpResponse(200, {"content-type": "text/plain"}, body, url)


def test_qualcomm_ai_hub_models_enumerates_pinned_exact_model_ids() -> None:
    client = _Client()
    page = QualcommAIHubModelsSourceAdapter(client=client).fetch_page({})

    assert page.complete
    assert not page.authoritative_snapshot
    assert page.upstream_count == 2
    assert page.next_state == {"completed_revision": _REVISION, "model_count": 2}
    records = {record.source_record_id: record for record in page.records}
    assert set(records) == {"mobilenet_v2", "yolov8_det"}
    model = records["mobilenet_v2"].models[0]
    assert model.name == "MobileNet-v2"
    assert model.identifiers[0].value == "mobilenet_v2"
    assert records["yolov8_det"].canonical_url == ("https://aihub.qualcomm.com/models/yolov8_det")
    assert len(client.calls) == 2


def test_qualcomm_ai_hub_models_rejects_changed_identity_mapping() -> None:
    readme = _README.replace(b"models/yolov8_det", b"models/yolov8").replace(
        b"models.yolov8_det", b"models.yolov8"
    )
    readme = readme.replace(b"models/yolov8/README.md", b"models/yolov8_det/README.md")
    with pytest.raises(ValueError, match="inconsistent exact model identity"):
        QualcommAIHubModelsSourceAdapter(client=_Client(readme)).fetch_page({})


def test_qualcomm_ai_hub_models_fails_closed_if_directory_format_changes() -> None:
    with pytest.raises(ValueError, match="format was not recognized"):
        QualcommAIHubModelsSourceAdapter(client=_Client(b"# changed layout\n")).fetch_page({})


def test_qualcomm_ai_hub_models_reuses_unchanged_revision_checkpoint() -> None:
    client = _Client()
    page = QualcommAIHubModelsSourceAdapter(client=client).fetch_page(
        {"completed_revision": _REVISION, "model_count": 17}
    )
    assert page.complete and not page.records
    assert page.upstream_count == 17
    assert len(client.calls) == 1


def test_qualcomm_ai_hub_models_rejects_bad_completed_count() -> None:
    with pytest.raises(ValueError, match="invalid completed model count"):
        QualcommAIHubModelsSourceAdapter(client=_Client()).fetch_page(
            {"completed_revision": _REVISION, "model_count": "17"}
        )
