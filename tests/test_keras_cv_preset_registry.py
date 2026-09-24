from __future__ import annotations

import io
import json
import tomllib
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.keras_hub_preset_registry import KerasHubPresetRegistrySourceAdapter

_REVISION = "e" * 40


class _Client:
    def __init__(self, archive: bytes) -> None:
        self.archive = archive

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        body = json.dumps({"sha": _REVISION}).encode() if "/commits/" in url else self.archive
        return HttpResponse(200, {}, body, url)


def test_keras_cv_proposal_scans_its_distinct_archive_for_exact_presets() -> None:
    proposal = tomllib.loads(
        (Path(__file__).parents[1] / "config/proposals/keras_cv_preset_registry.toml").read_text()
    )["source"][0]
    assert proposal["enabled"] is False
    assert proposal["repository"] == "keras-team/keras-cv"
    assert proposal["preset_root"] == "keras_cv/src/models"

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(
            f"keras-cv-{_REVISION}/keras_cv/src/models/backbones/backbone_presets.py",
            '''from keras_cv.src.models.backbones.resnet_v1.resnet_v1_backbone_presets import (
    backbone_presets as resnet_presets,
)
from keras_cv.src.models.backbones.vit_det.vit_det_backbone_presets import (
    backbone_presets as vit_det_presets,
)

backbone_presets = {**resnet_presets, **vit_det_presets}
''',
        )
        package.writestr(
            f"keras-cv-{_REVISION}/keras_cv/src/models/segmentation/basnet/basnet_presets.py",
            '''presets_no_weights = {
    "basnet": {"backbone": build_resnet_backbone()},
}
presets_with_weights = {}  # TODO: add pretrained weights
basnet_presets = {**presets_no_weights, **presets_with_weights}
''',
        )
        package.writestr(
            f"keras-cv-{_REVISION}/keras_cv/src/models/backbones/efficientnet_lite/efficientnet_lite_backbone_presets.py",
            '''backbone_presets_no_weights = {
    "efficientnetlite_b0": {
        "metadata": {"description": "EfficientNet Lite B0"},
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetlite_b0",
    },
}
''',
        )
        package.writestr(
            f"keras-cv-{_REVISION}/keras_cv/src/models/backbones/vit_det/vit_det_backbone_presets.py",
            '''backbone_presets = {
    "vit_det_base_imagenet": {
        "metadata": {"description": "Vision Transformer Detector"},
        "kaggle_handle": "kaggle://keras/vit_det/keras/vit_det_base_imagenet/1",
    },
}''',
        )

    adapter = KerasHubPresetRegistrySourceAdapter(
        name=proposal["name"],
        repository=proposal["repository"],
        branch=proposal["branch"],
        preset_root=proposal["preset_root"],
        client=_Client(archive.getvalue()),
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )
    page = adapter.fetch_page({})

    assert page.complete and page.upstream_count == 1
    record = page.records[0]
    assert record.identifiers[0].namespace == "keras-hub:preset"
    assert record.identifiers[0].value == (
        "keras_cv/src/models/backbones/vit_det/vit_det_backbone_presets.py:"
        "backbone_presets:vit_det_base_imagenet"
    )
    assert record.raw["handles"] == [
        {
            "field": "kaggle_handle",
            "handle": "kaggle://keras/vit_det/keras/vit_det_base_imagenet/1",
        }
    ]
    assert all(record.title != "basnet" for record in page.records)
    assert {record.title for record in page.records} == {"vit_det_base_imagenet"}


def test_keras_hub_still_rejects_dynamic_preset_collections() -> None:
    from modelome.sources.keras_hub_preset_registry import _parse_presets

    source = "from other import presets\nmodel_presets = {**presets}\n"
    try:
        _parse_presets(source, "model_presets.py", "keras-hub-preset-registry")
    except ValueError as error:
        assert "is not a literal dictionary" in str(error)
    else:
        raise AssertionError("KerasHub dynamic preset collections must remain rejected")


def test_keras_cv_retains_literal_handle_when_sibling_fields_are_computed() -> None:
    from modelome.sources.keras_hub_preset_registry import _parse_presets

    source = '''model_presets = {
    "model_with_weights": {
        "backbone": build_backbone(),
        "kaggle_handle": "kaggle://keras/model/keras/model_variant/4",
    },
}
'''
    presets = _parse_presets(
        source,
        "model_presets.py",
        "keras-cv-preset-registry",
        allow_dynamic_preset_collections=True,
    )
    assert len(presets) == 1
    assert presets[0].handles == (
        ("kaggle_handle", "kaggle://keras/model/keras/model_variant/4"),
    )


def test_keras_cv_skips_internal_gcs_handle_without_losing_public_handle() -> None:
    from modelome.sources.keras_hub_preset_registry import _parse_presets

    source = '''model_presets = {
    "gcs_only": {"kaggle_handle": "gs://keras-cv-kaggle/internal"},
    "public": {"kaggle_handle": "kaggle://keras/model/keras/public/2"},
}
'''
    presets = _parse_presets(
        source,
        "model_presets.py",
        "keras-cv-preset-registry",
        allow_dynamic_preset_collections=True,
    )
    assert [preset.name for preset in presets] == ["public"]
