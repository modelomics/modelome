from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.timm_model_registry import TimmModelRegistrySourceAdapter

_REVISION = "c" * 40


class _Client:
    def __init__(self, archive: bytes) -> None:
        self.responses = [
            HttpResponse(200, {}, json.dumps({"sha": _REVISION}).encode(), "https://test"),
            HttpResponse(200, {}, archive, "https://test"),
        ]

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        return self.responses.pop(0)


def _archive() -> bytes:
    files = {
        "timm/models/__init__.py": "from .vision import *\n",
        "timm/models/vision.py": """\
from ._registry import generate_default_cfgs, register_model

WEIGHTS_URL = 'https://weights.example.test/vision.pth'
MODEL_HUB_ID = 'timm/vision_base.v1'
def _gcfg(**kwargs):
    return {'origin_url': 'https://github.com/example/original-model', **kwargs}

default_cfgs = generate_default_cfgs({
    'vision_base.v1': _gcfg(
        url=WEIGHTS_URL, hf_hub_id=MODEL_HUB_ID,
        hf_hub_filename='checkpoints/model.safetensors',
    ),
})

@register_model
def vision_base(pretrained=False):
    return None
""",
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as package:
        for path, source in files.items():
            package.writestr(f"pytorch-image-models-{_REVISION}/{path}", source)
    return output.getvalue()


def test_timm_registry_resolves_module_literal_checkpoint_references() -> None:
    adapter = TimmModelRegistrySourceAdapter(
        client=_Client(_archive()),
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    record = page.records[0]
    assert len(record.releases) == 1
    assert record.releases[0].metadata["config"]["url"] == (
        "https://weights.example.test/vision.pth"
    )
    assert record.releases[0].metadata["config"]["hf_hub_id"] == "timm/vision_base.v1"
    assert any(link.relation == "weights" for link in record.links)
    assert any(link.url == "https://huggingface.co/timm/vision_base.v1" for link in record.links)
    assert any(
        link.url == (
            "https://huggingface.co/timm/vision_base.v1/resolve/main/"
            "checkpoints/model.safetensors"
        )
        and link.relation == "weights"
        for link in record.links
    )
    assert any(
        link.url == "https://github.com/example/original-model"
        and link.relation == "official_implementation"
        for link in record.links
    )
