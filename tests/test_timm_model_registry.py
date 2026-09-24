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
from modelome.sources.timm_model_registry import TimmModelRegistrySourceAdapter

_REVISION = "b" * 40


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
    return HttpResponse(200, {}, body, "https://fixtures.test/timm")


def _archive(files: Mapping[str, str]) -> bytes:
    output = io.BytesIO()
    root = f"pytorch-image-models-{_REVISION}"
    with zipfile.ZipFile(output, "w") as package:
        for path, text in files.items():
            package.writestr(f"{root}/{path}", text)
    return output.getvalue()


_INITIALIZER = """\
from .vision import *
from .untrained import *
from ._builder import build_model_with_cfg
"""

_VISION = """\
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

def _cfg(**kwargs):
    return kwargs

default_cfgs = generate_default_cfgs({
    'vision_base.foo*': _cfg(
        hf_hub_id='timm/',
        url=('https://weights.example.test/vision-a.pth', 'https://weights.example.test/vision-b.pth'),
        license='https://licenses.example.test/vision',
        input_size=(3, 224, 224),
    ),
    'vision_base.untrained': _cfg(),
})

@register_model
def vision_base(pretrained=False):
    return None

register_model_deprecations(__name__, {'old_vision_base': 'vision_base'})
"""

_UNTRAINED = """\
from ._registry import register_model

@register_model
def bare_architecture(pretrained=False):
    return None
"""


def _catalog_archive() -> bytes:
    return _archive(
        {
            "timm/models/__init__.py": _INITIALIZER,
            "timm/models/vision.py": _VISION,
            "timm/models/untrained.py": _UNTRAINED,
        }
    )


def test_timm_registry_enumerates_static_models_and_declared_artifacts() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_catalog_archive()))
    adapter = TimmModelRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["completed_revision"] == _REVISION
    assert page.next_state["module_count"] == 2
    assert page.next_state["model_count"] == 2
    assert page.next_state["pretrained_config_count"] == 2
    assert len(client.calls) == 2
    assert client.calls[1][0].endswith(f"/archive/{_REVISION}.zip")

    released, documented = page.records
    assert released.kind is ArtifactKind.CATALOG_RECORD
    assert released.source_record_id == "model:vision_base"
    assert released.identifiers == (Identifier("timm:model", "vision_base"),)
    assert released.models[0].status is ModelStatus.RELEASED
    assert released.models[0].aliases == ("old_vision_base",)
    assert released.releases[0].identifiers == (
        Identifier("timm:pretrained-config", "vision_base.foo"),
    )
    assert released.releases[0].version == "foo"
    assert released.releases[0].metadata["source_config_key"] == "vision_base.foo*"
    assert released.releases[0].metadata["config"]["hf_hub_id"] == "timm/vision_base.foo"
    assert {(link.url, link.relation, link.crawl) for link in released.links} == {
        ("https://github.com/huggingface/pytorch-image-models", "source_repository", False),
        (
            f"https://github.com/huggingface/pytorch-image-models/blob/{_REVISION}/timm/models/vision.py",
            "model_definition",
            False,
        ),
        ("https://huggingface.co/timm/vision_base.foo", "linked_model_artifact", False),
        ("https://weights.example.test/vision-a.pth", "weights", False),
        ("https://weights.example.test/vision-b.pth", "weights", False),
        ("https://licenses.example.test/vision", "license", False),
    }
    assert documented.source_record_id == "model:bare_architecture"
    assert documented.models[0].status is ModelStatus.DOCUMENTED
    assert documented.releases == ()


def test_timm_registry_skips_archive_when_commit_is_unchanged() -> None:
    first_client = _QueuedClient(_response({"sha": _REVISION}), _response(_catalog_archive()))
    adapter = TimmModelRegistrySourceAdapter(client=first_client)
    first = adapter.fetch_page({})

    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client
    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.records == ()
    assert second.next_state["checked_at"]
    assert len(second_client.calls) == 1


def test_timm_registry_fails_if_declared_module_is_missing_from_the_archive() -> None:
    archive = _archive({"timm/models/__init__.py": "from .missing import *\n"})
    client = _QueuedClient(_response({"sha": _REVISION}), _response(archive))

    with pytest.raises(ValueError, match="archive lacks"):
        TimmModelRegistrySourceAdapter(client=client).fetch_page({})
