from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.static_python_checkpoint_registry import (
    StaticPythonCheckpointRegistrySourceAdapter,
)

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
    return HttpResponse(200, {}, body, "https://fixtures.test/registry.py")


_REGISTRY = '''\
import untrusted_package

_MODELS = {
    "RN50": "https://download.example.test/rn50.pt",
    "ViT-L/14@336px": "https://download.example.test/vitl-336.pth",
}
'''


def _adapter(client: _QueuedClient) -> StaticPythonCheckpointRegistrySourceAdapter:
    return StaticPythonCheckpointRegistrySourceAdapter(
        name="fixture-python-checkpoints",
        repository="example-org/checkpoints",
        branch="main",
        source_path="package/checkpoints.py",
        mapping_variable="_MODELS",
        provider_namespace="fixture:python-checkpoint",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_static_python_registry_parses_one_literal_mapping_without_execution() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_REGISTRY))

    page = _adapter(client).fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert client.calls[1][0].endswith(f"/{_REVISION}/package/checkpoints.py")
    rn50, vit = page.records
    assert rn50.kind is ArtifactKind.MODEL_CARD
    assert rn50.models[0].identifiers == (
        Identifier("fixture:python-checkpoint", "RN50"),
    )
    assert vit.title == "ViT-L/14@336px"
    assert {
        (link.url, link.relation, link.crawl, link.model_local_ids)
        for link in vit.links
    } == {
        (
            f"https://github.com/example-org/checkpoints/blob/{_REVISION}/package/checkpoints.py",
            "model_card",
            False,
            ("model:ViT-L/14@336px",),
        ),
        (
            "https://github.com/example-org/checkpoints",
            "source_implementation",
            False,
            ("model:ViT-L/14@336px",),
        ),
        (
            "https://download.example.test/vitl-336.pth",
            "weights",
            False,
            ("model:ViT-L/14@336px",),
        ),
    }


def test_static_python_registry_skips_an_unchanged_source_file() -> None:
    adapter = _adapter(_QueuedClient(_response({"sha": _REVISION}), _response(_REGISTRY)))
    first = adapter.fetch_page({})
    adapter.client = _QueuedClient(_response({"sha": _REVISION}))

    second = adapter.fetch_page(first.next_state)

    assert second.records == ()
    assert second.upstream_count == 2


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ("_MODELS = build_registry()", "must be a literal dictionary"),
        ("_MODELS = {'name': MODEL_URL}", "literal string keys and values"),
        ("_MODELS = {'name': '/weights.pt'}", "absolute checkpoint URL"),
        ("_MODELS = {'name': 'https://example.test/paper.pdf'}", "recognised checkpoint"),
        (
            "_MODELS = {'name': 'https://example.test/a.pt', 'name': 'https://example.test/b.pt'}",
            "duplicate handle",
        ),
        ("_MODELS = {'../escape': 'https://example.test/a.pt'}", "invalid checkpoint handle"),
        ("_MODELS = {'name': 'https://example.test/a.pt'}\n_MODELS = {}", "exactly one"),
        ("OTHER = {'name': 'https://example.test/a.pt'}", "exactly one"),
    ],
)
def test_static_python_registry_rejects_dynamic_or_ambiguous_source_shapes(
    document: str,
    match: str,
) -> None:
    adapter = _adapter(_QueuedClient(_response({"sha": _REVISION}), _response(document)))

    with pytest.raises(ValueError, match=match):
        adapter.fetch_page({})
