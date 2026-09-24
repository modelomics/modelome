from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
)

_REVISION = "e" * 40


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
    return HttpResponse(200, {}, body, "https://fixtures.test/checkpoints.json")


_REGISTRY = {
    "classification/resnet50": "https://download.example.test/resnet50-1234abcd.pth",
    "detection/retinanet_r50": "https://download.example.test/retinanet-r50.pkl",
}


def _adapter(client: _QueuedClient) -> StaticJsonCheckpointRegistrySourceAdapter:
    return StaticJsonCheckpointRegistrySourceAdapter(
        name="fixture-checkpoints",
        repository="example-org/checkpoints",
        branch="main",
        source_path="package/hub/models.json",
        provider_namespace="fixture:checkpoint",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_static_json_registry_preserves_one_model_scoped_release_per_literal_handle() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_REGISTRY))

    page = _adapter(client).fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["model_count"] == 2
    assert client.calls[1][0].endswith(f"/{_REVISION}/package/hub/models.json")
    classification, detection = page.records
    assert classification.kind is ArtifactKind.MODEL_CARD
    assert classification.source_record_id == "checkpoint:classification/resnet50"
    assert classification.models[0].name == "resnet50"
    assert classification.models[0].identifiers == (
        Identifier("fixture:checkpoint", "classification/resnet50"),
    )
    assert classification.releases[0].identifiers == (
        Identifier("fixture:checkpoint:release", "classification/resnet50"),
    )
    assert {
        (link.url, link.relation, link.crawl, link.model_local_ids)
        for link in classification.links
    } == {
        (
            f"https://github.com/example-org/checkpoints/blob/{_REVISION}/package/hub/models.json",
            "model_card",
            False,
            ("model:classification/resnet50",),
        ),
        (
            "https://github.com/example-org/checkpoints",
            "source_implementation",
            False,
            ("model:classification/resnet50",),
        ),
        (
            "https://download.example.test/resnet50-1234abcd.pth",
            "weights",
            False,
            ("model:classification/resnet50",),
        ),
    }
    assert detection.title == "retinanet_r50"


def test_static_json_registry_skips_unchanged_source_file() -> None:
    first_client = _QueuedClient(_response({"sha": _REVISION}), _response(_REGISTRY))
    adapter = _adapter(first_client)
    first = adapter.fetch_page({})
    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client

    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.records == ()
    assert second.upstream_count == 2
    assert len(second_client.calls) == 1


@pytest.mark.parametrize(
    ("registry", "match"),
    [
        (["not", "a", "map"], "must be a JSON object"),
        ({"relative": "/weights.pth"}, "absolute checkpoint URL"),
        ({"paper": "https://example.test/paper.pdf"}, "recognised checkpoint file"),
        ({"../escape": "https://example.test/weights.pth"}, "invalid checkpoint handle"),
        ({"valid": ["https://example.test/weights.pth"]}, "absolute checkpoint URL"),
    ],
)
def test_static_json_registry_fails_closed_on_nonliteral_or_noncheckpoint_rows(
    registry: Any,
    match: str,
) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(registry))

    with pytest.raises(ValueError, match=match):
        _adapter(client).fetch_page({})
