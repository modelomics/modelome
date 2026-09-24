from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.graph_ml_registry import GraphMLRegistrySourceAdapter

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
    return HttpResponse(200, {}, body, "https://fixtures.test/registry")


def _adapter(client: _QueuedClient) -> GraphMLRegistrySourceAdapter:
    return GraphMLRegistrySourceAdapter(
        name="dgl-lifesci-graph-checkpoints",
        repository="awslabs/dgl-lifesci",
        branch="master",
        source_path="python/dgllife/model/pretrain/property_prediction.py",
        mapping_variable="property_url",
        provider_namespace="dgllife:checkpoint",
        checkpoint_base_url="https://data.dgl.ai/",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_graph_ml_registry_emits_source_declared_checkpoint_handles() -> None:
    body = """property_url = {
    'gin_supervised_contextpred': 'dgllife/pre_trained/gin_supervised_contextpred.pth',
    'gin_supervised_masking': 'dgllife/pre_trained/gin_supervised_masking.pth',
}
"""
    client = _QueuedClient(_response({"sha": _REVISION}), _response(body))

    page = _adapter(client).fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert client.calls[1][0].endswith(
        f"/{_REVISION}/python/dgllife/model/pretrain/property_prediction.py"
    )
    first, second = page.records
    assert first.kind is ArtifactKind.MODEL_CARD
    assert first.source_record_id == "checkpoint:gin_supervised_contextpred"
    assert first.models[0].identifiers == (
        Identifier("dgllife:checkpoint", "gin_supervised_contextpred"),
    )
    assert first.releases[0].identifiers == (
        Identifier("dgllife:checkpoint:release", "gin_supervised_contextpred"),
    )
    assert {
        (link.url, link.relation, link.model_local_ids)
        for link in first.links
    } == {
        (
            f"https://github.com/awslabs/dgl-lifesci/blob/{_REVISION}/"
            "python/dgllife/model/pretrain/property_prediction.py",
            "model_card",
            ("model:gin_supervised_contextpred",),
        ),
        (
            "https://github.com/awslabs/dgl-lifesci",
            "source_implementation",
            ("model:gin_supervised_contextpred",),
        ),
        (
            "https://data.dgl.ai/dgllife/pre_trained/gin_supervised_contextpred.pth",
            "weights",
            ("model:gin_supervised_contextpred",),
        ),
    }
    assert second.title == "gin_supervised_masking"


@pytest.mark.parametrize(
    "value",
    [
        "https://attacker.example/model.pth",
        "../outside/model.pth",
        "dgllife/model.txt",
    ],
)
def test_graph_ml_registry_rejects_non_relative_or_non_checkpoint_values(value: str) -> None:
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(f"property_url = {{'model': {value!r}}}\n"),
    )

    with pytest.raises(ValueError, match="unsafe checkpoint path"):
        _adapter(client).fetch_page({})


def test_graph_ml_registry_rejects_non_https_checkpoint_base() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        GraphMLRegistrySourceAdapter(
            name="fixture",
            repository="example-org/example",
            branch="main",
            source_path="models.py",
            mapping_variable="models",
            provider_namespace="fixture:checkpoint",
            checkpoint_base_url="http://data.example.test/",
            client=_QueuedClient(),
        )
