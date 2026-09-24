from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.mlx_registry import MlxRegistrySourceAdapter, _parse

SHA = "a" * 40


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(payload: Any) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixture.test")


def test_mlx_type_remapping_is_pinned_and_keeps_architecture_semantics() -> None:
    source = b'MODEL_REMAPPING = {"mistral": "llama", "llava": "mistral3"}\n'
    client = Client(response({"sha": SHA}), response(source))
    adapter = MlxRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert client.calls[1].endswith(f"/{SHA}/mlx_lm/utils.py")
    record = page.records[0]
    assert record.kind is ArtifactKind.CATALOG_RECORD
    assert record.models[0].identifiers == (Identifier("mlx-lm:model-type-alias", "mistral"),)
    assert record.models[0].aliases == ("llama",)
    assert record.models[0].status is ModelStatus.DOCUMENTED
    assert record.releases == ()


def test_mlx_skips_unchanged_commit() -> None:
    adapter = MlxRegistrySourceAdapter(
        client=Client(response({"sha": SHA}), response(b'MODEL_REMAPPING = {"mistral": "llama"}'))
    )
    first = adapter.fetch_page({})
    adapter.client = Client(response({"sha": SHA}))

    page = adapter.fetch_page(first.next_state)

    assert page.records == ()
    assert page.upstream_count == 1


@pytest.mark.parametrize(
    "source",
    [
        "MODEL_REMAPPING = build_map()",
        'MODEL_REMAPPING = {"mistral": dynamic}',
        'MODEL_REMAPPING = {"Bad-Type": "llama"}',
        'MODEL_REMAPPING = {"mistral": "llama", "mistral": "mistral3"}',
        'OTHER = {"mistral": "llama"}',
    ],
)
def test_mlx_rejects_dynamic_or_ambiguous_maps(source: str) -> None:
    with pytest.raises(ValueError):
        _parse(source)
