from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.torchxrayvision_registry import (
    TorchXRayVisionRegistrySourceAdapter,
)

_REVISION = "a" * 40


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def _response(payload: bytes | Mapping[str, Any]) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixture.test/source")


_SOURCE = '''\
model_urls = {}
model_urls['all'] = {
    "weights_url": "https://github.com/mlmed/torchxrayvision/releases/download/v1/all.pt",
    "labels": ['Atelectasis'],
}
model_urls['densenet121-res224-all'] = model_urls['all']
model_urls['rsna'] = {
    "weights_url": "https://github.com/mlmed/torchxrayvision/releases/download/v1/rsna.pt",
}
model_urls['densenet121-res224-rsna'] = model_urls['rsna']
'''


def _adapter(client: _QueuedClient) -> TorchXRayVisionRegistrySourceAdapter:
    return TorchXRayVisionRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_torchxrayvision_records_declared_weight_urls_and_groups_constructor_aliases() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_SOURCE.encode()))

    page = _adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    all_weights, rsna = page.records
    assert all_weights.models[0].identifiers == (
        Identifier("torchxrayvision:weight", "densenet121-res224-all"),
    )
    assert all_weights.models[0].aliases == ("all",)
    assert rsna.models[0].aliases == ("rsna",)
    assert {
        link.url
        for link in all_weights.links
        if link.relation == "weights"
    } == {
        "https://github.com/mlmed/torchxrayvision/releases/download/v1/all.pt"
    }
    assert client.calls[1].endswith(
        f"/{_REVISION}/torchxrayvision/models.py"
    )


@pytest.mark.parametrize(
    ("source", "match"),
    [
        (
            "model_urls['a'] = {'weights_url': 'https://example.test/a.pt'}\n"
            "model_urls['bad'] = model_urls['missing']",
            "unknown model",
        ),
        ("model_urls['a'] = {'weights_url': make_url()}", "no literal model_urls"),
        (
            "model_urls['a'] = {'weights_url': 'https://example.test/a.pt'}\n"
            "model_urls['a'] = {'weights_url': 'https://example.test/b.pt'}",
            "duplicate model_urls key",
        ),
    ],
)
def test_torchxrayvision_rejects_unknown_aliases_dynamic_urls_and_duplicate_keys(
    source: str, match: str
) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(source.encode()))

    with pytest.raises(ValueError, match=match):
        _adapter(client).fetch_page({})
