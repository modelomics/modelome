from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.graphormer_checkpoint_registry import (
    GraphormerCheckpointRegistrySourceAdapter,
)

_REVISION = "b" * 40
_SOURCE = '''\
from torch.hub import load_state_dict_from_url

PRETRAINED_MODEL_URLS = {
    "pcqm4mv1_graphormer_base": "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv1.pt",
    "pcqm4mv2_graphormer_base": "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv2.pt",
    "oc20is2re_graphormer3d_base": (
        "https://szheng.blob.core.windows.net/graphormer/modelzoo/oc20is2re/"
        "checkpoint_last_oc20_is2re.pt"
    ), # temporarily unavailable
    "pcqm4mv1_graphormer_base_for_molhiv": "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_base_preln_pcqm4mv1_for_hiv.pt",
}
'''


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
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: str | Mapping[str, Any]) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/source")


def _adapter(client: _QueuedClient) -> GraphormerCheckpointRegistrySourceAdapter:
    return GraphormerCheckpointRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_graphormer_registry_emits_four_exact_first_party_model_artifacts() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_SOURCE))

    page = _adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    assert client.calls[1].endswith(
        f"/{_REVISION}/graphormer/pretrain/__init__.py"
    )
    by_handle = {record.models[0].identifiers[0].value: record for record in page.records}
    assert set(by_handle) == {
        "pcqm4mv1_graphormer_base",
        "pcqm4mv2_graphormer_base",
        "oc20is2re_graphormer3d_base",
        "pcqm4mv1_graphormer_base_for_molhiv",
    }
    assert by_handle["pcqm4mv2_graphormer_base"].models[0].identifiers == (
        Identifier("microsoft-graphormer:checkpoint", "pcqm4mv2_graphormer_base"),
    )
    assert by_handle["pcqm4mv2_graphormer_base"].releases[0].metadata["weight_url"] == (
        "https://ml2md.blob.core.windows.net/graphormer-ckpts/checkpoint_best_pcqm4mv2.pt"
    )
    assert by_handle["oc20is2re_graphormer3d_base"].releases[0].metadata["weight_url"] == (
        "https://szheng.blob.core.windows.net/graphormer/modelzoo/oc20is2re/"
        "checkpoint_last_oc20_is2re.pt"
    )


@pytest.mark.parametrize(
    "source, message",
    [
        ("PRETRAINED_MODEL_URLS = build_registry()", "literal dictionary"),
        (
            'PRETRAINED_MODEL_URLS = {"not-a-checkpoint": "https://example.test/readme"}',
            "recognised checkpoint file",
        ),
    ],
)
def test_graphormer_registry_rejects_dynamic_or_non_checkpoint_entries(
    source: str, message: str
) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(source))

    with pytest.raises(ValueError, match=message):
        _adapter(client).fetch_page({})
