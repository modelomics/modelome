from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.dgl_core_tutorial_checkpoint import (
    DGLCoreTutorialCheckpointSourceAdapter,
)

_REVISION = "d" * 40
_SOURCE = '''\
class DGMG:
    pass

import torch.utils.model_zoo as model_zoo
state_dict = model_zoo.load_url(
    "https://data.dgl.ai/model/dgmg_cycles-5a0c40be.pth"
)
model = DGMG(v_max=20, node_hidden_size=16, num_prop_rounds=2)
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


def _adapter(client: _QueuedClient) -> DGLCoreTutorialCheckpointSourceAdapter:
    return DGLCoreTutorialCheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_dgl_core_tutorial_emits_its_literal_cycle_checkpoint() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_SOURCE))

    page = _adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    assert client.calls[1].endswith(
        f"/{_REVISION}/tutorials/models/3_generative_model/5_dgmg.py"
    )
    record = page.records[0]
    assert record.title == "DGL DGMG cycles (10–20 nodes)"
    assert record.models[0].identifiers == (
        Identifier("dgl:core-checkpoint", "dgmg_cycles-5a0c40be.pth"),
    )
    assert record.releases[0].metadata["weight_url"] == (
        "https://data.dgl.ai/model/dgmg_cycles-5a0c40be.pth"
    )


@pytest.mark.parametrize(
    "source",
    [
        _SOURCE.replace("class DGMG:", "class Other:"),
        _SOURCE.replace("https://data.dgl.ai/model/dgmg_cycles-5a0c40be.pth", "https://example.test/model.pth"),
        _SOURCE.replace(
            "model_zoo.load_url(\n    \"https://data.dgl.ai/model/dgmg_cycles-5a0c40be.pth\"\n)",
            "model_zoo.load_url(make_url())",
        ),
    ],
)
def test_dgl_core_tutorial_rejects_nonmatching_or_dynamic_asset(source: str) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(source))

    with pytest.raises(ValueError):
        _adapter(client).fetch_page({})
