from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.dgl_lifesci_registry import (
    DglLifeSciCheckpointRegistrySourceAdapter,
)

_REVISION = "c" * 40
_GENERATIVE = """generative_url = {
    'DGMG_ChEMBL_canonical': 'pre_trained/dgmg_ChEMBL_canonical.pth',
    'DGMG_ChEMBL_random': 'pre_trained/dgmg_ChEMBL_random.pth',
    'DGMG_ZINC_canonical': 'pre_trained/dgmg_ZINC_canonical.pth',
    'DGMG_ZINC_random': 'pre_trained/dgmg_ZINC_random.pth',
    'JTVAE_ZINC_no_kl': 'pre_trained/jtvae_ZINC_no_kl.pth'
}
"""


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
    return HttpResponse(200, {}, body, "https://fixtures.test/registry")


def test_dgl_lifesci_generative_registry_emits_five_direct_pretrained_models() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_GENERATIVE))
    adapter = DglLifeSciCheckpointRegistrySourceAdapter(
        name="dgllife-generative-checkpoints",
        provider_namespace="dgllife:generative-checkpoint",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot is True
    assert page.upstream_count == 5
    assert client.calls[1].endswith(
        f"/{_REVISION}/python/dgllife/model/pretrain/generative_models.py"
    )
    by_name = {record.title: record for record in page.records}
    assert set(by_name) == {
        "DGMG_ChEMBL_canonical",
        "DGMG_ChEMBL_random",
        "DGMG_ZINC_canonical",
        "DGMG_ZINC_random",
        "JTVAE_ZINC_no_kl",
    }
    model = by_name["DGMG_ZINC_random"]
    assert model.models[0].identifiers == (
        Identifier("dgllife:generative-checkpoint", "DGMG_ZINC_random"),
    )
    assert model.releases[0].metadata["weight_url"] == (
        "https://data.dgl.ai/pre_trained/dgmg_ZINC_random.pth"
    )
