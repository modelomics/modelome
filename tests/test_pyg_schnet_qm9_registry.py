from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.pyg_schnet_qm9_registry import PyGSchNetQM9RegistrySourceAdapter

_REVISION = "f" * 40
_SOURCE = '''\
from typing import Dict

qm9_target_dict: Dict[int, str] = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
}

class SchNet:
    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    @staticmethod
    def from_qm9_pretrained(target):
        name = f'qm9_{qm9_target_dict[target]}'
        path = osp.join(root, 'trained_schnet_models', name, 'best_model')
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


def _adapter(client: _QueuedClient) -> PyGSchNetQM9RegistrySourceAdapter:
    return PyGSchNetQM9RegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_pyg_schnet_registry_maps_qm9_targets_to_archive_members() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_SOURCE))

    page = _adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert client.calls[1].endswith(
        f"/{_REVISION}/torch_geometric/nn/models/schnet.py"
    )
    entries = {record.releases[0].metadata["qm9_target"]: record for record in page.records}
    assert set(entries) == {"dipole_moment", "isotropic_polarizability", "homo"}
    model = entries["isotropic_polarizability"]
    assert model.models[0].identifiers == (
        Identifier(
            "pytorch-geometric:schnet-qm9-checkpoint",
            "schnet:qm9:isotropic_polarizability",
        ),
    )
    assert model.releases[0].metadata["archive_member"] == (
        "trained_schnet_models/qm9_isotropic_polarizability/best_model"
    )
    assert model.releases[0].metadata["weight_url"] == (
        "http://www.quantum-machine.org/datasets/trained_schnet_models.zip"
    )


@pytest.mark.parametrize(
    "source",
    [
        _SOURCE.replace("0: 'dipole_moment',", "0: get_target(),"),
        _SOURCE.replace("trained_schnet_models.zip", "other_models.zip"),
        _SOURCE.replace("class SchNet:", "class Other:"),
    ],
)
def test_pyg_schnet_registry_rejects_dynamic_or_unexpected_mapping(source: str) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(source))

    with pytest.raises(ValueError):
        _adapter(client).fetch_page({})
