from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.pyg_gpse_registry import PyGGPSECheckpointRegistrySourceAdapter

_REVISION = "a" * 40
_SOURCE = '''
class GPSE:
    url_dict = {
        'molpcba': 'https://zenodo.org/record/8145095/files/gpse_model_molpcba_1.0.pt',
        'zinc': 'https://zenodo.org/record/8145095/files/gpse_model_zinc_1.0.pt',
        'pcqm4mv2': 'https://zenodo.org/record/8145095/files/gpse_model_pcqm4mv2_1.0.pt',
        'geom': 'https://zenodo.org/record/8145095/files/gpse_model_geom_1.0.pt',
        'chembl': 'https://zenodo.org/record/8145095/files/gpse_model_chembl_1.0.pt'
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
    return HttpResponse(200, {}, body, "https://fixtures.test/registry")


def _adapter(client: _QueuedClient) -> PyGGPSECheckpointRegistrySourceAdapter:
    return PyGGPSECheckpointRegistrySourceAdapter(
        name="pyg-gpse-pretrained-checkpoints",
        provider_namespace="pytorch-geometric:gpse-checkpoint",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_pyg_gpse_registry_emits_all_five_literal_pretrained_checkpoints() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_SOURCE))

    page = _adapter(client).fetch_page({})

    assert page.authoritative_snapshot is True
    assert page.upstream_count == 5
    assert client.calls[1].endswith(f"/{_REVISION}/torch_geometric/nn/models/gpse.py")
    by_name = {record.title: record for record in page.records}
    assert set(by_name) == {"molpcba", "zinc", "pcqm4mv2", "geom", "chembl"}
    zinc = by_name["zinc"]
    assert zinc.models[0].identifiers == (
        Identifier("pytorch-geometric:gpse-checkpoint", "zinc"),
    )
    assert zinc.releases[0].metadata["weight_url"] == (
        "https://zenodo.org/record/8145095/files/gpse_model_zinc_1.0.pt"
    )


def test_pyg_gpse_registry_requires_class_scoped_literal_map() -> None:
    source = "class Other:\n    url_dict = {'wrong': 'https://host.test/model.pt'}\n"
    client = _QueuedClient(_response({"sha": _REVISION}), _response(source))

    with pytest.raises(ValueError, match="GPSE class"):
        _adapter(client).fetch_page({})


def test_pyg_gpse_registry_rejects_dynamic_map_values() -> None:
    source = "class GPSE:\n    url_dict = {'model': make_url()}\n"
    client = _QueuedClient(_response({"sha": _REVISION}), _response(source))

    with pytest.raises(ValueError, match="literal strings"):
        _adapter(client).fetch_page({})
