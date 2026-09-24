from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.pyg_dimenet_checkpoints import PyGDimeNetCheckpointSourceAdapter

_REVISION = "e" * 40
_SOURCE = '''\
from typing import Dict

qm9_target_dict: Dict[int, str] = {0: 'mu', 1: 'alpha', 2: 'homo'}

class DimeNet:
    url = ('https://github.com/klicperajo/dimenet/raw/master/pretrained/'
           'dimenet')

class DimeNetPlusPlus:
    url = ('https://raw.githubusercontent.com/gasteigerjo/dimenet/'
           'master/pretrained/dimenet_pp')
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
    return HttpResponse(200, {}, body, "https://fixtures.test/dimenet.py")


def _adapter(client: _QueuedClient) -> PyGDimeNetCheckpointSourceAdapter:
    return PyGDimeNetCheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_pyg_dimenet_registry_expands_exact_target_checkpoint_parts() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_SOURCE))

    page = _adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 6
    assert client.calls[1].endswith(
        f"/{_REVISION}/torch_geometric/nn/models/dimenet.py"
    )
    records = {record.title: record for record in page.records}
    assert set(records) == {
        "DimeNet QM9 mu checkpoint",
        "DimeNet QM9 alpha checkpoint",
        "DimeNet QM9 homo checkpoint",
        "DimeNetPlusPlus QM9 mu checkpoint",
        "DimeNetPlusPlus QM9 alpha checkpoint",
        "DimeNetPlusPlus QM9 homo checkpoint",
    }
    record = records["DimeNetPlusPlus QM9 alpha checkpoint"]
    assert record.models[0].identifiers == (
        Identifier("pytorch-geometric:dimenet-checkpoint", "dimenet++:qm9:alpha"),
    )
    assert record.releases[0].metadata["checkpoint_files"] == (
        "https://raw.githubusercontent.com/gasteigerjo/dimenet/master/pretrained/"
        "dimenet_pp/alpha/checkpoint",
        "https://raw.githubusercontent.com/gasteigerjo/dimenet/master/pretrained/"
        "dimenet_pp/alpha/ckpt.data-00000-of-00002",
        "https://raw.githubusercontent.com/gasteigerjo/dimenet/master/pretrained/"
        "dimenet_pp/alpha/ckpt.data-00001-of-00002",
        "https://raw.githubusercontent.com/gasteigerjo/dimenet/master/pretrained/"
        "dimenet_pp/alpha/ckpt.index",
    )


@pytest.mark.parametrize(
    "source",
    [
        _SOURCE.replace("{0: 'mu', 1: 'alpha', 2: 'homo'}", "make_targets()"),
        _SOURCE.replace("master/pretrained/dimenet_pp", "main/pretrained/dimenet_pp"),
        _SOURCE.replace("class DimeNetPlusPlus:", "class OtherVariant:"),
    ],
)
def test_pyg_dimenet_registry_rejects_dynamic_or_unexpected_maps(source: str) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(source))

    with pytest.raises(ValueError):
        _adapter(client).fetch_page({})
