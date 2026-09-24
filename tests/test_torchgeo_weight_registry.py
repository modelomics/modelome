from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.torchgeo_weight_registry import TorchGeoWeightRegistrySourceAdapter

_REVISION = "d" * 40


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
    return HttpResponse(200, {}, body, "https://fixture.test/torchgeo")


def _archive() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as package:
        package.writestr(
            f"torchgeo-{_REVISION}/torchgeo/models/resnet.py",
            """from torchvision.models._api import Weights, WeightsEnum\n\n"""
            """class ResNet50_Weights(WeightsEnum):\n"""
            """    SENTINEL2_MOCO = Weights(url="https://weights.example.test/s2-moco.pth")\n"""
            """    DEFAULT = SENTINEL2_MOCO\n""",
        )
    return output.getvalue()


def test_torchgeo_weight_enum_records_exact_declared_checkpoint_url() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_archive()))
    adapter = TorchGeoWeightRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.source_record_id == "torchgeo-weight-enum:ResNet50_Weights"
    assert record.identifiers == (Identifier("torchgeo:weight-enum", "ResNet50_Weights"),)
    assert record.releases[0].identifiers == (
        Identifier("torchgeo:weight-enum-member", "ResNet50_Weights.SENTINEL2_MOCO"),
    )
    assert any(
        link.url == "https://weights.example.test/s2-moco.pth" and link.relation == "weights"
        for link in record.links
    )
    assert client.calls[1].endswith(f"/archive/{_REVISION}.zip")
