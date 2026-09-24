from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.graphcore_gpspp_checkpoints import (
    GraphcoreGPSPlusPlusCheckpointSourceAdapter,
)

_REVISION = "c" * 40
_HOST = "https://graphcore-ogblsc-pcqm4mv2.s3.us-west-1.amazonaws.com/"
_TABLE = "\n".join(
    (
        "## Performance",
        "",
        "| Model config | Parameters | No. layers | Train MAE | Valid MAE | "
        "Config file name | Checkpoint |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| GPS++ 11M | 11M | 4 | ~0.075 | ~0.090 | GPS_PCQ_4gps_11M.yaml | "
        f"[11M ckpt]({_HOST}GPS_PCQ_4gps_11M.tar.gz) |",
        "| GPS++ 22M | 22M | 8 | ~0.056 | ~0.082 | GPS_PCQ_8gps_22M.yaml | "
        f"[22M ckpt]({_HOST}GPS_PCQ_8gps_22M.tar.gz) |",
        "| GPS++ | 44M | 16 | ~0.044 | ~0.077 | GPS_PCQ_16gps_44M.yaml | "
        f"[gps++ ckpt1]({_HOST}GPS_PCQ_16gps_44M.tar.gz) |",
        "| GPS++ trained on valid split | 44M | 16 | ~0.044 | NA | "
        "GPS_PCQ_16gps_44M.yaml | "
        f"[gps++ ckpt2]({_HOST}GPS_PCQ_16gps_44M_inc_valid.tar.gz) |",
        "",
        "## Our submission to OGB-LSC PCQM4Mv2",
        "The full challenge ensemble contains 112 models.",
    )
)


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
    return HttpResponse(200, {}, body, "https://fixtures.test/README.md")


def _adapter(client: _QueuedClient) -> GraphcoreGPSPlusPlusCheckpointSourceAdapter:
    return GraphcoreGPSPlusPlusCheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_graphcore_gpspp_checkpoint_table_emits_exact_rows() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_TABLE))

    page = _adapter(client).fetch_page({})

    assert page.authoritative_snapshot is True
    assert page.upstream_count == 4
    expected = (
        ("GPS++ 11M", "GPS_PCQ_4gps_11M.tar.gz"),
        ("GPS++ 22M", "GPS_PCQ_8gps_22M.tar.gz"),
        ("GPS++", "GPS_PCQ_16gps_44M.tar.gz"),
        ("GPS++ trained on valid split", "GPS_PCQ_16gps_44M_inc_valid.tar.gz"),
    )
    for record, (handle, filename) in zip(page.records, expected, strict=True):
        assert record.title == handle
        assert record.models[0].identifiers == (Identifier("graphcore-gpspp:checkpoint", handle),)
        assert record.releases[0].metadata["weight_url"] == _HOST + filename


@pytest.mark.parametrize(
    "document",
    [
        _TABLE.replace("## Performance", "## Missing table"),
        _TABLE.replace("GPS_PCQ_16gps_44M_inc_valid.tar.gz", "unlisted.tar.gz"),
        _TABLE.replace("GPS_PCQ_4gps_11M.tar.gz", "GPS_PCQ_8gps_22M.tar.gz"),
    ],
)
def test_graphcore_gpspp_rejects_missing_or_altered_artifact_rows(document: str) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(document))

    with pytest.raises(ValueError):
        _adapter(client).fetch_page({})
