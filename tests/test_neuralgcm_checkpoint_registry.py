from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.neuralgcm_checkpoint_registry import (
    NeuralGCMCheckpointRegistrySourceAdapter,
    _parse_table,
)

REV = "b" * 40
TABLE = "\n".join(
    (
        "# Pre-trained model checkpoints",
        "",
        "Available on Google Cloud Storage at `gs://neuralgcm/models/`.",
        "| Reference | Model Name | Path |",
        "| -- | -- | -- |",
        "| [NeuralGCM weather and climate]("
        "https://www.nature.com/articles/s41586-024-07744-y) | "
        "0.7° deterministic | `v1/deterministic_0_7_deg.pkl` |",
        "| | 1.4° deterministic | `v1/deterministic_1_4_deg.pkl` |",
        "| | 1.4° stochastic | `v1/stochastic_1_4_deg.pkl` |",
        "| [NeuralGCM precipitation](https://arxiv.org/abs/2412.11973) | "
        "2.8° stochastic (precipitation) | `v1_precip/stochastic_precip_2_8_deg.pkl` |",
    )
)


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.urls.append(url)
        return self.responses.pop(0)


def response(value: Any) -> HttpResponse:
    body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
    return HttpResponse(200, {}, body, "https://fixture.test")


def test_first_party_neuralgcm_inventory_emits_exact_declared_objects() -> None:
    client = Client(response({"sha": REV}), response(TABLE))
    adapter = NeuralGCMCheckpointRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot and page.upstream_count == 4
    record = page.records[0]
    assert record.identifiers == (
        Identifier("neuralgcm:checkpoint", "v1/deterministic_0_7_deg.pkl"),
    )
    assert any(
        link.relation == "weights"
        and link.url
        == "https://storage.googleapis.com/neuralgcm/models/v1/deterministic_0_7_deg.pkl"
        for link in record.links
    )
    assert page.records[3].releases[0].metadata["checkpoint_handle"] == (
        "v1_precip/stochastic_precip_2_8_deg.pkl"
    )
    assert client.urls[1].endswith(f"/{REV}/docs/checkpoints.md")


@pytest.mark.parametrize(
    "document",
    [
        "| Reference | Model | Path |\n| -- | -- | -- |\n| x | y | `v1/a.pkl` |",
        (
            "| Reference | Model Name | Path |\n| -- | -- | -- |\n"
            "| [a](https://example.org) | x | `../a.pkl` |"
        ),
        (
            "| Reference | Model Name | Path |\n| -- | -- | -- |\n"
            "| [a](https://example.org) | x | `v1/a.pt` |"
        ),
    ],
)
def test_checkpoint_table_rejects_schema_and_unsafe_paths(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_table(document, "test", "docs/checkpoints.md", 100)
