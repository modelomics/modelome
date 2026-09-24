from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.geospatial_registry import GeospatialRegistrySourceAdapter, _parse_table

REV = "a" * 40
TABLE = (
    "| Model  | Details  | Weights\n"
    "| --- | --- | --- |\n"
    "| Prithvi-EO-2.0-300M | Pretrained 300M parameter model | "
    "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M |\n"
    "| Prithvi-EO-2.0-600M | Pretrained 600M parameter model | "
    "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M |\n"
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


def adapter(client: Client) -> GeospatialRegistrySourceAdapter:
    return GeospatialRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )


def test_first_party_prithvi_table_emits_declared_model_repositories() -> None:
    client = Client(response({"sha": REV}), response(TABLE))
    page = adapter(client).fetch_page({})
    assert page.authoritative_snapshot and page.upstream_count == 2
    model = page.records[0]
    assert model.identifiers == (Identifier("nasa-prithvi:model", "Prithvi-EO-2.0-300M"),)
    assert any(
        link.url == "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
        and link.relation == "model_repository"
        for link in model.links
    )
    assert not any(link.relation == "weights" for link in model.links)
    assert client.urls[1].endswith(f"/{REV}/README.md")


@pytest.mark.parametrize(
    "document",
    [
        (
            "| Model  | Details  | Weights\n| --- | --- | --- |\n"
            "| Other | x | https://huggingface.co/a/b |\n"
        ),
        (
            "| Model  | Details  | Weights\n| --- | --- | --- |\n"
            "| Prithvi-EO-x | x | https://example.org/a.pt |\n"
        ),
    ],
)
def test_table_rejects_unrecognized_or_noncanonical_rows(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_table(document, "test", 100)
