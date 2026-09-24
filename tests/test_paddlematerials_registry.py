from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.paddlematerials_registry import (
    PaddleMaterialsRegistryAdapter,
    _parse_registry,
)

REVISION = "a" * 40
COMMIT_URL = "https://api.github.com/repos/PaddlePaddle/PaddleMaterials/commits/develop"
SOURCE_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleMaterials/"
    f"{REVISION}/ppmat/models/__init__.py"
)


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(url: str, body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, url)


def registry_source(count: int = 22) -> str:
    rows = [
        f'"model_{index}": "https://paddle-org.bj.bcebos.com/paddlematerials/'
        f'checkpoints/model_{index}/model_{index}.zip"'
        for index in range(count)
    ]
    rows.extend(
        [
            '"chgnet_mptrj": "https://paddle-org.bj.bcebos.com/paddlematerial/'
            'checkpoints/interatomic_potentials/chgnet/chgnet_mptrj.zip"',
            '"mattersim_1M": "https://paddle-org.bj.bcebos.com/paddlematerial/'
            'checkpoints/interatomic_potentials/mattersim/mattersim_1M.zip"',
        ]
    )
    return "MODEL_REGISTRY = {\n" + ",\n".join(rows) + "\n}\n"


def test_parses_literal_names_and_exact_first_party_urls_excluding_duplicates() -> None:
    rows = _parse_registry(registry_source(), "test")

    assert len(rows) == 22
    assert rows[0] == (
        "model_0",
        "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/model_0/model_0.zip",
    )


def test_rejects_non_literal_or_non_first_party_registry_entries() -> None:
    with pytest.raises(ValueError, match="literal string"):
        _parse_registry("MODEL_REGISTRY = {get_name(): get_url()}\n", "test")
    with pytest.raises(ValueError, match="invalid first-party"):
        _parse_registry(
            'MODEL_REGISTRY = {"chem_model": "https://example.org/chem_model.zip"}\n',
            "test",
        )


def test_fetches_current_source_and_records_exact_package_handles() -> None:
    client = QueueClient(
        response(COMMIT_URL, json.dumps({"sha": REVISION}).encode()),
        response(SOURCE_URL, registry_source().encode()),
    )
    page = PaddleMaterialsRegistryAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 22
    assert len(client.calls) == 2
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.raw["package_name"] == "model_0"
    assert record.identifiers == (Identifier("paddlematerials:model-package", "model_0"),)
    assert record.raw["package_url"].endswith("/model_0.zip")
    assert record.raw
    assert record.releases[0].metadata["binary_reachability_checked"] is False


def test_live_paddlematerials_registry_metadata_smoke() -> None:
    """Fetch current first-party source metadata; do not request package bytes."""
    try:
        page = PaddleMaterialsRegistryAdapter().fetch_page({})
    except Exception as error:  # pragma: no cover - network-dependent smoke
        pytest.skip(f"live PaddleMaterials metadata unavailable: {error}")
    assert page.authoritative_snapshot
    assert page.upstream_count >= 40
