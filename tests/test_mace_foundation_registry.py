from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.mace_foundation_registry import (
    MaceFoundationCheckpointRegistrySourceAdapter,
)

REVISION = "8" * 40
MP = "https://github.com/ACEsuit/mace-mp/releases/download/"
FOUNDATIONS = "https://github.com/ACEsuit/mace-foundations/releases/download/"


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None,
            headers: dict[str, str] | None = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, "https://api.github.com")


SOURCE = f'''mace_mp_urls = {{
    "small": "{MP}mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "medium-mpa-0": "{MP}mace_mpa_0/mace-mpa-0-medium.model",
    "mace-matpes-pbe-0": "{FOUNDATIONS}mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
}}
polar_model_urls = {{
    "polar-1-s": "{FOUNDATIONS}mace_polar_1/MACE-POLAR-1-S.model",
    "polar-1-m": "{FOUNDATIONS}mace_polar_1/MACE-POLAR-1-M.model",
    "polar-1-l": "{FOUNDATIONS}mace_polar_1/MACE-POLAR-1-L.model",
}}
'''.encode()


def test_indexes_mace_mp_and_polar_literal_keys_with_exact_release_urls() -> None:
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()), response(SOURCE))
    adapter = MaceFoundationCheckpointRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 6
    records = {record.source_record_id: record for record in page.records}
    assert "checkpoint:mace-mp:medium-mpa-0" in records
    assert "checkpoint:mace-polar:polar-1-l" in records
    assert records["checkpoint:mace-mp:medium-mpa-0"].raw["weight_url"] == (
        f"{MP}mace_mpa_0/mace-mpa-0-medium.model"
    )
    polar = records["checkpoint:mace-polar:polar-1-l"]
    assert polar.raw["weight_url"] == (
        f"{FOUNDATIONS}mace_polar_1/MACE-POLAR-1-L.model"
    )
    assert polar.models[0].identifiers[0].namespace == "mace:polar-checkpoint"
    assert len(client.calls) == 2


def test_rejects_nonliteral_mapping_values_without_importing_source() -> None:
    source = b'''mace_mp_urls = {"small": resolve_model_url("small")}
polar_model_urls = {"polar-1-s": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-S.model"}
'''
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()), response(source))
    with pytest.raises(ValueError, match="literal handles and URLs"):
        MaceFoundationCheckpointRegistrySourceAdapter(client=client).fetch_page({})


def test_rejects_nonfirstparty_or_wrong_polar_assets() -> None:
    source = f'''mace_mp_urls = {{"small": "{MP}mace_mp_0/small.model"}}
polar_model_urls = {{"polar-1-s": "{FOUNDATIONS}mace_polar_1/other.model"}}
'''.encode()
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()), response(source))
    with pytest.raises(ValueError, match="unapproved checkpoint URL"):
        MaceFoundationCheckpointRegistrySourceAdapter(client=client).fetch_page({})


def test_skips_source_fetch_when_commit_is_unchanged() -> None:
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()))
    page = MaceFoundationCheckpointRegistrySourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 19}
    )
    assert page.records == ()
    assert page.upstream_count == 19
    assert len(client.calls) == 1
