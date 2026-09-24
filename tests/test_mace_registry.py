from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.mace_registry import MaceOff23CheckpointRegistrySourceAdapter

REVISION = "9" * 40
BASE = "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/"


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


def test_indexes_only_exact_first_party_mace_off23_checkpoint_urls() -> None:
    source = f'''mace_off_urls = {{
    "small": "{BASE}MACE-OFF23_small.model",
    "medium": "{BASE}MACE-OFF23_medium.model",
    "large": "{BASE}MACE-OFF23_large.model",
}}
'''.encode()
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()), response(source))
    adapter = MaceOff23CheckpointRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert {record.source_record_id for record in page.records} == {
        "checkpoint:small", "checkpoint:medium", "checkpoint:large"
    }
    records = {record.raw["checkpoint_handle"]: record for record in page.records}
    for handle in ("small", "medium", "large"):
        assert records[handle].raw["weight_url"] == f"{BASE}MACE-OFF23_{handle}.model"
        assert records[handle].releases[0].metadata["weight_url"] == (
            f"{BASE}MACE-OFF23_{handle}.model"
        )
    assert len(client.calls) == 2


def test_rejects_literal_checkpoint_url_outside_official_mace_off_path() -> None:
    source = (
        b'mace_off_urls = {"small": '
        b'"https://example.org/MACE-OFF23_small.model"}\n'
    )
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()), response(source))
    with pytest.raises(ValueError, match="official checkpoint URL"):
        MaceOff23CheckpointRegistrySourceAdapter(client=client).fetch_page({})


def test_rejects_nonliteral_registry_values_without_executing_them() -> None:
    source = b'mace_off_urls = {"small": build_url("small")}\n'
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()), response(source))
    with pytest.raises(ValueError, match="literal string"):
        MaceOff23CheckpointRegistrySourceAdapter(client=client).fetch_page({})


def test_skips_source_fetch_when_revision_is_unchanged() -> None:
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()))
    page = MaceOff23CheckpointRegistrySourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 3}
    )
    assert page.records == ()
    assert page.upstream_count == 3
    assert len(client.calls) == 1
