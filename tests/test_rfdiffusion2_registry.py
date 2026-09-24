from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.rfdiffusion2_registry import (
    RFDiffusion2CheckpointRegistryAdapter,
    _parse_manifest,
)

REVISION = "f" * 40
BASE_URL = "https://files.ipd.uw.edu/pub/rfdiffusion2/"
MANIFEST = '''BASE_URL = "https://files.ipd.uw.edu/pub/rfdiffusion2/"
WEIGHTS = [
    "model_weights/RFD_173.pt",
    "model_weights/RFD_140.pt",
    "third_party_model_weights/ligand_mpnn/s25_r010_t300_p.pt",
    "third_party_model_weights/ligand_mpnn/s_300756.pt",
]
'''


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Any = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(value: Any) -> HttpResponse:
    body = json.dumps(value).encode() if isinstance(value, dict) else value.encode()
    return HttpResponse(200, {}, body, "https://example.test")


def test_first_party_installer_names_exact_rfdiffusion2_model_weights() -> None:
    client = QueuedClient(response({"sha": REVISION}), response(MANIFEST))
    adapter = RFDiffusion2CheckpointRegistryAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    record = page.records[0]
    assert [model.name for model in record.models] == [
        "RFdiffusion2 RFD_140",
        "RFdiffusion2 RFD_173",
    ]
    assert [item["path"] for item in record.raw["checkpoints"]] == [
        "model_weights/RFD_140.pt", "model_weights/RFD_173.pt"
    ]
    assert [item["url"] for item in record.raw["checkpoints"]] == [
        f"{BASE_URL}model_weights/RFD_140.pt",
        f"{BASE_URL}model_weights/RFD_173.pt",
    ]
    assert len(record.releases) == 2
    assert all(release.revision is None for release in record.releases)
    assert len(client.calls) == 2


def test_skips_manifest_when_installer_commit_is_unchanged() -> None:
    client = QueuedClient(response({"sha": REVISION}))
    page = RFDiffusion2CheckpointRegistryAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "checkpoint_count": 2}
    )
    assert page.records == ()
    assert page.upstream_count == 2
    assert len(client.calls) == 1


def test_manifest_parser_rejects_mutable_or_ambiguous_first_party_inventory() -> None:
    with pytest.raises(ValueError, match="base URL"):
        _parse_manifest(
            MANIFEST.replace(BASE_URL, "https://example.test/"), max_entries=32
        )
    duplicated = MANIFEST.replace(
        '"model_weights/RFD_140.pt",',
        '"model_weights/RFD_140.pt", "model_weights/RFD_140.pt",',
    )
    with pytest.raises(ValueError, match="repeats"):
        _parse_manifest(duplicated, max_entries=32)


def test_rejects_unsupported_paths_inside_own_model_directory() -> None:
    malformed = MANIFEST.replace("RFD_173.pt", "latest.pt")
    with pytest.raises(ValueError, match="checkpoint path"):
        _parse_manifest(malformed, max_entries=32)
