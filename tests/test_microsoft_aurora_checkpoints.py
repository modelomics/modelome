from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.microsoft_aurora_checkpoints import (
    MicrosoftAuroraCheckpointSourceAdapter,
)

_REVISION = "a" * 40


class _Client:
    def __init__(self, payload: Any, *, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        body = json.dumps(self.payload).encode()
        return HttpResponse(self.status, {}, body, url)


def _payload(files: list[str]) -> dict[str, Any]:
    return {
        "id": "microsoft/aurora",
        "sha": _REVISION,
        "siblings": [{"rfilename": filename} for filename in files],
    }


def test_projects_only_officially_documented_checkpoint_files_at_pinned_revision() -> None:
    client = _Client(
        _payload(
            [
                "aurora-0.25-pretrained.ckpt",
                "aurora-0.25-v1.5.ckpt",
                "aurora-0.25-v1.5-ensemble.ckpt",
                "aurora-0.25-v1.5-static.pickle",
                "README.md",
                "unlisted.ckpt",
            ]
        )
    )
    adapter = MicrosoftAuroraCheckpointSourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert [record.source_record_id for record in page.records] == [
        "microsoft-aurora-checkpoints:aurora-0.25-pretrained.ckpt",
        "microsoft-aurora-checkpoints:aurora-0.25-v1.5.ckpt",
        "microsoft-aurora-checkpoints:aurora-0.25-v1.5-ensemble.ckpt",
    ]
    record = page.records[1]
    assert record.models[0].name == "Aurora 1.5 (0.25°)"
    assert record.models[0].identifiers == (
        Identifier("microsoft:aurora-model", "aurora-0.25-v1.5.ckpt"),
    )
    assert record.releases[0].revision == _REVISION
    assert record.releases[0].identifiers == (
        Identifier("microsoft:aurora-checkpoint", "aurora-0.25-v1.5.ckpt"),
    )
    assert record.links[0].url == (
        f"https://huggingface.co/microsoft/aurora/resolve/{_REVISION}/aurora-0.25-v1.5.ckpt"
    )
    assert record.raw["documentation_url"] == "https://microsoft.github.io/aurora/models.html"
    assert client.calls == [
        (
            "https://huggingface.co/api/models/microsoft/aurora",
            {"full": "true", "config": "false"},
        )
    ]


def test_unchanged_completed_inventory_does_not_reemit_records() -> None:
    files = ["aurora-0.25-pretrained.ckpt"]
    client = _Client(_payload(files))
    adapter = MicrosoftAuroraCheckpointSourceAdapter(client=client)
    first = adapter.fetch_page({})

    page = adapter.fetch_page(first.next_state)

    assert page.records == ()
    assert page.complete and page.upstream_count == 1
    assert len(client.calls) == 2


def test_changed_revision_or_file_inventory_reemits_current_checkpoint_set() -> None:
    first_client = _Client(_payload(["aurora-0.25-pretrained.ckpt"]))
    adapter = MicrosoftAuroraCheckpointSourceAdapter(client=first_client)
    first = adapter.fetch_page({})
    changed_revision = "b" * 40
    second_client = _Client(
        {
            **_payload(
                [
                    "aurora-0.25-pretrained.ckpt",
                    "aurora-0.25-v1.5.ckpt",
                ]
            ),
            "sha": changed_revision,
        }
    )
    adapter.client = second_client

    updated = adapter.fetch_page(first.next_state)

    assert updated.complete and updated.authoritative_snapshot
    assert updated.upstream_count == 2
    assert updated.next_state["revision"] == changed_revision
    assert updated.next_state["checkpoint_files"] == [
        "aurora-0.25-pretrained.ckpt",
        "aurora-0.25-v1.5.ckpt",
    ]
    assert [record.releases[0].revision for record in updated.records] == [
        changed_revision,
        changed_revision,
    ]
    assert updated.records[1].links[0].url.endswith(f"/{changed_revision}/aurora-0.25-v1.5.ckpt")


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "somebody/else", "sha": _REVISION, "siblings": []},
        {"id": "microsoft/aurora", "sha": "", "siblings": []},
        {"id": "microsoft/aurora", "sha": _REVISION, "siblings": []},
    ],
)
def test_rejects_invalid_repository_or_missing_documented_checkpoint(payload: Any) -> None:
    with pytest.raises(ValueError):
        MicrosoftAuroraCheckpointSourceAdapter(client=_Client(payload)).fetch_page({})


def test_rejects_non_success_http_status() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        MicrosoftAuroraCheckpointSourceAdapter(client=_Client({}, status=503)).fetch_page({})


def test_disabled_proposal_matches_adapter_configuration() -> None:
    proposal_path = Path(__file__).parents[1] / "config/proposals/microsoft_aurora_checkpoints.toml"
    source = tomllib.loads(proposal_path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = MicrosoftAuroraCheckpointSourceAdapter(
        name=source["name"], max_response_bytes=source["max_response_bytes"]
    )
    assert adapter.name == source["name"]
    assert adapter.max_response_bytes == source["max_response_bytes"]
