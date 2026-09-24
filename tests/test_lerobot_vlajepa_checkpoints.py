from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.lerobot_vlajepa_checkpoints import (
    LeRobotVLAJEPACheckpointSourceAdapter,
    _parse_checkpoints,
)

_SHA = "f" * 40
_ROWS = (
    ("VLA-JEPA-LIBERO", "LIBERO-10", "2 (agentview + wrist)", "Enabled", "7"),
    ("VLA-JEPA-Pretrain", "DROID 1.0.1", "2 (exterior left views)", "Enabled", "7"),
    ("VLA-JEPA-SimplerEnv", "OXE Bridge / RT-1", "1 (view duplicated ×2)", "Enabled", "7"),
)
_README = "\n".join(
    (
        "# VLA-JEPA",
        "## Pretrained Checkpoints",
        "Three checkpoints are available:",
        "Checkpoint | Dataset | Cameras | World model | Action dim",
        "--- | --- | --- | --- | ---",
        *[
            f"`lerobot/{slug}` | {dataset} | {cameras} | {world_model} | {action_dim}"
            for slug, dataset, cameras, world_model, action_dim in _ROWS
        ],
        "## Configuration",
        "`lerobot/VLA-JEPA-Other` | Other | 1 | Enabled | 7",
    )
)


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        del params, headers
        self.calls.append(url)
        body = (
            json.dumps({"sha": _SHA}).encode()
            if url.endswith("/commits/main")
            else _README.encode()
        )
        return HttpResponse(200, {}, body, url)


def test_parser_keeps_only_exact_rows_from_pretrained_checkpoints_table() -> None:
    entries = _parse_checkpoints(_README, source="test", maximum=5)

    assert [entry[0] for entry in entries] == [f"lerobot/{row[0]}" for row in _ROWS]
    assert [entry[1] for entry in entries] == [row[1] for row in _ROWS]
    assert [entry[2] for entry in entries] == [row[2] for row in _ROWS]
    assert [entry[5] for entry in entries] == [6, 7, 8]


def test_adapter_emits_three_exact_checkpoint_references() -> None:
    client = _Client()
    adapter = LeRobotVLAJEPACheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/huggingface/lerobot/commits/main",
        f"https://raw.githubusercontent.com/huggingface/lerobot/{_SHA}/docs/source/vla_jepa.mdx",
    ]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    libero = next(item for item in page.records if item.title.endswith("VLA-JEPA-LIBERO"))
    assert libero.kind is ArtifactKind.WEIGHTS
    assert libero.identifiers == (Identifier("huggingface:model", "lerobot/VLA-JEPA-LIBERO"),)
    assert libero.releases[0].identifiers == (
        Identifier("lerobot:checkpoint", "lerobot/VLA-JEPA-LIBERO"),
    )
    assert libero.releases[0].metadata["dataset_label"] == "LIBERO-10"
    assert libero.releases[0].metadata["camera_configuration"] == "2 (agentview + wrist)"


@pytest.mark.parametrize(
    "document,maximum",
    [
        (_README, 2),
        (
            "## Pretrained Checkpoints\nCheckpoint | Dataset | Cameras | World model | Action dim\n"
            "--- | --- | --- | --- | ---\n"
            "`lerobot/OTHER` | Other | 1 | Enabled | 7",
            5,
        ),
    ],
)
def test_parser_enforces_table_bounds_and_scope(document: str, maximum: int) -> None:
    if maximum == 2:
        with pytest.raises(ValueError, match="exceeds 2"):
            _parse_checkpoints(document, source="test", maximum=maximum)
    else:
        assert _parse_checkpoints(document, source="test", maximum=maximum) == ()
