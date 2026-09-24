from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.robotics_registry_v3 import RoboticsTransformerCheckpointSourceAdapter

_REVISION = "a" * 40


class _Client:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return HttpResponse(200, {}, json.dumps(self.payload).encode(), url)


def _blob(path: str, index: int) -> dict[str, Any]:
    return {
        "path": path,
        "mode": "100644",
        "type": "blob",
        "sha": f"{index:040x}",
        "size": 100 + index,
        "url": f"https://api.github.com/blobs/{index}",
    }


def _tree() -> dict[str, Any]:
    files = []
    for index, name in enumerate(("rt1main", "rt1multirobot", "rt1simreal"), start=1):
        root = f"trained_checkpoints/{name}"
        files.extend(
            (
                _blob(f"{root}/policy_specs.pbtxt", index),
                _blob(f"{root}/saved_model.pb", index + 1),
                _blob(f"{root}/variables/variables.index", index + 2),
                _blob(f"{root}/variables/variables.data-00000-of-00001", index + 3),
            )
        )
    files.append(_blob("transformer.py", 100))
    return {"sha": _REVISION, "truncated": False, "tree": files}


def test_rt1_tree_enumerates_three_first_party_savedmodels_and_exact_files() -> None:
    client = _Client(_tree())
    adapter = RoboticsTransformerCheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls[0][0] == (
        "https://api.github.com/repos/google-research/robotics_transformer/"
        "git/trees/master?recursive=1"
    )
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 3
    assert page.next_state["file_count"] == 12
    assert [record.source_record_id for record in page.records] == [
        f"rt1:{name}@{_REVISION}" for name in ("rt1main", "rt1multirobot", "rt1simreal")
    ]

    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.identifiers == (Identifier("rt1:model", "rt1main"),)
    assert record.releases[0].metadata["revision"] == _REVISION
    assert len(record.releases[0].metadata["saved_model_files"]) == 4
    artifact_urls = {link.url for link in record.links if link.relation == "model_artifact"}
    assert artifact_urls == {
        f"https://raw.githubusercontent.com/google-research/robotics_transformer/{_REVISION}/trained_checkpoints/rt1main/policy_specs.pbtxt",
        f"https://raw.githubusercontent.com/google-research/robotics_transformer/{_REVISION}/trained_checkpoints/rt1main/saved_model.pb",
        f"https://raw.githubusercontent.com/google-research/robotics_transformer/{_REVISION}/trained_checkpoints/rt1main/variables/variables.index",
        f"https://raw.githubusercontent.com/google-research/robotics_transformer/{_REVISION}/trained_checkpoints/rt1main/variables/variables.data-00000-of-00001",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"sha": "short", "truncated": False, "tree": []},
        {"sha": _REVISION, "truncated": True, "tree": []},
        {
            "sha": _REVISION,
            "truncated": False,
            "tree": [_blob("trained_checkpoints/rt1main/policy_specs.pbtxt", 1)],
        },
    ],
)
def test_rt1_tree_rejects_incomplete_catalogs(payload: Mapping[str, Any]) -> None:
    adapter = RoboticsTransformerCheckpointSourceAdapter(client=_Client(payload))

    with pytest.raises(ValueError):
        adapter.fetch_page({})
