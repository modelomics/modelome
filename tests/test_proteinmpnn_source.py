from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.proteinmpnn import ProteinMpnSourceAdapter

REVISION = "a" * 40


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


def response(payload: Any) -> HttpResponse:
    return HttpResponse(200, {}, json.dumps(payload).encode(), "https://api.github.com")


def test_indexes_only_official_root_checkpoint_files_without_fetching_weights() -> None:
    client = QueuedClient(
        response({"sha": REVISION}),
        response({"truncated": False, "tree": [
            {"path": "vanilla_model_weights/v_48_020.pt", "type": "blob"},
            {"path": "soluble_model_weights/v_48_010.pt", "type": "blob"},
            {"path": "ca_model_weights/v_48_020.pt", "type": "blob"},
            {"path": "vanilla_model_weights/notes.txt", "type": "blob"},
            {"path": "other/v_48_020.pt", "type": "blob"},
            {"path": "vanilla_model_weights/archive/old.pt", "type": "blob"},
        ]}),
    )
    adapter = ProteinMpnSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert [record.source_record_id for record in page.records] == [
        "checkpoint:ca_model_weights/v_48_020",
        "checkpoint:soluble_model_weights/v_48_010",
        "checkpoint:vanilla_model_weights/v_48_020",
    ]
    vanilla = page.records[-1]
    assert vanilla.releases[0].revision == REVISION
    assert vanilla.links[-1].url.endswith("/vanilla_model_weights/v_48_020.pt")
    assert len(client.calls) == 2


def test_rejects_truncated_tree() -> None:
    client = QueuedClient(response({"sha": REVISION}), response({"truncated": True, "tree": []}))
    with pytest.raises(ValueError, match="truncated"):
        ProteinMpnSourceAdapter(client=client).fetch_page({})


def test_skips_tree_fetch_when_revision_is_unchanged() -> None:
    client = QueuedClient(response({"sha": REVISION}))
    page = ProteinMpnSourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 6}
    )
    assert page.records == ()
    assert page.upstream_count == 6
    assert len(client.calls) == 1
