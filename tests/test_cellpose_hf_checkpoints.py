from __future__ import annotations

import json
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.cellpose_hf_checkpoints import CellposeHubCheckpointSourceAdapter

REVISION = "a" * 40
REPOSITORY = "mouseland/cellpose-sam"
CHECKPOINTS = ("cpsam_v2", "cpdino", "cpdino-vitb", "cpsam")


class QueuedClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append((url, dict(params) if params is not None else None))
        return HttpResponse(
            status=200,
            headers={},
            body=json.dumps(self.payload).encode(),
            url=url,
        )


def test_enumerates_only_the_four_source_declared_cellpose_hub_files() -> None:
    client = QueuedClient(
        {
            "id": REPOSITORY,
            "sha": REVISION,
            "siblings": [{"rfilename": name} for name in (*CHECKPOINTS, "README.md")],
        }
    )
    source = CellposeHubCheckpointSourceAdapter(client=client)

    page = source.fetch_page({})

    assert page.authoritative_snapshot and page.complete
    assert page.upstream_count == 4
    assert [record.source_record_id for record in page.records] == [
        f"cellpose-hub-checkpoints:{name}" for name in CHECKPOINTS
    ]
    for record, name in zip(page.records, CHECKPOINTS, strict=True):
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        assert record.canonical_url == url
        assert record.raw["checkpoint_file"] == name
        assert record.raw["revision"] == REVISION
        assert record.links[0].url == url
    assert client.calls == [
        (
            f"https://huggingface.co/api/models/{REPOSITORY}",
            {"full": "true", "config": "false"},
        )
    ]


def test_unchanged_hub_revision_avoids_reemitting_checkpoint_records() -> None:
    payload = {
        "id": REPOSITORY,
        "sha": REVISION,
        "siblings": [{"rfilename": name} for name in CHECKPOINTS],
    }
    source = CellposeHubCheckpointSourceAdapter(
        client=QueuedClient(payload)
    )

    page = source.fetch_page(
        {"completed": True, "revision": REVISION, "checkpoint_files": list(CHECKPOINTS)}
    )

    assert page.records == ()
    assert page.upstream_count == 4
