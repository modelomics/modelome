from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.molecular_registry import OpenFoldCheckpointRegistrySourceAdapter

REVISION = "c" * 40


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


def response(value: Any, status: int = 200) -> HttpResponse:
    body = json.dumps(value).encode() if isinstance(value, dict) else value.encode()
    return HttpResponse(status, {}, body, "https://example.test")


def test_parses_only_explicit_checkpoint_filenames_from_first_party_guides() -> None:
    client = QueuedClient(
        response({"sha": REVISION}),
        response("Run --openfold_checkpoint_path openfold_params/finetuning_ptm_2.pt"),
        response("Use seq_model_esm1b_ptm.pt. Also see finetuning_ptm_2.pt."),
    )
    source = OpenFoldCheckpointRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    by_id = {record.source_record_id: record for record in page.records}
    assert set(by_id) == {
        "checkpoint:finetuning_ptm_2",
        "checkpoint:seq_model_esm1b_ptm",
    }
    solo = by_id["checkpoint:seq_model_esm1b_ptm"]
    assert solo.models[0].identifiers[0].value == "seq_model_esm1b_ptm"
    assert solo.releases[0].revision == REVISION
    assert solo.raw["guide_path"] == "docs/source/Single_Sequence_Inference.md"
    assert len(client.calls) == 3


def test_skips_guides_when_revision_is_unchanged() -> None:
    client = QueuedClient(response({"sha": REVISION}))
    page = OpenFoldCheckpointRegistrySourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 2}
    )
    assert page.records == ()
    assert page.upstream_count == 2
    assert len(client.calls) == 1


def test_rejects_missing_documented_checkpoint_names() -> None:
    client = QueuedClient(
        response({"sha": REVISION}),
        response("No files here."),
        response("No files here either."),
    )
    with pytest.raises(ValueError, match="no documented"):
        OpenFoldCheckpointRegistrySourceAdapter(client=client).fetch_page({})
