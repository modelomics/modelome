from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.fs_mol_checkpoints import FSMolCheckpointSourceAdapter

REVISION = "e" * 40
MODELS = {
    "GNN-MAML": ("MAML-Support16_best_validation.pkl", "31346701"),
    "GNN-MT": ("multitask_best_model.pt", "31338334"),
    "PN": ("PN-Support64_best_validation.pt", "31307479"),
}
SOURCE_URL = (
    f"https://raw.githubusercontent.com/microsoft/FS-Mol/{REVISION}/README.md"
)


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: bytes, url: str = "https://api.github.com") -> HttpResponse:
    return HttpResponse(200, {}, body, url)


def readme(*, override: dict[str, tuple[str, str]] | None = None) -> bytes:
    models = MODELS | (override or {})
    lines = [
        "## Available Model Checkpoints",
        "| Model Name | Description | Checkpoint File |",
        "| --- | --- | --- |",
    ]
    descriptions = {
        "GNN-MAML": "Support set size 16.",
        "GNN-MT": "PNA message passing.",
        "PN": "Prototypical networks.",
    }
    for model, (filename, file_id) in models.items():
        lines.append(
            f"| {model} | {descriptions[model]} | "
            f"[{filename}](https://figshare.com/ndownloader/files/{file_id}) |"
        )
    return ("\n".join(lines) + "\n").encode()


def test_indexes_exact_ids_files_and_figshare_file_urls() -> None:
    api = "https://api.github.com/repos/microsoft/FS-Mol/commits/main"
    client = QueueClient(
        response(json.dumps({"sha": REVISION}).encode(), api),
        response(readme(), SOURCE_URL),
    )
    adapter = FSMolCheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 3
    records = {record.raw["checkpoint_handle"]: record for record in page.records}
    assert set(records) == set(MODELS)
    for model_name, (filename, file_id) in MODELS.items():
        record = records[model_name]
        weight_url = f"https://figshare.com/ndownloader/files/{file_id}"
        assert record.raw["checkpoint_filename"] == filename
        assert record.raw["weight_url"] == weight_url
        assert record.links[0].url == weight_url
        assert record.models[0].identifiers == (
            Identifier("fs-mol:checkpoint", model_name),
        )
    assert client.calls == [api, SOURCE_URL]


def test_skips_source_fetch_when_revision_is_unchanged() -> None:
    api = "https://api.github.com/repos/microsoft/FS-Mol/commits/main"
    client = QueueClient(response(json.dumps({"sha": REVISION}).encode(), api))

    page = FSMolCheckpointSourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 3}
    )

    assert page.records == ()
    assert page.upstream_count == 3
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "document",
    [
        readme(override={"GNN-MT": ("multitask_best_model.pt", "1")}),
        readme(override={"PN": ("PN-Support64_best_validation.pt", "1")}),
        readme().replace(b"GNN-MAML", b"GNN-MALM"),
    ],
)
def test_rejects_missing_or_changed_checkpoint_rows(document: bytes) -> None:
    api = "https://api.github.com/repos/microsoft/FS-Mol/commits/main"
    client = QueueClient(
        response(json.dumps({"sha": REVISION}).encode(), api),
        response(document, SOURCE_URL),
    )
    with pytest.raises(ValueError):
        FSMolCheckpointSourceAdapter(client=client).fetch_page({})
