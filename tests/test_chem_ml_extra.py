from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.chem_ml_extra import (
    ChempropCheMeleonCheckpointSourceAdapter,
    UniMofCheckpointSourceAdapter,
    _checkpoint_rows,
)

REVISION = "e" * 40


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


README = b"""# Uni-MOF

Model | File Size | Update Date | Download Link
--- | --- | --- | ---
nanoporous material pretrain | 303 MB | May 10 2023 | https://github.com/dptech-corp/Uni-MOF/releases/download/v0.1/unimof_pretrain_best.pt
hMOF_MOFX_DB | 304 MB | May 10 2023 | https://github.com/dptech-corp/Uni-MOF/releases/download/v0.1/unimof_hMOF_MOFX_DB_finetune_best.pt
CoRE_MOFX_DB | 304 MB | May 10 2023 | https://github.com/dptech-corp/Uni-MOF/releases/download/v0.1/unimof_CoRE_MOFX_DB_finetune_best.pt
CoRE_MAP_DB | 168 MB | May 10 2023 | https://github.com/dptech-corp/Uni-MOF/releases/download/v0.1/unimof_CoRE_MAP_DB_fintune_best.pt
Data | 5.77 MB | May 10 2023 | https://github.com/dptech-corp/Uni-MOF/releases/download/v0.1/MOF_structure_data.zip
Weight | 303 MB | May 10 2023 | https://github.com/dptech-corp/Uni-MOF/releases/download/v0.1/CoRE_PLD_bset.pt
"""


def test_indexes_only_checkpoint_rows_from_pinned_first_party_readme() -> None:
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()), response(README))
    adapter = UniMofCheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 5
    assert [record.source_record_id for record in page.records] == [
        "checkpoint:CoRE_PLD_bset",
        "checkpoint:unimof_CoRE_MAP_DB_fintune_best",
        "checkpoint:unimof_CoRE_MOFX_DB_finetune_best",
        "checkpoint:unimof_hMOF_MOFX_DB_finetune_best",
        "checkpoint:unimof_pretrain_best",
    ]
    record = page.records[-1]
    assert record.releases[0].revision == REVISION
    assert record.releases[0].metadata["revision"] == REVISION
    assert record.links[-1].url.endswith("/unimof_pretrain_best.pt")
    assert len(client.calls) == 2


def test_checkpoint_parser_rejects_other_hosts_and_non_pt_assets() -> None:
    markdown = """| name | link |
| --- | --- |
| outside | https://example.org/releases/download/v1/model.pt |
| dataset | https://github.com/dptech-corp/Uni-MOF/releases/download/v0.1/data.zip |
"""
    assert _checkpoint_rows(markdown, "dptech-corp/Uni-MOF") == ()


def test_rejects_invalid_revision() -> None:
    client = QueuedClient(response(json.dumps({"sha": "bad"}).encode()))
    with pytest.raises(ValueError, match="invalid commit"):
        UniMofCheckpointSourceAdapter(client=client).fetch_page({})


def test_skips_readme_fetch_when_revision_is_unchanged() -> None:
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()))
    page = UniMofCheckpointSourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 5}
    )
    assert page.records == ()
    assert page.upstream_count == 5
    assert len(client.calls) == 1


def test_indexes_chemeleon_only_when_first_party_docs_list_exact_zenodo_file() -> None:
    doc = (
        b'<p>Download the model with</p>'
        b'<code>https://zenodo.org/records/15460715/files/chemeleon_mp.pt</code>'
    )
    client = QueuedClient(response(doc))
    adapter = ChempropCheMeleonCheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.source_record_id == "checkpoint:chemeleon_mp"
    assert record.canonical_url == "https://zenodo.org/records/15460715"
    assert record.links[-1].url == adapter.weight_url
    assert record.models[0].aliases == ("chemeleon_mp.pt",)


def test_chemeleon_adapter_rejects_changed_or_unrelated_documentation() -> None:
    doc = b"https://zenodo.org/records/15460715/files/other_file.pt"
    adapter = ChempropCheMeleonCheckpointSourceAdapter(client=QueuedClient(response(doc)))
    with pytest.raises(ValueError, match="no longer lists"):
        adapter.fetch_page({})


def test_chemeleon_adapter_uses_documentation_hash_as_change_token() -> None:
    doc = b"https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
    client = QueuedClient(response(doc))
    adapter = ChempropCheMeleonCheckpointSourceAdapter(client=client)
    from modelome.normalize import content_hash

    page = adapter.fetch_page({"documentation_sha256": content_hash(doc), "model_count": 1})
    assert page.records == ()
    assert page.upstream_count == 1
    assert len(client.calls) == 1
