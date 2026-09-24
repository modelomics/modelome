from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.normalize import content_hash
from modelome.sources.zatom_checkpoint_registry import ZatomCheckpointRegistrySourceAdapter
from modelome.sources.zatom_zenodo_checkpoint_record import (
    _API_URL,
    _README_DECLARED_FILES,
    ZatomZenodoCheckpointRecordSourceAdapter,
)

EXTRAS = {
    "platom_1_joint_pretraining_paper_weights.ckpt": "md5:51858170267bdb7b29f294e194b7b467",
    "zatom_1_from_omol25_mol_prop_pred_paper_weights.ckpt": (
        "md5:2e6bca003814e438f2e4ff3b2a32eccf"
    ),
}


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def payload() -> bytes:
    files = []
    for filename in sorted(_README_DECLARED_FILES | EXTRAS.keys()):
        checksum = EXTRAS.get(filename, "md5:" + "a" * 32)
        files.append(
            {
                "key": filename,
                "checksum": checksum,
                "links": {
                    "self": (
                        f"https://zenodo.org/api/records/19766997/files/{filename}/content"
                    )
                },
            }
        )
    files.append(
        {"key": "zatom_1_materials_mp20.zip", "checksum": "md5:" + "b" * 32, "links": {}}
    )
    return json.dumps({"id": 19766997, "files": files}).encode()


def response(body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, _API_URL)


def adapter(client: QueueClient) -> ZatomZenodoCheckpointRecordSourceAdapter:
    return ZatomZenodoCheckpointRecordSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )


def test_indexes_all_26_record_files_and_preserves_file_metadata() -> None:
    body = payload()
    client = QueueClient(response(body))

    page = adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 26
    assert client.calls == [_API_URL]
    records = {record.raw["zenodo_file_key"]: record for record in page.records}
    assert len(records) == 26
    assert all(record.kind is ArtifactKind.WEIGHTS for record in records.values())
    assert records["platom_1_joint_pretraining_paper_weights.ckpt"].raw["checksum"] == EXTRAS[
        "platom_1_joint_pretraining_paper_weights.ckpt"
    ]
    record_only = records["zatom_1_from_omol25_mol_prop_pred_paper_weights.ckpt"]
    assert record_only.raw["readme_coverage"] == (
        "record_only_not_in_readme_checkpoint_section"
    )
    declared = records["zatom_1_joint_paper_weights.ckpt"]
    assert declared.raw["readme_coverage"] == "declared_in_readme_checkpoint_section"
    assert declared.raw["checkpoint_url"] == (
        "https://zenodo.org/api/records/19766997/files/"
        "zatom_1_joint_paper_weights.ckpt/content"
    )


def test_unchanged_record_metadata_skips_emitting_duplicate_rows() -> None:
    body = payload()
    client = QueueClient(response(body))

    page = adapter(client).fetch_page({"record_sha256": content_hash(body), "model_count": 26})

    assert page.records == ()
    assert page.upstream_count == 26


def test_entry_assembly_joins_24_readme_pairs_and_keeps_two_record_only_files() -> None:
    page = adapter(QueueClient(response(payload()))).fetch_page({})
    readme_adapter = ZatomCheckpointRegistrySourceAdapter(client=QueueClient())
    readme_seeds = [
        source_record_to_entry_seed(
            readme_adapter._record(
                filename,
                f"https://zenodo.org/records/19766997/files/{filename}",
                "c" * 40,
                b"README fixture",
            ),
            source="zatom-checkpoints",
        )
        for filename in _README_DECLARED_FILES
    ]
    zenodo_seeds = [
        source_record_to_entry_seed(record, source="zatom-zenodo-checkpoint-record")
        for record in page.records
    ]

    assembled = build_entries([*readme_seeds, *zenodo_seeds])

    assert assembled.candidate_count == 50
    assert len(assembled.entries) == 26
    assert sum(len(entry.members) == 2 for entry in assembled.entries) == 24
    assert sum(len(entry.members) == 1 for entry in assembled.entries) == 2
    for filename in EXTRAS:
        entry = next(
            entry
            for entry in assembled.entries
            if any(
                identifier.value == filename.removesuffix(".ckpt")
                for identifier in entry.identifiers
            )
        )
        assert len(entry.members) == 1
        assert entry.members[0].source == "zatom-zenodo-checkpoint-record"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update(id=1),
        lambda data: data["files"].append(data["files"][0]),
        lambda data: data["files"][0].update(checksum="md5:abc"),
        lambda data: data["files"][0]["links"].update(self="https://evil.example/file.ckpt"),
    ],
)
def test_rejects_record_identity_duplicates_bad_checksums_and_changed_urls(mutate: Any) -> None:
    data = json.loads(payload())
    mutate(data)
    client = QueueClient(response(json.dumps(data).encode()))

    with pytest.raises(ValueError):
        adapter(client).fetch_page({})


def test_live_zenodo_metadata_smoke() -> None:
    """Read only Zenodo's JSON file metadata; never request a checkpoint URL."""
    try:
        page = ZatomZenodoCheckpointRecordSourceAdapter().fetch_page({})
    except Exception as error:  # pragma: no cover - network-dependent smoke
        pytest.skip(f"live Zenodo checkpoint metadata unavailable: {error}")
    assert page.upstream_count == 26
    record_only = {
        record.raw["zenodo_file_key"] for record in page.records
    } - _README_DECLARED_FILES
    assert record_only == set(EXTRAS)
    extra_checksums = {
        record.raw["checksum"]
        for record in page.records
        if record.raw["zenodo_file_key"] in EXTRAS
    }
    assert extra_checksums == set(EXTRAS.values())
