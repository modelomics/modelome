from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

import modelome.semantic_scholar_materialize as semantic_scholar_materialize
from modelome.lake import LakeRecord, ParquetLandingZone, ShardApplicationOrder, ShardReceipt
from modelome.semantic_scholar_materialize import (
    PROJECTION_SCHEMA,
    MaterializationLimits,
    SemanticScholarProjectionMaterializer,
)

SOURCE = "semantic-scholar"
RELEASE = "2026-08-25"
BASE_RELEASE = "2026-08-18"
NEXT_RELEASE = "2026-09-01"


def limits(**overrides: int) -> MaterializationLimits:
    values = {
        "bucket_count": 4,
        "scan_batch_rows": 1,
        "max_scan_batch_bytes": 1024 * 1024,
        "partition_buffer_rows": 2,
        "partition_buffer_bytes": 1024 * 1024,
        "join_batch_rows": 1,
        "max_join_batch_bytes": 1024 * 1024,
        "max_bucket_rows": 100,
        "max_bucket_payload_bytes": 1024 * 1024,
        "max_payload_json_bytes": 512 * 1024,
        "output_part_rows": 1,
        "max_output_buffer_bytes": 4 * 1024 * 1024,
        "max_output_row_bytes": 2 * 1024 * 1024,
    }
    values.update(overrides)
    return MaterializationLimits(**values)


def source_id(dataset: str, payload: Mapping[str, Any]) -> str:
    if dataset == "paper-ids":
        return f"semantic-scholar:paper-id:{str(payload['sha']).strip().casefold()}"
    return f"semantic-scholar:corpus:{int(payload['corpusid'])}"


def seal_dataset(
    lake: ParquetLandingZone,
    dataset: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    mode: str = "snapshot",
    split_every: int = 2,
    release: str = RELEASE,
    base_release: str = BASE_RELEASE,
) -> tuple[ShardReceipt, ...]:
    values = list(rows)
    chunks = [values[index : index + split_every] for index in range(0, len(values), split_every)]
    if not chunks:
        chunks = [[]]
    receipts = []
    for index, chunk in enumerate(chunks):
        records = [
            LakeRecord(source_record_id=source_id(dataset, payload), payload=payload)
            for payload in chunk
        ]
        if mode == "snapshot":
            order = ShardApplicationOrder.snapshot(index)
        else:
            order = ShardApplicationOrder.diff(
                diff_index=0,
                operation="upsert",
                operation_index=index,
                from_release=base_release,
                to_release=release,
            )
        control = hashlib.sha256(
            f"control:{dataset}:{release}:{mode}:{index}".encode()
        ).hexdigest()
        upstream = hashlib.sha256(
            json.dumps(chunk, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        receipts.append(
            lake.commit_shard(
                source=SOURCE,
                dataset=dataset,
                release=release,
                shard=f"{dataset}-shard-{index}",
                control_sha256=control,
                upstream_sha256=upstream,
                records=records,
                application_order=order,
                expected_rows=len(records),
                batch_rows=1,
            )
        )
    lake.seal_release(
        source=SOURCE,
        dataset=dataset,
        release=release,
        expected_shards={receipt.shard: receipt for receipt in receipts},
    )
    return tuple(receipts)


def seal_complete_snapshot(
    lake: ParquetLandingZone,
    *,
    papers: Iterable[Mapping[str, Any]],
    abstracts: Iterable[Mapping[str, Any]] = (),
    paper_ids: Iterable[Mapping[str, Any]] = (),
    release: str = RELEASE,
) -> dict[str, tuple[ShardReceipt, ...]]:
    return {
        "papers": seal_dataset(lake, "papers", papers, release=release),
        "abstracts": seal_dataset(lake, "abstracts", abstracts, release=release),
        "paper-ids": seal_dataset(lake, "paper-ids", paper_ids, release=release),
    }


def seal_diff_dataset(
    lake: ParquetLandingZone,
    dataset: str,
    transitions: Iterable[
        tuple[
            str,
            str,
            Iterable[Mapping[str, Any]],
            Iterable[Mapping[str, Any]],
        ]
    ],
) -> tuple[ShardReceipt, ...]:
    values = list(transitions)
    target_release = values[-1][1]
    receipts: list[ShardReceipt] = []
    for diff_index, (start, end, raw_upserts, raw_deletes) in enumerate(values):
        upserts = list(raw_upserts)
        deletes = list(raw_deletes)
        operations = [("upsert", upserts)]
        if deletes:
            operations.append(("delete", deletes))
        for operation, rows in operations:
            records = [
                LakeRecord(
                    source_record_id=source_id(dataset, payload),
                    payload=payload,
                    operation=operation,
                )
                for payload in rows
            ]
            control = hashlib.sha256(
                f"control:{dataset}:{target_release}:{diff_index}:{operation}".encode()
            ).hexdigest()
            upstream = hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            receipt = lake.commit_shard(
                source=SOURCE,
                dataset=dataset,
                release=target_release,
                shard=f"{dataset}-{diff_index}-{operation}",
                control_sha256=control,
                upstream_sha256=upstream,
                records=records,
                application_order=ShardApplicationOrder.diff(
                    diff_index=diff_index,
                    operation=operation,
                    operation_index=0,
                    from_release=start,
                    to_release=end,
                ),
                expected_rows=len(records),
                batch_rows=1,
            )
            receipts.append(receipt)
    lake.seal_release(
        source=SOURCE,
        dataset=dataset,
        release=target_release,
        expected_shards={receipt.shard: receipt for receipt in receipts},
    )
    return tuple(receipts)


def seal_diff_release(
    lake: ParquetLandingZone,
    *,
    base_release: str,
    target_release: str,
    changes: Mapping[
        str,
        tuple[Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
    ],
) -> None:
    for dataset in ("papers", "abstracts", "paper-ids"):
        upserts, deletes = changes.get(dataset, ((), ()))
        seal_diff_dataset(
            lake,
            dataset,
            [(base_release, target_release, upserts, deletes)],
        )


def projection_rows(materializer, receipt) -> list[dict[str, Any]]:
    return [
        row
        for batch in materializer.iter_batches(receipt, batch_size=1)
        for row in batch.to_pylist()
    ]


def test_snapshot_materializer_joins_bounded_hash_buckets_and_is_idempotent(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    papers = [
        {
            "corpusid": corpus_id,
            "title": f"Paper {corpus_id}",
            "externalids": {
                "arbitrary-provider-id": f"provider:{corpus_id}",
                "DOI": f"10.1000/{corpus_id}",
                **(
                    {
                        "ArXiv": "2401.01234v2",
                        "ACL": "2024.acl-long.123",
                        "PubMed": "12345678",
                        "PubMedCentral": "PMC2345678",
                    }
                    if corpus_id == 1
                    else {"PubMed": "not-a-pmid"} if corpus_id == 2 else {}
                ),
            },
            "authors": [{"authorId": f"a-{corpus_id}", "name": "Researcher"}],
            "venue": "A venue",
            "year": 2025,
            "publicationdate": "2025-01-02",
            "url": f"https://papers.example/{corpus_id}",
        }
        for corpus_id in range(1, 9)
    ]
    abstracts = [
        {
            "corpusid": corpus_id,
            "abstract": f"Abstract {corpus_id}",
            "openaccessinfo": {"url": f"https://fulltext.example/{corpus_id}"},
        }
        for corpus_id in range(1, 9)
    ]
    aliases = [
        {"corpusid": 1, "sha": "a" * 40, "primary": True},
        {"corpusid": 1, "sha": "b" * 40, "primary": False},
        {"corpusid": 7, "sha": "c" * 40, "primary": True},
    ]
    seal_complete_snapshot(
        lake,
        papers=papers,
        abstracts=abstracts,
        paper_ids=aliases,
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )

    first = materializer.materialize(RELEASE)
    rows = projection_rows(materializer, first)
    repeated = materializer.materialize(RELEASE)

    assert first.row_count == 8
    assert first.part_count == 8
    assert len(rows) == 8
    assert repeated.path == first.path
    assert repeated.artifact_id == first.artifact_id
    assert repeated.already_materialized is True
    assert len(list(first.path.rglob("*.parquet"))) == 8
    assert len({path.parent for path in first.path.rglob("*.parquet")}) > 1
    assert all(
        pq.read_schema(path) == PROJECTION_SCHEMA
        for path in first.path.rglob("*.parquet")
    )

    first_row = next(row for row in rows if row["corpus_id"] == "1")
    assert first_row["source_record_id"] == "semantic-scholar:corpus:1"
    assert first_row["abstract"] == "Abstract 1"
    assert first_row["text"] == "Abstract 1"
    assert first_row["sha_aliases"] == ["a" * 40, "b" * 40]
    assert json.loads(first_row["external_ids_json"])["arbitrary-provider-id"] == "provider:1"
    assert json.loads(first_row["authors_json"])[0]["name"] == "Researcher"
    assert first_row["canonical_url"] == "https://papers.example/1"
    urls = json.loads(first_row["urls_json"])
    assert {item["url"] for item in urls} == {
        "https://papers.example/1",
        "https://fulltext.example/1",
        "https://arxiv.org/abs/2401.01234v2",
        "https://doi.org/10.1000/1",
        "https://aclanthology.org/2024.acl-long.123/",
        "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC2345678/",
    }
    assert {
        item["locator"]: item["url"]
        for item in urls
        if item["locator"].startswith("$.paper.externalids.")
    } == {
        "$.paper.externalids.ArXiv": "https://arxiv.org/abs/2401.01234v2",
        "$.paper.externalids.DOI": "https://doi.org/10.1000/1",
        "$.paper.externalids.ACL": "https://aclanthology.org/2024.acl-long.123/",
        "$.paper.externalids.PubMed": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "$.paper.externalids.PubMedCentral": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2345678/",
    }
    second_row = next(row for row in rows if row["corpus_id"] == "2")
    assert all(
        "pubmed.ncbi.nlm.nih.gov" not in item["url"]
        for item in json.loads(second_row["urls_json"])
    )
    raw = json.loads(first_row["raw_payload_json"])
    assert raw["paper"]["corpusid"] == 1
    assert len(raw["paper_ids"]) == 2
    evidence = json.loads(first_row["evidence_json"])
    assert evidence["projection_artifact_id"] == first.artifact_id
    assert [item["dataset"] for item in evidence["inputs"]] == [
        "papers",
        "abstracts",
        "paper-ids",
        "paper-ids",
    ]
    assert {item["raw_payload_locator"] for item in evidence["inputs"]} == {
        "$.paper",
        "$.abstract",
        "$.paper_ids[0]",
        "$.paper_ids[1]",
    }
    assert list((tmp_path / "projection" / ".staging").iterdir()) == []


def test_missing_or_unsealed_required_dataset_fails_closed(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_dataset(lake, "papers", [{"corpusid": 1, "title": "One"}])
    seal_dataset(lake, "abstracts", [{"corpusid": 1, "abstract": "Text"}])
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    assert materializer.list_ready_releases() == ()
    assert materializer.latest_ready_release() is None

    with pytest.raises(ValueError, match="release is not sealed"):
        materializer.materialize(RELEASE)

    assert list((tmp_path / "projection" / ".staging").iterdir()) == []


def test_diff_release_requires_an_explicit_base_projection(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_dataset(
        lake,
        "papers",
        [{"corpusid": 1, "title": "One"}],
        mode="diff",
    )
    seal_dataset(lake, "abstracts", [], mode="diff")
    seal_dataset(lake, "paper-ids", [], mode="diff")
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )

    with pytest.raises(ValueError, match="requires a base projection"):
        materializer.materialize(RELEASE)


def test_diff_replay_updates_children_emits_tombstone_and_is_idempotent(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    old_sha = "1" * 40
    new_sha = "2" * 40
    third_sha = "3" * 40
    seal_complete_snapshot(
        lake,
        release=BASE_RELEASE,
        papers=[
            {"corpusid": 1, "title": "Old one", "externalids": {"DOI": "old"}},
            {"corpusid": 2, "title": "Delete me"},
        ],
        abstracts=[
            {"corpusid": 1, "abstract": "Old abstract"},
            {"corpusid": 2, "abstract": "Deleted abstract"},
        ],
        paper_ids=[{"corpusid": 1, "sha": old_sha}],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    base = materializer.materialize(BASE_RELEASE)
    # A later daily process reopens the sealed base; it does not depend on the
    # in-memory receipt returned by the snapshot process.
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    assert materializer.list_projections(BASE_RELEASE) == (
        materializer.open_projection(BASE_RELEASE, artifact_id=base.artifact_id),
    )
    base = materializer.open_projection(BASE_RELEASE)
    assert materializer.list_ready_releases() == (BASE_RELEASE,)
    assert materializer.latest_ready_release() == BASE_RELEASE
    seal_diff_release(
        lake,
        base_release=BASE_RELEASE,
        target_release=RELEASE,
        changes={
            "papers": (
                [
                    {
                        "corpusid": 1,
                        "title": "Updated one",
                        "externalids": {"DOI": "new"},
                    },
                    {"corpusid": 3, "title": "New three"},
                ],
                [{"corpusid": 2}],
            ),
            "abstracts": (
                [
                    {"corpusid": 1, "abstract": "Updated abstract"},
                    {"corpusid": 3, "abstract": "New abstract"},
                ],
                [{"corpusid": 2}],
            ),
            "paper-ids": (
                [
                    {"corpusid": 1, "sha": new_sha},
                    {"corpusid": 3, "sha": third_sha},
                ],
                [{"corpusid": 1, "sha": old_sha}],
            ),
        },
    )

    target = materializer.materialize(RELEASE, base=base)
    plan = materializer.plan(RELEASE)
    rows = projection_rows(materializer, target)
    repeated = materializer.materialize(RELEASE, base=base)

    assert repeated.path == target.path
    assert materializer.list_ready_releases() == (BASE_RELEASE, RELEASE)
    assert materializer.latest_ready_release() == RELEASE
    assert repeated.already_materialized is True
    assert target.artifact_id != base.artifact_id
    assert plan.mode == "diff"
    assert plan.base_release == BASE_RELEASE
    assert plan.transitions == ((0, BASE_RELEASE, RELEASE),)
    assert {dataset for dataset, _digest in plan.input_release_sha256} == {
        "papers",
        "abstracts",
        "paper-ids",
    }
    assert target.row_count == 3
    by_id = {row["corpus_id"]: row for row in rows}
    assert by_id["1"]["title"] == "Updated one"
    assert by_id["1"]["text"] == "Updated abstract"
    assert by_id["1"]["sha_aliases"] == [new_sha]
    assert by_id["3"]["title"] == "New three"
    assert by_id["3"]["sha_aliases"] == [third_sha]
    assert by_id["2"]["operation"] == "delete"
    assert by_id["2"]["tombstone"] is True
    assert by_id["2"]["title"] is None
    assert by_id["2"]["sha_aliases"] == []
    tombstone_raw = json.loads(by_id["2"]["raw_payload_json"])
    assert tombstone_raw["tombstone"] == {"corpusid": 2}
    assert [event["operation"] for event in tombstone_raw["applied_events"]] == [
        "delete",
        "delete",
    ]
    manifest = json.loads((target.path / "manifest.json").read_text())
    assert manifest["projection_mode"] == "diff"
    assert manifest["lineage"] == {
        "base_artifact_id": base.artifact_id,
        "base_release": BASE_RELEASE,
        "target_release": RELEASE,
        "transitions": [
            {
                "diff_index": 0,
                "from_release": BASE_RELEASE,
                "to_release": RELEASE,
            }
        ],
    }


def test_diff_replay_reactivates_a_prior_tombstone(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        release=BASE_RELEASE,
        papers=[{"corpusid": 4, "title": "Original"}],
        abstracts=[{"corpusid": 4, "abstract": "Original abstract"}],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    snapshot = materializer.materialize(BASE_RELEASE)
    seal_diff_release(
        lake,
        base_release=BASE_RELEASE,
        target_release=RELEASE,
        changes={"papers": ((), [{"corpusid": 4}])},
    )
    deleted = materializer.materialize(RELEASE, base=snapshot)
    assert projection_rows(materializer, deleted)[0]["tombstone"] is True

    seal_diff_release(
        lake,
        base_release=RELEASE,
        target_release=NEXT_RELEASE,
        changes={
            "papers": ([{"corpusid": 4, "title": "Reactivated"}], ()),
            "abstracts": ([{"corpusid": 4, "abstract": "New abstract"}], ()),
        },
    )
    reactivated = materializer.materialize(NEXT_RELEASE, base=deleted)
    row = projection_rows(materializer, reactivated)[0]

    assert row["operation"] == "upsert"
    assert row["tombstone"] is False
    assert row["title"] == "Reactivated"
    assert row["text"] == "New abstract"
    manifest = json.loads((reactivated.path / "manifest.json").read_text())
    assert manifest["lineage"]["base_artifact_id"] == deleted.artifact_id
    assert manifest["lineage"]["base_release"] == RELEASE


def test_diff_dataset_transition_chains_must_match_exactly(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        release=BASE_RELEASE,
        papers=[{"corpusid": 1, "title": "One"}],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    base = materializer.materialize(BASE_RELEASE)
    seal_diff_dataset(
        lake,
        "papers",
        [
            (BASE_RELEASE, RELEASE, (), ()),
            (RELEASE, NEXT_RELEASE, (), ()),
        ],
    )
    seal_diff_dataset(
        lake,
        "abstracts",
        [(BASE_RELEASE, NEXT_RELEASE, (), ())],
    )
    seal_diff_dataset(
        lake,
        "paper-ids",
        [(BASE_RELEASE, NEXT_RELEASE, (), ())],
    )

    with pytest.raises(ValueError, match="transition chains disagree"):
        materializer.materialize(NEXT_RELEASE, base=base)


def test_diff_base_release_must_match_first_transition(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        release=BASE_RELEASE,
        papers=[{"corpusid": 1, "title": "One"}],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    base = materializer.materialize(BASE_RELEASE)
    seal_diff_release(
        lake,
        base_release="2026-08-11",
        target_release=RELEASE,
        changes={},
    )

    with pytest.raises(ValueError, match="does not match the first transition"):
        materializer.materialize(RELEASE, base=base)


def test_diff_delete_of_missing_child_fails_and_cleans_stage(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        release=BASE_RELEASE,
        papers=[{"corpusid": 1, "title": "One"}],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    base = materializer.materialize(BASE_RELEASE)
    seal_diff_release(
        lake,
        base_release=BASE_RELEASE,
        target_release=RELEASE,
        changes={"abstracts": ((), [{"corpusid": 1}])},
    )

    with pytest.raises(ValueError, match="abstract delete references missing"):
        materializer.materialize(RELEASE, base=base)

    assert list((tmp_path / "projection" / ".staging").iterdir()) == []


def test_contradictory_diff_upserts_in_one_transition_fail_closed(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        release=BASE_RELEASE,
        papers=[{"corpusid": 1, "title": "One"}],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    base = materializer.materialize(BASE_RELEASE)
    seal_diff_release(
        lake,
        base_release=BASE_RELEASE,
        target_release=RELEASE,
        changes={
            "papers": (
                [
                    {"corpusid": 1, "title": "First update"},
                    {"corpusid": 1, "title": "Contradictory update"},
                ],
                (),
            )
        },
    )

    with pytest.raises(ValueError, match="contradictory upsert events"):
        materializer.materialize(RELEASE, base=base)


def test_tampered_base_projection_is_rejected_before_diff_replay(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        release=BASE_RELEASE,
        papers=[{"corpusid": 1, "title": "One"}],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    base = materializer.materialize(BASE_RELEASE)
    seal_diff_release(
        lake,
        base_release=BASE_RELEASE,
        target_release=RELEASE,
        changes={},
    )
    base_part = next(base.path.rglob("*.parquet"))
    base_part.write_bytes(base_part.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="byte count mismatch"):
        materializer.materialize(RELEASE, base=base)


def test_orphan_abstract_and_sha_alias_rows_fail_without_paper_anchor(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        papers=[{"corpusid": 1, "title": "One"}],
        abstracts=[{"corpusid": 2, "abstract": "Orphan"}],
        paper_ids=[{"corpusid": 3, "sha": "d" * 40}],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )

    with pytest.raises(ValueError, match="no paper anchor"):
        materializer.materialize(RELEASE)

    assert list((tmp_path / "projection" / ".staging").iterdir()) == []


def test_contradictory_duplicate_papers_fail_closed(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        papers=[
            {"corpusid": 5, "title": "First"},
            {"corpusid": 5, "title": "Contradiction"},
        ],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )

    with pytest.raises(ValueError, match="contradictory duplicate"):
        materializer.materialize(RELEASE)


def test_contradictory_sha_ownership_fails_closed(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    shared_sha = "e" * 40
    seal_complete_snapshot(
        lake,
        papers=[
            {"corpusid": 1, "title": "One"},
            {"corpusid": 2, "title": "Two"},
        ],
        paper_ids=[
            {"corpusid": 1, "sha": shared_sha},
            {"corpusid": 2, "sha": shared_sha},
        ],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        # Corpus IDs 1 and 2 land in different corpus buckets. The second
        # external-memory SHA pass must still detect their conflicting owner.
        limits=limits(bucket_count=4),
    )

    with pytest.raises(ValueError, match="contradictory duplicate|contradictory corpus"):
        materializer.materialize(RELEASE)


def test_malformed_core_types_fail_closed_and_stage_is_cleaned(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        papers=[{"corpusid": 1, "title": ["not", "text"]}],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )

    with pytest.raises(ValueError, match="paper title must be text"):
        materializer.materialize(RELEASE)

    assert list((tmp_path / "projection" / ".staging").iterdir()) == []


def test_bucket_memory_ceiling_fails_closed_and_cleans_stage(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        papers=[
            {"corpusid": 1, "title": "One"},
            {"corpusid": 2, "title": "Two"},
        ],
        abstracts=[
            {"corpusid": 1, "abstract": "One abstract"},
            {"corpusid": 2, "abstract": "Two abstract"},
        ],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(bucket_count=1, max_bucket_rows=3),
    )

    with pytest.raises(ValueError, match="exceeded max_bucket_rows"):
        materializer.materialize(RELEASE)

    assert list((tmp_path / "projection" / ".staging").iterdir()) == []


def test_projection_and_input_checksum_tampering_are_detected(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    receipts = seal_complete_snapshot(
        lake,
        papers=[{"corpusid": 1, "title": "One"}],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    projection = materializer.materialize(RELEASE)
    projection_part = next(projection.path.rglob("*.parquet"))
    projection_part.write_bytes(projection_part.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="byte count mismatch"):
        materializer.materialize(RELEASE)

    # Restore the projection so the next assertion reaches input verification.
    projection_part.write_bytes(projection_part.read_bytes()[: -len(b"tampered")])
    input_part = next((receipts["papers"][0].path / "parts").glob("*.parquet"))
    input_part.write_bytes(input_part.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="byte count mismatch"):
        materializer.materialize(RELEASE)


def test_unexpected_projection_path_entry_is_rejected(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(lake, papers=[{"corpusid": 1, "title": "One"}])
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )
    projection = materializer.materialize(RELEASE)
    (projection.path / "unexpected.txt").write_text("path tampering")

    with pytest.raises(ValueError, match="unexpected entries"):
        materializer.materialize(RELEASE)


def test_v5_projection_rematerializes_release_with_sealed_v4_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    seal_complete_snapshot(
        lake,
        papers=[
            {
                "corpusid": 1,
                "title": "Model paper",
                "externalids": {"ArXiv": "2401.01234"},
            }
        ],
    )
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        output_root=tmp_path / "projection",
        limits=limits(),
    )

    monkeypatch.setattr(
        semantic_scholar_materialize,
        "_ALGORITHM",
        "sha256-corpus-bucket-stateful-join-v4",
    )
    old = materializer.materialize(RELEASE)
    monkeypatch.setattr(
        semantic_scholar_materialize,
        "_ALGORITHM",
        "sha256-corpus-bucket-stateful-join-v5",
    )

    assert materializer.list_projections(RELEASE) == ()
    current = materializer.materialize(RELEASE)
    assert current.artifact_id != old.artifact_id
    listed = materializer.list_projections(RELEASE)
    assert [receipt.artifact_id for receipt in listed] == [current.artifact_id]
    row = next(materializer.iter_batches(current, batch_size=1)).to_pylist()[0]
    assert json.loads(row["urls_json"])[0]["url"] == "https://arxiv.org/abs/2401.01234"
