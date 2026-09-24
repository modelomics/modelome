from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from modelome.lake import LakeRecord, ParquetLandingZone, ShardApplicationOrder
from modelome.semantic_scholar_citation_materialize import (
    CitationProjectionLimits,
    SemanticScholarCitationMaterializer,
)
from modelome.semantic_scholar_citation_state import (
    CitationStateLimits,
    SemanticScholarCitationStateMaterializer,
)

SOURCE = "semantic-scholar"
BASE = "2026-09-10"
RELEASE = "2026-09-17"


def _edge_id(payload):
    return (
        f"semantic-scholar:citation:{int(payload['citingPaperId'])}:"
        f"{int(payload['citedPaperId'])}"
    )


def _seal_citation_snapshot(lake, payloads, *, release=RELEASE):
    receipts = []
    for index, payload in enumerate(payloads):
        receipt = lake.commit_shard(
            source=SOURCE,
            dataset="citations",
            release=release,
            shard=f"citations-{index}",
            control_sha256=hashlib.sha256(f"control-{index}".encode()).hexdigest(),
            upstream_sha256=hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
            records=[LakeRecord(source_record_id=_edge_id(payload), payload=payload)],
            application_order=ShardApplicationOrder.snapshot(index),
            expected_rows=1,
            batch_rows=1,
        )
        receipts.append(receipt)
    lake.seal_release(
        source=SOURCE,
        dataset="citations",
        release=release,
        expected_shards={item.shard: item for item in receipts},
    )


def _projected_rows(receipt) -> list[dict[str, Any]]:
    manifest = json.loads((receipt.path / "manifest.json").read_text())
    rows = []
    for part in manifest["parts"]:
        rows.extend(pq.read_table(receipt.path / "parts" / part["name"]).to_pylist())
    return rows


def test_snapshot_projection_streams_exact_paper_edges_with_source_provenance(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    payloads = [
        {"citingPaperId": 101, "citedPaperId": 202, "contexts": ["uses"]},
        {"citingPaperId": "303", "citedPaperId": 404, "intents": ["background"]},
    ]
    _seal_citation_snapshot(lake, payloads)
    materializer = SemanticScholarCitationMaterializer(
        lake,
        output_root=tmp_path / "projections",
        limits=CitationProjectionLimits(output_part_rows=1),
    )

    receipt = materializer.materialize(RELEASE)
    rows = _projected_rows(receipt)
    assert [(row["citing_paper_id"], row["cited_paper_id"]) for row in rows] == [
        ("101", "202"),
        ("303", "404"),
    ]
    assert [row["operation"] for row in rows] == ["upsert", "upsert"]
    assert [row["event_index"] for row in rows] == [0, 1]
    assert [row["source_row_index"] for row in rows] == [0, 0]
    assert rows[0]["source_shard"] == "citations-0"
    assert json.loads(rows[0]["application_order_json"]) == {
        "mode": "snapshot",
        "manifest_index": 0,
    }
    assert json.loads(rows[0]["payload_json"]) == payloads[0]
    assert receipt.event_count == 2
    assert receipt.part_count == 2
    assert materializer.materialize(RELEASE).already_materialized is True


def test_diff_projection_retains_both_edge_events_in_application_order(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    edge = {"citingPaperId": 101, "citedPaperId": 202}
    receipts = []
    for operation in ("upsert", "delete"):
        receipts.append(
            lake.commit_shard(
                source=SOURCE,
                dataset="citations",
                release=RELEASE,
                shard=f"citations-diff-{operation}",
                control_sha256=hashlib.sha256(
                    f"control-{operation}".encode()
                ).hexdigest(),
                upstream_sha256=hashlib.sha256(operation.encode()).hexdigest(),
                records=[
                    LakeRecord(
                        source_record_id=_edge_id(edge),
                        payload=edge,
                        operation=operation,
                    )
                ],
                application_order=ShardApplicationOrder.diff(
                    diff_index=0,
                    operation=operation,
                    operation_index=0,
                    from_release=BASE,
                    to_release=RELEASE,
                ),
                expected_rows=1,
                batch_rows=1,
            )
        )
    lake.seal_release(
        source=SOURCE,
        dataset="citations",
        release=RELEASE,
        expected_shards={item.shard: item for item in receipts},
    )

    receipt = SemanticScholarCitationMaterializer(
        lake, output_root=tmp_path / "projections"
    ).materialize(RELEASE)
    rows = _projected_rows(receipt)
    assert [row["operation"] for row in rows] == ["upsert", "delete"]
    assert [row["event_index"] for row in rows] == [0, 1]
    assert [json.loads(row["application_order_json"])["operation"] for row in rows] == [
        "upsert",
        "delete",
    ]
    assert rows[0]["source_record_id"] == rows[1]["source_record_id"]


def test_projection_rejects_rows_whose_source_identity_does_not_match_exact_ids(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    payload = {"citingPaperId": 101, "citedPaperId": 202}
    receipt = lake.commit_shard(
        source=SOURCE,
        dataset="citations",
        release=RELEASE,
        shard="citations-0",
        control_sha256=hashlib.sha256(b"control").hexdigest(),
        upstream_sha256=hashlib.sha256(b"upstream").hexdigest(),
        records=[LakeRecord(source_record_id="wrong-edge-id", payload=payload)],
        application_order=ShardApplicationOrder.snapshot(0),
        expected_rows=1,
        batch_rows=1,
    )
    lake.seal_release(
        source=SOURCE,
        dataset="citations",
        release=RELEASE,
        expected_shards={receipt.shard: receipt},
    )
    with pytest.raises(ValueError, match="does not match its edge IDs"):
        SemanticScholarCitationMaterializer(
            lake, output_root=tmp_path / "projections"
        ).materialize(RELEASE)


def test_active_snapshot_contains_exact_edges_and_is_queryable_without_tombstones(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    _seal_citation_snapshot(
        lake,
        [
            {"citingPaperId": 101, "citedPaperId": 202, "contexts": ["use"]},
            {"citingPaperId": 303, "citedPaperId": 404},
        ],
    )
    event_projection = SemanticScholarCitationMaterializer(
        lake, output_root=tmp_path / "events"
    ).materialize(RELEASE)
    state_materializer = SemanticScholarCitationStateMaterializer(
        lake,
        output_root=tmp_path / "states",
        limits=CitationStateLimits(
            bucket_count=4,
            event_partition_rows=1,
            output_part_rows=1,
            max_bucket_rows=20,
            max_bucket_bytes=1024 * 1024,
        ),
    )

    receipt = state_materializer.materialize(event_projection)
    active_rows = [
        row
        for batch in state_materializer.iter_batches(receipt)
        for row in batch.to_pylist()
    ]
    all_rows = [
        row
        for batch in state_materializer.iter_batches(receipt, include_tombstones=True)
        for row in batch.to_pylist()
    ]
    assert {(row["citing_paper_id"], row["cited_paper_id"]) for row in active_rows} == {
        ("101", "202"),
        ("303", "404"),
    }
    assert len(all_rows) == len(active_rows) == receipt.active_count == 2
    assert all(row["tombstone"] is False for row in all_rows)
    reopened = state_materializer.open_projection(RELEASE, receipt.artifact_id)
    assert reopened.artifact_id == receipt.artifact_id
    assert reopened.path == receipt.path
    assert reopened.active_count == receipt.active_count
    assert reopened.already_materialized is True


def test_active_diff_applies_deletes_as_provenanced_tombstones_and_requires_base(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    base_release = "2026-09-10"
    edges = [
        {"citingPaperId": 101, "citedPaperId": 202, "contexts": ["prior"]},
        {"citingPaperId": 303, "citedPaperId": 404},
    ]
    _seal_citation_snapshot(lake, edges, release=base_release)
    source_receipt = SemanticScholarCitationMaterializer(
        lake, output_root=tmp_path / "events"
    ).materialize(base_release)
    state_materializer = SemanticScholarCitationStateMaterializer(
        lake,
        output_root=tmp_path / "states",
        limits=CitationStateLimits(
            bucket_count=4,
            event_partition_rows=1,
            output_part_rows=1,
            max_bucket_rows=20,
            max_bucket_bytes=1024 * 1024,
        ),
    )
    base = state_materializer.materialize(source_receipt)

    target_release = "2026-09-17"
    delete_edge = edges[0]
    diff_receipt = lake.commit_shard(
        source=SOURCE,
        dataset="citations",
        release=target_release,
        shard="citations-delete",
        control_sha256=hashlib.sha256(b"control-delete").hexdigest(),
        upstream_sha256=hashlib.sha256(b"upstream-delete").hexdigest(),
        records=[
            LakeRecord(
                source_record_id=_edge_id(delete_edge),
                payload={"citingPaperId": 101, "citedPaperId": 202},
                operation="delete",
            )
        ],
        application_order=ShardApplicationOrder.diff(
            diff_index=0,
            operation="delete",
            operation_index=0,
            from_release=base_release,
            to_release=target_release,
        ),
        expected_rows=1,
        batch_rows=1,
    )
    lake.seal_release(
        source=SOURCE,
        dataset="citations",
        release=target_release,
        expected_shards={diff_receipt.shard: diff_receipt},
    )
    diff_events = SemanticScholarCitationMaterializer(
        lake, output_root=tmp_path / "events"
    ).materialize(target_release)

    with pytest.raises(ValueError, match="requires a base"):
        state_materializer.materialize(diff_events)
    target = state_materializer.materialize(diff_events, base=base)
    all_rows = [
        row
        for batch in state_materializer.iter_batches(target, include_tombstones=True)
        for row in batch.to_pylist()
    ]
    assert target.active_count == 1
    assert target.deleted_count == 1
    deleted = next(row for row in all_rows if row["tombstone"])
    active = next(row for row in all_rows if not row["tombstone"])
    assert (deleted["citing_paper_id"], deleted["cited_paper_id"]) == ("101", "202")
    assert deleted["last_operation"] == "delete"
    assert deleted["event_ledger_artifact_id"] == diff_events.artifact_id
    assert json.loads(deleted["application_order_json"])["from_release"] == base_release
    assert (active["citing_paper_id"], active["cited_paper_id"]) == ("303", "404")
