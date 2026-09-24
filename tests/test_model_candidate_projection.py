from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from modelome.lake import LakeRecord, ParquetLandingZone, ShardApplicationOrder
from modelome.model_candidate_projection import (
    MODEL_CANDIDATE_SCHEMA,
    CandidateProjectionLimits,
    SemanticScholarModelCandidateProjector,
)
from modelome.semantic_scholar_materialize import (
    MaterializationLimits,
    ProjectionReceipt,
    SemanticScholarProjectionMaterializer,
)

SOURCE = "semantic-scholar"
RELEASE = "2026-09-01"


def _source_id(dataset: str, payload: Mapping[str, Any]) -> str:
    if dataset == "paper-ids":
        return f"semantic-scholar:paper-id:{payload['sha']}"
    return f"semantic-scholar:corpus:{payload['corpusid']}"


def _seal_dataset(
    lake: ParquetLandingZone,
    dataset: str,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    values = list(rows)
    serialized = json.dumps(values, sort_keys=True, separators=(",", ":"))
    receipt = lake.commit_shard(
        source=SOURCE,
        dataset=dataset,
        release=RELEASE,
        shard=f"{dataset}-0000",
        control_sha256=hashlib.sha256(f"control:{dataset}".encode()).hexdigest(),
        upstream_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        records=(
            LakeRecord(source_record_id=_source_id(dataset, row), payload=row)
            for row in values
        ),
        application_order=ShardApplicationOrder.snapshot(0),
        expected_rows=len(values),
        batch_rows=1,
    )
    lake.seal_release(
        source=SOURCE,
        dataset=dataset,
        release=RELEASE,
        expected_shards={receipt.shard: receipt},
    )


def _source_projection(tmp_path: Path, *, long_text: str | None = None) -> tuple[
    SemanticScholarProjectionMaterializer,
    ProjectionReceipt,
]:
    lake = ParquetLandingZone(tmp_path / "lake")
    papers = [
        {
            "corpusid": 1,
            "title": "A graph method",
            "externalids": {"DOI": "10.1000/one"},
            "url": "https://papers.example/one",
            "publicationdate": "2025-03-04",
        },
        {
            "corpusid": 2,
            "title": "Cell Oracle Engine: A deep-learning model for cell states",
            "externalids": {"PubMed": "12345"},
        },
        {"corpusid": 3, "title": "Bayesian kinetics"},
        {"corpusid": 4, "title": "A measurement dataset"},
    ]
    abstracts = [
        {
            "corpusid": 1,
            "abstract": long_text
            or (
                "We introduce IonWeaver-7, a deep neural network for graph inference. "
                "Its code is evaluated separately."
            ),
        },
        {"corpusid": 2, "abstract": "We evaluate the learned representation."},
        {
            "corpusid": 3,
            "abstract": (
                "We developed BayesKinetics-X, a hierarchical statistical model "
                "fitted by MCMC."
            ),
        },
        {"corpusid": 4, "abstract": "We release a cohort for evaluation."},
    ]
    _seal_dataset(lake, "papers", papers)
    _seal_dataset(lake, "abstracts", abstracts)
    _seal_dataset(lake, "paper-ids", [])
    materializer = SemanticScholarProjectionMaterializer(
        lake,
        limits=MaterializationLimits(
            bucket_count=4,
            scan_batch_rows=1,
            partition_buffer_rows=2,
            join_batch_rows=1,
            output_part_rows=2,
        ),
    )
    return materializer, materializer.materialize(RELEASE)


def _candidate_rows(
    projector: SemanticScholarModelCandidateProjector,
    receipt,
) -> list[dict[str, Any]]:
    return [
        row
        for batch in projector.iter_batches(receipt, batch_size=1)
        for row in batch.to_pylist()
    ]


def test_projects_candidates_with_exact_source_evidence_and_is_idempotent(
    tmp_path: Path,
) -> None:
    materializer, source = _source_projection(tmp_path)
    projector = SemanticScholarModelCandidateProjector(
        materializer,
        limits=CandidateProjectionLimits(
            input_batch_rows=1,
            output_part_rows=1,
        ),
    )

    first = projector.materialize(source)
    rows = _candidate_rows(projector, first)
    repeated = projector.materialize(source)
    reopened = projector.open_projection(source)

    assert first.row_count == 2
    assert first.part_count == 2
    assert first.document_count == 4
    assert repeated.path == first.path
    assert repeated.artifact_id == first.artifact_id
    assert repeated.already_materialized is True
    assert reopened == repeated
    assert [row["name"] for row in rows] == ["IonWeaver-7", "Cell Oracle Engine"]
    assert all(row["status"] == "candidate" for row in rows)
    assert all(row["source_projection_artifact_id"] == source.artifact_id for row in rows)
    assert len({row["candidate_assertion_id"] for row in rows}) == 2
    assert all(
        pq.read_schema(path) == MODEL_CANDIDATE_SCHEMA
        for path in first.path.glob("parts/*.parquet")
    )

    introduced = rows[0]
    assert introduced["source_record_id"] == "semantic-scholar:corpus:1"
    assert introduced["evidence_field"] == "text"
    assert introduced["evidence_text"] == "IonWeaver-7"
    assert "deep neural network" in introduced["supporting_text"]
    assert json.loads(introduced["external_ids_json"]) == {"DOI": "10.1000/one"}
    assert introduced["document_url"] == "https://papers.example/one"
    source_evidence = json.loads(introduced["source_evidence_json"])
    assert source_evidence["projection_artifact_id"] == source.artifact_id
    assert {item["dataset"] for item in source_evidence["inputs"]} == {
        "papers",
        "abstracts",
    }
    derivation = json.loads(introduced["derivation_json"])
    assert derivation["source_projection"]["row_ordinal"] == introduced[
        "source_projection_row_ordinal"
    ]
    assert derivation["evidence"]["text"] == "IonWeaver-7"

    manifest = json.loads((first.path / "manifest.json").read_text())
    assert manifest["stats"] == {
        "active_document_count": 4,
        "candidate_document_count": 2,
        "candidate_row_count": 2,
        "documents_without_candidates": 2,
        "input_row_count": 4,
        "tombstone_document_count": 0,
    }
    assert list((projector.output_root / ".staging").iterdir()) == []


def test_candidate_assertion_ids_do_not_depend_on_parquet_part_layout(
    tmp_path: Path,
) -> None:
    materializer, source = _source_projection(tmp_path)
    one_per_part = SemanticScholarModelCandidateProjector(
        materializer,
        limits=CandidateProjectionLimits(output_part_rows=1),
    )
    many_per_part = SemanticScholarModelCandidateProjector(
        materializer,
        limits=CandidateProjectionLimits(output_part_rows=100),
    )

    first = one_per_part.materialize(source)
    second = many_per_part.materialize(source)

    assert first.artifact_id != second.artifact_id
    assert {
        row["candidate_assertion_id"] for row in _candidate_rows(one_per_part, first)
    } == {
        row["candidate_assertion_id"] for row in _candidate_rows(many_per_part, second)
    }


def test_document_limit_fails_without_publishing_partial_output(tmp_path: Path) -> None:
    materializer, source = _source_projection(
        tmp_path,
        long_text=(
            "We introduce BoundaryLearner-4, a deep neural network. " + "x" * 100
        ),
    )
    projector = SemanticScholarModelCandidateProjector(
        materializer,
        limits=CandidateProjectionLimits(max_document_text_bytes=64),
    )

    with pytest.raises(ValueError, match="max_document_text_bytes"):
        projector.materialize(source)

    assert list((projector.output_root / ".staging").iterdir()) == []
    assert not list(projector.output_root.rglob("manifest.json"))


def test_repeated_materialization_detects_a_tampered_part(tmp_path: Path) -> None:
    materializer, source = _source_projection(tmp_path)
    projector = SemanticScholarModelCandidateProjector(materializer)
    receipt = projector.materialize(source)
    part = next((receipt.path / "parts").glob("*.parquet"))
    with part.open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(ValueError, match="checksum mismatch"):
        projector.materialize(source)


def test_limits_reject_boolean_and_incoherent_values() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CandidateProjectionLimits(input_batch_rows=True)
    with pytest.raises(ValueError, match="must not exceed"):
        CandidateProjectionLimits(
            max_output_row_bytes=3,
            max_output_buffer_bytes=2,
        )
