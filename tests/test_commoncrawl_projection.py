from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from modelome.commoncrawl_projection import (
    DOCUMENT_SCHEMA,
    MODEL_CANDIDATE_SCHEMA,
    URL_RELATION_SCHEMA,
    CommonCrawlDiscoveryLimits,
    CommonCrawlWetDiscoveryProjector,
)
from modelome.lake import (
    LakeRecord,
    ParquetLandingZone,
    ReleaseReceipt,
    ShardApplicationOrder,
    ShardReceipt,
    canonical_control_sha256,
)

SOURCE = "commoncrawl"
DATASET = "wet"
RELEASE = "CC-MAIN-2026-34"
OBJECT_URL = (
    "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-34/segments/"
    "1723456789012.0/wet/CC-MAIN-20260812010203-00000.warc.wet.gz"
)
OBJECT_PATH = OBJECT_URL.removeprefix("https://data.commoncrawl.org/")


def _payload(
    *,
    source_url: str,
    text: str,
    index: int,
) -> tuple[str, dict[str, Any]]:
    warc_record_id = f"urn:uuid:00000000-0000-0000-0000-{index:012d}"
    key = canonical_control_sha256(
        {
            "collection_id": RELEASE,
            "object_url": OBJECT_URL,
            "warc_record_id": warc_record_id,
        }
    )
    payload = {
        "uri": source_url,
        "date": "2026-08-12T01:02:03Z",
        "content": text,
        "content_type": "text/plain",
        "content_bytes": len(text.encode()),
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "warc": {
            "version": "WARC/1.0",
            "record_id": warc_record_id,
            "type": "conversion",
            "headers": [],
        },
        "bulk": {
            "collection_id": RELEASE,
            "collection_from": "2026-08-07T10:18:45",
            "collection_to": "2026-08-20T01:52:41",
            "manifest_index": 0,
            "total_shards": 1,
            "object_url": OBJECT_URL,
            "path": OBJECT_PATH,
            "record_index": index,
            "conversion_index": index,
        },
    }
    return f"commoncrawl:wet-record:{key}", payload


def _sealed_release(
    tmp_path: Path,
) -> tuple[ParquetLandingZone, ReleaseReceipt, ShardReceipt]:
    records = [
        _payload(
            source_url="https://github.com/example-lab/source-only/tree/main",
            text=(
                "This repository contains the complete project. "
                "We introduce CellForge-7, a deep neural network for cellular response. "
                "The official implementation is available at "
                "https://github.com/example-lab/cellforge/tree/main. "
                "Trained weights: https://weights.example/cellforge."
            ),
            index=0,
        ),
        _payload(
            source_url="https://unlisted.example/research/orbit",
            text=(
                "We developed OrbitNet-X, a deep neural network for plasma dynamics. "
                "Documentation: https://another-unlisted.example/models/orbit."
            ),
            index=1,
        ),
        _payload(
            source_url="https://statistics.example/bayes",
            text=(
                "We developed BayesTrack-X, a hierarchical statistical model fitted "
                "with MCMC."
            ),
            index=2,
        ),
    ]
    lake = ParquetLandingZone(tmp_path / "lake")
    shard = lake.commit_shard(
        source=SOURCE,
        dataset=DATASET,
        release=RELEASE,
        shard="wet-shard-0000",
        control_sha256=hashlib.sha256(b"control").hexdigest(),
        upstream_sha256=hashlib.sha256(b"upstream").hexdigest(),
        upstream_url=OBJECT_URL,
        upstream_bytes=123,
        application_order=ShardApplicationOrder.snapshot(0),
        records=(
            LakeRecord(source_record_id=record_id, payload=payload)
            for record_id, payload in records
        ),
        expected_rows=len(records),
        batch_rows=1,
    )
    release = lake.seal_release(
        source=SOURCE,
        dataset=DATASET,
        release=RELEASE,
        expected_shards={shard.shard: shard},
    )
    return lake, release, shard


def _rows(
    projector: CommonCrawlWetDiscoveryProjector,
    receipt,
    table: str,
) -> list[dict[str, Any]]:
    return [
        row
        for batch in projector.iter_batches(receipt, table, batch_size=1)
        for row in batch.to_pylist()
    ]


def test_projects_all_documents_urls_and_neural_candidates_with_exact_evidence(
    tmp_path: Path,
) -> None:
    lake, source, _shard = _sealed_release(tmp_path)
    projector = CommonCrawlWetDiscoveryProjector(
        lake,
        limits=CommonCrawlDiscoveryLimits(input_batch_rows=1, output_part_rows=1),
    )

    first = projector.materialize(source)
    documents = _rows(projector, first, "documents")
    relations = _rows(projector, first, "url_relations")
    candidates = _rows(projector, first, "model_candidates")
    repeated = projector.materialize(source)
    reopened = projector.open_projection(source)

    assert first.document_count == 3
    assert first.url_relation_count == 3
    assert first.candidate_count == 2
    assert repeated.path == first.path
    assert repeated.artifact_id == first.artifact_id
    assert repeated.already_materialized is True
    assert reopened == repeated
    assert len({row["document_id"] for row in documents}) == 3
    assert len({row["relation_assertion_id"] for row in relations}) == 3
    assert len({row["candidate_assertion_id"] for row in candidates}) == 2

    github_document = documents[0]
    assert github_document["source_url"] == (
        "https://github.com/example-lab/source-only/tree/main"
    )
    assert github_document["source_identifier_namespace"] == "github:repository"
    assert github_document["source_identifier_value"] == "example-lab/source-only"
    assert github_document["source_repository_url"] == (
        "https://github.com/example-lab/source-only"
    )
    assert github_document["content_sha256"] == hashlib.sha256(
        github_document["text"].encode()
    ).hexdigest()
    locator = json.loads(github_document["crawl_locator_json"])
    assert locator == {
        "collection_id": RELEASE,
        "conversion_index": 0,
        "manifest_index": 0,
        "object_url": OBJECT_URL,
        "path": OBJECT_PATH,
        "record_index": 0,
        "warc_record_id": "urn:uuid:00000000-0000-0000-0000-000000000000",
    }
    assert github_document["source_input_kind"] == "release"
    assert github_document["source_input_sha256"] == first.source_input_sha256
    assert github_document["source_shard"] is None
    assert github_document["projection_artifact_id"] == first.artifact_id

    github_relation = next(
        row
        for row in relations
        if row["target_identifier_namespace"] == "github:repository"
    )
    assert github_relation["predicate"] == "official_implementation"
    assert github_relation["target_url"] == (
        "https://github.com/example-lab/cellforge/tree/main"
    )
    assert github_relation["target_identifier_value"] == "example-lab/cellforge"
    assert github_relation["target_repository_url"] == (
        "https://github.com/example-lab/cellforge"
    )
    start, end = github_relation["evidence_start"], github_relation["evidence_end"]
    assert github_relation["evidence_text"] == github_document["text"][start:end]
    assert "official implementation" in github_relation["context_text"]
    assert any(
        row["target_url"]
        == "https://another-unlisted.example/models/orbit"
        for row in relations
    )

    assert {row["name"] for row in candidates} == {"CellForge-7", "OrbitNet-X"}
    assert "BayesTrack-X" not in {row["name"] for row in candidates}
    github_candidate = next(row for row in candidates if row["name"] == "CellForge-7")
    assert github_candidate["source_identifier_namespace"] == "github:repository"
    assert github_candidate["source_repository_url"] == (
        "https://github.com/example-lab/source-only"
    )
    assert github_candidate["evidence_text"] == "CellForge-7"
    assert "deep neural network" in github_candidate["context_text"]
    assert json.loads(github_candidate["derivation_json"])["crawl_locator"] == locator

    schemas = {
        "documents": DOCUMENT_SCHEMA,
        "model_candidates": MODEL_CANDIDATE_SCHEMA,
        "url_relations": URL_RELATION_SCHEMA,
    }
    for table, schema in schemas.items():
        assert all(
            pq.read_schema(path) == schema
            for path in (first.path / table).glob("*.parquet")
        )
    manifest = json.loads((first.path / "manifest.json").read_text())
    assert manifest["stats"] == {
        "document_row_count": 3,
        "documents_with_candidates": 2,
        "documents_with_urls": 2,
        "documents_without_candidates": 1,
        "documents_without_urls": 1,
        "input_row_count": 3,
        "model_candidate_row_count": 2,
        "url_relation_row_count": 3,
    }
    assert list((projector.output_root / ".staging").iterdir()) == []


def test_assertion_ids_do_not_depend_on_parquet_part_layout(tmp_path: Path) -> None:
    lake, source, _shard = _sealed_release(tmp_path)
    split = CommonCrawlWetDiscoveryProjector(
        lake,
        output_root=tmp_path / "split",
        limits=CommonCrawlDiscoveryLimits(output_part_rows=1),
    )
    packed = CommonCrawlWetDiscoveryProjector(
        lake,
        output_root=tmp_path / "packed",
        limits=CommonCrawlDiscoveryLimits(output_part_rows=100),
    )

    first = split.materialize(source)
    second = packed.materialize(source)

    assert first.artifact_id != second.artifact_id
    assert {
        row["relation_assertion_id"]
        for row in _rows(split, first, "url_relations")
    } == {
        row["relation_assertion_id"]
        for row in _rows(packed, second, "url_relations")
    }
    assert {
        row["candidate_assertion_id"]
        for row in _rows(split, first, "model_candidates")
    } == {
        row["candidate_assertion_id"]
        for row in _rows(packed, second, "model_candidates")
    }


def test_projection_limits_fail_without_publishing_partial_output(tmp_path: Path) -> None:
    lake, source, _shard = _sealed_release(tmp_path)
    projector = CommonCrawlWetDiscoveryProjector(
        lake,
        limits=CommonCrawlDiscoveryLimits(max_document_text_bytes=32),
    )

    with pytest.raises(ValueError, match="max_document_text_bytes"):
        projector.materialize(source)

    assert list((projector.output_root / ".staging").iterdir()) == []
    assert not list(projector.output_root.rglob("manifest.json"))


def test_repeated_materialization_detects_a_tampered_part(tmp_path: Path) -> None:
    lake, source, _shard = _sealed_release(tmp_path)
    projector = CommonCrawlWetDiscoveryProjector(lake)
    receipt = projector.materialize(source)
    part = next((receipt.path / "documents").glob("*.parquet"))
    with part.open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(ValueError, match="checksum mismatch"):
        projector.materialize(source)


def test_source_receipt_and_limit_configuration_are_validated(tmp_path: Path) -> None:
    lake, source, _shard = _sealed_release(tmp_path)
    projector = CommonCrawlWetDiscoveryProjector(lake)
    conflicting = replace(source, row_count=source.row_count + 1)

    with pytest.raises(ValueError, match="does not match"):
        projector.materialize(conflicting)
    with pytest.raises(ValueError, match="positive integer"):
        CommonCrawlDiscoveryLimits(input_batch_rows=True)
    with pytest.raises(ValueError, match="must not exceed"):
        CommonCrawlDiscoveryLimits(
            max_output_buffer_bytes=1,
            max_output_row_bytes=2,
        )


def test_a_committed_shard_is_actionable_before_its_release_is_sealed(
    tmp_path: Path,
) -> None:
    lake, release, shard = _sealed_release(tmp_path)
    release.path.unlink()
    projector = CommonCrawlWetDiscoveryProjector(lake)

    receipt = projector.materialize(shard)
    documents = _rows(projector, receipt, "documents")
    candidates = _rows(projector, receipt, "model_candidates")

    assert receipt.source_input_kind == "shard"
    assert receipt.source_shard == shard.shard
    assert receipt.document_count == shard.row_count
    assert {row["source_input_kind"] for row in documents} == {"shard"}
    assert {row["source_shard"] for row in documents} == {shard.shard}
    assert {row["name"] for row in candidates} == {"CellForge-7", "OrbitNet-X"}
