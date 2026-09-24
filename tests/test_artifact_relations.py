from __future__ import annotations

from pathlib import Path

import pytest

from modelome.artifact_relations import (
    RELATION_SCHEMA,
    ArtifactRelationLimits,
    ArtifactRelationMaterializer,
)
from modelome.models import ArtifactKind, Identifier, Link, ModelHint, ReleaseHint, SourceRecord
from modelome.storage import Database


def _rows(materializer, receipt) -> list[dict]:
    return [
        row
        for batch in materializer.iter_batches(receipt, batch_size=1)
        for row in batch.to_pylist()
    ]


def _limits(**overrides: int) -> ArtifactRelationLimits:
    values = {
        "bucket_count": 4,
        "scan_batch_rows": 1,
        "max_scan_batch_bytes": 2 * 1024 * 1024,
        "partition_buffer_rows": 2,
        "partition_buffer_bytes": 2 * 1024 * 1024,
        "max_work_row_bytes": 1024 * 1024,
        "join_batch_rows": 1,
        "max_join_batch_bytes": 2 * 1024 * 1024,
        "max_bucket_rows": 100,
        "max_bucket_bytes": 4 * 1024 * 1024,
        "output_part_rows": 1,
        "max_output_buffer_bytes": 2 * 1024 * 1024,
        "max_output_row_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return ArtifactRelationLimits(**values)


def test_materializes_current_evidence_into_restart_safe_parquet_relations(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    model_identifier = Identifier("provider:model", "research/model")
    release_identifier = Identifier("weights:sha256", "a" * 64)
    paper_identifier = Identifier("doi", "10.1000/research")
    repository_url = "https://github.com/research/implementation"

    paper = SourceRecord(
        source_record_id="paper-a",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.example/research",
        title="A neural architecture",
        raw={"version": 1},
        text=f"The official implementation is available at {repository_url}.",
        identifiers=(paper_identifier,),
        models=(
            ModelHint(
                "introduced-model",
                "A Neural Architecture",
                identifiers=(model_identifier,),
            ),
        ),
    )
    duplicate_paper = SourceRecord(
        source_record_id="paper-b",
        kind=ArtifactKind.PAPER,
        canonical_url="https://index.example/record/research",
        title="Indexed copy",
        raw={},
        identifiers=(paper_identifier,),
    )
    repository = SourceRecord(
        source_record_id="repository",
        kind=ArtifactKind.CODE_REPOSITORY,
        canonical_url=repository_url,
        title="Implementation",
        raw={},
    )
    card = SourceRecord(
        source_record_id="card",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://models.example/research/card",
        title="Model card",
        raw={},
        models=(ModelHint("model", "Provider Display Name", identifiers=(model_identifier,)),),
        releases=(
            ReleaseHint(
                "checkpoint",
                "model",
                identifiers=(release_identifier,),
            ),
        ),
    )
    weights = SourceRecord(
        source_record_id="weights",
        kind=ArtifactKind.WEIGHTS,
        canonical_url="https://weights.example/research/checkpoint",
        title="Checkpoint",
        raw={},
        models=(ModelHint("model", "Another Display Name", identifiers=(model_identifier,)),),
        releases=(
            ReleaseHint(
                "checkpoint",
                "model",
                identifiers=(release_identifier,),
            ),
        ),
    )
    for source, record in (
        ("papers-a", paper),
        ("papers-b", duplicate_paper),
        ("repositories", repository),
        ("cards", card),
        ("weights", weights),
    ):
        database.ingest_page(source, (record,), {}, extractor="fixture")

    materializer = ArtifactRelationMaterializer(
        database.root,
        output_root=tmp_path / "relations",
        limits=_limits(),
    )
    first = materializer.materialize()
    rows = _rows(materializer, first)
    repeated = materializer.materialize()

    assert first.already_materialized is False
    assert repeated.already_materialized is True
    assert repeated.artifact_id == first.artifact_id
    assert repeated.path == first.path
    assert first.row_count == len(rows)
    assert all(batch.schema == RELATION_SCHEMA for batch in materializer.iter_batches(first))
    assert {row["evidence_type"] for row in rows} == {
        "shared_artifact_identifier",
        "shared_model_identity",
        "shared_release_identity",
        "url_mention",
    }
    assert dict(first.relation_counts) == {
        "shared_artifact_identifier": 1,
        "shared_model_identity": 2,
        "shared_release_identity": 1,
        "url_mention": 1,
    }
    url_relation = next(row for row in rows if row["evidence_type"] == "url_mention")
    assert url_relation["predicate"] == "official_implementation"
    assert url_relation["subject_kind"] == "paper"
    assert url_relation["target_kind"] == "code_repository"
    assert url_relation["match_value"] == repository_url
    assert url_relation["locator"].startswith("text:")
    assert url_relation["symmetric"] is False
    identifier_relation = next(
        row for row in rows if row["evidence_type"] == "shared_artifact_identifier"
    )
    assert identifier_relation["predicate"] == "same_artifact"
    assert identifier_relation["match_namespace"] == "doi"
    assert identifier_relation["match_value"] == paper_identifier.value
    assert identifier_relation["symmetric"] is True
    assert all(row["subject_artifact_id"] != row["target_artifact_id"] for row in rows)

    database.ingest_page(
        "blogs",
        (
            SourceRecord(
                source_record_id="unrelated",
                kind=ArtifactKind.BLOG_POST,
                canonical_url="https://blog.example/unrelated",
                title="Unrelated",
                raw={},
            ),
        ),
        {},
        extractor="fixture",
    )
    refreshed = materializer.materialize()
    refreshed_rows = _rows(materializer, refreshed)
    assert refreshed.source_commit != first.source_commit
    assert {row["relation_id"] for row in refreshed_rows} == {row["relation_id"] for row in rows}
    assert {row["source_commit"] for row in refreshed_rows} == {refreshed.source_commit}


def test_shared_license_locators_remain_metadata_not_artifact_relations(tmp_path: Path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    license_url = "https://licenses.example/public-terms"
    for record_id in ("paper-a", "paper-b"):
        database.ingest_page(
            "papers",
            (
                SourceRecord(
                    source_record_id=record_id,
                    kind=ArtifactKind.PAPER,
                    canonical_url=f"https://papers.example/{record_id}",
                    title=record_id,
                    raw={},
                    links=(Link(license_url, relation="license", crawl=False),),
                ),
            ),
            {},
            extractor="fixture",
        )

    materializer = ArtifactRelationMaterializer(
        database.root,
        output_root=tmp_path / "relations",
        limits=_limits(),
    )
    receipt = materializer.materialize()

    assert len(database.table_rows("url_discoveries")) == 2
    assert _rows(materializer, receipt) == []


def test_rebuild_uses_only_current_revisions_and_old_projection_remains_readable(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    target_url = "https://code.example/project"
    target = SourceRecord(
        source_record_id="code",
        kind=ArtifactKind.CODE_REPOSITORY,
        canonical_url=target_url,
        title="Code",
        raw={},
    )
    initial = SourceRecord(
        source_record_id="paper",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.example/item",
        title="Paper",
        raw={"revision": 1},
        text=f"Source code is available at {target_url}.",
    )
    database.ingest_page("code", (target,), {}, extractor="fixture")
    database.ingest_page("papers", (initial,), {}, extractor="fixture")
    materializer = ArtifactRelationMaterializer(
        database.root,
        output_root=tmp_path / "relations",
        limits=_limits(),
    )
    before = materializer.materialize()
    assert [row["predicate"] for row in _rows(materializer, before)] == ["implementation"]

    corrected = SourceRecord(
        source_record_id="paper",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.example/item",
        title="Paper",
        raw={"revision": 2},
        text="The repository assertion was withdrawn.",
    )
    database.ingest_page("papers", (corrected,), {}, extractor="fixture")
    after = materializer.materialize()

    assert after.source_commit != before.source_commit
    assert after.row_count == 0
    assert _rows(materializer, after) == []
    assert [row["predicate"] for row in _rows(materializer, before)] == ["implementation"]


def test_same_name_without_an_exact_identity_does_not_create_a_relation(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    for source in ("one", "two"):
        database.ingest_page(
            source,
            (
                SourceRecord(
                    source_record_id="record",
                    kind=ArtifactKind.MODEL_CARD,
                    canonical_url=f"https://{source}.example/model",
                    title="Same Display Name",
                    raw={},
                    models=(ModelHint("model", "Same Display Name"),),
                ),
            ),
            {},
            extractor="fixture",
        )
    materializer = ArtifactRelationMaterializer(
        database.root,
        output_root=tmp_path / "relations",
        limits=_limits(),
    )

    receipt = materializer.materialize()

    assert receipt.row_count == 0
    assert _rows(materializer, receipt) == []


def test_fails_closed_when_a_join_bucket_exceeds_its_declared_bound(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    for index in range(3):
        database.ingest_page(
            f"source-{index}",
            (
                SourceRecord(
                    source_record_id="record",
                    kind=ArtifactKind.PAPER,
                    canonical_url=f"https://papers.example/{index}",
                    title=f"Paper {index}",
                    raw={},
                    identifiers=(Identifier("doi", "10.1000/shared"),),
                ),
            ),
            {},
            extractor="fixture",
        )
    materializer = ArtifactRelationMaterializer(
        database.root,
        output_root=tmp_path / "relations",
        limits=_limits(bucket_count=1, max_bucket_rows=2),
    )

    with pytest.raises(ValueError, match="max_bucket_rows"):
        materializer.materialize()
