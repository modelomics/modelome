from pathlib import Path

import pytest

from modelome.coverage import evaluate_manifest
from modelome.models import ArtifactKind, ModelHint, SourcePage, SourceRecord
from modelome.storage import Database


def _database() -> Database:
    database = Database(":memory:")
    database.initialize()
    record = SourceRecord(
        source_record_id="record-1",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url="https://example.test/record-1",
        title="NovelNet",
        raw={"name": "NovelNet"},
        models=(ModelHint(local_id="model", name="NovelNet", aliases=("Novel Net",)),),
    )
    database.ingest_page("catalog", SourcePage((record,), {}, True))
    return database


def test_evaluate_manifest_reports_exact_recall_by_bucket(tmp_path: Path) -> None:
    manifest = tmp_path / "coverage.csv"
    manifest.write_text(
        "Model,Bucket,Domain\nNovel Net,new_family,vision\nMissingNet,new_family,text\n",
        encoding="utf-8",
    )
    database = _database()
    try:
        report = evaluate_manifest(database, manifest)
    finally:
        database.close()

    assert report["found"] == 1
    assert report["missing"] == 1
    assert report["recall"] == 0.5
    assert report["buckets"]["new_family"]["recall"] == 0.5
    assert report["expectations"][0]["sources"] == ["catalog"]
    assert not report["expectations"][1]["found"]


def test_evaluate_manifest_rejects_a_malformed_or_duplicate_suite(tmp_path: Path) -> None:
    database = _database()
    missing = tmp_path / "missing.csv"
    missing.write_text("Name,Bucket\nA,b\n", encoding="utf-8")
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("Model,Bucket\nA,b\na,b\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="missing column"):
            evaluate_manifest(database, missing)
        with pytest.raises(ValueError, match="duplicate expectation"):
            evaluate_manifest(database, duplicate)
    finally:
        database.close()


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("Model,Bucket,Model\nA,b,A\n", "duplicate column"),
        ("Model,Bucket\n---,b\n", "letters or numbers"),
        ("Model,Bucket\nA,b,extra\n", "unexpected extra columns"),
    ],
)
def test_evaluate_manifest_rejects_ambiguous_rows(
    tmp_path: Path, contents: str, message: str
) -> None:
    manifest = tmp_path / "ambiguous.csv"
    manifest.write_text(contents, encoding="utf-8")
    database = _database()
    try:
        with pytest.raises(ValueError, match=message):
            evaluate_manifest(database, manifest)
    finally:
        database.close()


def test_evaluate_manifest_groups_bucket_labels_case_insensitively(tmp_path: Path) -> None:
    manifest = tmp_path / "coverage.csv"
    manifest.write_text(
        "Model,Bucket\nNovelNet,Vision\nMissingNet,vision\n", encoding="utf-8"
    )
    database = _database()
    try:
        report = evaluate_manifest(database, manifest)
    finally:
        database.close()

    assert report["buckets"] == {
        "Vision": {"total": 2, "found": 1, "missing": 1, "recall": 0.5}
    }


def test_evaluate_manifest_does_not_count_tombstoned_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "coverage.csv"
    manifest.write_text("Model,Bucket\nNovelNet,vision\n", encoding="utf-8")
    database = _database()
    run_id = database.start_run("catalog")
    database.ingest_page(
        "catalog",
        SourcePage(records=(), next_state={}, complete=True, authoritative_snapshot=True),
        run_id=run_id,
        extractor="fixture",
    )
    database.finish_run(run_id, "complete")
    try:
        report = evaluate_manifest(database, manifest)
    finally:
        database.close()

    assert report["found"] == 0
    assert report["expectations"][0]["sources"] == []


def test_evaluate_manifest_does_not_count_historical_revision_only(tmp_path: Path) -> None:
    manifest = tmp_path / "coverage.csv"
    manifest.write_text("Model,Bucket\nNovelNet,vision\n", encoding="utf-8")
    database = _database()
    corrected = SourceRecord(
        source_record_id="record-1",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url="https://example.test/record-1",
        title="Corrected record",
        raw={"name": "NovelNet", "correction": True},
    )
    database.ingest_page("catalog", (corrected,), {}, extractor="fixture")
    try:
        report = evaluate_manifest(database, manifest)
    finally:
        database.close()

    assert report["found"] == 0
    assert report["expectations"][0]["sources"] == []
