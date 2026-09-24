from __future__ import annotations

import hashlib
import json
from pathlib import Path

from modelome.commoncrawl_projection import (
    CommonCrawlDiscoveryLimits,
    CommonCrawlWetDiscoveryProjector,
)
from modelome.commoncrawl_search_runtime import (
    CommonCrawlSearchLimits,
    commoncrawl_search_checkpoint_source,
    run_commoncrawl_search_ingestion,
)
from modelome.lake import (
    LakeRecord,
    ParquetLandingZone,
    ShardApplicationOrder,
    canonical_control_sha256,
)
from modelome.storage import Database

SOURCE = "commoncrawl"
DATASET = "wet"
RELEASE = "CC-MAIN-2026-34"
OBJECT_URL = (
    "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-34/segments/"
    "1723456789012.0/wet/CC-MAIN-20260812010203-00000.warc.wet.gz"
)
OBJECT_PATH = OBJECT_URL.removeprefix("https://data.commoncrawl.org/")


def _payload(source_url: str, text: str, index: int) -> tuple[str, dict[str, object]]:
    warc_record_id = f"urn:uuid:00000000-0000-0000-0000-{index:012d}"
    key = canonical_control_sha256(
        {
            "collection_id": RELEASE,
            "object_url": OBJECT_URL,
            "warc_record_id": warc_record_id,
        }
    )
    payload: dict[str, object] = {
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


def _projection(tmp_path: Path):
    source_rows = (
        _payload(
            "https://github.com/biology-lab/git-only-model/tree/main",
            "We introduce GitOnlyNet-3, a deep neural network for cell state inference.",
            0,
        ),
        _payload(
            "https://research.example/software",
            "The official implementation is available at "
            "https://github.com/physics-lab/plasma-code/tree/main.",
            1,
        ),
        _payload(
            "https://github.com/uncovered/classical-method",
            "We fitted a classical hierarchical model with MCMC.",
            2,
        ),
    )
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
            for record_id, payload in source_rows
        ),
        expected_rows=len(source_rows),
        batch_rows=1,
    )
    release = lake.seal_release(
        source=SOURCE,
        dataset=DATASET,
        release=RELEASE,
        expected_shards={shard.shard: shard},
    )
    projector = CommonCrawlWetDiscoveryProjector(
        lake,
        limits=CommonCrawlDiscoveryLimits(input_batch_rows=1, output_part_rows=1),
    )
    return projector, projector.materialize(release)


def _database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "store")
    database.initialize()
    return database


def test_github_only_model_becomes_searchable_without_dropping_crawl_documents(
    tmp_path: Path,
) -> None:
    projector, receipt = _projection(tmp_path)
    database = _database(tmp_path)

    outcome = run_commoncrawl_search_ingestion(database, projector, receipt)

    assert outcome.status == "complete"
    assert outcome.total_rows == 3
    assert outcome.actionable_documents == 2
    assert outcome.candidate_count == 1
    assert receipt.document_count == 3
    assert len(database.table_rows("artifacts")) == 2
    [model] = database.search_models("GitOnlyNet-3")
    assert model["canonical_name"] == "GitOnlyNet-3"
    repository = next(
        row
        for row in database.table_rows("artifacts")
        if row["canonical_url"].startswith("https://github.com/")
    )
    assert repository["kind"] == "code_repository"
    revision = next(
        row
        for row in database.table_rows("artifact_revisions")
        if row["id"] == repository["current_revision_id"]
    )
    provenance = json.loads(revision["raw_json"])
    assert provenance["document"]["projection_artifact_id"] == receipt.artifact_id
    assert provenance["document"]["source_row_ordinal"] == 0
    assert provenance["candidate_assertions"][0]["name"] == "GitOnlyNet-3"
    assert repository["modified_at"] is None
    assert revision["source_modified_at"] is None
    assert revision["text"] == ""

    assertion = provenance["candidate_assertions"][0]
    [document] = [
        row
        for batch in projector.iter_batches(receipt, "documents", batch_size=1)
        for row in batch.to_pylist()
        if row["source_row_ordinal"] == provenance["document"]["source_row_ordinal"]
    ]
    start, end = (int(value) for value in assertion["locator"].split(":")[1:])
    assert document["text"][start:end] == assertion["name"]
    assert document["content_sha256"] == provenance["document"]["content_sha256"]
    assert document["crawl_locator_json"] == provenance["document"]["crawl_locator_json"]

    assert not any(
        row["canonical_url"] == "https://github.com/uncovered/classical-method"
        for row in database.table_rows("artifacts")
    )


def test_external_page_enqueues_only_the_exact_github_repository_root(
    tmp_path: Path,
) -> None:
    projector, receipt = _projection(tmp_path)
    database = _database(tmp_path)

    outcome = run_commoncrawl_search_ingestion(database, projector, receipt)

    assert outcome.github_relation_count == 1
    assert outcome.links_discovered == 1
    assert [row["url"] for row in database.list_frontier()] == [
        "https://github.com/physics-lab/plasma-code"
    ]
    [discovery] = database.table_rows("url_discoveries")
    assert discovery["relation"] == "official_implementation"
    assert discovery["locator"].startswith("text:")


def test_runtime_resumes_by_source_row_ordinal_and_completed_replay_is_noop(
    tmp_path: Path,
) -> None:
    projector, receipt = _projection(tmp_path)
    database = _database(tmp_path)
    limits = CommonCrawlSearchLimits(max_document_rows=1)

    first = run_commoncrawl_search_ingestion(
        database,
        projector,
        receipt,
        limits=limits,
    )
    second = run_commoncrawl_search_ingestion(
        database,
        projector,
        receipt,
        limits=limits,
    )
    third = run_commoncrawl_search_ingestion(
        database,
        projector,
        receipt,
        limits=limits,
    )
    replay = run_commoncrawl_search_ingestion(
        database,
        projector,
        receipt,
        limits=limits,
    )

    assert (first.start_row, first.next_row, first.status) == (0, 1, "partial")
    assert (second.start_row, second.next_row, second.status) == (1, 2, "partial")
    assert (third.start_row, third.next_row, third.status) == (2, 3, "complete")
    assert replay.status == "complete"
    assert replay.run_id is None
    assert replay.documents_examined == 0
    assert database.get_source_state(first.checkpoint_source)["next_row"] == 3
    assert len(database.table_rows("artifacts")) == 2
    assert len(database.search_models("GitOnlyNet-3")) == 1


def test_projection_artifacts_have_isolated_completed_checkpoints(
    tmp_path: Path,
) -> None:
    projector, receipt = _projection(tmp_path)
    [source_receipt] = projector.landing_zone.list_releases(
        source=SOURCE,
        dataset=DATASET,
        verify_shards=True,
    )
    alternate_projector = CommonCrawlWetDiscoveryProjector(
        projector.landing_zone,
        output_root=tmp_path / "alternate-projection",
        limits=CommonCrawlDiscoveryLimits(
            input_batch_rows=1,
            output_part_rows=2,
        ),
    )
    alternate_receipt = alternate_projector.materialize(source_receipt)
    assert alternate_receipt.artifact_id != receipt.artifact_id
    database = _database(tmp_path)

    first = run_commoncrawl_search_ingestion(database, projector, receipt)
    alternate = run_commoncrawl_search_ingestion(
        database,
        alternate_projector,
        alternate_receipt,
    )
    replay = run_commoncrawl_search_ingestion(database, projector, receipt)

    assert first.complete is True
    assert alternate.complete is True
    assert first.checkpoint_source != alternate.checkpoint_source
    assert first.checkpoint_source == commoncrawl_search_checkpoint_source(receipt)
    assert replay.run_id is None
    assert database.get_source_state(first.checkpoint_source)["complete"] is True
    assert database.get_source_state(alternate.checkpoint_source)["complete"] is True


def test_tampered_projection_fails_without_advancing_search_checkpoint(
    tmp_path: Path,
) -> None:
    projector, receipt = _projection(tmp_path)
    database = _database(tmp_path)
    candidate_part = next((receipt.path / "model_candidates").glob("*.parquet"))
    with candidate_part.open("ab") as stream:
        stream.write(b"tamper")

    outcome = run_commoncrawl_search_ingestion(database, projector, receipt)

    assert outcome.status == "failed"
    assert "checksum mismatch" in (outcome.error or "")
    assert database.get_source_state(outcome.checkpoint_source) == {}
    assert database.table_rows("artifacts") == []
