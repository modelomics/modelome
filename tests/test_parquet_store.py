from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pyarrow.parquet as pq
import pytest

from modelome.models import ArtifactKind, Identifier, Link, ModelHint, SourceRecord
from modelome.storage import _TABLES, Database, _state_and_table_digests


def _head(store: Path) -> dict[str, object]:
    return json.loads((store / "HEAD.json").read_text(encoding="utf-8"))


def _record(record_id: str, name: str, *, identifier: str | None = None) -> SourceRecord:
    identifiers = (
        (Identifier("provider:model", identifier),) if identifier is not None else ()
    )
    return SourceRecord(
        source_record_id=record_id,
        kind=ArtifactKind.MODEL_CARD,
        canonical_url=f"https://models.example/{record_id}",
        title=name,
        raw={"id": record_id, "name": name},
        links=(Link(f"https://code.example/{record_id}", "code"),),
        models=(ModelHint("model", name, identifiers=identifiers),),
    )


def test_initialize_creates_a_headed_commit_directory(tmp_path: Path) -> None:
    store = tmp_path / "registry-store"
    database = Database(store)

    database.initialize()

    assert store.is_dir()
    head = _head(store)
    commit = head["commit"]
    assert isinstance(commit, str) and commit
    commit_dir = store / "commits" / commit
    assert commit_dir.is_dir()
    assert (commit_dir / "manifest.json").is_file()
    assert list((commit_dir / "tables").glob("*.parquet"))
    manifest = json.loads((commit_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["table_digests"]) == set(_TABLES)
    assert set(manifest["parquet_sha256"]) == set(_TABLES)


def test_ingest_publishes_real_readable_parquet_tables(tmp_path: Path) -> None:
    store = tmp_path / "registry-store"
    database = Database(store)
    database.initialize()
    initial_commit = _head(store)["commit"]

    database.ingest_page(
        "provider",
        (_record("parquet-model", "Parquet Model", identifier="lab/parquet-model"),),
        {"cursor": "complete"},
        extractor="fixture",
    )

    current_commit = _head(store)["commit"]
    assert current_commit != initial_commit
    table_paths = list((store / "commits" / str(current_commit) / "tables").glob("*.parquet"))
    assert table_paths
    tables = {path.stem: pq.read_table(path) for path in table_paths}
    assert all(table.num_rows >= 0 for table in tables.values())
    assert {row["canonical_name"] for row in tables["models"].to_pylist()} == {
        "Parquet Model"
    }
    assert {row["source_record_id"] for row in tables["artifacts"].to_pylist()} == {
        "parquet-model"
    }
    assert tables["artifact_revisions"].num_rows == 1


def test_run_only_commit_hardlinks_unchanged_immutable_tables(tmp_path: Path) -> None:
    store = tmp_path / "registry-store"
    database = Database(store)
    database.initialize()
    parent_commit = str(_head(store)["commit"])

    database.start_run("catalog")

    child_commit = str(_head(store)["commit"])
    parent_artifacts = store / "commits" / parent_commit / "tables" / "artifacts.parquet"
    child_artifacts = store / "commits" / child_commit / "tables" / "artifacts.parquet"
    assert parent_artifacts.stat().st_ino == child_artifacts.stat().st_ino
    assert parent_artifacts.stat().st_nlink >= 2
    assert (store / "commits" / child_commit / "tables" / "sync_runs.parquet").stat().st_ino != (
        store / "commits" / parent_commit / "tables" / "sync_runs.parquet"
    ).stat().st_ino
    reopened = Database(store)
    reopened.initialize()
    assert reopened.source_status()[0]["source"] == "catalog"


def test_fresh_instance_reopens_identical_registry_state(tmp_path: Path) -> None:
    store = tmp_path / "registry-store"
    first = Database(store)
    first.initialize()
    first.ingest_page(
        "provider",
        (_record("reopen", "Reopened Model", identifier="lab/reopened"),),
        {"cursor": "next-page"},
        extractor="fixture",
    )
    expected_stats = first.stats()
    expected_search = first.search_models("provider:model:lab/reopened")
    expected_detail = first.model_detail(expected_search[0]["id"])
    expected_state = first.get_source_state("provider")

    reopened = Database(store)
    reopened.initialize()

    assert reopened.stats() == expected_stats
    assert reopened.search_models("provider:model:lab/reopened") == expected_search
    assert reopened.model_detail(expected_search[0]["id"]) == expected_detail
    assert reopened.get_source_state("provider") == expected_state


def test_fresh_instance_reopens_the_predecessor_store_format(tmp_path: Path) -> None:
    store = tmp_path / "registry-store"
    database = Database(store)
    database.initialize()
    database.ingest_page(
        "provider",
        (_record("legacy", "Legacy Model", identifier="lab/legacy"),),
        {"cursor": "saved"},
        extractor="fixture",
    )
    head = _head(store)
    commit = str(head["commit"])
    manifest_path = store / "commits" / commit / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables = {
        table_name: pq.read_table(
            store / "commits" / commit / "tables" / f"{table_name}.parquet"
        ).to_pylist()
        for table_name in _TABLES
    }
    legacy_state_digest, _ = _state_and_table_digests(tables)
    head["format"] = "ulr-parquet-snapshot-v1"
    head["state_digest"] = legacy_state_digest
    manifest["format"] = "ulr-parquet-snapshot-v1"
    manifest["state_digest"] = legacy_state_digest
    manifest.pop("table_digests")
    manifest.pop("parquet_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (store / "HEAD.json").write_text(json.dumps(head), encoding="utf-8")

    reopened = Database(store)
    reopened.initialize()

    assert reopened.search_models("Legacy Model")


def test_reopen_rejects_a_parquet_file_checksum_mismatch(tmp_path: Path) -> None:
    store = tmp_path / "registry-store"
    database = Database(store)
    database.initialize()
    head = _head(store)
    manifest_path = store / "commits" / str(head["commit"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parquet_sha256"]["artifacts"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        Database(store).initialize()


def test_two_instances_serialize_writes_and_enforce_source_lease(tmp_path: Path) -> None:
    store = tmp_path / "registry-store"
    first = Database(store)
    second = Database(store)
    first.initialize()
    second.initialize()
    run_id = first.start_run("leased-source")
    head_before_rejected_lease = _head(store)

    with pytest.raises(RuntimeError, match="another sync run is active"):
        second.start_run("leased-source")

    assert _head(store) == head_before_rejected_lease
    start = Barrier(2)

    def write_leased_source() -> None:
        start.wait()
        first.ingest_page(
            "leased-source",
            (_record("leased", "Leased Model"),),
            {"cursor": "leased"},
            run_id=run_id,
            extractor="fixture",
        )

    def write_independent_source() -> None:
        start.wait()
        second.ingest_page(
            "independent-source",
            (_record("independent", "Independent Model"),),
            {"cursor": "independent"},
            extractor="fixture",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write_leased_source), executor.submit(write_independent_source)]
        for future in futures:
            future.result(timeout=30)

    reopened = Database(store)
    reopened.initialize()
    assert reopened.stats()["artifacts"] == 2
    assert {item["canonical_name"] for item in reopened.search_models("Model")} == {
        "Independent Model",
        "Leased Model",
    }


def test_failed_strict_page_does_not_publish_a_new_head(tmp_path: Path) -> None:
    store = tmp_path / "registry-store"
    database = Database(store)
    database.initialize()
    good = _record("strict-good", "Strict Good")
    bad = SourceRecord(
        source_record_id="strict-bad",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://models.example/strict-bad",
        title="Strict Bad",
        raw={"not-json-serializable": object()},
        models=(ModelHint("model", "Strict Bad"),),
    )
    head_before = _head(store)
    commits_before = {path.name for path in (store / "commits").iterdir()}

    with pytest.raises(TypeError):
        database.ingest_page(
            "strict-source",
            (good, bad),
            {"cursor": "must-not-publish"},
            extractor="fixture",
            quarantine_errors=False,
        )

    assert _head(store) == head_before
    assert {path.name for path in (store / "commits").iterdir()} == commits_before
    assert database.get_source_state("strict-source") == {}
    assert database.stats()["artifacts"] == 0
    assert database.stats()["models"] == 0


def test_missing_head_never_resets_a_populated_store(tmp_path: Path) -> None:
    store = tmp_path / "registry-store"
    database = Database(store)
    database.initialize()
    database.ingest_page(
        "provider",
        (_record("survivor", "Surviving Model"),),
        {"cursor": "saved"},
        extractor="fixture",
    )
    commits_before = {path.name for path in (store / "commits").iterdir()}
    (store / "HEAD.json").unlink()

    with pytest.raises(ValueError, match="HEAD is missing"):
        database.start_run("must-not-reset")
    with pytest.raises(ValueError, match="HEAD is missing but immutable commits exist"):
        Database(store).initialize()

    assert {path.name for path in (store / "commits").iterdir()} == commits_before


@pytest.mark.parametrize("child", ["commits", ".staging", ".write.lock"])
def test_store_rejects_symlinked_internal_paths(tmp_path: Path, child: str) -> None:
    store = tmp_path / "registry-store"
    store.mkdir()
    outside = tmp_path / "outside"
    if child == ".write.lock":
        outside.write_text("outside", encoding="utf-8")
    else:
        outside.mkdir()
    (store / child).symlink_to(outside, target_is_directory=child != ".write.lock")

    with pytest.raises(ValueError, match="must not be a symlink"):
        Database(store).initialize()
