import hashlib
import json

import pytest

from modelome.cli import main
from modelome.dataset import build_dataset
from modelome.models import ArtifactKind, ModelHint, SourcePage, SourceRecord
from modelome.storage import Database


def paper(record_id="one"):
    return {
        "source": "example",
        "source_record_id": record_id,
        "kind": "paper",
        "canonical_url": f"https://example.test/{record_id}",
        "title": "A declared technique",
        "models": [{"local_id": "model", "name": f"Technique {record_id}"}],
        "links": [{
            "url": f"https://github.com/example/{record_id}",
            "relation": "official_implementation",
        }],
    }


def test_papers_produce_relocatable_bundle_and_idempotent_entries(tmp_path):
    store = tmp_path / "store"
    first = build_dataset(store, tmp_path / "first", papers=[paper()], source_configs=[])
    second = build_dataset(store, tmp_path / "second", papers=[paper()], source_configs=[])
    assert first.entry_count == second.entry_count == 1
    assert first.status == "complete"
    assert first.papers_ingested == 1
    assert (tmp_path / "first/entries/entries.jsonl").read_bytes() == (
        tmp_path / "second/entries/entries.jsonl"
    ).read_bytes()
    manifest = json.loads((tmp_path / "first/manifest.json").read_text())
    assert manifest["source_commit"] == first.source_commit
    assert manifest["mode"] == "papers"
    for name, info in manifest["files"].items():
        body = (tmp_path / "first" / name).read_bytes()
        assert info == {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    metadata = json.loads((tmp_path / "first/metadata/export-receipt.json").read_text())
    assert metadata["source_commit"] == first.source_commit
    assert metadata["output"] == "metadata"


class PagedSource:
    name = "paged"

    def fetch_page(self, state):
        record_id = "two" if state else "one"
        return SourcePage(
            records=(SourceRecord(
                source_record_id=record_id,
                kind=ArtifactKind.CATALOG_RECORD,
                canonical_url=f"https://example.test/{record_id}",
                title=record_id,
                raw={},
                models=(ModelHint(local_id="model", name=record_id),),
            ),),
            next_state={"cursor": "two"},
            complete=bool(state),
        )


def test_source_budget_resumes_and_marks_partial_bundle(tmp_path):
    store = tmp_path / "store"
    sources = {"paged": PagedSource()}
    first = build_dataset(store, tmp_path / "first", sources=sources, source_configs=[])
    second = build_dataset(store, tmp_path / "second", sources=sources, source_configs=[])
    assert (first.status, first.entry_count) == ("partial", 1)
    assert (second.status, second.entry_count) == ("complete", 2)
    manifest = json.loads((tmp_path / "first/manifest.json").read_text())
    assert manifest["sync"][0]["status"] == "partial"


def test_failure_keeps_ingestion_but_does_not_publish_bundle(tmp_path):
    store = tmp_path / "store"
    with pytest.raises(ValueError):
        build_dataset(store, tmp_path / "failed", papers=[paper(), {}], source_configs=[])
    assert not (tmp_path / "failed").exists()
    recovered = build_dataset(store, tmp_path / "recovered", source_configs=[])
    assert recovered.entry_count == 1


def test_failed_source_does_not_publish(tmp_path):
    class FailedSource:
        name = "failed"

        def fetch_page(self, state):
            raise RuntimeError("unavailable")

    with pytest.raises(RuntimeError, match="source ingestion failed: failed"):
        build_dataset(
            tmp_path / "store", tmp_path / "dataset",
            sources={"failed": FailedSource()}, source_configs=[],
        )
    assert not (tmp_path / "dataset").exists()


def test_offline_export_does_not_mutate_store_and_handles_no_entries(tmp_path):
    store = tmp_path / "store"
    database = Database(store)
    database.initialize()
    database.close()
    before = (store / "HEAD.json").read_bytes()
    receipt = build_dataset(store, tmp_path / "dataset", source_configs=[])
    assert receipt.entry_count == receipt.seed_count == 0
    assert (store / "HEAD.json").read_bytes() == before
    assert (tmp_path / "dataset/entries/entries.jsonl").read_text() == ""


def test_existing_output_rejected_before_ingestion(tmp_path):
    output = tmp_path / "dataset"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_dataset(tmp_path / "store", output, papers=[paper()], source_configs=[])
    assert not (tmp_path / "store").exists()


def test_concurrent_write_aborts_publication(tmp_path, monkeypatch):
    import modelome.dataset as module

    original = module.export_public_metadata

    def export_then_write(database, output, **kwargs):
        result = original(database, output, **kwargs)
        database.start_run("concurrent-writer")
        return result

    monkeypatch.setattr(module, "export_public_metadata", export_then_write)
    with pytest.raises(RuntimeError, match="registry changed"):
        build_dataset(tmp_path / "store", tmp_path / "dataset", papers=[paper()], source_configs=[])
    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset-*"))


def test_cli_paper_worklist(tmp_path, capsys):
    worklist = tmp_path / "papers.jsonl"
    worklist.write_text(json.dumps(paper()) + "\n" + json.dumps(paper("two")) + "\n")
    with pytest.raises(SystemExit) as result:
        main([
            "--store", str(tmp_path / "store"), "--json", "build-dataset",
            "--input", str(worklist), "--output", str(tmp_path / "dataset"),
        ])
    assert result.value.code == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["entry_count"] == 2
    assert receipt["papers_ingested"] == 2


def test_cli_source_selection_and_partial_exit(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"paged": PagedSource()})
    with pytest.raises(SystemExit) as result:
        main([
            "--store", str(tmp_path / "store"), "--json", "build-dataset",
            "--source", "paged", "--output", str(tmp_path / "dataset"),
        ])
    assert result.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "partial"


def test_cli_rejects_unknown_source_without_creating_store(tmp_path, capsys):
    with pytest.raises(SystemExit) as result:
        main([
            "--store", str(tmp_path / "store"), "--json", "build-dataset",
            "--source", "does-not-exist", "--output", str(tmp_path / "dataset"),
        ])
    assert result.value.code == 2
    assert "unknown, disabled, or credential-gated" in capsys.readouterr().out
    assert not (tmp_path / "store").exists()
