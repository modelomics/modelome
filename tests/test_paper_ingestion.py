from __future__ import annotations

import json

import pytest

from modelome.cli import main
from modelome.paper_ingestion import ingest_paper, prepare_paper_ingestion
from modelome.storage import Database


def _paper_seed(*, models: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "source": "arxiv",
        "source_record_id": "2401.00001",
        "kind": "paper",
        "canonical_url": "https://arxiv.org/abs/2401.00001",
        "title": "ExampleNet: an exact paper observation",
        "raw": {"provider": "fixture"},
        "text": (
            "We introduce ExampleNet, a neural network for image segmentation. "
            "Official implementation: https://github.com/example/examplenet."
        ),
        "identifiers": [{"namespace": "arxiv", "value": "2401.00001"}],
        "links": [
            {
                "url": "https://zenodo.org/records/12345",
                "relation": "dataset",
                "locator": "metadata:data",
                "crawl": False,
            }
        ],
        "models": models or [],
        "model_relations": [],
        "releases": [],
        "tags": ["field:computer-vision", "project:fixture"],
    }


def _invoke(arguments: list[str], capsys) -> tuple[int, object]:
    with pytest.raises(SystemExit) as exit_info:
        main(arguments)
    return exit_info.value.code, capsys.readouterr()


def test_prepare_paper_ingestion_derives_a_candidate_and_retains_every_direct_link() -> None:
    prepared = prepare_paper_ingestion(_paper_seed())

    assert prepared.extractor_name == "introduction-cues-v4"
    assert prepared.candidate_count == 1
    assert prepared.direct_resource_count == 2
    assert prepared.entry_seed["models"][0]["name"] == "ExampleNet"
    assert "raw" not in prepared.entry_seed
    assert "text" not in prepared.entry_seed
    resources = [
        action
        for action in prepared.plan["actions"]
        if action["action"] == "attach_resource"
    ]
    assert {action["url"] for action in resources} == {
        "https://arxiv.org/abs/2401.00001",
        "https://github.com/example/examplenet",
        "https://zenodo.org/records/12345",
    }
    code_action = next(action for action in resources if "github.com" in action["url"])
    assert code_action["relation"] == "official_implementation"
    assert code_action["category"] == "code"


def test_prepare_paper_ingestion_does_not_mix_source_declarations_with_derived_candidates() -> None:
    source_model = {
        "local_id": "source-model",
        "name": "ExampleNet",
        "identifiers": [{"namespace": "arxiv", "value": "2401.00001"}],
        "status": "documented",
    }

    prepared = prepare_paper_ingestion(_paper_seed(models=[source_model]))

    assert prepared.extractor_name == "source"
    assert prepared.candidate_count == 1
    assert prepared.entry_seed["models"] == [source_model]


def test_ingest_paper_persists_one_source_record_and_only_returns_an_entry_plan(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()

    outcome = ingest_paper(database, _paper_seed())

    assert outcome.source == "arxiv"
    assert outcome.source_record_id == "2401.00001"
    assert outcome.candidate_count == 1
    assert outcome.materialized_entries == 0
    assert outcome.stats["new_artifacts"] == 1
    assert outcome.entry_seed["tags"] == ["field:computer-vision", "project:fixture"]
    assert database.table_rows("artifacts")[0]["source"] == "arxiv"
    assert database.search_models("ExampleNet")
    pending_urls = {row["url"] for row in database.list_frontier()}
    observed_urls = {row["url"] for row in database.list_frontier(status="observed")}
    assert "https://github.com/example/examplenet" in pending_urls
    assert "https://zenodo.org/records/12345" in observed_urls
    checkpoint_sources = {row["source"] for row in database.table_rows("source_checkpoints")}
    assert "paper-ingest" in checkpoint_sources


def test_ingest_paper_dry_run_does_not_open_or_write_a_store(tmp_path, capsys) -> None:
    input_path = tmp_path / "paper.json"
    input_path.write_text(json.dumps(_paper_seed()), encoding="utf-8")
    store = tmp_path / "store"

    code, captured = _invoke(
        [
            "--store",
            str(store),
            "--json",
            "ingest-paper",
            "--input",
            str(input_path),
            "--dry-run",
        ],
        capsys,
    )

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["dry_run"] is True
    assert payload["candidate_count"] == 1
    assert payload["materialized_entries"] == 0
    assert "text" not in payload["entry_seed"]
    assert not store.exists()


def test_ingest_paper_rejects_non_paper_input() -> None:
    seed = _paper_seed()
    seed["kind"] = "model_card"

    with pytest.raises(ValueError, match="kind must be 'paper'"):
        prepare_paper_ingestion(seed)
