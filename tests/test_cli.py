import json

import pytest

from modelome.alphaxiv_runtime import AlphaXivOutcome
from modelome.backfill import DEFAULT_BACKFILL_MAX_PAGES, BackfillOutcome
from modelome.bootstrap import DEFAULT_BOOTSTRAP_MAX_PAGES, BootstrapOutcome
from modelome.bulk import BulkError, BulkOutcome
from modelome.cli import main
from modelome.daily import DailyOutcome
from modelome.frontier import FrontierOutcome
from modelome.lake import LakeRecord, ParquetLandingZone, ReleaseReceipt
from modelome.models import ArtifactKind, Identifier, ModelHint, SourceRecord
from modelome.pmc_bootstrap import PmcBootstrapOutcome
from modelome.projection_runtime import ProjectionOutcome
from modelome.semantic_scholar_materialize import ProjectionPlan, ProjectionReceipt
from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.sources.biorxiv import BioRxivSourceAdapter
from modelome.sources.commoncrawl import CommonCrawlWetSourceAdapter
from modelome.sources.crossref import CrossrefSourceAdapter
from modelome.sources.datacite import DataCiteSourceAdapter
from modelome.sources.eartharxiv import EarthArxivSourceAdapter
from modelome.sources.europe_pmc import EuropePmcSourceAdapter
from modelome.sources.gharchive import GhArchiveSourceAdapter
from modelome.sources.hal import HalSourceAdapter
from modelome.sources.openalex import OpenAlexSourceAdapter
from modelome.sources.osf_preprints import OsfPreprintSourceAdapter
from modelome.sources.pmc import PmcSourceAdapter
from modelome.sources.pubmed import PubMedBulkSourceAdapter
from modelome.sources.semantic_scholar import SemanticScholarDatasetSourceAdapter
from modelome.sources.software_heritage import SoftwareHeritageOriginSourceAdapter
from modelome.storage import Database


def invoke(arguments, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(arguments)
    return exit_info.value.code, capsys.readouterr()


def test_init_and_stats_json(tmp_path, capsys) -> None:
    store = tmp_path / "store"
    code, output = invoke(
        ["--store", str(store), "--json", "init"],
        capsys,
    )
    assert code == 0
    initialized = json.loads(output.out)
    assert initialized["initialized"] is True
    assert initialized["store"] == str(store)

    code, output = invoke(
        ["--store", str(store), "--json", "stats"],
        capsys,
    )
    assert code == 0
    assert json.loads(output.out)["models"] == 0


def test_entry_commands_are_offline_and_do_not_initialize_the_registry(tmp_path, capsys) -> None:
    seeds = tmp_path / "seeds.json"
    seeds.write_text(
        json.dumps(
            {
                "source": "example",
                "source_record_id": "paper-1",
                "canonical_url": "https://example.test/paper-1",
                "title": "Example technique",
                "kind": "paper",
                "tags": ["field:computer-vision"],
                "links": [
                    {
                        "url": "https://github.com/example/technique",
                        "relation": "official_implementation",
                    }
                ],
                "models": [
                    {
                        "local_id": "technique",
                        "name": "Example technique",
                        "identifiers": [{"namespace": "arxiv", "value": "2401.00001"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store = tmp_path / "unopened-store"

    code, output = invoke(
        ["--store", str(store), "--json", "entry-plan", "--input", str(seeds)], capsys
    )

    assert code == 0
    plan = json.loads(output.out)
    assert plan[0]["actions"][0]["action"] == "upsert_entry"
    assert not (store / "HEAD.json").exists()

    code, output = invoke(
        [
            "--store",
            str(store),
            "--json",
            "build-entry-corpus",
            "--input",
            str(seeds),
            "--dry-run",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["dry_run"] is True
    assert not (store / "HEAD.json").exists()


def test_cli_exports_current_assertions_as_entry_seeds(tmp_path, capsys) -> None:
    store_path = tmp_path / "store"
    store = Database(store_path)
    store.initialize()
    store.ingest_page(
        "arxiv",
        (
            SourceRecord(
                source_record_id="2401.00001",
                kind=ArtifactKind.PAPER,
                canonical_url="https://arxiv.org/abs/2401.00001",
                title="Example technique",
                raw={},
                models=(
                    ModelHint(
                        "example-technique",
                        "Example technique",
                        identifiers=(Identifier("arxiv", "2401.00001"),),
                    ),
                ),
            ),
        ),
        extractor="fixture",
    )
    output_path = tmp_path / "entry-seeds.jsonl"

    code, output = invoke(
        [
            "--store",
            str(store_path),
            "--json",
            "export-entry-seeds",
            "--source",
            "arxiv",
            "--output",
            str(output_path),
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["seed_count"] == 1
    assert output_path.is_file()


def test_cli_imports_one_explicitly_authorized_institutional_text(tmp_path, capsys) -> None:
    store = tmp_path / "store"
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "We introduce ClinicNet-X, a neural network for clinical prediction.",
        encoding="utf-8",
    )

    code, output = invoke(
        [
            "--store",
            str(store),
            "--json",
            "ingest-institutional-text",
            "--doi",
            "10.1126/ClinicNet.1",
            "--title",
            "ClinicNet-X",
            "--text-file",
            str(text_path),
            "--access-confirmed",
        ],
        capsys,
    )

    assert code == 0
    result = json.loads(output.out)
    assert result["doi"] == "10.1126/clinicnet.1"
    assert result["access_scope"] == "institutional-restricted"
    assert result["stats"]["models_touched"] == 1


def test_sources_lists_declarative_catalog(capsys) -> None:
    code, output = invoke(["--json", "sources"], capsys)

    assert code == 0
    sources = json.loads(output.out)
    names = [source["name"] for source in sources]
    assert len(names) == len(set(names))
    assert {
        "huggingface",
        "openrouter-models",
        "tensorflow-model-garden",
        "t5x-model-registry",
        "proteinmpnn-checkpoints",
        "openml-flows",
        "xai-models",
    } <= set(names)


def test_benchmarks_lists_epoch_separately_from_discovery_sources(capsys) -> None:
    code, output = invoke(["--json", "benchmarks"], capsys)

    assert code == 0
    benchmarks = json.loads(output.out)
    assert [benchmark["name"] for benchmark in benchmarks] == ["epoch"]
    assert benchmarks[0]["adapter"] == "csv"
    assert benchmarks[0]["schedule"] == "daily"


@pytest.mark.parametrize(
    ("recall", "minimum_recall", "expected_code"),
    [(0.75, 0.75, 0), (0.749, 0.75, 1)],
)
def test_benchmark_command_uses_recall_as_a_gate_and_selects_sole_benchmark(
    tmp_path,
    capsys,
    monkeypatch,
    recall,
    minimum_recall,
    expected_code,
) -> None:
    store = tmp_path / "store"
    init_code, _ = invoke(["--store", str(store), "--json", "init"], capsys)
    assert init_code == 0

    epoch = {
        "name": "epoch",
        "adapter": "csv",
        "url": "https://example.test/epoch.csv",
        "mapping": {"model_field": "Model"},
    }
    captured = {}

    def evaluate(database, config):
        captured["config"] = config
        return {"benchmark": config["name"], "recall": recall}

    monkeypatch.setattr("modelome.cli.load_benchmark_configs", lambda path: (epoch,))
    monkeypatch.setattr("modelome.cli.evaluate_epoch_benchmark", evaluate)

    code, output = invoke(
        [
            "--store",
            str(store),
            "--json",
            "benchmark",
            "--minimum-recall",
            str(minimum_recall),
        ],
        capsys,
    )

    assert code == expected_code
    report = json.loads(output.out)
    assert report["recall"] == recall
    assert report["minimum_recall"] == minimum_recall
    assert report["passed"] is (expected_code == 0)
    assert captured["config"] is epoch


def test_benchmark_command_selects_an_explicit_benchmark(tmp_path, capsys, monkeypatch) -> None:
    store = tmp_path / "store"
    init_code, _ = invoke(["--store", str(store), "--json", "init"], capsys)
    assert init_code == 0

    configs = (
        {"name": "first", "adapter": "csv"},
        {"name": "epoch", "adapter": "csv"},
    )
    captured = {}

    def evaluate(database, config):
        captured["name"] = config["name"]
        return {"benchmark": config["name"], "recall": 1.0}

    monkeypatch.setattr("modelome.cli.load_benchmark_configs", lambda path: configs)
    monkeypatch.setattr("modelome.cli.evaluate_epoch_benchmark", evaluate)

    code, output = invoke(
        ["--store", str(store), "--json", "benchmark", "--name", "epoch"],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["passed"] is True
    assert captured["name"] == "epoch"


def test_benchmark_fails_closed_when_the_live_corpus_has_invalid_rows(
    tmp_path, capsys, monkeypatch
) -> None:
    store = tmp_path / "store"
    init_code, _ = invoke(["--store", str(store), "--json", "init"], capsys)
    assert init_code == 0
    config = {"name": "epoch", "adapter": "csv"}
    monkeypatch.setattr("modelome.cli.load_benchmark_configs", lambda path: (config,))
    monkeypatch.setattr(
        "modelome.cli.evaluate_epoch_benchmark",
        lambda database, selected: {
            "benchmark": selected["name"],
            "recall": 1.0,
            "integrity_ok": False,
            "invalid_count": 1,
        },
    )

    code, output = invoke(
        ["--store", str(store), "--json", "benchmark"],
        capsys,
    )

    assert code == 1
    assert json.loads(output.out)["passed"] is False


@pytest.mark.parametrize(
    "minimum_recall",
    ["-0.01", "1.01", "nan"],
)
def test_benchmark_rejects_invalid_minimum_recall(capsys, minimum_recall) -> None:
    code, output = invoke(
        ["benchmark", "--minimum-recall", minimum_recall],
        capsys,
    )

    assert code == 2
    assert "must be between 0 and 1" in output.err


@pytest.mark.parametrize(
    ("configs", "arguments", "error"),
    [
        (
            ({"name": "epoch", "adapter": "csv", "enabled": False},),
            ["--name", "epoch"],
            "unknown or disabled benchmark: epoch",
        ),
        (
            (
                {"name": "one", "adapter": "csv"},
                {"name": "two", "adapter": "csv"},
            ),
            [],
            "multiple benchmarks are enabled",
        ),
    ],
)
def test_benchmark_reports_invalid_selection_as_a_usage_error(
    tmp_path,
    capsys,
    monkeypatch,
    configs,
    arguments,
    error,
) -> None:
    store = tmp_path / "store"
    init_code, _ = invoke(["--store", str(store), "--json", "init"], capsys)
    assert init_code == 0
    monkeypatch.setattr("modelome.cli.load_benchmark_configs", lambda path: configs)

    code, output = invoke(
        ["--store", str(store), "--json", "benchmark", *arguments],
        capsys,
    )

    assert code == 2
    assert error in json.loads(output.out)["error"]


def test_benchmark_does_not_implicitly_create_a_registry_store(tmp_path, capsys) -> None:
    store = tmp_path / "missing-store"

    code, output = invoke(
        ["--store", str(store), "--json", "benchmark"],
        capsys,
    )

    assert code == 2
    assert "registry store is not initialized" in json.loads(output.out)["error"]
    assert not store.exists()


def test_coverage_command_uses_missing_as_a_gate(tmp_path, capsys) -> None:
    store = tmp_path / "store"
    manifest = tmp_path / "coverage.csv"
    manifest.write_text("Model,Bucket\nMissingNet,smoke\n", encoding="utf-8")

    code, output = invoke(
        [
            "--store",
            str(store),
            "--json",
            "coverage",
            "--manifest",
            str(manifest),
        ],
        capsys,
    )

    assert code == 1
    report = json.loads(output.out)
    assert report["total"] == 1
    assert report["missing"] == 1


def test_coverage_command_reports_invalid_manifest_as_usage_error(tmp_path, capsys) -> None:
    manifest = tmp_path / "coverage.csv"
    manifest.write_text("Name,Bucket\nA,smoke\n", encoding="utf-8")

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "coverage",
            "--manifest",
            str(manifest),
        ],
        capsys,
    )

    assert code == 2
    assert "missing column" in json.loads(output.out)["error"]


def test_backfill_command_delegates_with_safe_default_budget(tmp_path, capsys, monkeypatch) -> None:
    source = OpenAlexSourceAdapter(client=object())
    captured = {}

    def run_backfill(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return BackfillOutcome(
            source="historical-window",
            status="partial",
            run_id=1,
            stats={"pages": 100},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"openalex": source})
    monkeypatch.setattr("modelome.cli.run_openalex_backfill", run_backfill)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "backfill",
            "--from",
            "1990-01-01",
            "--to",
            "1990-12-31",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "partial"
    assert captured == {
        "adapter": source,
        "from_date": "1990-01-01",
        "to_date": "1990-12-31",
        "max_pages": DEFAULT_BACKFILL_MAX_PAGES,
        "namespace": None,
    }


def test_backfill_command_dispatches_biorxiv_source(tmp_path, capsys, monkeypatch) -> None:
    source = BioRxivSourceAdapter(
        name="medrxiv",
        url="https://api.biorxiv.org/details",
        server="medrxiv",
        client=object(),
    )
    captured = {}

    def run_backfill(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return BackfillOutcome(
            source="medrxiv:backfill:2020-01-01:2020-12-31",
            status="complete",
            run_id=1,
            stats={"pages": 1},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"medrxiv": source})
    monkeypatch.setattr("modelome.cli.run_biorxiv_backfill", run_backfill)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "backfill",
            "--source",
            "medrxiv",
            "--from",
            "2020-01-01",
            "--to",
            "2020-12-31",
            "--max-pages",
            "17",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "complete"
    assert captured == {
        "adapter": source,
        "from_date": "2020-01-01",
        "to_date": "2020-12-31",
        "max_pages": 17,
        "namespace": None,
    }


def test_backfill_command_dispatches_osf_preprints_source(
    tmp_path, capsys, monkeypatch
) -> None:
    source = OsfPreprintSourceAdapter(client=object())
    captured = {}

    def run_backfill(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return BackfillOutcome(
            source="osf-preprints:backfill:2020-01-01:2020-12-31",
            status="complete",
            run_id=1,
            stats={"pages": 1},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"osf-preprints": source})
    monkeypatch.setattr("modelome.cli.run_osf_preprints_backfill", run_backfill)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "backfill",
            "--source",
            "osf-preprints",
            "--from",
            "2020-01-01",
            "--to",
            "2020-12-31",
            "--max-pages",
            "17",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "complete"
    assert captured == {
        "adapter": source,
        "from_date": "2020-01-01",
        "to_date": "2020-12-31",
        "max_pages": 17,
        "namespace": None,
    }


def test_backfill_command_dispatches_eartharxiv_source(tmp_path, capsys, monkeypatch) -> None:
    source = EarthArxivSourceAdapter(client=object())
    captured = {}

    def run_backfill(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return BackfillOutcome(
            source="eartharxiv:backfill:2020-01-01:2020-12-31",
            status="complete",
            run_id=1,
            stats={"pages": 1},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"eartharxiv": source})
    monkeypatch.setattr("modelome.cli.run_eartharxiv_backfill", run_backfill)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "backfill",
            "--source",
            "eartharxiv",
            "--from",
            "2020-01-01",
            "--to",
            "2020-12-31",
            "--max-pages",
            "17",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "complete"
    assert captured == {
        "adapter": source,
        "from_date": "2020-01-01",
        "to_date": "2020-12-31",
        "max_pages": 17,
        "namespace": None,
    }


def test_backfill_command_dispatches_hal_source(tmp_path, capsys, monkeypatch) -> None:
    source = HalSourceAdapter(client=object())
    captured = {}

    def run_backfill(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return BackfillOutcome(
            source="hal:backfill:2020-01-01:2020-12-31",
            status="complete",
            run_id=1,
            stats={"pages": 1},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"hal": source})
    monkeypatch.setattr("modelome.cli.run_hal_backfill", run_backfill)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "backfill",
            "--source",
            "hal",
            "--from",
            "2020-01-01",
            "--to",
            "2020-12-31",
            "--max-pages",
            "17",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "complete"
    assert captured == {
        "adapter": source,
        "from_date": "2020-01-01",
        "to_date": "2020-12-31",
        "max_pages": 17,
        "namespace": None,
    }


def test_backfill_command_dispatches_arxiv_source(tmp_path, capsys, monkeypatch) -> None:
    source = ArxivSourceAdapter(client=object())
    captured = {}

    def run_backfill(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return BackfillOutcome(
            source="arxiv:backfill:2017-01-01:2017-12-31",
            status="partial",
            run_id=1,
            stats={"pages": 11},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"arxiv": source})
    monkeypatch.setattr("modelome.cli.run_arxiv_backfill", run_backfill)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "backfill",
            "--source",
            "arxiv",
            "--from",
            "2017-01-01",
            "--to",
            "2017-12-31",
            "--max-pages",
            "11",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "partial"
    assert captured == {
        "adapter": source,
        "from_date": "2017-01-01",
        "to_date": "2017-12-31",
        "max_pages": 11,
        "namespace": None,
    }


def test_backfill_command_dispatches_crossref_source(tmp_path, capsys, monkeypatch) -> None:
    source = CrossrefSourceAdapter(client=object())
    captured = {}

    def run_backfill(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return BackfillOutcome(
            source="crossref:backfill:2020-01-01:2020-12-31",
            status="complete",
            run_id=1,
            stats={"pages": 5},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"crossref": source})
    monkeypatch.setattr("modelome.cli.run_crossref_backfill", run_backfill)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "backfill",
            "--source",
            "crossref",
            "--from",
            "2020-01-01",
            "--to",
            "2020-12-31",
            "--max-pages",
            "5",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "complete"
    assert captured == {
        "adapter": source,
        "from_date": "2020-01-01",
        "to_date": "2020-12-31",
        "max_pages": 5,
        "namespace": None,
    }


def test_backfill_command_dispatches_europe_pmc_source(
    tmp_path, capsys, monkeypatch
) -> None:
    source = EuropePmcSourceAdapter(client=object())
    captured = {}

    def run_backfill(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return BackfillOutcome(
            source="europe-pmc:backfill:2020-01-01:2020-12-31",
            status="complete",
            run_id=1,
            stats={"pages": 5},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"europe-pmc": source})
    monkeypatch.setattr("modelome.cli.run_europe_pmc_backfill", run_backfill)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "backfill",
            "--source",
            "europe-pmc",
            "--from",
            "2020-01-01",
            "--to",
            "2020-12-31",
            "--max-pages",
            "5",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "complete"
    assert captured == {
        "adapter": source,
        "from_date": "2020-01-01",
        "to_date": "2020-12-31",
        "max_pages": 5,
        "namespace": None,
    }


def test_backfill_command_dispatches_datacite_source(
    tmp_path, capsys, monkeypatch
) -> None:
    source = DataCiteSourceAdapter(client=object())
    captured = {}

    def run_backfill(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return BackfillOutcome(
            source="datacite:backfill:2020-01-01:2020-12-31",
            status="complete",
            run_id=1,
            stats={"pages": 2},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"datacite": source})
    monkeypatch.setattr("modelome.cli.run_datacite_backfill", run_backfill)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "backfill",
            "--source",
            "datacite",
            "--from",
            "2020-01-01",
            "--to",
            "2020-12-31",
            "--max-pages",
            "2",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "complete"
    assert captured == {
        "adapter": source,
        "from_date": "2020-01-01",
        "to_date": "2020-12-31",
        "max_pages": 2,
        "namespace": None,
    }


def test_bootstrap_command_discovers_arxiv_history_with_safe_default_budget(
    tmp_path, capsys, monkeypatch
) -> None:
    source = ArxivSourceAdapter(client=object())
    captured = {}

    def run_bootstrap(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return BootstrapOutcome(
            source="arxiv:bootstrap",
            status="partial",
            run_id=1,
            stats={"pages": DEFAULT_BOOTSTRAP_MAX_PAGES},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"arxiv": source})
    monkeypatch.setattr("modelome.cli.run_arxiv_bootstrap", run_bootstrap)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "bootstrap",
            "--source",
            "arxiv",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "partial"
    assert captured == {
        "adapter": source,
        "max_pages": DEFAULT_BOOTSTRAP_MAX_PAGES,
        "namespace": None,
    }


def test_bootstrap_command_discovers_complete_pmc_history(
    tmp_path, capsys, monkeypatch
) -> None:
    source = PmcSourceAdapter(client=object())
    captured = {}

    def run_bootstrap(database, adapter, **kwargs):
        captured.update(kwargs)
        captured["adapter"] = adapter
        return PmcBootstrapOutcome(
            source="pmc:bootstrap",
            status="partial",
            run_id=1,
            stats={"pages": 7},
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"pmc": source})
    monkeypatch.setattr("modelome.cli.run_pmc_bootstrap", run_bootstrap)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "bootstrap",
            "--source",
            "pmc",
            "--max-pages",
            "7",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "partial"
    assert captured == {
        "adapter": source,
        "max_pages": 7,
        "namespace": None,
    }


def test_bootstrap_rejects_sources_without_a_complete_baseline(
    tmp_path, capsys, monkeypatch
) -> None:
    source = CrossrefSourceAdapter(client=object())
    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {"crossref": source})

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "bootstrap",
            "--source",
            "crossref",
        ],
        capsys,
    )

    assert code == 2
    assert "complete bootstrap is not implemented" in json.loads(output.out)["error"]


def test_crawl_command_exits_nonzero_on_systematic_failure(tmp_path, capsys, monkeypatch) -> None:
    class FailedCrawler:
        def __init__(self, database):
            pass

        def crawl(self, **kwargs):
            return FrontierOutcome(
                status="failed",
                run_id=1,
                stats={"complete": False, "errors": ["all fetches failed"]},
            )

    monkeypatch.setattr("modelome.cli.FrontierCrawler", FailedCrawler)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "crawl",
        ],
        capsys,
    )

    assert code == 1
    assert json.loads(output.out)["status"] == "failed"


@pytest.mark.parametrize(
    ("errors", "expected_code"),
    [(["one URL failed"], 1), ([], 0)],
)
def test_crawl_partial_exit_reflects_errors_not_only_backlog(
    tmp_path, capsys, monkeypatch, errors, expected_code
) -> None:
    class PartialCrawler:
        def __init__(self, database):
            pass

        def crawl(self, **kwargs):
            return FrontierOutcome(
                status="partial",
                run_id=1,
                stats={"complete": False, "errors": errors},
            )

    monkeypatch.setattr("modelome.cli.FrontierCrawler", PartialCrawler)

    code, _ = invoke(
        ["--store", str(tmp_path / "store"), "--json", "crawl"],
        capsys,
    )

    assert code == expected_code


def test_bulk_load_defaults_to_every_enabled_bulk_control_source(
    tmp_path, capsys, monkeypatch
) -> None:
    sources = {
        "semantic-scholar-datasets": SemanticScholarDatasetSourceAdapter(
            api_key="fixture-key",
            client=object(),
        ),
        "pubmed-bulk": PubMedBulkSourceAdapter(client=object()),
        "commoncrawl-wet": CommonCrawlWetSourceAdapter(client=object()),
        "gharchive": GhArchiveSourceAdapter(),
        "software-heritage-origins": SoftwareHeritageOriginSourceAdapter(),
        "crossref": CrossrefSourceAdapter(client=object()),
    }
    monkeypatch.setattr("modelome.cli.load_sources", lambda path: sources)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--lake",
            str(tmp_path / "lake"),
            "--json",
            "bulk-load",
            "--max-new-shards",
            "0",
        ],
        capsys,
    )

    assert code == 0
    payload = json.loads(output.out)
    assert [item["control_source"] for item in payload["sources"]] == [
        "semantic-scholar-datasets",
        "pubmed-bulk",
        "commoncrawl-wet",
        "gharchive",
        "software-heritage-origins",
    ]
    assert all(item["selected"] == 0 for item in payload["sources"])
    assert (tmp_path / "lake" / "shards").is_dir()


def test_lake_status_reports_persistent_physical_counts_without_calling_them_models(
    tmp_path, capsys
) -> None:
    lake_path = tmp_path / "lake"
    lake = ParquetLandingZone(lake_path)
    shard = lake.commit_shard(
        source="semantic-scholar",
        dataset="papers",
        release="2026-09-01",
        shard="papers-000",
        control_sha256="a" * 64,
        upstream_sha256="b" * 64,
        records=(
            LakeRecord("corpus:1", {"corpusid": 1}),
            LakeRecord("corpus:2", {"corpusid": 2}),
        ),
    )
    lake.seal_release(
        source=shard.source,
        dataset=shard.dataset,
        release=shard.release,
        expected_shards={shard.shard: shard},
    )

    code, output = invoke(
        ["--lake", str(lake_path), "--json", "lake-status"],
        capsys,
    )

    assert code == 0
    payload = json.loads(output.out)
    assert payload["payload_verified"] is False
    assert payload["sealed_releases"] == 1
    assert payload["sealed_shards"] == 1
    assert payload["physical_rows"] == 2
    assert "not deduplicated" in payload["count_semantics"]
    assert payload["sources"] == [
        {
            "dataset": "papers",
            "physical_rows": 2,
            "sealed_releases": 1,
            "sealed_shards": 1,
            "source": "semantic-scholar",
        }
    ]


def test_project_command_is_restart_safe_and_serializes_the_projection_receipt(
    tmp_path, capsys, monkeypatch
) -> None:
    release = "2026-09-01"
    artifact_id = "a" * 64
    plan = ProjectionPlan(
        source="semantic-scholar",
        release=release,
        mode="diff",
        base_release="2026-08-01",
        transitions=((0, "2026-08-01", release),),
        input_release_sha256=(("papers", "b" * 64),),
    )
    receipt = ProjectionReceipt(
        source="semantic-scholar",
        release=release,
        artifact_id=artifact_id,
        row_count=123,
        part_count=4,
        bucket_count=256,
        path=tmp_path / "projection",
        already_materialized=False,
    )
    captured = {}

    def project(lake, **kwargs):
        captured["lake"] = lake
        captured.update(kwargs)
        return ProjectionOutcome(
            source="semantic-scholar",
            release=release,
            status="complete",
            plan=plan,
            receipt=receipt,
            created=True,
        )

    monkeypatch.setattr("modelome.cli.run_semantic_scholar_projection", project)
    lake_path = tmp_path / "lake"

    code, output = invoke(
        [
            "--lake",
            str(lake_path),
            "--json",
            "project",
            "--source",
            "semantic-scholar",
            "--release",
            release,
            "--base-artifact-id",
            "c" * 64,
        ],
        capsys,
    )

    assert code == 0
    payload = json.loads(output.out)
    assert payload["created"] is True
    assert payload["receipt"]["row_count"] == 123
    assert payload["receipt"]["path"] == str(receipt.path)
    assert captured["lake"].root == lake_path.resolve()
    assert {key: value for key, value in captured.items() if key != "lake"} == {
        "release": release,
        "base_artifact_id": "c" * 64,
        "target_artifact_id": None,
    }


def test_bulk_load_serializes_release_paths_and_reports_success(
    tmp_path, capsys, monkeypatch
) -> None:
    source = PubMedBulkSourceAdapter(client=object())
    release_path = tmp_path / "lake" / "release" / "RELEASE.json"

    class SuccessfulOrchestrator:
        def __init__(self, database, lake):
            pass

        def run(self, **kwargs):
            return BulkOutcome(
                control_source=kwargs["control_source"],
                selected=1,
                examined=1,
                represented=1,
                new=1,
                cached=0,
                rows=17,
                releases=(
                    ReleaseReceipt(
                        source="pubmed",
                        dataset="citations",
                        release="2026-update-00000001",
                        shard_count=1,
                        row_count=17,
                        application_mode="single",
                        path=release_path,
                    ),
                ),
                errors=(),
                control_complete=True,
                complete=True,
                budget_exhausted=False,
            )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {source.name: source})
    monkeypatch.setattr(
        "modelome.bulk_runtime.BulkControlOrchestrator", SuccessfulOrchestrator
    )

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--lake",
            str(tmp_path / "lake"),
            "--json",
            "bulk-load",
            "--source",
            source.name,
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["sources"][0]["releases"][0]["path"] == str(
        release_path
    )


def test_bulk_load_exits_nonzero_on_loader_error(tmp_path, capsys, monkeypatch) -> None:
    source = PubMedBulkSourceAdapter(client=object())

    class FailedOrchestrator:
        def __init__(self, database, lake):
            pass

        def run(self, **kwargs):
            return BulkOutcome(
                control_source=kwargs["control_source"],
                selected=1,
                examined=1,
                represented=0,
                new=0,
                cached=0,
                rows=0,
                releases=(),
                errors=(BulkError("control-1", "load", "RuntimeError: failed"),),
                control_complete=True,
                complete=False,
                budget_exhausted=False,
            )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {source.name: source})
    monkeypatch.setattr("modelome.bulk_runtime.BulkControlOrchestrator", FailedOrchestrator)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--lake",
            str(tmp_path / "lake"),
            "--json",
            "bulk-load",
            "--source",
            source.name,
        ],
        capsys,
    )

    assert code == 1
    assert json.loads(output.out)["sources"][0]["errors"][0]["stage"] == "load"


def test_bulk_load_rejects_a_non_bulk_source(tmp_path, capsys, monkeypatch) -> None:
    source = CrossrefSourceAdapter(client=object())
    monkeypatch.setattr("modelome.cli.load_sources", lambda path: {source.name: source})

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "bulk-load",
            "--source",
            source.name,
        ],
        capsys,
    )

    assert code == 2
    assert "bulk loading is not implemented" in json.loads(output.out)["error"]


@pytest.mark.parametrize(("status", "expected_code"), [("partial", 0), ("failed", 1)])
def test_daily_command_runs_every_phase_with_bounded_budgets(
    tmp_path,
    capsys,
    monkeypatch,
    status,
    expected_code,
) -> None:
    sources = {"configured": object()}
    captured = {}

    def daily(database, lake, configured_sources, **kwargs):
        captured["database"] = database
        captured["lake"] = lake
        captured["sources"] = configured_sources
        captured.update(kwargs)
        return DailyOutcome(
            status=status,
            sync=(),
            bulk=(),
            bootstraps=(),
            frontier=None,
        )

    monkeypatch.setattr("modelome.cli.load_sources", lambda path: sources)
    monkeypatch.setattr("modelome.cli.run_daily", daily)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--lake",
            str(tmp_path / "lake"),
            "--json",
            "daily",
            "--max-pages",
            "11",
            "--max-new-shards",
            "3",
            "--bootstrap-max-pages",
            "13",
            "--no-alphaxiv",
            "--alphaxiv-limit",
            "19",
            "--no-artifact-relations",
            "--no-frontier",
            "--frontier-limit",
            "17",
            "--frontier-max-depth",
            "2",
        ],
        capsys,
    )

    assert code == expected_code
    assert json.loads(output.out)["status"] == status
    assert captured["sources"] is sources
    assert captured["lake"].root == (tmp_path / "lake").resolve()
    assert {
        key: captured[key]
        for key in (
            "max_pages",
            "max_new_shards",
            "bootstrap_max_pages",
            "frontier",
            "frontier_limit",
            "frontier_max_depth",
            "alphaxiv_enricher",
            "alphaxiv_limit",
            "artifact_relations",
        )
    } == {
        "max_pages": 11,
        "max_new_shards": 3,
        "bootstrap_max_pages": 13,
        "frontier": False,
        "frontier_limit": 17,
        "frontier_max_depth": 2,
        "alphaxiv_enricher": None,
        "alphaxiv_limit": 19,
        "artifact_relations": False,
    }


def test_enrich_alphaxiv_command_is_bounded_and_reports_outcome(
    tmp_path, capsys, monkeypatch
) -> None:
    captured = {}

    def enrich(database, **kwargs):
        captured["database"] = database
        captured.update(kwargs)
        return AlphaXivOutcome(
            source="alphaxiv",
            status="complete",
            run_id=1,
            discovered=0,
            eligible=0,
            selected=0,
            enriched=0,
            new_artifacts=0,
            new_revisions=0,
            models_touched=0,
            links_discovered=0,
            remaining=0,
        )

    monkeypatch.setattr("modelome.cli.run_alphaxiv_enrichment", enrich)

    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "enrich-alphaxiv",
            "--limit",
            "7",
            "--refresh-after-days",
            "30",
        ],
        capsys,
    )

    assert code == 0
    assert json.loads(output.out)["status"] == "complete"
    assert captured["limit"] == 7
    assert captured["refresh_after_days"] == 30


def test_link_artifacts_command_materializes_an_empty_current_projection(
    tmp_path, capsys
) -> None:
    code, output = invoke(
        [
            "--store",
            str(tmp_path / "store"),
            "--json",
            "link-artifacts",
        ],
        capsys,
    )

    assert code == 0
    payload = json.loads(output.out)
    assert payload["status"] == "complete"
    assert payload["receipt"]["row_count"] == 0
