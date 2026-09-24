from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from modelome.alphaxiv_runtime import run_alphaxiv_enrichment
from modelome.models import ArtifactKind, Identifier, SourceRecord
from modelome.storage import Database

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


class FakeEnricher:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[str] = []

    def enrich(self, arxiv_id: str) -> SourceRecord:
        self.calls.append(arxiv_id)
        if arxiv_id in self.failures:
            raise RuntimeError("upstream paper is temporarily unavailable")
        return SourceRecord(
            source_record_id=arxiv_id,
            kind=ArtifactKind.PAPER,
            canonical_url=f"https://arxiv.org/abs/{arxiv_id}",
            title=f"Paper {arxiv_id}",
            raw={"provider": "alphaXiv"},
            text="We introduce a neural architecture named Example Network.",
            identifiers=(Identifier("arxiv", arxiv_id),),
        )


def _database(path: Path, *arxiv_ids: str) -> Database:
    database = Database(path)
    database.initialize()
    records = tuple(
        SourceRecord(
            source_record_id=f"paper-{index}",
            kind=ArtifactKind.PAPER,
            canonical_url=f"https://papers.example/{index}",
            title=f"Primary paper {index}",
            raw={},
            identifiers=(Identifier("arxiv", arxiv_id),),
        )
        for index, arxiv_id in enumerate(arxiv_ids)
    )
    database.ingest_page("primary", records, next_state={"done": True}, complete=True)
    return database


def test_bounded_runs_enrich_every_discovered_exact_id_once(tmp_path: Path) -> None:
    database = _database(tmp_path / "store", "1706.03762", "2001.00001")
    # A duplicate identity from another primary source must not duplicate work.
    database.ingest_page(
        "secondary",
        (
            SourceRecord(
                source_record_id="duplicate",
                kind=ArtifactKind.PAPER,
                canonical_url="https://secondary.example/paper",
                title="Duplicate",
                raw={},
                identifiers=(Identifier("arxiv", "1706.03762"),),
            ),
        ),
        complete=True,
    )
    enricher = FakeEnricher()

    first = run_alphaxiv_enrichment(
        database,
        enricher=enricher,
        limit=1,
        clock=lambda: NOW,
    )
    second = run_alphaxiv_enrichment(
        database,
        enricher=enricher,
        limit=1,
        clock=lambda: NOW,
    )
    third = run_alphaxiv_enrichment(
        database,
        enricher=enricher,
        limit=1,
        clock=lambda: NOW,
    )

    assert first.status == "partial"
    assert first.discovered == 2
    assert first.selected == first.enriched == 1
    assert second.status == "complete"
    assert second.selected == second.enriched == 1
    assert third.status == "complete"
    assert third.selected == third.enriched == 0
    assert sorted(enricher.calls) == ["1706.03762", "2001.00001"]
    alpha_artifacts = [
        row for row in database.table_rows("artifacts") if row["source"] == "alphaxiv"
    ]
    assert {row["source_record_id"] for row in alpha_artifacts} == {
        "1706.03762",
        "2001.00001",
    }
    assert database.search_models("Example Network")[0]["name"].rstrip(".") == (
        "Example Network"
    )


def test_prior_failure_does_not_starve_new_exact_ids(tmp_path: Path) -> None:
    database = _database(tmp_path / "store", "1706.03762", "2001.00001")
    enricher = FakeEnricher({"1706.03762"})

    first = run_alphaxiv_enrichment(
        database,
        enricher=enricher,
        limit=1,
        clock=lambda: NOW,
    )
    second = run_alphaxiv_enrichment(
        database,
        enricher=enricher,
        limit=1,
        clock=lambda: NOW,
    )

    assert first.status == "partial"
    assert first.enriched == 0
    assert first.errors[0].arxiv_id == "1706.03762"
    assert second.enriched == 1
    assert enricher.calls == ["1706.03762", "2001.00001"]
    dead_letters = database.table_rows("dead_letters")
    assert dead_letters[0]["stage"] == "alphaxiv_exact_id_enrichment"
    assert dead_letters[0]["summary_json"] == '{"arxiv_id":"1706.03762"}'


def test_rejects_identity_substitution_without_poisoning_store(tmp_path: Path) -> None:
    database = _database(tmp_path / "store", "1706.03762")

    class WrongIdentity:
        def enrich(self, arxiv_id: str) -> SourceRecord:
            return SourceRecord(
                source_record_id="2001.00001",
                kind=ArtifactKind.PAPER,
                canonical_url="https://arxiv.org/abs/2001.00001",
                title="Wrong paper",
                raw={},
                identifiers=(Identifier("arxiv", "2001.00001"),),
            )

    outcome = run_alphaxiv_enrichment(
        database,
        enricher=WrongIdentity(),
        clock=lambda: NOW,
    )

    assert outcome.status == "partial"
    assert outcome.enriched == 0
    assert "different arXiv identity" in outcome.errors[0].error
    assert not [
        row for row in database.table_rows("artifacts") if row["source"] == "alphaxiv"
    ]


def test_refresh_is_explicit_and_validates_budgets_before_writes(tmp_path: Path) -> None:
    database = _database(tmp_path / "store", "1706.03762")
    enricher = FakeEnricher()
    run_alphaxiv_enrichment(database, enricher=enricher, clock=lambda: NOW)

    current_only = run_alphaxiv_enrichment(
        database,
        enricher=enricher,
        refresh_after_days=None,
        clock=lambda: NOW,
    )
    immediate_refresh = run_alphaxiv_enrichment(
        database,
        enricher=enricher,
        refresh_after_days=0,
        clock=lambda: NOW,
    )

    assert current_only.selected == 0
    assert immediate_refresh.selected == 1
    assert len(enricher.calls) == 2

    before = len(database.table_rows("sync_runs"))
    try:
        run_alphaxiv_enrichment(database, enricher=enricher, limit=0)
    except ValueError:
        pass
    else:  # pragma: no cover - assertion helper
        raise AssertionError("zero limit must fail")
    assert len(database.table_rows("sync_runs")) == before
