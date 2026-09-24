from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from modelome.alphaxiv_runtime import run_alphaxiv_enrichment
from modelome.models import ArtifactKind, Identifier, SourceRecord
from modelome.storage import Database


def test_rejects_canonical_url_that_conflicts_with_exact_arxiv_identity(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    database.ingest_page(
        "primary",
        (
            SourceRecord(
                source_record_id="paper-1",
                kind=ArtifactKind.PAPER,
                canonical_url="https://papers.example/paper-1",
                title="Primary paper",
                raw={},
                identifiers=(Identifier("arxiv", "1706.03762"),),
            ),
        ),
        complete=True,
    )

    class ConflictingCanonicalUrl:
        def enrich(self, arxiv_id: str) -> SourceRecord:
            return SourceRecord(
                source_record_id=arxiv_id,
                kind=ArtifactKind.PAPER,
                canonical_url="https://arxiv.org/abs/2001.00001",
                title="Different paper",
                raw={},
                identifiers=(Identifier("arxiv", arxiv_id),),
            )

    outcome = run_alphaxiv_enrichment(
        database,
        enricher=ConflictingCanonicalUrl(),
        clock=lambda: datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
    )

    assert outcome.status == "partial"
    assert outcome.enriched == 0
    assert "canonical URL" in outcome.errors[0].error
    assert not [row for row in database.table_rows("artifacts") if row["source"] == "alphaxiv"]
