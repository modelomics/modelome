from __future__ import annotations

from pathlib import Path

import pytest

from modelome.alphaxiv_runtime import AlphaXivOutcome
from modelome.artifact_relation_runtime import ArtifactRelationOutcome
from modelome.bootstrap import BootstrapOutcome
from modelome.bulk import BulkError, BulkOutcome
from modelome.daily import run_daily
from modelome.frontier import FrontierOutcome
from modelome.gharchive_projection import GhArchiveProjectionOutcome
from modelome.lake import ParquetLandingZone, ReleaseReceipt
from modelome.pipeline import SyncOutcome
from modelome.pmc_bootstrap import PmcBootstrapOutcome
from modelome.projection_runtime import ProjectionOutcome
from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.sources.gharchive import GhArchiveSourceAdapter
from modelome.sources.pmc import PmcSourceAdapter
from modelome.sources.pubmed import PubMedBulkSourceAdapter
from modelome.storage import Database


def test_daily_runs_current_bulk_history_and_frontier_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    arxiv = ArxivSourceAdapter(client=object())
    pubmed = PubMedBulkSourceAdapter(client=object())
    sources = {arxiv.name: arxiv, pubmed.name: pubmed}

    class FakeSyncEngine:
        def __init__(self, database, configured_sources):
            assert configured_sources is sources

        def sync(self, **kwargs):
            events.append("sync")
            assert kwargs == {"max_pages": 11, "fail_fast": False}
            return [
                SyncOutcome(
                    source=source.name,
                    status="complete",
                    run_id=index,
                    stats={"complete": True, "errors": []},
                )
                for index, source in enumerate(sources.values(), start=1)
            ]

    def bulk(database, lake, source, **kwargs):
        events.append("bulk")
        assert source is pubmed
        assert kwargs == {"max_new_shards": 3}
        return BulkOutcome(
            control_source=source.name,
            selected=4,
            examined=3,
            represented=3,
            new=3,
            cached=0,
            rows=30,
            releases=(),
            errors=(),
            control_complete=True,
            complete=False,
            budget_exhausted=True,
        )

    def bootstrap(database, source, **kwargs):
        events.append("bootstrap")
        assert source is arxiv
        assert kwargs == {"max_pages": 13}
        return BootstrapOutcome(
            source="arxiv:bootstrap",
            status="partial",
            run_id=3,
            stats={"complete": False, "errors": []},
        )

    def project(lake):
        events.append("project")
        return ProjectionOutcome(
            source="semantic-scholar",
            release=None,
            status="unavailable",
            plan=None,
            receipt=None,
        )

    alphaxiv_enricher = object()

    def enrich(database, **kwargs):
        events.append("alphaxiv")
        assert kwargs == {"enricher": alphaxiv_enricher, "limit": 23}
        return AlphaXivOutcome(
            source="alphaxiv",
            status="complete",
            run_id=5,
            discovered=2,
            eligible=2,
            selected=2,
            enriched=2,
            new_artifacts=2,
            new_revisions=2,
            models_touched=1,
            links_discovered=1,
            remaining=0,
        )

    class FakeCrawler:
        def __init__(self, database):
            pass

        def crawl(self, **kwargs):
            events.append("frontier")
            assert kwargs == {"limit": 17, "max_depth": 2}
            return FrontierOutcome(
                status="complete",
                run_id=4,
                stats={"complete": True, "errors": []},
            )

    def link(store_root):
        events.append("relations")
        assert store_root == database.root
        return ArtifactRelationOutcome(
            status="complete",
            source_commit="commit",
            receipt=None,
        )

    monkeypatch.setattr("modelome.daily.SyncEngine", FakeSyncEngine)
    monkeypatch.setattr("modelome.daily.run_bulk_source", bulk)
    monkeypatch.setattr("modelome.daily.run_semantic_scholar_projection", project)
    monkeypatch.setattr("modelome.daily.run_arxiv_bootstrap", bootstrap)
    monkeypatch.setattr("modelome.daily.run_alphaxiv_enrichment", enrich)
    monkeypatch.setattr("modelome.daily.FrontierCrawler", FakeCrawler)
    monkeypatch.setattr("modelome.daily.run_artifact_relation_projection", link)

    database = Database(tmp_path / "store")
    database.initialize()
    outcome = run_daily(
        database,
        ParquetLandingZone(tmp_path / "lake"),
        sources,
        max_pages=11,
        max_new_shards=3,
        bootstrap_max_pages=13,
        frontier_limit=17,
        frontier_max_depth=2,
        alphaxiv_enricher=alphaxiv_enricher,
        alphaxiv_limit=23,
    )

    assert events == [
        "sync",
        "bulk",
        "project",
        "bootstrap",
        "alphaxiv",
        "frontier",
        "relations",
    ]
    assert outcome.status == "partial"
    assert [item.source for item in outcome.sync] == ["arxiv", "pubmed-bulk"]
    assert outcome.bulk[0].budget_exhausted is True
    assert outcome.bootstraps[0].source == "arxiv:bootstrap"
    assert outcome.projections[0].status == "unavailable"
    assert outcome.enrichments[0].enriched == 2
    assert outcome.relations[0].source_commit == "commit"
    assert outcome.frontier is not None


def test_daily_isolates_bulk_and_bootstrap_setup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arxiv = ArxivSourceAdapter(client=object())
    pubmed = PubMedBulkSourceAdapter(client=object())
    sources = {arxiv.name: arxiv, pubmed.name: pubmed}

    class FakeSyncEngine:
        def __init__(self, database, configured_sources):
            pass

        def sync(self, **kwargs):
            return []

    monkeypatch.setattr("modelome.daily.SyncEngine", FakeSyncEngine)
    monkeypatch.setattr(
        "modelome.daily.run_bulk_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bulk setup")),
    )
    monkeypatch.setattr(
        "modelome.daily.run_arxiv_bootstrap",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad checkpoint")),
    )
    monkeypatch.setattr(
        "modelome.daily.run_semantic_scholar_projection",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad projection")),
    )

    outcome = run_daily(
        Database(tmp_path / "store"),
        ParquetLandingZone(tmp_path / "lake"),
        sources,
        frontier=False,
        artifact_relations=False,
    )

    assert outcome.status == "failed"
    assert outcome.bulk[0].errors == (
        BulkError(None, "setup", "RuntimeError: bulk setup"),
    )
    assert outcome.bootstraps[0].error == "ValueError: bad checkpoint"
    assert outcome.projections[0].error == "ValueError: bad projection"
    assert outcome.frontier is None


def test_daily_projects_sealed_github_activity_hours_in_chronological_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = GhArchiveSourceAdapter()
    sources = {source.name: source}
    lake = ParquetLandingZone(tmp_path / "lake")
    releases = tuple(
        ReleaseReceipt(
            source="gharchive",
            dataset="events",
            release=release,
            shard_count=1,
            row_count=1,
            application_mode="single",
            path=tmp_path / release / "RELEASE.json",
        )
        for release in ("2026-09-04-10", "2026-09-04-9")
    )

    class FakeSyncEngine:
        def __init__(self, database, configured_sources):
            assert configured_sources is sources

        def sync(self, **kwargs):
            return []

    monkeypatch.setattr("modelome.daily.SyncEngine", FakeSyncEngine)
    monkeypatch.setattr(
        "modelome.daily.run_bulk_source",
        lambda *args, **kwargs: BulkOutcome(
            control_source=source.name,
            selected=2,
            examined=2,
            represented=2,
            new=2,
            cached=0,
            rows=2,
            releases=releases,
            errors=(),
            control_complete=True,
            complete=True,
            budget_exhausted=False,
        ),
    )
    projected: list[str] = []

    def project(database, landing_zone, *, release):
        assert landing_zone is lake
        projected.append(release)
        return GhArchiveProjectionOutcome(
            status="complete",
            run_id=len(projected),
            release=release,
            start_row=0,
            next_row=1,
            total_rows=1,
            rows_examined=1,
            repositories=1,
            links_discovered=1,
            complete=True,
        )

    monkeypatch.setattr("modelome.daily.run_gharchive_repository_projection", project)
    monkeypatch.setattr(
        "modelome.daily.run_semantic_scholar_projection",
        lambda _lake: ProjectionOutcome(
            source="semantic-scholar",
            release=None,
            status="unavailable",
            plan=None,
            receipt=None,
        ),
    )

    outcome = run_daily(
        Database(tmp_path / "store"),
        lake,
        sources,
        frontier=False,
        artifact_relations=False,
    )

    assert projected == ["2026-09-04-9", "2026-09-04-10"]
    assert [item.release for item in outcome.repository_projections] == projected


def test_daily_bootstraps_complete_pmc_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pmc = PmcSourceAdapter(client=object())
    sources = {pmc.name: pmc}
    captured: dict[str, object] = {}

    class FakeSyncEngine:
        def __init__(self, database, configured_sources):
            assert configured_sources is sources

        def sync(self, **kwargs):
            return []

    def bootstrap(database, source, **kwargs):
        captured.update(kwargs)
        captured["source"] = source
        return PmcBootstrapOutcome(
            source="pmc:bootstrap",
            status="complete",
            run_id=1,
            stats={"complete": True, "errors": []},
        )

    monkeypatch.setattr("modelome.daily.SyncEngine", FakeSyncEngine)
    monkeypatch.setattr("modelome.daily.run_pmc_bootstrap", bootstrap)

    outcome = run_daily(
        Database(tmp_path / "store"),
        ParquetLandingZone(tmp_path / "lake"),
        sources,
        bootstrap_max_pages=19,
        frontier=False,
        artifact_relations=False,
    )

    assert captured == {"source": pmc, "max_pages": 19}
    assert outcome.status == "complete"
    assert outcome.bootstraps == (
        PmcBootstrapOutcome(
            source="pmc:bootstrap",
            status="complete",
            run_id=1,
            stats={"complete": True, "errors": []},
        ),
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_pages", 0),
        ("max_new_shards", -1),
        ("bootstrap_max_pages", 0),
        ("frontier_limit", 0),
        ("frontier_max_depth", -1),
        ("alphaxiv_limit", 0),
    ],
)
def test_daily_validates_every_budget_before_mutating(
    tmp_path: Path,
    keyword: str,
    value: int,
) -> None:
    arguments = {keyword: value}
    database = Database(tmp_path / "store")

    with pytest.raises(ValueError):
        run_daily(
            database,
            ParquetLandingZone(tmp_path / "lake"),
            {},
            **arguments,
        )

    assert not (tmp_path / "lake").exists()
