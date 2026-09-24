from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from modelome.alphaxiv_runtime import AlphaXivOutcome, ExactIdEnricher, run_alphaxiv_enrichment
from modelome.artifact_relation_runtime import (
    ArtifactRelationOutcome,
    run_artifact_relation_projection,
)
from modelome.bootstrap import BootstrapOutcome, run_arxiv_bootstrap
from modelome.bulk import BulkError, BulkOutcome
from modelome.bulk_runtime import run_bulk_source, select_bulk_sources
from modelome.europe_pmc_bootstrap import (
    EuropePmcBootstrapOutcome,
    run_europe_pmc_bootstrap,
)
from modelome.frontier import FrontierCrawler, FrontierOutcome
from modelome.gharchive_projection import (
    GhArchiveProjectionOutcome,
    run_gharchive_repository_projection,
)
from modelome.lake import ParquetLandingZone
from modelome.pipeline import SyncEngine, SyncOutcome
from modelome.pmc_bootstrap import PmcBootstrapOutcome, run_pmc_bootstrap
from modelome.projection_runtime import ProjectionOutcome, run_semantic_scholar_projection
from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.sources.base import SourceAdapter
from modelome.sources.europe_pmc import EuropePmcSourceAdapter
from modelome.sources.gharchive import GhArchiveSourceAdapter
from modelome.sources.pmc import PmcSourceAdapter
from modelome.storage import Database


@dataclass(frozen=True, slots=True)
class DailyOutcome:
    """One bounded pass over current feeds, bulk shards, history, and links."""

    status: str
    sync: tuple[SyncOutcome, ...]
    bulk: tuple[BulkOutcome, ...]
    bootstraps: tuple[BootstrapOutcome | PmcBootstrapOutcome | EuropePmcBootstrapOutcome, ...]
    frontier: FrontierOutcome | None
    projections: tuple[ProjectionOutcome, ...] = ()
    enrichments: tuple[AlphaXivOutcome, ...] = ()
    relations: tuple[ArtifactRelationOutcome, ...] = ()
    repository_projections: tuple[GhArchiveProjectionOutcome, ...] = ()


def run_daily(
    database: Database,
    lake: ParquetLandingZone,
    sources: Mapping[str, SourceAdapter],
    *,
    max_pages: int = 1_000,
    max_new_shards: int | None = None,
    bootstrap_max_pages: int = 1_000,
    frontier: bool = True,
    frontier_limit: int = 200,
    frontier_max_depth: int = 1,
    alphaxiv_enricher: ExactIdEnricher | None = None,
    alphaxiv_limit: int = 100,
    artifact_relations: bool = True,
) -> DailyOutcome:
    """Advance every enabled coverage plane while keeping each phase resumable."""

    _positive(max_pages, "max_pages")
    if max_new_shards is not None:
        _nonnegative(max_new_shards, "max_new_shards")
    _positive(bootstrap_max_pages, "bootstrap_max_pages")
    _positive(frontier_limit, "frontier_limit")
    _nonnegative(frontier_max_depth, "frontier_max_depth")
    _positive(alphaxiv_limit, "alphaxiv_limit")
    lake.initialize()

    sync_outcomes = tuple(
        SyncEngine(database, sources).sync(max_pages=max_pages, fail_fast=False)
    )

    bulk_outcomes = []
    gharchive_releases = []
    for source in select_bulk_sources(sources):
        try:
            outcome = run_bulk_source(
                database,
                lake,
                source,
                max_new_shards=max_new_shards,
            )
        except Exception as error:
            outcome = _bulk_failure(source.name, error)
        bulk_outcomes.append(outcome)
        if isinstance(source, GhArchiveSourceAdapter):
            gharchive_releases.extend(outcome.releases)

    repository_projection_outcomes = _project_gharchive_releases(
        database,
        lake,
        gharchive_releases,
    )

    try:
        projection_outcomes = (run_semantic_scholar_projection(lake),)
    except Exception as error:
        projection_outcomes = (_projection_failure("semantic-scholar", error),)

    bootstrap_outcomes = []
    for source in sources.values():
        try:
            if isinstance(source, ArxivSourceAdapter):
                outcome = run_arxiv_bootstrap(
                    database,
                    source,
                    max_pages=bootstrap_max_pages,
                )
            elif isinstance(source, PmcSourceAdapter):
                outcome = run_pmc_bootstrap(
                    database,
                    source,
                    max_pages=bootstrap_max_pages,
                )
            elif isinstance(source, EuropePmcSourceAdapter):
                outcome = run_europe_pmc_bootstrap(
                    database,
                    source,
                    max_pages=bootstrap_max_pages,
                )
            else:
                continue
        except Exception as error:
            outcome = _bootstrap_failure(source, error)
        bootstrap_outcomes.append(outcome)

    enrichment_outcomes = (
        (
            run_alphaxiv_enrichment(
                database,
                enricher=alphaxiv_enricher,
                limit=alphaxiv_limit,
            ),
        )
        if alphaxiv_enricher is not None
        else ()
    )

    frontier_outcome = (
        FrontierCrawler(database).crawl(
            limit=frontier_limit,
            max_depth=frontier_max_depth,
        )
        if frontier
        else None
    )
    if artifact_relations:
        try:
            relation_outcomes = (run_artifact_relation_projection(database.root),)
        except Exception as error:
            relation_outcomes = (_relation_failure(error),)
    else:
        relation_outcomes = ()
    return DailyOutcome(
        status=_daily_status(
            sync_outcomes,
            tuple(bulk_outcomes),
            tuple(bootstrap_outcomes),
            frontier_outcome,
            projection_outcomes,
            enrichment_outcomes,
            relation_outcomes,
            repository_projection_outcomes,
        ),
        sync=sync_outcomes,
        bulk=tuple(bulk_outcomes),
        bootstraps=tuple(bootstrap_outcomes),
        frontier=frontier_outcome,
        projections=projection_outcomes,
        enrichments=enrichment_outcomes,
        relations=relation_outcomes,
        repository_projections=repository_projection_outcomes,
    )


def _project_gharchive_releases(
    database: Database,
    lake: ParquetLandingZone,
    releases: list[Any],
) -> tuple[GhArchiveProjectionOutcome, ...]:
    outcomes: list[GhArchiveProjectionOutcome] = []
    for receipt in sorted(releases, key=lambda item: _gharchive_hour(item.release)):
        try:
            outcome = run_gharchive_repository_projection(
                database,
                lake,
                release=receipt.release,
            )
        except Exception as error:
            outcome = _gharchive_projection_failure(receipt.release, error)
        if outcome.run_id is not None or outcome.status != "complete":
            outcomes.append(outcome)
        if outcome.status != "complete" or not outcome.complete:
            break
    return tuple(outcomes)


def _gharchive_hour(value: str) -> datetime:
    date, separator, raw_hour = value.rpartition("-")
    if not separator or not raw_hour.isascii() or not raw_hour.isdecimal():
        raise ValueError(f"invalid GH Archive release hour: {value!r}")
    hour = int(raw_hour)
    if not 0 <= hour <= 23:
        raise ValueError(f"invalid GH Archive release hour: {value!r}")
    try:
        return datetime.strptime(date, "%Y-%m-%d").replace(hour=hour)
    except ValueError as error:
        raise ValueError(f"invalid GH Archive release hour: {value!r}") from error


def _bulk_failure(source: str, error: Exception) -> BulkOutcome:
    return BulkOutcome(
        control_source=source,
        selected=0,
        examined=0,
        represented=0,
        new=0,
        cached=0,
        rows=0,
        releases=(),
        errors=(
            BulkError(
                source_record_id=None,
                stage="setup",
                error=f"{type(error).__name__}: {error}",
            ),
        ),
        control_complete=False,
        complete=False,
        budget_exhausted=False,
    )


def _bootstrap_failure(
    source: ArxivSourceAdapter | PmcSourceAdapter | EuropePmcSourceAdapter,
    error: Exception,
) -> BootstrapOutcome | PmcBootstrapOutcome | EuropePmcBootstrapOutcome:
    namespace = f"{source.name}:bootstrap"
    message = f"{type(error).__name__}: {error}"
    outcome_type = (
        PmcBootstrapOutcome
        if isinstance(source, PmcSourceAdapter)
        else EuropePmcBootstrapOutcome
        if isinstance(source, EuropePmcSourceAdapter)
        else BootstrapOutcome
    )
    return outcome_type(
        source=namespace,
        status="failed",
        run_id=None,
        stats={
            "source": namespace,
            "complete": False,
            "errors": [message],
        },
        error=message,
    )


def _projection_failure(source: str, error: Exception) -> ProjectionOutcome:
    return ProjectionOutcome(
        source=source,
        release=None,
        status="failed",
        plan=None,
        receipt=None,
        error=f"{type(error).__name__}: {error}",
    )


def _relation_failure(error: Exception) -> ArtifactRelationOutcome:
    return ArtifactRelationOutcome(
        status="failed",
        source_commit=None,
        receipt=None,
        error=f"{type(error).__name__}: {error}",
    )


def _gharchive_projection_failure(
    release: str,
    error: Exception,
) -> GhArchiveProjectionOutcome:
    return GhArchiveProjectionOutcome(
        status="failed",
        run_id=None,
        release=release,
        start_row=0,
        next_row=0,
        total_rows=0,
        rows_examined=0,
        repositories=0,
        links_discovered=0,
        complete=False,
        error=f"{type(error).__name__}: {error}",
    )


def _daily_status(
    sync: tuple[SyncOutcome, ...],
    bulk: tuple[BulkOutcome, ...],
    bootstraps: tuple[BootstrapOutcome | PmcBootstrapOutcome, ...],
    frontier: FrontierOutcome | None,
    projections: tuple[ProjectionOutcome, ...],
    enrichments: tuple[AlphaXivOutcome, ...],
    relations: tuple[ArtifactRelationOutcome, ...],
    repository_projections: tuple[GhArchiveProjectionOutcome, ...],
) -> str:
    failed = (
        any(outcome.status == "failed" for outcome in sync)
        or any(outcome.errors for outcome in bulk)
        or any(outcome.status == "failed" for outcome in bootstraps)
        or any(outcome.status == "failed" for outcome in projections)
        or any(outcome.status == "failed" for outcome in enrichments)
        or any(outcome.status == "failed" for outcome in relations)
        or any(outcome.status == "failed" for outcome in repository_projections)
        or (
            frontier is not None
            and (frontier.status == "failed" or bool(frontier.stats.get("errors")))
        )
    )
    if failed:
        return "failed"
    partial = (
        any(outcome.status == "partial" for outcome in sync)
        or any(not outcome.complete for outcome in bulk)
        or any(outcome.status == "partial" for outcome in bootstraps)
        or any(outcome.status == "partial" for outcome in projections)
        or any(outcome.status == "partial" for outcome in enrichments)
        or any(outcome.status == "partial" for outcome in repository_projections)
        or (frontier is not None and frontier.status == "partial")
    )
    return "partial" if partial else "complete"


def _positive(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


__all__ = ["DailyOutcome", "run_daily"]
