from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from modelome.models import SyncStats
from modelome.pipeline import SyncEngine, SyncOutcome
from modelome.sources.europe_pmc import EuropePmcBootstrapSource, EuropePmcSourceAdapter
from modelome.storage import Database

DEFAULT_EUROPE_PMC_BOOTSTRAP_MAX_PAGES = 1_000


@dataclass(frozen=True, slots=True)
class EuropePmcBootstrapOutcome:
    source: str
    status: str
    run_id: int | None
    stats: Mapping[str, Any]
    error: str | None = None
    already_complete: bool = False


class EuropePmcBootstrap:
    """Run finite, resumable monthly chunks of Europe PMC history."""

    def __init__(
        self,
        database: Database,
        adapter: EuropePmcSourceAdapter,
        *,
        earliest_update_date: date = date(1900, 1, 1),
        namespace: str | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.database = database
        self.source = EuropePmcBootstrapSource(
            adapter,
            earliest_update_date=earliest_update_date,
            namespace=namespace,
        )
        self.extractor = extractor

    @property
    def namespace(self) -> str:
        return self.source.name

    def run(
        self,
        *,
        max_pages: int = DEFAULT_EUROPE_PMC_BOOTSTRAP_MAX_PAGES,
    ) -> EuropePmcBootstrapOutcome:
        if not isinstance(max_pages, int) or isinstance(max_pages, bool) or max_pages < 1:
            raise ValueError("Europe PMC bootstrap requires a finite, positive max_pages budget")
        state = self.database.get_source_state(self.namespace)
        if self.source.is_complete(state):
            stats = SyncStats(source=self.namespace, complete=True)
            if "upstream_count" in state:
                stats.upstream_count = _nonnegative_int(
                    state["upstream_count"], "upstream_count", self.namespace
                )
            return EuropePmcBootstrapOutcome(
                source=self.namespace,
                status="complete",
                run_id=None,
                stats=stats.as_dict(),
                already_complete=True,
            )
        outcome = SyncEngine(
            self.database,
            {self.namespace: self.source},
            extractor=self.extractor,
        ).sync_source(self.source, max_pages=max_pages)
        return _outcome(outcome)


def run_europe_pmc_bootstrap(
    database: Database,
    adapter: EuropePmcSourceAdapter,
    *,
    max_pages: int = DEFAULT_EUROPE_PMC_BOOTSTRAP_MAX_PAGES,
    earliest_update_date: date = date(1900, 1, 1),
    namespace: str | None = None,
    extractor: Any | None = None,
) -> EuropePmcBootstrapOutcome:
    return EuropePmcBootstrap(
        database,
        adapter,
        earliest_update_date=earliest_update_date,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def _nonnegative_int(value: Any, field: str, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{source}: {field} must be a nonnegative integer")
    return value


def _outcome(outcome: SyncOutcome) -> EuropePmcBootstrapOutcome:
    return EuropePmcBootstrapOutcome(
        source=outcome.source,
        status=outcome.status,
        run_id=outcome.run_id,
        stats=outcome.stats,
        error=outcome.error,
    )


__all__ = [
    "DEFAULT_EUROPE_PMC_BOOTSTRAP_MAX_PAGES",
    "EuropePmcBootstrap",
    "EuropePmcBootstrapOutcome",
    "run_europe_pmc_bootstrap",
]
