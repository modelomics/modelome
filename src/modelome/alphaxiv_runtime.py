from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from modelome.alphaxiv import (
    AlphaXivExactIdEnricher,
    AlphaXivMcpClient,
    normalize_discovered_arxiv_id,
)
from modelome.extract import IntroductionCueExtractor
from modelome.models import Identifier, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url
from modelome.storage import Database


class ExactIdEnricher(Protocol):
    def enrich(self, arxiv_id: str) -> SourceRecord: ...


@dataclass(frozen=True, slots=True)
class AlphaXivItemError:
    arxiv_id: str
    error: str


@dataclass(frozen=True, slots=True)
class AlphaXivOutcome:
    """One bounded pass over arXiv identities discovered by primary sources."""

    source: str
    status: str
    run_id: int
    discovered: int
    eligible: int
    selected: int
    enriched: int
    new_artifacts: int
    new_revisions: int
    models_touched: int
    links_discovered: int
    remaining: int
    errors: tuple[AlphaXivItemError, ...] = ()
    fatal_error: str | None = None


def run_alphaxiv_enrichment(
    database: Database,
    *,
    enricher: ExactIdEnricher | None = None,
    source: str = "alphaxiv",
    limit: int = 100,
    refresh_after_days: int | None = None,
    clock: Callable[[], datetime] | None = None,
) -> AlphaXivOutcome:
    """Enrich only exact arXiv IDs already retained from another source.

    alphaXiv is intentionally not an enumerator. Candidate IDs come exclusively
    from active, non-alphaXiv artifacts in the evidence store. New IDs are tried
    before prior failures so one unavailable paper cannot starve the queue.
    """

    if not isinstance(database, Database):
        raise TypeError("database must be a Database")
    source = _required_text(source, "source")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if refresh_after_days is not None and (
        isinstance(refresh_after_days, bool)
        or not isinstance(refresh_after_days, int)
        or refresh_after_days < 0
    ):
        raise ValueError("refresh_after_days must be a nonnegative integer or None")
    now = _utc_now(clock)
    database.initialize()
    discovered, eligible = _candidate_ids(
        database,
        source=source,
        now=now,
        refresh_after_days=refresh_after_days,
    )
    selected = eligible[:limit]
    run_id = database.start_run(source)
    worker = enricher
    errors: list[AlphaXivItemError] = []
    counts = {
        "enriched": 0,
        "new_artifacts": 0,
        "new_revisions": 0,
        "models_touched": 0,
        "links_discovered": 0,
    }

    try:
        if selected and worker is None:
            worker = AlphaXivExactIdEnricher(AlphaXivMcpClient())
        for arxiv_id in selected:
            try:
                if worker is None:  # pragma: no cover - guarded by selected
                    raise RuntimeError("alphaXiv enricher is unavailable")
                record = worker.enrich(arxiv_id)
                _validate_enrichment_record(record, arxiv_id)
            except Exception as error:
                message = _error_message(error)
                errors.append(AlphaXivItemError(arxiv_id=arxiv_id, error=message))
                database.ingest_page(
                    source,
                    SourcePage(
                        records=(),
                        next_state=database.get_source_state(source),
                        complete=False,
                        upstream_count=len(discovered),
                        issues=(
                            SourceIssue(
                                source_record_id=arxiv_id,
                                stage="alphaxiv_exact_id_enrichment",
                                error=message,
                                summary={"arxiv_id": arxiv_id},
                            ),
                        ),
                        retry_state=database.get_source_state(source),
                    ),
                    run_id=run_id,
                    extractor="source",
                    enqueue_links=False,
                )
                continue

            result = database.ingest_page(
                source,
                (record,),
                next_state=database.get_source_state(source),
                run_id=run_id,
                extractor=IntroductionCueExtractor(),
                complete=False,
                upstream_count=len(discovered),
            )
            counts["enriched"] += 1
            for field in (
                "new_artifacts",
                "new_revisions",
                "models_touched",
                "links_discovered",
            ):
                counts[field] += int(result[field])

        remaining = len(eligible) - counts["enriched"]
        status = "complete" if remaining == 0 else "partial"
        state = {
            "completed_at": _isoformat(now),
            "discovered": len(discovered),
            "eligible": len(eligible),
            "selected": len(selected),
            "enriched": counts["enriched"],
            "remaining": remaining,
        }
        database.ingest_page(
            source,
            (),
            next_state=state,
            run_id=run_id,
            extractor="source",
            complete=status == "complete",
            upstream_count=len(discovered),
            enqueue_links=False,
        )
        database.finish_run(
            run_id,
            status,
            {
                "source": source,
                **state,
                **counts,
                "errors": [item.error for item in errors],
                "complete": status == "complete",
            },
        )
        return AlphaXivOutcome(
            source=source,
            status=status,
            run_id=run_id,
            discovered=len(discovered),
            eligible=len(eligible),
            selected=len(selected),
            remaining=remaining,
            errors=tuple(errors),
            **counts,
        )
    except Exception as error:
        message = _error_message(error)
        database.finish_run(
            run_id,
            "failed",
            {
                "source": source,
                **counts,
                "errors": [*[item.error for item in errors], message],
                "complete": False,
            },
            error=message,
        )
        return AlphaXivOutcome(
            source=source,
            status="failed",
            run_id=run_id,
            discovered=len(discovered),
            eligible=len(eligible),
            selected=len(selected),
            remaining=len(eligible) - counts["enriched"],
            errors=tuple(errors),
            fatal_error=message,
            **counts,
        )


def _candidate_ids(
    database: Database,
    *,
    source: str,
    now: datetime,
    refresh_after_days: int | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    artifacts = {row["id"]: row for row in database.table_rows("artifacts")}
    discovered: set[str] = set()
    for row in database.table_rows("artifact_identifiers"):
        if row.get("namespace") != "arxiv":
            continue
        artifact = artifacts.get(row.get("artifact_id"))
        if artifact is None or artifact.get("active") != 1 or artifact.get("source") == source:
            continue
        try:
            discovered.add(normalize_discovered_arxiv_id(str(row.get("value", ""))))
        except ValueError:
            continue

    existing = {
        str(row["source_record_id"]): row
        for row in artifacts.values()
        if row.get("source") == source
    }
    cutoff = now - timedelta(days=refresh_after_days) if refresh_after_days is not None else None
    eligible: list[str] = []
    for arxiv_id in discovered:
        artifact = existing.get(arxiv_id)
        if artifact is None or artifact.get("active") != 1:
            eligible.append(arxiv_id)
            continue
        if cutoff is not None and (
            refresh_after_days == 0 or _stored_time(artifact.get("updated_at")) <= cutoff
        ):
            eligible.append(arxiv_id)

    failure_attempts = {
        str(row["source_record_id"]): int(row.get("attempts", 0))
        for row in database.table_rows("dead_letters")
        if row.get("source") == source and row.get("stage") == "alphaxiv_exact_id_enrichment"
    }
    eligible.sort(
        key=lambda arxiv_id: (
            arxiv_id in failure_attempts,
            failure_attempts.get(arxiv_id, 0),
            arxiv_id,
        )
    )
    return tuple(sorted(discovered)), tuple(eligible)


def _validate_enrichment_record(record: SourceRecord, arxiv_id: str) -> None:
    if not isinstance(record, SourceRecord):
        raise TypeError("alphaXiv enricher must return a SourceRecord")
    if record.deleted:
        raise ValueError("alphaXiv enrichment must not tombstone a primary paper identity")
    if record.source_record_id != arxiv_id:
        raise ValueError("alphaXiv enrichment returned a different arXiv identity")
    if Identifier("arxiv", arxiv_id) not in record.identifiers:
        raise ValueError("alphaXiv enrichment omitted its exact arXiv identifier")
    if canonicalize_url(record.canonical_url) != f"https://arxiv.org/abs/{arxiv_id}":
        raise ValueError(
            "alphaXiv enrichment returned a canonical URL that does not match "
            "its exact arXiv identity"
        )


def _utc_now(clock: Callable[[], datetime] | None) -> datetime:
    value = (clock or (lambda: datetime.now(UTC)))()
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _stored_time(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("stored alphaXiv artifact has no update timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("stored alphaXiv artifact has an invalid update timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("stored alphaXiv artifact timestamp is not timezone-aware")
    return parsed.astimezone(UTC)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _error_message(error: Exception) -> str:
    detail = str(error).strip() or type(error).__name__
    return f"{type(error).__name__}: {detail}"[:2_000]


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "AlphaXivItemError",
    "AlphaXivOutcome",
    "run_alphaxiv_enrichment",
]
