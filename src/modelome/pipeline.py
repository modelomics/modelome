from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any

from modelome.extract import IntroductionCueExtractor
from modelome.models import SyncStats
from modelome.sources.base import SourceAdapter
from modelome.storage import Database


@dataclass(frozen=True, slots=True)
class SyncOutcome:
    source: str
    status: str
    run_id: int
    stats: Mapping[str, Any]
    error: str | None = None


class SyncEngine:
    """Run independent, resumable source scans with page-level checkpoints."""

    def __init__(
        self,
        database: Database,
        sources: Mapping[str, SourceAdapter],
        *,
        extractor: Any | None = None,
    ) -> None:
        self.database = database
        self.sources = dict(sources)
        self.extractor = extractor or IntroductionCueExtractor()

    def sync(
        self,
        source_names: Iterable[str] | None = None,
        *,
        max_pages: int | None = None,
        fail_fast: bool = False,
    ) -> list[SyncOutcome]:
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be positive")
        names = list(source_names) if source_names is not None else list(self.sources)
        unknown = sorted(set(names) - self.sources.keys())
        if unknown:
            raise KeyError(f"unknown sources: {', '.join(unknown)}")

        outcomes = []
        for name in names:
            try:
                outcome = self.sync_source(self.sources[name], max_pages=max_pages)
            except Exception as error:
                if fail_fast:
                    raise
                message = f"{type(error).__name__}: {error}"
                outcome = SyncOutcome(
                    name,
                    "failed",
                    -1,
                    SyncStats(source=name, errors=[message]).as_dict(),
                    message,
                )
            outcomes.append(outcome)
            if fail_fast and outcome.status == "failed":
                break
        return outcomes

    def sync_source(
        self,
        source: SourceAdapter,
        *,
        max_pages: int | None = None,
    ) -> SyncOutcome:
        """Run one adapter directly, including adapters with isolated namespaces."""

        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be positive")
        return self._sync_one(source, max_pages=max_pages)

    def _sync_one(
        self,
        source: SourceAdapter,
        *,
        max_pages: int | None,
    ) -> SyncOutcome:
        name = source.name
        artifact_source = _artifact_source(source)
        run_id = self.database.start_run(name)
        stats = SyncStats(source=name)

        try:
            state = self.database.get_source_state(name)
            source_signature = _checkpoint_signature(source)
            stored_signature = state.get("_modelome_source_signature")
            if (
                source_signature is not None
                and stored_signature is not None
                and stored_signature != source_signature
            ):
                message = (
                    f"{name}: checkpoint belongs to a different source configuration; "
                    "use a new source name or explicitly reset that checkpoint"
                )
                stats.errors.append(message)
                self.database.finish_run(run_id, "failed", stats, error=message)
                return SyncOutcome(name, "failed", run_id, stats.as_dict(), message)

            comparison_state = dict(state)
            if source_signature is not None:
                comparison_state["_modelome_source_signature"] = source_signature
            prior_state_fingerprint = repr(sorted(comparison_state.items()))
            commit_pages = int(getattr(source, "commit_pages", 1))
            if commit_pages < 1:
                raise ValueError(f"{name}: commit_pages must be positive")

            while max_pages is None or stats.pages < max_pages:
                page_batch = []
                fetch_state = state
                fetch_error: Exception | None = None
                while (
                    len(page_batch) < commit_pages
                    and (max_pages is None or stats.pages + len(page_batch) < max_pages)
                ):
                    try:
                        page = source.fetch_page(fetch_state)
                    except Exception as error:
                        if not page_batch:
                            raise
                        fetch_error = error
                        break
                    retry_page = bool(page.issues) and not page.advance_on_source_issues
                    if retry_page:
                        # A quarantined upstream item is still part of this page. Keep
                        # the source cursor at the page boundary so a corrected item is
                        # observed on the next run instead of being skipped forever.
                        held_state = dict(
                            page.retry_state if page.retry_state is not None else fetch_state
                        )
                        if source_signature is not None:
                            held_state["_modelome_source_signature"] = source_signature
                        page = replace(
                            page,
                            next_state=held_state,
                            complete=False,
                            authoritative_snapshot=False,
                            retry_state=held_state,
                        )
                    elif source_signature is not None:
                        signed_retry_state = (
                            {
                                **page.retry_state,
                                "_modelome_source_signature": source_signature,
                            }
                            if page.retry_state is not None
                            else None
                        )
                        page = replace(
                            page,
                            next_state={
                                **page.next_state,
                                "_modelome_source_signature": source_signature,
                            },
                            retry_state=signed_retry_state,
                        )
                    page_batch.append(page)
                    next_fingerprint = repr(sorted(page.next_state.items()))
                    # Do not prefetch the same incomplete page repeatedly. The
                    # persisted checkpoint check below still raises as before.
                    if page.complete or retry_page or next_fingerprint == repr(
                        sorted(fetch_state.items())
                    ):
                        break
                    fetch_state = dict(page.next_state)

                transaction = (
                    self.database.ingest_batch()
                    if len(page_batch) > 1
                    else nullcontext()
                )
                failure_message: str | None = None
                checkpoint_error: str | None = None
                with transaction:
                    for page in page_batch:
                        page_result = self.database.ingest_page(
                            name,
                            page,
                            run_id=run_id,
                            extractor=(
                                "source-declared"
                                if getattr(source, "disable_derived_extraction", False)
                                else self.extractor
                            ),
                            artifact_source=artifact_source,
                        )
                        stats.pages += 1
                        stats.records_seen += page_result["records_seen"]
                        stats.new_artifacts += page_result["new_artifacts"]
                        stats.new_revisions += page_result["new_revisions"]
                        stats.models_touched += page_result["models_touched"]
                        stats.links_discovered += page_result["links_discovered"]
                        page_errors = int(page_result.get("errors", 0))
                        if page_errors and not (
                            page.advance_on_source_issues
                            and int(page_result.get("record_errors", 0)) == 0
                        ):
                            stats.complete = False
                            message = (
                                f"{name}: {page_errors} item(s) quarantined on page "
                                f"{stats.pages}; checkpoint retained for retry"
                            )
                            stats.errors.append(message)
                            failure_message = message
                            break
                        if page_errors:
                            stats.errors.append(
                                f"{name}: {page_errors} immutable source item(s) quarantined "
                                f"on page {stats.pages}; checkpoint advanced"
                            )
                        if not page_result.get("checkpoint_advanced", 1):
                            checkpoint_error = (
                                f"{name}: {page_result.get('record_errors', 0)} record(s) "
                                "failed extraction or persistence; checkpoint retained for retry"
                            )
                            break
                        stats.complete = page.complete
                        if page.upstream_count is not None:
                            stats.upstream_count = page.upstream_count
                        state = dict(page.next_state)

                        if page.complete:
                            break
                        fingerprint = repr(sorted(state.items()))
                        if fingerprint == prior_state_fingerprint:
                            raise RuntimeError(
                                f"{name}: incomplete source did not advance checkpoint state"
                            )
                        prior_state_fingerprint = fingerprint

                if failure_message is not None:
                    self.database.finish_run(run_id, "failed", stats, error=failure_message)
                    return SyncOutcome(
                        name, "failed", run_id, stats.as_dict(), failure_message
                    )
                if checkpoint_error is not None:
                    raise RuntimeError(checkpoint_error)
                if fetch_error is not None:
                    raise fetch_error
                if stats.complete:
                    break

            status = "complete" if stats.complete else "partial"
            self.database.finish_run(run_id, status, stats)
            return SyncOutcome(name, status, run_id, stats.as_dict())
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            stats.errors.append(message)
            self.database.finish_run(run_id, "failed", stats, error=message)
            return SyncOutcome(name, "failed", run_id, stats.as_dict(), message)


def _checkpoint_signature(source: SourceAdapter) -> str | None:
    value = getattr(source, "checkpoint_signature", None)
    if value is None:
        return None
    if callable(value):
        value = value()
    signature = str(value).strip()
    if not signature:
        raise ValueError(f"{source.name}: checkpoint signature must not be empty")
    return signature


def _artifact_source(source: SourceAdapter) -> str:
    value = getattr(source, "artifact_source", source.name)
    if callable(value):
        value = value()
    result = str(value).strip()
    if not result:
        raise ValueError(f"{source.name}: artifact_source must not be empty")
    return result


__all__ = ["SyncEngine", "SyncOutcome"]
