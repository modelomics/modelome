from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any

from modelome.models import SourcePage, SyncStats
from modelome.normalize import content_hash
from modelome.pipeline import SyncEngine, SyncOutcome
from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.sources.biorxiv import BioRxivPublicationSourceAdapter, BioRxivSourceAdapter
from modelome.sources.crossref import CrossrefSourceAdapter
from modelome.sources.datacite import DataCiteSourceAdapter
from modelome.sources.eartharxiv import EarthArxivSourceAdapter
from modelome.sources.europe_pmc import EuropePmcSourceAdapter
from modelome.sources.hal import HalSourceAdapter
from modelome.sources.openalex import OpenAlexSourceAdapter
from modelome.sources.osf_preprints import OsfPreprintSourceAdapter
from modelome.storage import Database

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_BACKFILL_STATE_KEY = "backfill"
_BACKFILL_COMPLETE_KEY = "backfill_complete"
DEFAULT_BACKFILL_MAX_PAGES = 100
PreprintAdapter = (
    BioRxivSourceAdapter
    | BioRxivPublicationSourceAdapter
    | EarthArxivSourceAdapter
    | HalSourceAdapter
    | OsfPreprintSourceAdapter
)


@dataclass(frozen=True, slots=True)
class BackfillOutcome:
    source: str
    status: str
    run_id: int | None
    stats: Mapping[str, Any]
    error: str | None = None
    already_complete: bool = False


class OpenAlexBackfillSource:
    """Freeze one OpenAlex adapter to an inclusive historical date window."""

    def __init__(
        self,
        adapter: OpenAlexSourceAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.from_date = _parse_date(from_date, "from_date")
        self.to_date = _parse_date(to_date, "to_date")
        if self.from_date > self.to_date:
            raise ValueError("from_date must be on or before to_date")
        date_field = "updated_date" if adapter.sync_mode == "updated" else "publication_date"
        lowered_filter = adapter.filter.casefold()
        if any(
            boundary in lowered_filter
            for boundary in (f"from_{date_field}:", f"to_{date_field}:")
        ):
            raise ValueError(
                "OpenAlex adapter filter must not contain its own backfill date boundaries"
            )

        default_namespace = (
            f"{adapter.name}:backfill:{adapter.sync_mode}:"
            f"{self.from_date.isoformat()}:{self.to_date.isoformat()}"
        )
        self.name = (namespace or default_namespace).strip()
        if not self.name:
            raise ValueError("backfill namespace must not be empty")
        namespace_prefix = f"{adapter.name}:backfill:"
        if not self.name.startswith(namespace_prefix) or self.name == namespace_prefix:
            raise ValueError(
                f"backfill namespace must begin with {namespace_prefix!r} "
                "and include a nonempty suffix"
            )

        self.window_start = _start_of_day(self.from_date)
        self.window_end = _end_of_day(self.to_date)
        self.signature = _signature(adapter, self.from_date, self.to_date)
        self.artifact_source = adapter.name
        self.checkpoint_signature = content_hash(self.signature)

    def validate_state(self, state: Mapping[str, Any]) -> None:
        self._validate_adapter()
        if not state:
            return
        if state.get(_BACKFILL_STATE_KEY) != self.signature:
            raise ValueError(
                f"{self.name}: checkpoint does not match this backfill window or source config"
            )
        if state.get("window_start") != self.window_start:
            raise ValueError(f"{self.name}: checkpoint has a different from_date")
        if state.get("window_end") != self.window_end:
            raise ValueError(f"{self.name}: checkpoint has a different to_date")

    def is_complete(self, state: Mapping[str, Any]) -> bool:
        self.validate_state(state)
        return _completion_flag(state, self.name)

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        self.validate_state(state)
        if _completion_flag(state, self.name):
            raise RuntimeError(f"{self.name}: completed backfill must not be fetched again")

        adapter_state = {
            key: value
            for key, value in state.items()
            if key not in {_BACKFILL_STATE_KEY, _BACKFILL_COMPLETE_KEY, "upstream_count"}
        }
        # Always overwrite these values. A cursor can resume, but it can never
        # move the immutable historical boundary stored with this namespace.
        adapter_state["window_start"] = self.window_start
        adapter_state["window_end"] = self.window_end
        page = self.adapter.fetch_page(adapter_state)

        next_state = dict(page.next_state)
        next_state["window_start"] = self.window_start
        next_state["window_end"] = self.window_end
        next_state[_BACKFILL_STATE_KEY] = dict(self.signature)
        next_state[_BACKFILL_COMPLETE_KEY] = page.complete
        if page.upstream_count is not None:
            next_state["upstream_count"] = page.upstream_count
        retry_state = None
        if page.retry_state is not None:
            retry_state = dict(page.retry_state)
            retry_state["window_start"] = self.window_start
            retry_state["window_end"] = self.window_end
            retry_state[_BACKFILL_STATE_KEY] = dict(self.signature)
            retry_state[_BACKFILL_COMPLETE_KEY] = False
            if state.get("upstream_count") is not None:
                retry_state["upstream_count"] = state["upstream_count"]
        return SourcePage(
            records=page.records,
            next_state=next_state,
            complete=page.complete,
            upstream_count=page.upstream_count,
            issues=page.issues,
            retry_state=retry_state,
        )

    def _validate_adapter(self) -> None:
        current = _signature(self.adapter, self.from_date, self.to_date)
        if current != self.signature:
            raise ValueError(f"{self.name}: OpenAlex adapter config changed during backfill")


class OpenAlexBackfill:
    """Run and resume a durable OpenAlex historical backfill."""

    def __init__(
        self,
        database: Database,
        adapter: OpenAlexSourceAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.database = database
        self.source = OpenAlexBackfillSource(
            adapter,
            from_date=from_date,
            to_date=to_date,
            namespace=namespace,
        )
        if namespace is not None:
            _reject_non_backfill_namespace_collision(database, self.source.name)
        self.extractor = extractor

    @property
    def namespace(self) -> str:
        return self.source.name

    def run(
        self,
        *,
        max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    ) -> BackfillOutcome:
        if max_pages is None or max_pages < 1:
            raise ValueError("backfills require a finite, positive max_pages budget")

        state = self.database.get_source_state(self.namespace)
        if self.source.is_complete(state):
            stats = SyncStats(source=self.namespace, complete=True)
            stats.upstream_count = _optional_int(state.get("upstream_count"))
            return BackfillOutcome(
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
        return _backfill_outcome(outcome)


class ArxivBackfillSource:
    """Freeze one arXiv OAI-PMH scan to an inclusive historical date window."""

    def __init__(
        self,
        adapter: ArxivSourceAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.from_date = _parse_date(from_date, "from_date")
        self.to_date = _parse_date(to_date, "to_date")
        if self.from_date > self.to_date:
            raise ValueError("from_date must be on or before to_date")

        default_namespace = (
            f"{adapter.name}:backfill:{self.from_date.isoformat()}:{self.to_date.isoformat()}"
        )
        self.name = (namespace or default_namespace).strip()
        if not self.name:
            raise ValueError("backfill namespace must not be empty")
        namespace_prefix = f"{adapter.name}:backfill:"
        if not self.name.startswith(namespace_prefix) or self.name == namespace_prefix:
            raise ValueError(
                f"backfill namespace must begin with {namespace_prefix!r} "
                "and include a nonempty suffix"
            )

        self.window_start = self.from_date.isoformat()
        self.window_end = self.to_date.isoformat()
        self.signature = _arxiv_signature(adapter, self.from_date, self.to_date)
        # Backfill checkpoints are isolated, but their records belong to the
        # daily source's artifact namespace so the same paper cannot duplicate.
        self.artifact_source = adapter.name
        self.checkpoint_signature = content_hash(self.signature)

    def validate_state(self, state: Mapping[str, Any]) -> None:
        self._validate_adapter()
        if not state:
            return
        if state.get(_BACKFILL_STATE_KEY) != self.signature:
            raise ValueError(
                f"{self.name}: checkpoint does not match this backfill window or source config"
            )
        if state.get("window_start") != self.window_start:
            raise ValueError(f"{self.name}: checkpoint has a different from_date")
        if state.get("window_end") != self.window_end:
            raise ValueError(f"{self.name}: checkpoint has a different to_date")

    def is_complete(self, state: Mapping[str, Any]) -> bool:
        self.validate_state(state)
        return _completion_flag(state, self.name)

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        self.validate_state(state)
        if _completion_flag(state, self.name):
            raise RuntimeError(f"{self.name}: completed backfill must not be fetched again")

        adapter_state = {
            key: value
            for key, value in state.items()
            if key not in {_BACKFILL_STATE_KEY, _BACKFILL_COMPLETE_KEY, "upstream_count"}
        }
        adapter_state["window_start"] = self.window_start
        adapter_state["window_end"] = self.window_end
        page = self.adapter.fetch_page(adapter_state)

        next_state = dict(page.next_state)
        next_state["window_start"] = self.window_start
        next_state["window_end"] = self.window_end
        next_state[_BACKFILL_STATE_KEY] = dict(self.signature)
        next_state[_BACKFILL_COMPLETE_KEY] = page.complete
        if page.upstream_count is not None:
            next_state["upstream_count"] = page.upstream_count

        retry_state = None
        if page.retry_state is not None:
            retry_state = dict(page.retry_state)
            retry_state["window_start"] = self.window_start
            retry_state["window_end"] = self.window_end
            retry_state[_BACKFILL_STATE_KEY] = dict(self.signature)
            retry_state[_BACKFILL_COMPLETE_KEY] = False
            if state.get("upstream_count") is not None:
                retry_state["upstream_count"] = state["upstream_count"]

        return SourcePage(
            records=page.records,
            next_state=next_state,
            complete=page.complete,
            upstream_count=page.upstream_count,
            issues=page.issues,
            retry_state=retry_state,
        )

    def _validate_adapter(self) -> None:
        current = _arxiv_signature(self.adapter, self.from_date, self.to_date)
        if current != self.signature:
            raise ValueError(f"{self.name}: arXiv adapter config changed during backfill")


class ArxivBackfill:
    """Run and resume a durable arXiv historical metadata backfill."""

    def __init__(
        self,
        database: Database,
        adapter: ArxivSourceAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.database = database
        self.source = ArxivBackfillSource(
            adapter,
            from_date=from_date,
            to_date=to_date,
            namespace=namespace,
        )
        if namespace is not None:
            _reject_non_backfill_namespace_collision(database, self.source.name)
        self.extractor = extractor

    @property
    def namespace(self) -> str:
        return self.source.name

    def run(
        self,
        *,
        max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    ) -> BackfillOutcome:
        if max_pages is None or max_pages < 1:
            raise ValueError("backfills require a finite, positive max_pages budget")

        state = self.database.get_source_state(self.namespace)
        if self.source.is_complete(state):
            stats = SyncStats(source=self.namespace, complete=True)
            stats.upstream_count = _optional_int(state.get("upstream_count"))
            return BackfillOutcome(
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
        return _backfill_outcome(outcome)


class CrossrefBackfillSource:
    """Freeze one Crossref index-date scan to an inclusive historical window."""

    def __init__(
        self,
        adapter: CrossrefSourceAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.from_date = _parse_date(from_date, "from_date")
        self.to_date = _parse_date(to_date, "to_date")
        if self.from_date > self.to_date:
            raise ValueError("from_date must be on or before to_date")

        default_namespace = (
            f"{adapter.name}:backfill:{self.from_date.isoformat()}:{self.to_date.isoformat()}"
        )
        self.name = (namespace or default_namespace).strip()
        if not self.name:
            raise ValueError("backfill namespace must not be empty")
        namespace_prefix = f"{adapter.name}:backfill:"
        if not self.name.startswith(namespace_prefix) or self.name == namespace_prefix:
            raise ValueError(
                f"backfill namespace must begin with {namespace_prefix!r} "
                "and include a nonempty suffix"
            )

        self.window_start = self.from_date.isoformat()
        self.window_end = self.to_date.isoformat()
        self.signature = _crossref_signature(adapter, self.from_date, self.to_date)
        self.artifact_source = adapter.name
        self.checkpoint_signature = content_hash(self.signature)

    def validate_state(self, state: Mapping[str, Any]) -> None:
        self._validate_adapter()
        if not state:
            return
        if state.get(_BACKFILL_STATE_KEY) != self.signature:
            raise ValueError(
                f"{self.name}: checkpoint does not match this backfill window or source config"
            )
        if state.get("window_start") != self.window_start:
            raise ValueError(f"{self.name}: checkpoint has a different from_date")
        if state.get("window_end") != self.window_end:
            raise ValueError(f"{self.name}: checkpoint has a different to_date")

    def is_complete(self, state: Mapping[str, Any]) -> bool:
        self.validate_state(state)
        return _completion_flag(state, self.name)

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        self.validate_state(state)
        if _completion_flag(state, self.name):
            raise RuntimeError(f"{self.name}: completed backfill must not be fetched again")

        adapter_state = {
            key: value
            for key, value in state.items()
            if key not in {_BACKFILL_STATE_KEY, _BACKFILL_COMPLETE_KEY, "upstream_count"}
        }
        adapter_state["window_start"] = self.window_start
        adapter_state["window_end"] = self.window_end
        page = self.adapter.fetch_page(adapter_state)

        next_state = dict(page.next_state)
        next_state["window_start"] = self.window_start
        next_state["window_end"] = self.window_end
        next_state[_BACKFILL_STATE_KEY] = dict(self.signature)
        next_state[_BACKFILL_COMPLETE_KEY] = page.complete
        if page.upstream_count is not None:
            next_state["upstream_count"] = page.upstream_count

        retry_state = None
        if page.retry_state is not None:
            retry_state = dict(page.retry_state)
            retry_state["window_start"] = self.window_start
            retry_state["window_end"] = self.window_end
            retry_state[_BACKFILL_STATE_KEY] = dict(self.signature)
            retry_state[_BACKFILL_COMPLETE_KEY] = False
            if state.get("upstream_count") is not None:
                retry_state["upstream_count"] = state["upstream_count"]

        return SourcePage(
            records=page.records,
            next_state=next_state,
            complete=page.complete,
            upstream_count=page.upstream_count,
            issues=page.issues,
            retry_state=retry_state,
        )

    def _validate_adapter(self) -> None:
        current = _crossref_signature(self.adapter, self.from_date, self.to_date)
        if current != self.signature:
            raise ValueError(f"{self.name}: Crossref adapter config changed during backfill")


class CrossrefBackfill:
    """Run and resume a durable Crossref historical metadata backfill."""

    def __init__(
        self,
        database: Database,
        adapter: CrossrefSourceAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.database = database
        self.source = CrossrefBackfillSource(
            adapter,
            from_date=from_date,
            to_date=to_date,
            namespace=namespace,
        )
        if namespace is not None:
            _reject_non_backfill_namespace_collision(database, self.source.name)
        self.extractor = extractor

    @property
    def namespace(self) -> str:
        return self.source.name

    def run(
        self,
        *,
        max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    ) -> BackfillOutcome:
        if max_pages is None or max_pages < 1:
            raise ValueError("backfills require a finite, positive max_pages budget")

        state = self.database.get_source_state(self.namespace)
        if self.source.is_complete(state):
            stats = SyncStats(source=self.namespace, complete=True)
            stats.upstream_count = _optional_int(state.get("upstream_count"))
            return BackfillOutcome(
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
        return _backfill_outcome(outcome)


class EuropePmcBackfillSource:
    """Freeze one Europe PMC update-date scan to an inclusive historical window."""

    def __init__(
        self,
        adapter: EuropePmcSourceAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.from_date = _parse_date(from_date, "from_date")
        self.to_date = _parse_date(to_date, "to_date")
        if self.from_date > self.to_date:
            raise ValueError("from_date must be on or before to_date")

        default_namespace = (
            f"{adapter.name}:backfill:{self.from_date.isoformat()}:"
            f"{self.to_date.isoformat()}"
        )
        self.name = (namespace or default_namespace).strip()
        if not self.name:
            raise ValueError("backfill namespace must not be empty")
        namespace_prefix = f"{adapter.name}:backfill:"
        if not self.name.startswith(namespace_prefix) or self.name == namespace_prefix:
            raise ValueError(
                f"backfill namespace must begin with {namespace_prefix!r} "
                "and include a nonempty suffix"
            )

        self.window_start = self.from_date.isoformat()
        self.window_end = self.to_date.isoformat()
        self.signature = _europe_pmc_signature(adapter, self.from_date, self.to_date)
        # Checkpoints remain isolated while daily and historical observations
        # share one artifact identity namespace.
        self.artifact_source = adapter.name
        self.checkpoint_signature = content_hash(self.signature)

    def validate_state(self, state: Mapping[str, Any]) -> None:
        self._validate_adapter()
        if not state:
            return
        if state.get(_BACKFILL_STATE_KEY) != self.signature:
            raise ValueError(
                f"{self.name}: checkpoint does not match this backfill window or source config"
            )
        if state.get("window_start") != self.window_start:
            raise ValueError(f"{self.name}: checkpoint has a different from_date")
        if state.get("window_end") != self.window_end:
            raise ValueError(f"{self.name}: checkpoint has a different to_date")

    def is_complete(self, state: Mapping[str, Any]) -> bool:
        self.validate_state(state)
        return _completion_flag(state, self.name)

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        self.validate_state(state)
        if _completion_flag(state, self.name):
            raise RuntimeError(f"{self.name}: completed backfill must not be fetched again")

        adapter_state = {
            key: value
            for key, value in state.items()
            if key not in {_BACKFILL_STATE_KEY, _BACKFILL_COMPLETE_KEY, "upstream_count"}
        }
        adapter_state["window_start"] = self.window_start
        adapter_state["window_end"] = self.window_end
        page = self.adapter.fetch_page(adapter_state)

        next_state = dict(page.next_state)
        next_state["window_start"] = self.window_start
        next_state["window_end"] = self.window_end
        next_state[_BACKFILL_STATE_KEY] = dict(self.signature)
        next_state[_BACKFILL_COMPLETE_KEY] = page.complete
        if page.upstream_count is not None:
            next_state["upstream_count"] = page.upstream_count

        retry_state = None
        if page.retry_state is not None:
            retry_state = dict(page.retry_state)
            retry_state["window_start"] = self.window_start
            retry_state["window_end"] = self.window_end
            retry_state[_BACKFILL_STATE_KEY] = dict(self.signature)
            retry_state[_BACKFILL_COMPLETE_KEY] = False
            if state.get("upstream_count") is not None:
                retry_state["upstream_count"] = state["upstream_count"]

        return SourcePage(
            records=page.records,
            next_state=next_state,
            complete=page.complete,
            upstream_count=page.upstream_count,
            issues=page.issues,
            retry_state=retry_state,
        )

    def _validate_adapter(self) -> None:
        current = _europe_pmc_signature(self.adapter, self.from_date, self.to_date)
        if current != self.signature:
            raise ValueError(f"{self.name}: Europe PMC adapter config changed during backfill")


class EuropePmcBackfill:
    """Run and resume a durable Europe PMC historical metadata backfill."""

    def __init__(
        self,
        database: Database,
        adapter: EuropePmcSourceAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.database = database
        self.source = EuropePmcBackfillSource(
            adapter,
            from_date=from_date,
            to_date=to_date,
            namespace=namespace,
        )
        if namespace is not None:
            _reject_non_backfill_namespace_collision(database, self.source.name)
        self.extractor = extractor

    @property
    def namespace(self) -> str:
        return self.source.name

    def run(
        self,
        *,
        max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    ) -> BackfillOutcome:
        if max_pages is None or max_pages < 1:
            raise ValueError("backfills require a finite, positive max_pages budget")

        state = self.database.get_source_state(self.namespace)
        if self.source.is_complete(state):
            stats = SyncStats(source=self.namespace, complete=True)
            stats.upstream_count = _optional_int(state.get("upstream_count"))
            return BackfillOutcome(
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
        return _backfill_outcome(outcome)


class DataCiteBackfillSource:
    """Freeze DataCite's public DOI index to an inclusive update-date window."""

    def __init__(
        self,
        adapter: DataCiteSourceAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
    ) -> None:
        if not isinstance(adapter, DataCiteSourceAdapter):
            raise TypeError("adapter must be a DataCiteSourceAdapter")
        self.adapter = adapter
        self.from_date = _parse_date(from_date, "from_date")
        self.to_date = _parse_date(to_date, "to_date")
        if self.from_date > self.to_date:
            raise ValueError("from_date must be on or before to_date")

        default_namespace = (
            f"{adapter.name}:backfill:{self.from_date.isoformat()}:"
            f"{self.to_date.isoformat()}"
        )
        self.name = (namespace or default_namespace).strip()
        if not self.name:
            raise ValueError("backfill namespace must not be empty")
        namespace_prefix = f"{adapter.name}:backfill:"
        if not self.name.startswith(namespace_prefix) or self.name == namespace_prefix:
            raise ValueError(
                f"backfill namespace must begin with {namespace_prefix!r} "
                "and include a nonempty suffix"
            )

        self.window_start = self.from_date.isoformat()
        self.window_end = self.to_date.isoformat()
        self.signature = _datacite_signature(adapter, self.from_date, self.to_date)
        self.artifact_source = adapter.name
        self.checkpoint_signature = content_hash(self.signature)

    def validate_state(self, state: Mapping[str, Any]) -> None:
        self._validate_adapter()
        if not state:
            return
        if state.get(_BACKFILL_STATE_KEY) != self.signature:
            raise ValueError(
                f"{self.name}: checkpoint does not match this backfill window or source config"
            )
        if state.get("window_start") != self.window_start:
            raise ValueError(f"{self.name}: checkpoint has a different from_date")
        if state.get("window_end") != self.window_end:
            raise ValueError(f"{self.name}: checkpoint has a different to_date")

    def is_complete(self, state: Mapping[str, Any]) -> bool:
        self.validate_state(state)
        return _completion_flag(state, self.name)

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        self.validate_state(state)
        if _completion_flag(state, self.name):
            raise RuntimeError(f"{self.name}: completed backfill must not be fetched again")

        adapter_state = {
            key: value
            for key, value in state.items()
            if key not in {_BACKFILL_STATE_KEY, _BACKFILL_COMPLETE_KEY, "upstream_count"}
        }
        adapter_state["window_start"] = self.window_start
        adapter_state["window_end"] = self.window_end
        page = self.adapter.fetch_page(adapter_state)

        next_state = dict(page.next_state)
        next_state["window_start"] = self.window_start
        next_state["window_end"] = self.window_end
        next_state[_BACKFILL_STATE_KEY] = dict(self.signature)
        next_state[_BACKFILL_COMPLETE_KEY] = page.complete
        if page.upstream_count is not None:
            next_state["upstream_count"] = page.upstream_count

        retry_state = None
        if page.retry_state is not None:
            retry_state = dict(page.retry_state)
            retry_state["window_start"] = self.window_start
            retry_state["window_end"] = self.window_end
            retry_state[_BACKFILL_STATE_KEY] = dict(self.signature)
            retry_state[_BACKFILL_COMPLETE_KEY] = False
            if state.get("upstream_count") is not None:
                retry_state["upstream_count"] = state["upstream_count"]

        return SourcePage(
            records=page.records,
            next_state=next_state,
            complete=page.complete,
            upstream_count=page.upstream_count,
            issues=page.issues,
            retry_state=retry_state,
        )

    def _validate_adapter(self) -> None:
        current = _datacite_signature(self.adapter, self.from_date, self.to_date)
        if current != self.signature:
            raise ValueError(f"{self.name}: DataCite adapter config changed during backfill")


class DataCiteBackfill:
    """Run and resume a durable DataCite historical metadata backfill."""

    def __init__(
        self,
        database: Database,
        adapter: DataCiteSourceAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.database = database
        self.source = DataCiteBackfillSource(
            adapter,
            from_date=from_date,
            to_date=to_date,
            namespace=namespace,
        )
        if namespace is not None:
            _reject_non_backfill_namespace_collision(database, self.source.name)
        self.extractor = extractor

    @property
    def namespace(self) -> str:
        return self.source.name

    def run(
        self,
        *,
        max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    ) -> BackfillOutcome:
        if max_pages is None or max_pages < 1:
            raise ValueError("backfills require a finite, positive max_pages budget")
        state = self.database.get_source_state(self.namespace)
        if self.source.is_complete(state):
            stats = SyncStats(source=self.namespace, complete=True)
            stats.upstream_count = _optional_int(state.get("upstream_count"))
            return BackfillOutcome(
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
        return _backfill_outcome(outcome)


class BioRxivBackfillSource:
    """Freeze one cursor-based preprint API stream to an inclusive historical window."""

    def __init__(
        self,
        adapter: PreprintAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.from_date = _parse_date(from_date, "from_date")
        self.to_date = _parse_date(to_date, "to_date")
        if self.from_date > self.to_date:
            raise ValueError("from_date must be on or before to_date")

        default_namespace = (
            f"{adapter.name}:backfill:{self.from_date.isoformat()}:{self.to_date.isoformat()}"
        )
        self.name = (namespace or default_namespace).strip()
        if not self.name:
            raise ValueError("backfill namespace must not be empty")
        namespace_prefix = f"{adapter.name}:backfill:"
        if not self.name.startswith(namespace_prefix) or self.name == namespace_prefix:
            raise ValueError(
                f"backfill namespace must begin with {namespace_prefix!r} "
                "and include a nonempty suffix"
            )

        self.window_start = self.from_date.isoformat()
        self.window_end = self.to_date.isoformat()
        self.signature = _preprint_signature(adapter, self.from_date, self.to_date)
        self.artifact_source = adapter.name
        self.checkpoint_signature = content_hash(self.signature)

    def validate_state(self, state: Mapping[str, Any]) -> None:
        self._validate_adapter()
        if not state:
            return
        if state.get(_BACKFILL_STATE_KEY) != self.signature:
            raise ValueError(
                f"{self.name}: checkpoint does not match this backfill window or source config"
            )
        if state.get("window_start") != self.window_start:
            raise ValueError(f"{self.name}: checkpoint has a different from_date")
        if state.get("window_end") != self.window_end:
            raise ValueError(f"{self.name}: checkpoint has a different to_date")

    def is_complete(self, state: Mapping[str, Any]) -> bool:
        self.validate_state(state)
        return _completion_flag(state, self.name)

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        self.validate_state(state)
        if _completion_flag(state, self.name):
            raise RuntimeError(f"{self.name}: completed backfill must not be fetched again")

        adapter_state = {
            key: value
            for key, value in state.items()
            if key not in {_BACKFILL_STATE_KEY, _BACKFILL_COMPLETE_KEY, "upstream_count"}
        }
        adapter_state["window_start"] = self.window_start
        adapter_state["window_end"] = self.window_end
        page = self.adapter.fetch_page(adapter_state)

        next_state = dict(page.next_state)
        next_state["window_start"] = self.window_start
        next_state["window_end"] = self.window_end
        next_state[_BACKFILL_STATE_KEY] = dict(self.signature)
        next_state[_BACKFILL_COMPLETE_KEY] = page.complete
        if page.upstream_count is not None:
            next_state["upstream_count"] = page.upstream_count

        retry_state = None
        if page.retry_state is not None:
            retry_state = dict(page.retry_state)
            retry_state["window_start"] = self.window_start
            retry_state["window_end"] = self.window_end
            retry_state[_BACKFILL_STATE_KEY] = dict(self.signature)
            retry_state[_BACKFILL_COMPLETE_KEY] = False
            if state.get("upstream_count") is not None:
                retry_state["upstream_count"] = state["upstream_count"]

        return SourcePage(
            records=page.records,
            next_state=next_state,
            complete=page.complete,
            upstream_count=page.upstream_count,
            issues=page.issues,
            retry_state=retry_state,
        )

    def _validate_adapter(self) -> None:
        current = _preprint_signature(self.adapter, self.from_date, self.to_date)
        if current != self.signature:
            raise ValueError(f"{self.name}: preprint adapter config changed during backfill")


class BioRxivBackfill:
    """Run and resume a durable cursor-based preprint metadata backfill."""

    def __init__(
        self,
        database: Database,
        adapter: PreprintAdapter,
        *,
        from_date: str | date,
        to_date: str | date,
        namespace: str | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.database = database
        self.source = BioRxivBackfillSource(
            adapter,
            from_date=from_date,
            to_date=to_date,
            namespace=namespace,
        )
        if namespace is not None:
            _reject_non_backfill_namespace_collision(database, self.source.name)
        self.extractor = extractor

    @property
    def namespace(self) -> str:
        return self.source.name

    def run(
        self,
        *,
        max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    ) -> BackfillOutcome:
        if max_pages is None or max_pages < 1:
            raise ValueError("backfills require a finite, positive max_pages budget")

        state = self.database.get_source_state(self.namespace)
        if self.source.is_complete(state):
            stats = SyncStats(source=self.namespace, complete=True)
            stats.upstream_count = _optional_int(state.get("upstream_count"))
            return BackfillOutcome(
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
        return _backfill_outcome(outcome)


def run_openalex_backfill(
    database: Database,
    adapter: OpenAlexSourceAdapter,
    *,
    from_date: str | date,
    to_date: str | date,
    max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> BackfillOutcome:
    """Run one resumable page budget for an inclusive historical window."""

    return OpenAlexBackfill(
        database,
        adapter,
        from_date=from_date,
        to_date=to_date,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def run_arxiv_backfill(
    database: Database,
    adapter: ArxivSourceAdapter,
    *,
    from_date: str | date,
    to_date: str | date,
    max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> BackfillOutcome:
    """Run one resumable arXiv OAI page budget for an inclusive date window."""

    return ArxivBackfill(
        database,
        adapter,
        from_date=from_date,
        to_date=to_date,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def run_crossref_backfill(
    database: Database,
    adapter: CrossrefSourceAdapter,
    *,
    from_date: str | date,
    to_date: str | date,
    max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> BackfillOutcome:
    """Run one resumable Crossref page budget for an inclusive index-date window."""

    return CrossrefBackfill(
        database,
        adapter,
        from_date=from_date,
        to_date=to_date,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def run_europe_pmc_backfill(
    database: Database,
    adapter: EuropePmcSourceAdapter,
    *,
    from_date: str | date,
    to_date: str | date,
    max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> BackfillOutcome:
    """Run one resumable Europe PMC page budget for an inclusive update window."""

    return EuropePmcBackfill(
        database,
        adapter,
        from_date=from_date,
        to_date=to_date,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def run_datacite_backfill(
    database: Database,
    adapter: DataCiteSourceAdapter,
    *,
    from_date: str | date,
    to_date: str | date,
    max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> BackfillOutcome:
    """Run one resumable DataCite page budget for an inclusive update window."""

    return DataCiteBackfill(
        database,
        adapter,
        from_date=from_date,
        to_date=to_date,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def run_biorxiv_backfill(
    database: Database,
    adapter: PreprintAdapter,
    *,
    from_date: str | date,
    to_date: str | date,
    max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> BackfillOutcome:
    """Run one resumable bioRxiv API page budget for an inclusive date window."""

    return BioRxivBackfill(
        database,
        adapter,
        from_date=from_date,
        to_date=to_date,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def run_osf_preprints_backfill(
    database: Database,
    adapter: OsfPreprintSourceAdapter,
    *,
    from_date: str | date,
    to_date: str | date,
    max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> BackfillOutcome:
    """Run one OSF preprints timestamp-window backfill page budget."""

    return BioRxivBackfill(
        database,
        adapter,
        from_date=from_date,
        to_date=to_date,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def run_eartharxiv_backfill(
    database: Database,
    adapter: EarthArxivSourceAdapter,
    *,
    from_date: str | date,
    to_date: str | date,
    max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> BackfillOutcome:
    """Run one EarthArXiv OAI-PMH timestamp-window backfill page budget."""

    return BioRxivBackfill(
        database,
        adapter,
        from_date=from_date,
        to_date=to_date,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def run_hal_backfill(
    database: Database,
    adapter: HalSourceAdapter,
    *,
    from_date: str | date,
    to_date: str | date,
    max_pages: int | None = DEFAULT_BACKFILL_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> BackfillOutcome:
    """Run one HAL OAI-PMH page budget for an inclusive historical date window."""

    return BioRxivBackfill(
        database,
        adapter,
        from_date=from_date,
        to_date=to_date,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def _signature(
    adapter: OpenAlexSourceAdapter,
    from_date: date,
    to_date: date,
) -> dict[str, Any]:
    return {
        "version": 1,
        "adapter": adapter.name,
        "url": adapter.url,
        "artifact_kind": adapter.artifact_kind.value,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "sync_mode": adapter.sync_mode,
        "filter": adapter.filter,
        "corpus": adapter.corpus,
    }


def _preprint_signature(
    adapter: PreprintAdapter,
    from_date: date,
    to_date: date,
) -> dict[str, Any]:
    signature = {
        "version": 1,
        "adapter": adapter.name,
        "adapter_type": type(adapter).__name__,
        "checkpoint_signature": adapter.checkpoint_signature,
        "url": adapter.url,
        "artifact_kind": adapter.artifact_kind.value,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
    }
    if isinstance(adapter, (BioRxivSourceAdapter, BioRxivPublicationSourceAdapter)):
        signature["server"] = adapter.server
    return signature


def _arxiv_signature(
    adapter: ArxivSourceAdapter,
    from_date: date,
    to_date: date,
) -> dict[str, Any]:
    return {
        "version": 1,
        "adapter": adapter.name,
        "adapter_type": type(adapter).__name__,
        "checkpoint_signature": adapter.checkpoint_signature,
        "url": adapter.url,
        "artifact_kind": adapter.artifact_kind.value,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
    }


def _crossref_signature(
    adapter: CrossrefSourceAdapter,
    from_date: date,
    to_date: date,
) -> dict[str, Any]:
    return {
        "version": 1,
        "adapter": adapter.name,
        "adapter_type": type(adapter).__name__,
        "checkpoint_signature": adapter.checkpoint_signature,
        "url": adapter.url,
        "artifact_kind": adapter.artifact_kind.value,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
    }


def _europe_pmc_signature(
    adapter: EuropePmcSourceAdapter,
    from_date: date,
    to_date: date,
) -> dict[str, Any]:
    return {
        "version": 1,
        "adapter": adapter.name,
        "adapter_type": type(adapter).__name__,
        "checkpoint_signature": adapter.checkpoint_signature,
        "url": adapter.url,
        "artifact_kind": adapter.artifact_kind.value,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
    }


def _datacite_signature(
    adapter: DataCiteSourceAdapter,
    from_date: date,
    to_date: date,
) -> dict[str, Any]:
    return {
        "version": 1,
        "adapter": adapter.name,
        "adapter_type": type(adapter).__name__,
        "checkpoint_signature": adapter.checkpoint_signature,
        "url": adapter.url,
        "artifact_kind": (
            adapter.artifact_kind.value
            if adapter.artifact_kind is not None
            else "resource_type"
        ),
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
    }


def _parse_date(value: str | date, field: str) -> date:
    if isinstance(value, datetime):
        raise ValueError(f"{field} must be a date without a time")
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or _DATE_RE.fullmatch(value.strip()) is None:
        raise ValueError(f"{field} must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as error:
        raise ValueError(f"invalid {field}: {value!r}") from error


def _start_of_day(value: date) -> str:
    return datetime.combine(value, time.min, tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _end_of_day(value: date) -> str:
    return datetime.combine(value, time.max, tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _completion_flag(state: Mapping[str, Any], source: str) -> bool:
    if _BACKFILL_COMPLETE_KEY not in state:
        return False
    value = state[_BACKFILL_COMPLETE_KEY]
    if not isinstance(value, bool):
        raise ValueError(f"{source}: backfill_complete checkpoint value must be boolean")
    return value


def _reject_non_backfill_namespace_collision(database: Database, namespace: str) -> None:
    existing = next(
        (item for item in database.source_status() if item["source"] == namespace),
        None,
    )
    if existing is None:
        return
    state = existing.get("state")
    if state and (not isinstance(state, Mapping) or _BACKFILL_STATE_KEY not in state):
        raise ValueError(
            f"{namespace}: custom backfill namespace collides with an existing "
            "non-backfill source checkpoint"
        )


def _backfill_outcome(outcome: SyncOutcome) -> BackfillOutcome:
    return BackfillOutcome(
        source=outcome.source,
        status=outcome.status,
        run_id=outcome.run_id,
        stats=outcome.stats,
        error=outcome.error,
    )


__all__ = [
    "ArxivBackfill",
    "ArxivBackfillSource",
    "BackfillOutcome",
    "BioRxivBackfill",
    "BioRxivBackfillSource",
    "CrossrefBackfill",
    "CrossrefBackfillSource",
    "DataCiteBackfill",
    "DataCiteBackfillSource",
    "DEFAULT_BACKFILL_MAX_PAGES",
    "EuropePmcBackfill",
    "EuropePmcBackfillSource",
    "OpenAlexBackfill",
    "OpenAlexBackfillSource",
    "run_arxiv_backfill",
    "run_biorxiv_backfill",
    "run_crossref_backfill",
    "run_datacite_backfill",
    "run_eartharxiv_backfill",
    "run_europe_pmc_backfill",
    "run_openalex_backfill",
    "run_osf_preprints_backfill",
]
