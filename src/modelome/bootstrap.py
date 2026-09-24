from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from modelome.models import SourcePage, SyncStats
from modelome.normalize import content_hash
from modelome.pipeline import SyncEngine, SyncOutcome
from modelome.sources.arxiv import ArxivRepositoryIdentity, ArxivSourceAdapter
from modelome.storage import Database

_BOOTSTRAP_KEY = "bootstrap"
_BOOTSTRAP_COMPLETE_KEY = "bootstrap_complete"
DEFAULT_BOOTSTRAP_MAX_PAGES = 1_000


@dataclass(frozen=True, slots=True)
class BootstrapOutcome:
    source: str
    status: str
    run_id: int | None
    stats: Mapping[str, Any]
    error: str | None = None
    already_complete: bool = False


class ArxivBootstrapSource:
    """Freeze an all-record arXiv harvest at repository-declared boundaries."""

    def __init__(
        self,
        adapter: ArxivSourceAdapter,
        *,
        namespace: str | None = None,
    ) -> None:
        if not isinstance(adapter, ArxivSourceAdapter):
            raise TypeError("adapter must be an ArxivSourceAdapter")
        self.adapter = adapter
        self.name = (namespace or f"{adapter.name}:bootstrap").strip()
        if not self.name:
            raise ValueError("bootstrap namespace must not be empty")
        self.artifact_source = adapter.name
        self.checkpoint_signature = content_hash(
            {
                "workflow": "arxiv-oai-bootstrap-v1",
                "adapter": adapter.checkpoint_signature,
                "namespace": self.name,
            }
        )

    def is_complete(self, state: Mapping[str, Any]) -> bool:
        if not state:
            return False
        self._descriptor(state)
        if _BOOTSTRAP_COMPLETE_KEY not in state:
            return False
        value = state[_BOOTSTRAP_COMPLETE_KEY]
        if not isinstance(value, bool):
            raise ValueError(
                f"{self.name}: bootstrap_complete checkpoint value must be boolean"
            )
        return value

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if self.is_complete(state):
            raise RuntimeError(f"{self.name}: completed bootstrap must not be fetched again")

        descriptor = self._descriptor(state) if state else self._start_descriptor()
        window_start = _required_date(
            descriptor.get("window_start"), "window_start", self.name
        )
        window_end = _required_date(
            descriptor.get("window_end"), "window_end", self.name
        )
        if window_start > window_end:
            return SourcePage(
                records=(),
                next_state={
                    _BOOTSTRAP_KEY: descriptor,
                    _BOOTSTRAP_COMPLETE_KEY: True,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "completed_at": _isoformat(_clock(self.adapter)),
                },
                complete=True,
                upstream_count=0,
            )

        adapter_state = {
            key: value
            for key, value in state.items()
            if key
            not in {
                _BOOTSTRAP_KEY,
                _BOOTSTRAP_COMPLETE_KEY,
                "_modelome_source_signature",
                "upstream_count",
            }
        }
        adapter_state["window_start"] = window_start.isoformat()
        adapter_state["window_end"] = window_end.isoformat()
        page = self.adapter.fetch_page(adapter_state)
        return SourcePage(
            records=page.records,
            next_state=self._wrapped_state(
                page.next_state,
                descriptor=descriptor,
                complete=page.complete,
                upstream_count=page.upstream_count,
            ),
            complete=page.complete,
            upstream_count=page.upstream_count,
            authoritative_snapshot=page.authoritative_snapshot,
            issues=page.issues,
            retry_state=(
                self._wrapped_state(
                    page.retry_state,
                    descriptor=descriptor,
                    complete=False,
                    upstream_count=state.get("upstream_count"),
                )
                if page.retry_state is not None
                else None
            ),
        )

    def _start_descriptor(self) -> dict[str, Any]:
        identity = self.adapter.identify()
        start = _datestamp_date(identity.earliest_datestamp, self.name)
        discovered_at = _clock(self.adapter)
        end = discovered_at.date() - timedelta(days=1)
        return {
            "version": 1,
            "adapter_checkpoint_signature": self.adapter.checkpoint_signature,
            "repository": _identity_payload(identity),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "discovered_at": _isoformat(discovered_at),
        }

    def _descriptor(self, state: Mapping[str, Any]) -> dict[str, Any]:
        raw = state.get(_BOOTSTRAP_KEY)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{self.name}: bootstrap checkpoint is missing its descriptor")
        descriptor = dict(raw)
        if descriptor.get("version") != 1:
            raise ValueError(f"{self.name}: unsupported bootstrap checkpoint version")
        if descriptor.get("adapter_checkpoint_signature") != self.adapter.checkpoint_signature:
            raise ValueError(
                f"{self.name}: bootstrap checkpoint belongs to a different arXiv adapter"
            )
        repository = descriptor.get("repository")
        if not isinstance(repository, Mapping):
            raise ValueError(f"{self.name}: bootstrap repository identity is invalid")
        if repository.get("base_url") != self.adapter.url:
            raise ValueError(f"{self.name}: bootstrap repository URL changed")
        _required_text(repository.get("earliest_datestamp"), "earliest_datestamp", self.name)
        start = _required_date(descriptor.get("window_start"), "window_start", self.name)
        end = _required_date(descriptor.get("window_end"), "window_end", self.name)
        expected_start = _datestamp_date(repository["earliest_datestamp"], self.name)
        if start != expected_start:
            raise ValueError(
                f"{self.name}: bootstrap lower boundary does not match OAI-PMH Identify"
            )
        _required_datetime(descriptor.get("discovered_at"), "discovered_at", self.name)
        if end >= _required_datetime(
            descriptor["discovered_at"], "discovered_at", self.name
        ).date():
            raise ValueError(f"{self.name}: bootstrap upper boundary is not a closed UTC day")
        return descriptor

    def _wrapped_state(
        self,
        adapter_state: Mapping[str, Any] | None,
        *,
        descriptor: Mapping[str, Any],
        complete: bool,
        upstream_count: Any,
    ) -> dict[str, Any]:
        result = dict(adapter_state or {})
        result["window_start"] = descriptor["window_start"]
        result["window_end"] = descriptor["window_end"]
        result[_BOOTSTRAP_KEY] = dict(descriptor)
        result[_BOOTSTRAP_COMPLETE_KEY] = complete
        if upstream_count is not None:
            result["upstream_count"] = upstream_count
        return result


class ArxivBootstrap:
    """Run bounded pieces of a durable, complete arXiv historical harvest."""

    def __init__(
        self,
        database: Database,
        adapter: ArxivSourceAdapter,
        *,
        namespace: str | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.database = database
        self.source = ArxivBootstrapSource(adapter, namespace=namespace)
        self.extractor = extractor

    @property
    def namespace(self) -> str:
        return self.source.name

    def run(
        self,
        *,
        max_pages: int | None = DEFAULT_BOOTSTRAP_MAX_PAGES,
    ) -> BootstrapOutcome:
        if max_pages is None or max_pages < 1:
            raise ValueError("bootstrap requires a finite, positive max_pages budget")
        state = self.database.get_source_state(self.namespace)
        if self.source.is_complete(state):
            stats = SyncStats(source=self.namespace, complete=True)
            stats.upstream_count = _optional_int(state.get("upstream_count"))
            return BootstrapOutcome(
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


def run_arxiv_bootstrap(
    database: Database,
    adapter: ArxivSourceAdapter,
    *,
    max_pages: int | None = DEFAULT_BOOTSTRAP_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> BootstrapOutcome:
    return ArxivBootstrap(
        database,
        adapter,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def _identity_payload(identity: ArxivRepositoryIdentity) -> dict[str, str]:
    return {
        "repository_name": identity.repository_name,
        "base_url": identity.base_url,
        "protocol_version": identity.protocol_version,
        "earliest_datestamp": identity.earliest_datestamp,
        "deleted_record": identity.deleted_record,
        "granularity": identity.granularity,
    }


def _datestamp_date(value: Any, source: str) -> date:
    text = _required_text(value, "earliest_datestamp", source)
    try:
        if len(text) == 10:
            return date.fromisoformat(text)
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC).date()
    except ValueError:
        raise ValueError(f"{source}: invalid earliest_datestamp {text!r}") from None


def _required_date(value: Any, field: str, source: str) -> date:
    text = _required_text(value, field, source)
    try:
        if len(text) != 10:
            raise ValueError
        return date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{source}: invalid {field} {text!r}") from None


def _required_datetime(value: Any, field: str, source: str) -> datetime:
    text = _required_text(value, field, source)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{source}: invalid {field} {text!r}") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{source}: {field} must include a UTC offset")
    return parsed.astimezone(UTC)


def _required_text(value: Any, field: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field} is required")
    return value.strip()


def _clock(adapter: ArxivSourceAdapter) -> datetime:
    value = adapter.clock()
    if not isinstance(value, datetime):
        raise TypeError(f"{adapter.name}: clock must return a datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _outcome(outcome: SyncOutcome) -> BootstrapOutcome:
    return BootstrapOutcome(
        source=outcome.source,
        status=outcome.status,
        run_id=outcome.run_id,
        stats=outcome.stats,
        error=outcome.error,
    )


__all__ = [
    "ArxivBootstrap",
    "ArxivBootstrapSource",
    "BootstrapOutcome",
    "DEFAULT_BOOTSTRAP_MAX_PAGES",
    "run_arxiv_bootstrap",
]
