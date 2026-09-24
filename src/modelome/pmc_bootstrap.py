from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from modelome.models import SourcePage, SyncStats
from modelome.normalize import canonicalize_url, content_hash
from modelome.pipeline import SyncEngine, SyncOutcome
from modelome.sources.pmc import PmcRepositoryIdentity, PmcSourceAdapter
from modelome.storage import Database

_BOOTSTRAP_KEY = "bootstrap"
_BOOTSTRAP_COMPLETE_KEY = "bootstrap_complete"
_STATE_SIGNATURE_KEY = "_modelome_source_signature"
_DESCRIPTOR_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
DEFAULT_PMC_BOOTSTRAP_MAX_PAGES = 1_000


@dataclass(frozen=True, slots=True)
class PmcBootstrapOutcome:
    source: str
    status: str
    run_id: int | None
    stats: Mapping[str, Any]
    error: str | None = None
    already_complete: bool = False


class PmcBootstrapSource:
    """Freeze PMC's complete reusable-full-text history for bounded harvesting."""

    def __init__(
        self,
        adapter: PmcSourceAdapter,
        *,
        namespace: str | None = None,
    ) -> None:
        if not isinstance(adapter, PmcSourceAdapter):
            raise TypeError("adapter must be a PmcSourceAdapter")
        self.adapter = adapter
        self.name = (namespace or f"{adapter.name}:bootstrap").strip()
        if not self.name:
            raise ValueError("bootstrap namespace must not be empty")
        if self.name == adapter.name:
            raise ValueError("bootstrap namespace must differ from the daily source name")
        # Checkpoint isolation must not create a second artifact identity space.
        self.artifact_source = adapter.name
        self.checkpoint_signature = content_hash(
            {
                "workflow": "pmc-open-oai-bootstrap-v1",
                "adapter_checkpoint_signature": adapter.checkpoint_signature,
                "artifact_source": self.artifact_source,
                "namespace": self.name,
            }
        )

    def is_complete(self, state: Mapping[str, Any]) -> bool:
        if not state:
            return False
        self._descriptor(state)
        if _BOOTSTRAP_COMPLETE_KEY not in state:
            return False
        complete = state[_BOOTSTRAP_COMPLETE_KEY]
        if not isinstance(complete, bool):
            raise ValueError(
                f"{self.name}: bootstrap_complete checkpoint value must be boolean"
            )
        if complete:
            _required_datetime(state.get("completed_at"), "completed_at", self.name)
        return complete

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
            completed_at = _isoformat(_clock(self.adapter))
            return SourcePage(
                records=(),
                next_state={
                    _BOOTSTRAP_KEY: descriptor,
                    _BOOTSTRAP_COMPLETE_KEY: True,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "completed_at": completed_at,
                    "upstream_count": 0,
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
                _STATE_SIGNATURE_KEY,
                "upstream_count",
            }
        }
        # The descriptor, not mutable outer checkpoint fields, owns the frozen
        # interval on every resumed request.
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
        repository = _identity_payload(identity)
        discovered_at = _clock(self.adapter)
        start = _datestamp_date(identity.earliest_datestamp, self.name)
        end = discovered_at.date() - timedelta(days=1)
        return {
            "version": _DESCRIPTOR_VERSION,
            "workflow_checkpoint_signature": self.checkpoint_signature,
            "adapter_checkpoint_signature": self.adapter.checkpoint_signature,
            "repository_identity_signature": content_hash(repository),
            "repository": repository,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "discovered_at": _isoformat(discovered_at),
        }

    def _descriptor(self, state: Mapping[str, Any]) -> dict[str, Any]:
        raw = state.get(_BOOTSTRAP_KEY)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{self.name}: bootstrap checkpoint is missing its descriptor")
        descriptor = dict(raw)
        version = descriptor.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"{self.name}: bootstrap checkpoint version must be an integer")
        if version != _DESCRIPTOR_VERSION:
            raise ValueError(f"{self.name}: unsupported bootstrap checkpoint version")
        if descriptor.get("workflow_checkpoint_signature") != self.checkpoint_signature:
            raise ValueError(
                f"{self.name}: bootstrap checkpoint belongs to a different workflow config"
            )
        if (
            descriptor.get("adapter_checkpoint_signature")
            != self.adapter.checkpoint_signature
        ):
            raise ValueError(
                f"{self.name}: bootstrap checkpoint belongs to a different PMC adapter"
            )
        stored_state_signature = state.get(_STATE_SIGNATURE_KEY)
        if (
            stored_state_signature is not None
            and stored_state_signature != self.checkpoint_signature
        ):
            raise ValueError(
                f"{self.name}: stored source signature does not match this workflow"
            )

        repository = _repository_payload(descriptor.get("repository"), self.name)
        identity_signature = _required_hash(
            descriptor.get("repository_identity_signature"),
            "repository_identity_signature",
            self.name,
        )
        if identity_signature != content_hash(repository):
            raise ValueError(
                f"{self.name}: bootstrap repository identity signature does not match"
            )
        if canonicalize_url(repository["base_url"]) != canonicalize_url(
            self.adapter.url
        ):
            raise ValueError(f"{self.name}: bootstrap repository URL changed")

        start = _required_date(descriptor.get("window_start"), "window_start", self.name)
        end = _required_date(descriptor.get("window_end"), "window_end", self.name)
        expected_start = _datestamp_date(repository["earliest_datestamp"], self.name)
        if start != expected_start:
            raise ValueError(
                f"{self.name}: bootstrap lower boundary does not match OAI-PMH Identify"
            )
        discovered_at = _required_datetime(
            descriptor.get("discovered_at"), "discovered_at", self.name
        )
        if end != discovered_at.date() - timedelta(days=1):
            raise ValueError(
                f"{self.name}: bootstrap upper boundary is not the last closed UTC day"
            )

        for field, expected in (
            ("window_start", start.isoformat()),
            ("window_end", end.isoformat()),
        ):
            if field in state and state[field] != expected:
                raise ValueError(
                    f"{self.name}: outer {field} does not match bootstrap descriptor"
                )
        if "upstream_count" in state:
            _nonnegative_int(state["upstream_count"], "upstream_count", self.name)
        if _BOOTSTRAP_COMPLETE_KEY in state and not isinstance(
            state[_BOOTSTRAP_COMPLETE_KEY], bool
        ):
            raise ValueError(
                f"{self.name}: bootstrap_complete checkpoint value must be boolean"
            )
        return descriptor

    def _wrapped_state(
        self,
        adapter_state: Mapping[str, Any] | None,
        *,
        descriptor: Mapping[str, Any],
        complete: bool,
        upstream_count: Any,
    ) -> dict[str, Any]:
        if not isinstance(complete, bool):
            raise TypeError("bootstrap completion state must be boolean")
        result = dict(adapter_state or {})
        result["window_start"] = descriptor["window_start"]
        result["window_end"] = descriptor["window_end"]
        result[_BOOTSTRAP_KEY] = dict(descriptor)
        result[_BOOTSTRAP_COMPLETE_KEY] = complete
        if upstream_count is not None:
            result["upstream_count"] = _nonnegative_int(
                upstream_count, "upstream_count", self.name
            )
        return result


class PmcBootstrap:
    """Run finite, resumable pieces of PMC's reusable-full-text history."""

    def __init__(
        self,
        database: Database,
        adapter: PmcSourceAdapter,
        *,
        namespace: str | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.database = database
        self.source = PmcBootstrapSource(adapter, namespace=namespace)
        self.extractor = extractor

    @property
    def namespace(self) -> str:
        return self.source.name

    def run(
        self,
        *,
        max_pages: int = DEFAULT_PMC_BOOTSTRAP_MAX_PAGES,
    ) -> PmcBootstrapOutcome:
        if not isinstance(max_pages, int) or isinstance(max_pages, bool) or max_pages < 1:
            raise ValueError("PMC bootstrap requires a finite, positive max_pages budget")
        state = self.database.get_source_state(self.namespace)
        if self.source.is_complete(state):
            stats = SyncStats(source=self.namespace, complete=True)
            if "upstream_count" in state:
                stats.upstream_count = _nonnegative_int(
                    state["upstream_count"], "upstream_count", self.namespace
                )
            return PmcBootstrapOutcome(
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


def run_pmc_bootstrap(
    database: Database,
    adapter: PmcSourceAdapter,
    *,
    max_pages: int = DEFAULT_PMC_BOOTSTRAP_MAX_PAGES,
    namespace: str | None = None,
    extractor: Any | None = None,
) -> PmcBootstrapOutcome:
    return PmcBootstrap(
        database,
        adapter,
        namespace=namespace,
        extractor=extractor,
    ).run(max_pages=max_pages)


def _identity_payload(identity: PmcRepositoryIdentity) -> dict[str, str]:
    if not isinstance(identity, PmcRepositoryIdentity):
        raise TypeError("PMC Identify response has an invalid type")
    return _repository_payload(
        {
            "repository_name": identity.repository_name,
            "base_url": identity.base_url,
            "protocol_version": identity.protocol_version,
            "earliest_datestamp": identity.earliest_datestamp,
            "deleted_record": identity.deleted_record,
            "granularity": identity.granularity,
        },
        "pmc",
    )


def _repository_payload(value: Any, source: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: bootstrap repository identity is invalid")
    expected_fields = {
        "repository_name",
        "base_url",
        "protocol_version",
        "earliest_datestamp",
        "deleted_record",
        "granularity",
    }
    if set(value) != expected_fields:
        raise ValueError(
            f"{source}: bootstrap repository identity fields are incomplete or unknown"
        )
    result = {
        field: _required_text(value.get(field), field, source)
        for field in sorted(expected_fields)
    }
    if result["protocol_version"] != "2.0":
        raise ValueError(f"{source}: unsupported repository protocol version")
    _datestamp_date(result["earliest_datestamp"], source)
    if result["deleted_record"] not in {"no", "persistent", "transient"}:
        raise ValueError(f"{source}: invalid repository deletion policy")
    if result["granularity"] not in {"YYYY-MM-DD", "YYYY-MM-DDThh:mm:ssZ"}:
        raise ValueError(f"{source}: invalid repository datestamp granularity")
    return result


def _datestamp_date(value: Any, source: str) -> date:
    text = _required_text(value, "earliest_datestamp", source)
    try:
        if len(text) == 10:
            return date.fromisoformat(text)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{source}: invalid earliest_datestamp {text!r}") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{source}: earliest_datestamp must include a UTC offset")
    return parsed.astimezone(UTC).date()


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


def _required_hash(value: Any, field: str, source: str) -> str:
    text = _required_text(value, field, source)
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{source}: {field} must be a SHA-256 digest")
    return text


def _required_text(value: Any, field: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field} is required")
    return value.strip()


def _nonnegative_int(value: Any, field: str, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{source}: {field} must be a nonnegative integer")
    return value


def _clock(adapter: PmcSourceAdapter) -> datetime:
    value = adapter.clock()
    if not isinstance(value, datetime):
        raise TypeError(f"{adapter.name}: clock must return a datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _outcome(outcome: SyncOutcome) -> PmcBootstrapOutcome:
    return PmcBootstrapOutcome(
        source=outcome.source,
        status=outcome.status,
        run_id=outcome.run_id,
        stats=outcome.stats,
        error=outcome.error,
    )


__all__ = [
    "DEFAULT_PMC_BOOTSTRAP_MAX_PAGES",
    "PmcBootstrap",
    "PmcBootstrapOutcome",
    "PmcBootstrapSource",
    "run_pmc_bootstrap",
]
