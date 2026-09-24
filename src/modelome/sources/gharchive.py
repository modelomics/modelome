from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from modelome.models import ArtifactKind, Identifier, Link, SourcePage, SourceRecord
from modelome.normalize import content_hash

Clock = Callable[[], datetime]

_DATA_URL = "https://data.gharchive.org/"
_SCAN_STAGE = "closed_hours"
_SCAN_KEYS = frozenset(
    {
        "stage",
        "scan_from",
        "scan_until",
        "scan_cursor",
        "scan_total_hours",
        "started_at",
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GhArchiveSourceAdapter:
    """Enumerate every eligible GH Archive UTC-hour object as a bulk control.

    The adapter contains no repository, topic, language, event-type, or model-name
    query. It advances a durable hour watermark and freezes the upper boundary of
    each scan, so an interrupted scan resumes without skipping or inventing an
    hour. Only hours whose end precedes the configured availability boundary are
    emitted.

    GH Archive is an activity log, not a repository catalog. A repository that
    has no public event in an observed hour cannot appear. The default bootstrap
    deliberately covers a recent activity window; even a longer backfill would
    not constitute a complete historical census of GitHub repositories.
    """

    def __init__(
        self,
        *,
        name: str = "gharchive",
        data_url: str = _DATA_URL,
        page_size: int = 24,
        initial_lookback_hours: int = 48,
        availability_lag_hours: int = 1,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.data_url = _https_directory_url(data_url, "data URL")
        self.page_size = _positive_integer(page_size, "page size")
        self.initial_lookback_hours = _positive_integer(
            initial_lookback_hours,
            "initial lookback hours",
        )
        self.availability_lag_hours = _nonnegative_integer(
            availability_lag_hours,
            "availability lag hours",
        )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "gharchive-closed-hours-v1",
                "data_url": self.data_url,
                "initial_lookback_hours": self.initial_lookback_hours,
                "availability_lag_hours": self.availability_lag_hours,
            }
        )

    @property
    def coverage_semantics(self) -> Mapping[str, Any]:
        """Describe the observable universe without overstating completeness."""

        return {
            "enumerates": "all public GitHub events in each emitted GH Archive hour",
            "discovery_basis": "repository activity",
            "historical_repository_census": False,
            "initial_lookback_hours": self.initial_lookback_hours,
            "availability_lag_hours": self.availability_lag_hours,
        }

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if not isinstance(state, Mapping):
            raise TypeError(f"{self.name}: checkpoint state must be a mapping")
        self._validate_signature(state)
        stage = _optional_text(state.get("stage"))
        if not stage:
            return self._begin_scan(state)
        if stage != _SCAN_STAGE:
            raise ValueError(f"{self.name}: unknown checkpoint stage {stage!r}")
        return self._emit_page(state)

    def _begin_scan(self, state: Mapping[str, Any]) -> SourcePage:
        boundary = _eligible_boundary(
            self.clock(),
            availability_lag_hours=self.availability_lag_hours,
        )
        watermark = _optional_hour(state.get("next_hour"), "next hour", self.name)
        start = watermark or boundary - timedelta(hours=self.initial_lookback_hours)
        if start > boundary:
            raise ValueError(f"{self.name}: next hour is ahead of the eligible boundary")
        if start == boundary:
            complete = _stable_state(state)
            complete.update(
                {
                    "checkpoint_signature": self.checkpoint_signature,
                    "next_hour": _isoformat_hour(boundary),
                    "checked_at": _isoformat(self.clock()),
                }
            )
            return SourcePage(
                records=(),
                next_state=complete,
                complete=True,
                upstream_count=0,
                retry_state=_stable_state(state),
            )

        total = _hours_between(start, boundary)
        scan = _stable_state(state)
        scan.update(
            {
                "checkpoint_signature": self.checkpoint_signature,
                "stage": _SCAN_STAGE,
                "scan_from": _isoformat_hour(start),
                "scan_until": _isoformat_hour(boundary),
                "scan_cursor": 0,
                "scan_total_hours": total,
                "started_at": _isoformat(self.clock()),
            }
        )
        return self._emit_page(scan)

    def _emit_page(self, state: Mapping[str, Any]) -> SourcePage:
        start = _required_hour(state.get("scan_from"), "scan start", self.name)
        stop = _required_hour(state.get("scan_until"), "scan end", self.name)
        if start >= stop:
            raise ValueError(f"{self.name}: scan window must contain at least one hour")
        total = _hours_between(start, stop)
        recorded_total = _integer(
            state.get("scan_total_hours"),
            "scan total hours",
            minimum=1,
        )
        if total != recorded_total:
            raise ValueError(f"{self.name}: scan total does not match its boundaries")
        cursor = _integer(state.get("scan_cursor"), "scan cursor", minimum=0)
        if cursor >= total:
            raise ValueError(f"{self.name}: scan cursor is outside its hour range")

        page_stop = min(cursor + self.page_size, total)
        hours = tuple(start + timedelta(hours=index) for index in range(cursor, page_stop))
        records = tuple(self._control_record(hour) for hour in hours)
        retry_state = _stable_state(state)
        if page_stop < total:
            next_state = dict(state)
            next_state["scan_cursor"] = page_stop
            return SourcePage(
                records=records,
                next_state=next_state,
                complete=False,
                upstream_count=total,
                retry_state=retry_state,
            )

        next_state = retry_state
        next_state.update(
            {
                "checkpoint_signature": self.checkpoint_signature,
                "next_hour": _isoformat_hour(stop),
                "last_hour": _isoformat_hour(stop - timedelta(hours=1)),
                "completed_at": _isoformat(self.clock()),
            }
        )
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=total,
            retry_state=retry_state,
        )

    def _control_record(self, hour: datetime) -> SourceRecord:
        hour_end = hour + timedelta(hours=1)
        hour_key = _hour_key(hour)
        archive_path = f"{hour_key}.json.gz"
        archive_url = f"{self.data_url}{archive_path}"
        return SourceRecord(
            source_record_id=f"gharchive:hour:{hour_key}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=archive_url,
            title=f"GitHub public events {hour:%Y-%m-%d %H}:00 UTC",
            published_at=_isoformat_hour(hour_end),
            identifiers=(Identifier("gharchive:hour", hour_key),),
            links=(Link(archive_url, relation="bulk_payload", crawl=False),),
            raw={
                "record_type": "gharchive_hour_shard",
                "hour_key": hour_key,
                "hour_start": _isoformat_hour(hour),
                "hour_end": _isoformat_hour(hour_end),
                "archive_path": archive_path,
                "stable_object_url": archive_url,
                "archive_format": "gzip_json_lines",
                "coverage_scope": "public_github_event_activity",
                "historical_repository_census": False,
            },
        )

    def _validate_signature(self, state: Mapping[str, Any]) -> None:
        signature = _optional_text(state.get("checkpoint_signature"))
        if signature and signature != self.checkpoint_signature:
            raise ValueError(
                f"{self.name}: checkpoint belongs to a different adapter configuration"
            )


def _eligible_boundary(now: datetime, *, availability_lag_hours: int) -> datetime:
    if not isinstance(now, datetime):
        raise TypeError("clock must return a datetime")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    utc = now.astimezone(UTC)
    closed_boundary = utc.replace(minute=0, second=0, microsecond=0)
    return closed_boundary - timedelta(hours=availability_lag_hours)


def _hours_between(start: datetime, stop: datetime) -> int:
    seconds = (stop - start).total_seconds()
    if seconds <= 0 or seconds % 3600:
        raise ValueError("hour boundaries must be positive whole-hour intervals")
    return int(seconds // 3600)


def _hour_key(value: datetime) -> str:
    return f"{value:%Y-%m-%d}-{value.hour}"


def _isoformat_hour(value: datetime) -> str:
    utc = value.astimezone(UTC)
    if any((utc.minute, utc.second, utc.microsecond)):
        raise ValueError("hour timestamp must be aligned to a UTC hour")
    return utc.strftime("%Y-%m-%dT%H:00:00Z")


def _isoformat(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _optional_hour(value: Any, label: str, source: str) -> datetime | None:
    if value is None:
        return None
    return _required_hour(value, label, source)


def _required_hour(value: Any, label: str, source: str) -> datetime:
    text = _required_text(value, label)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:00:00Z").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"{source}: {label} must be an exact UTC-hour timestamp") from error
    if _isoformat_hour(parsed) != text:
        raise ValueError(f"{source}: {label} must be canonical")
    return parsed


def _stable_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in state.items() if key not in _SCAN_KEYS}


def _https_directory_url(value: Any, label: str) -> str:
    text = _required_text(value, label)
    parts = urlsplit(text)
    try:
        port = parts.port
    except ValueError:
        raise ValueError(f"{label} has an invalid port") from None
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or port not in {None, 443}
    ):
        raise ValueError(f"{label} must be a credential-free HTTPS directory URL")
    host = parts.hostname.casefold().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"{label} host must be public")
    netloc = f"[{host}]" if ":" in host else host
    return urlunsplit(("https", netloc, parts.path.rstrip("/") + "/", "", ""))


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return value


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_integer(value: Any, label: str) -> int:
    return _integer(value, label, minimum=1)


def _nonnegative_integer(value: Any, label: str) -> int:
    return _integer(value, label, minimum=0)


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


__all__ = ["GhArchiveSourceAdapter"]
