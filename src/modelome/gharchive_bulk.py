from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import hmac
import ipaddress
import json
import os
import socket
import tempfile
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from modelome.bulk import BulkShardPlan
from modelome.lake import (
    LakeRecord,
    ParquetLandingZone,
    ShardApplicationOrder,
    ShardReceipt,
    canonical_control_sha256,
)
from modelome.models import ArtifactKind, SourceRecord

_DATA_URL = "https://data.gharchive.org/"
_DATASET = "events"
_RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
GHARCHIVE_DEFAULT_NEW_SHARD_BUDGET = 48


class StreamingResponse(Protocol):
    status: int
    headers: Mapping[str, str]
    url: str

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> StreamingResponse: ...

    def __exit__(self, *args: Any) -> None: ...


class GhArchiveBulkTransport(Protocol):
    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> AbstractContextManager[StreamingResponse]: ...


class HttpsGhArchiveBulkTransport:
    """Open public GH Archive byte streams with retry-safe redirects."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        attempts: int = 4,
        user_agent: str = "modelome/0.1",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("transport timeout must be positive")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError("transport attempts must be a positive integer")
        self.timeout = float(timeout)
        self.attempts = attempts
        self.user_agent = _required_text(user_agent, "user agent")
        self._sleep = sleep

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> AbstractContextManager[StreamingResponse]:
        _require_public_https(url)
        opener = build_opener(_ValidatedRedirectHandler(redirect_validator))
        request_headers = {"User-Agent": self.user_agent, **dict(headers)}
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            request = Request(url, headers=request_headers, method="GET")
            try:
                return opener.open(request, timeout=self.timeout)  # type: ignore[return-value]
            except HTTPError as error:
                last_error = error
                if error.code not in _RETRYABLE_HTTP:
                    break
            except (TimeoutError, URLError, OSError, ValueError) as error:
                last_error = error
            if attempt + 1 < self.attempts:
                self._sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"GH Archive shard transfer failed ({type(last_error).__name__})"
        ) from None


class _ValidatedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, validator: Callable[[str], None]) -> None:
        super().__init__()
        self.validator = validator

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> Request | None:
        _require_public_https(newurl)
        self.validator(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        for name, _ in redirected.header_items():
            if name.casefold() not in {
                "accept",
                "accept-encoding",
                "accept-language",
                "user-agent",
            }:
                redirected.remove_header(name)
        return redirected


@dataclass(frozen=True, slots=True)
class GhArchiveBulkLimits:
    max_compressed_bytes: int = 4 * 1024 * 1024 * 1024
    max_uncompressed_bytes: int = 32 * 1024 * 1024 * 1024
    max_event_bytes: int = 32 * 1024 * 1024
    max_events: int = 20_000_000
    download_chunk_bytes: int = 1024 * 1024
    parquet_batch_rows: int = 2_048

    def __post_init__(self) -> None:
        for field in (
            "max_compressed_bytes",
            "max_uncompressed_bytes",
            "max_event_bytes",
            "max_events",
            "download_chunk_bytes",
            "parquet_batch_rows",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.download_chunk_bytes > self.max_compressed_bytes:
            raise ValueError("download_chunk_bytes must not exceed max_compressed_bytes")
        if self.max_event_bytes > self.max_uncompressed_bytes:
            raise ValueError("max_event_bytes must not exceed max_uncompressed_bytes")


@dataclass(frozen=True, slots=True)
class GhArchiveBulkReceipt:
    source: str
    dataset: str
    release: str
    shard: str
    hour_key: str
    control_sha256: str
    upstream_sha256: str
    compressed_bytes: int
    uncompressed_bytes: int | None
    event_count: int | None
    row_count: int
    path: Path
    already_committed: bool
    response_etag: str | None = None


@dataclass(frozen=True, slots=True)
class _ControlShard:
    hour_key: str
    hour_start: datetime
    hour_end: datetime
    archive_path: str
    stable_url: str
    source_record_id: str
    control_sha256: str

    @property
    def application_order(self) -> ShardApplicationOrder:
        # One release per UTC hour avoids falsely sealing a partial UTC day.
        return ShardApplicationOrder.single()


@dataclass(frozen=True, slots=True)
class _Download:
    path: Path
    sha256: str
    byte_count: int
    response_etag: str | None


@dataclass(slots=True)
class _ParseStats:
    uncompressed_bytes: int = 0
    events: int = 0


class GhArchiveEventBulkLoader:
    """Verify and stream one complete GH Archive hourly JSONL gzip to Parquet.

    Every event is retained. Repository candidates are derived only from the
    event's exact ``repo`` object; no model names, keywords, topics, languages,
    or GitHub search queries participate in discovery.
    """

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        source: str = "gharchive",
        dataset: str = _DATASET,
        data_url: str = _DATA_URL,
        default_max_new_shards: int = GHARCHIVE_DEFAULT_NEW_SHARD_BUDGET,
        limits: GhArchiveBulkLimits | None = None,
        transport: GhArchiveBulkTransport | None = None,
    ) -> None:
        if not isinstance(landing_zone, ParquetLandingZone):
            raise TypeError("landing_zone must be a ParquetLandingZone")
        self.landing_zone = landing_zone
        self.source = _required_text(source, "source")
        self.dataset = _required_text(dataset, "dataset")
        self.data_url = _https_directory_url(data_url, "data URL")
        self.default_max_new_shards = _positive_integer(
            default_max_new_shards,
            "default_max_new_shards",
        )
        self.limits = limits or GhArchiveBulkLimits()
        if not isinstance(self.limits, GhArchiveBulkLimits):
            raise TypeError("limits must be GhArchiveBulkLimits")
        self.transport = transport or HttpsGhArchiveBulkTransport()

    def shard_budget(self, explicit: int | None = None) -> int:
        """Return the operator override or a two-day hourly catch-up budget."""

        if explicit is None:
            return self.default_max_new_shards
        if isinstance(explicit, bool) or not isinstance(explicit, int):
            raise TypeError("explicit shard budget must be an integer")
        if explicit < 0:
            raise ValueError("explicit shard budget must be nonnegative")
        return explicit

    def load(self, control_record: SourceRecord) -> GhArchiveBulkReceipt:
        shard = _control_shard(control_record, data_url=self.data_url)
        cached = self.landing_zone.lookup_committed_shard(
            source=self.source,
            dataset=self.dataset,
            release=shard.hour_key,
            shard=shard.source_record_id,
            control_sha256=shard.control_sha256,
            upstream_url=shard.stable_url,
            application_order=shard.application_order,
        )
        if cached is not None:
            return _bulk_receipt(cached, shard, download=None, stats=None)

        self.landing_zone.initialize()
        with tempfile.TemporaryDirectory(
            prefix="gharchive-events-",
            dir=self.landing_zone.staging_root,
        ) as temporary:
            path = Path(temporary) / "events.json.gz"
            download = self._download(shard, path)
            stats = _ParseStats()
            receipt = self.landing_zone.commit_shard(
                source=self.source,
                dataset=self.dataset,
                release=shard.hour_key,
                shard=shard.source_record_id,
                control_sha256=shard.control_sha256,
                upstream_sha256=download.sha256,
                upstream_url=shard.stable_url,
                upstream_bytes=download.byte_count,
                application_order=shard.application_order,
                records=self._records(download.path, download.sha256, shard, stats),
                batch_rows=self.limits.parquet_batch_rows,
            )
        return _bulk_receipt(
            receipt,
            shard,
            download=download,
            stats=stats if not receipt.already_committed else None,
        )

    def plan_shard(self, control_record: SourceRecord) -> BulkShardPlan | None:
        if control_record.raw.get("record_type") != "gharchive_hour_shard":
            return None
        shard = _control_shard(control_record, data_url=self.data_url)
        return BulkShardPlan(
            source=self.source,
            dataset=self.dataset,
            release=shard.hour_key,
            shard=shard.source_record_id,
            control_sha256=shard.control_sha256,
        )

    def shard_order(self, control_record: SourceRecord) -> tuple[datetime, str]:
        shard = _control_shard(control_record, data_url=self.data_url)
        return shard.hour_start, shard.source_record_id

    def _download(self, shard: _ControlShard, destination: Path) -> _Download:
        sha256 = hashlib.sha256()
        md5 = None
        content_md5: bytes | None = None
        byte_count = 0
        headers = {
            "Accept": "application/gzip,application/octet-stream",
            "Accept-Encoding": "identity",
        }
        with self.transport.open(
            shard.stable_url,
            headers=headers,
            redirect_validator=lambda url: _validate_exact_object_url(
                url,
                expected=shard.stable_url,
                data_url=self.data_url,
            ),
        ) as response:
            status = int(getattr(response, "status", 0))
            if status != 200:
                raise ValueError(f"GH Archive object returned HTTP {status}")
            _validate_exact_object_url(
                str(getattr(response, "url", "")),
                expected=shard.stable_url,
                data_url=self.data_url,
            )
            response_headers = {
                str(key).casefold(): str(value) for key, value in response.headers.items()
            }
            encoding = response_headers.get("content-encoding", "").strip().casefold()
            if encoding not in {"", "identity"}:
                raise ValueError("GH Archive object must not use HTTP content encoding")
            content_length = _content_length(response_headers.get("content-length"))
            if (
                content_length is not None
                and content_length > self.limits.max_compressed_bytes
            ):
                raise ValueError("GH Archive object exceeds max_compressed_bytes")
            if raw_content_md5 := _optional_text(response_headers.get("content-md5")):
                content_md5 = _base64_digest(raw_content_md5, "Content-MD5")
                if len(content_md5) != 16:
                    raise ValueError("Content-MD5 has the wrong digest length")
                md5 = hashlib.md5(usedforsecurity=False)
            response_etag = _optional_text(response_headers.get("etag")) or None

            with destination.open("xb") as output:
                while True:
                    block = response.read(self.limits.download_chunk_bytes)
                    if not isinstance(block, bytes):
                        raise TypeError("streaming transport returned a non-bytes block")
                    if not block:
                        break
                    byte_count += len(block)
                    if byte_count > self.limits.max_compressed_bytes:
                        raise ValueError("GH Archive object exceeds max_compressed_bytes")
                    sha256.update(block)
                    if md5 is not None:
                        md5.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())

        if byte_count == 0:
            raise ValueError("GH Archive object download was empty")
        if content_length is not None and content_length != byte_count:
            raise ValueError(
                f"GH Archive object length mismatch: expected {content_length}, "
                f"received {byte_count}"
            )
        if content_md5 is not None and md5 is not None and not hmac.compare_digest(
            md5.digest(), content_md5
        ):
            raise ValueError("GH Archive object failed Content-MD5 verification")
        return _Download(destination, sha256.hexdigest(), byte_count, response_etag)

    def _records(
        self,
        path: Path,
        archive_sha256: str,
        shard: _ControlShard,
        stats: _ParseStats,
    ) -> Iterator[LakeRecord]:
        try:
            with (
                path.open("rb") as compressed,
                gzip.GzipFile(fileobj=compressed, mode="rb") as decompressed,
            ):
                reader = _BoundedJsonLinesReader(
                    decompressed,
                    max_uncompressed_bytes=self.limits.max_uncompressed_bytes,
                    max_line_bytes=self.limits.max_event_bytes,
                )
                for line_number, raw_json in reader:
                    stats.events += 1
                    if stats.events > self.limits.max_events:
                        raise ValueError("GH Archive shard exceeds max_events")
                    yield _event_record(
                        raw_json,
                        line_number=line_number,
                        archive_sha256=archive_sha256,
                        shard=shard,
                    )
                stats.uncompressed_bytes = reader.byte_count
        except (gzip.BadGzipFile, EOFError, zlib.error, OSError) as error:
            raise ValueError(f"invalid or truncated GH Archive gzip: {error}") from error
        if stats.events == 0:
            raise ValueError("GH Archive shard contains no events")


class _BoundedJsonLinesReader:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        max_uncompressed_bytes: int,
        max_line_bytes: int,
    ) -> None:
        self.stream = stream
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_line_bytes = max_line_bytes
        self.byte_count = 0

    def __iter__(self) -> Iterator[tuple[int, bytes]]:
        line_number = 0
        while True:
            line = self.stream.readline(self.max_line_bytes + 1)
            if not isinstance(line, bytes):
                raise TypeError("GH Archive gzip reader returned a non-bytes line")
            if not line:
                return
            line_number += 1
            self.byte_count += len(line)
            if self.byte_count > self.max_uncompressed_bytes:
                raise ValueError("GH Archive shard exceeds max_uncompressed_bytes")
            if len(line) > self.max_line_bytes:
                raise ValueError("GH Archive event exceeds max_event_bytes")
            raw_json = line[:-1] if line.endswith(b"\n") else line
            if raw_json.endswith(b"\r"):
                raw_json = raw_json[:-1]
            if not raw_json.strip():
                raise ValueError(f"GH Archive line {line_number} is empty")
            yield line_number, raw_json


def _event_record(
    raw_json: bytes,
    *,
    line_number: int,
    archive_sha256: str,
    shard: _ControlShard,
) -> LakeRecord:
    try:
        text = raw_json.decode("utf-8", errors="strict")
        event = json.loads(text, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"GH Archive line {line_number} is not valid JSON") from error
    if not isinstance(event, Mapping):
        raise ValueError(f"GH Archive line {line_number} is not a JSON object")

    event_id = _bounded_exact_text(event.get("id"), "event id", maximum=128)
    event_type = _bounded_exact_text(event.get("type"), "event type", maximum=256)
    created_at = _utc_timestamp(event.get("created_at"), "event created_at")
    created = _timestamp_value(created_at)
    if not shard.hour_start <= created < shard.hour_end:
        raise ValueError(
            f"GH Archive event {event_id} timestamp is outside control hour {shard.hour_key}"
        )

    repo = event.get("repo")
    if not isinstance(repo, Mapping):
        raise ValueError(f"GH Archive event {event_id} has no repository object")
    repository_id = _positive_decimal(repo.get("id"), "repository id")
    repository_name = _repository_name(repo.get("name"))
    repository_url = _repository_web_url(repository_name)
    repository_api_url = _repository_api_url(repo.get("url"), repository_name)
    event_sha256 = hashlib.sha256(raw_json).hexdigest()
    locator = (
        f"gharchive:event:{event_id}:archive:{shard.hour_key}:"
        f"line:{line_number}:sha256:{event_sha256}"
    )

    return LakeRecord(
        source_record_id=f"gharchive:event:{event_id}",
        payload={
            "record_type": "gharchive_public_event",
            "event_id": event_id,
            "event_type": event_type,
            "created_at": created_at,
            "repository": {
                "id": repository_id,
                "name": repository_name,
                "api_url": repository_api_url,
                "html_url": repository_url,
                "identifier": {
                    "namespace": "github:repository",
                    "value": repository_name,
                },
                "stable_identifier": {
                    "namespace": "github:repository-id",
                    "value": repository_id,
                },
            },
            "evidence": {
                "locator": locator,
                "event_json_sha256": event_sha256,
                "archive_sha256": archive_sha256,
                "archive_url": shard.stable_url,
                "archive_hour": shard.hour_key,
                "line_number": line_number,
            },
            "event": dict(event),
        },
    )


def _control_shard(record: SourceRecord, *, data_url: str) -> _ControlShard:
    if not isinstance(record, SourceRecord):
        raise TypeError("control_record must be a SourceRecord")
    if record.kind is not ArtifactKind.CATALOG_RECORD:
        raise ValueError("GH Archive control must be a catalog record")
    raw = record.raw
    if not isinstance(raw, Mapping):
        raise TypeError("GH Archive control raw value must be a mapping")
    if raw.get("record_type") != "gharchive_hour_shard":
        raise ValueError("control record is not a GH Archive hourly shard")
    if raw.get("archive_format") != "gzip_json_lines":
        raise ValueError("GH Archive control has an unsupported archive format")
    if raw.get("coverage_scope") != "public_github_event_activity":
        raise ValueError("GH Archive control has an unknown coverage scope")
    if raw.get("historical_repository_census") is not False:
        raise ValueError("GH Archive control must not claim historical census coverage")

    hour_start = _exact_hour(raw.get("hour_start"), "hour_start")
    hour_end = _exact_hour(raw.get("hour_end"), "hour_end")
    if hour_end != hour_start + timedelta(hours=1):
        raise ValueError("GH Archive control must cover exactly one UTC hour")
    hour_key = _hour_key(hour_start)
    if raw.get("hour_key") != hour_key:
        raise ValueError("GH Archive hour_key disagrees with hour_start")
    archive_path = _required_exact_text(raw.get("archive_path"), "archive_path")
    if archive_path != f"{hour_key}.json.gz":
        raise ValueError("GH Archive archive_path is not canonical")
    stable_url = _required_exact_text(raw.get("stable_object_url"), "stable_object_url")
    expected_url = f"{data_url}{archive_path}"
    if stable_url != expected_url or record.canonical_url != stable_url:
        raise ValueError("GH Archive control URL, path, and canonical URL disagree")
    _validate_exact_object_url(stable_url, expected=expected_url, data_url=data_url)
    source_record_id = f"gharchive:hour:{hour_key}"
    if record.source_record_id != source_record_id:
        raise ValueError("GH Archive control source_record_id is inconsistent")
    control_sha256 = canonical_control_sha256(
        {
            "source_record_id": source_record_id,
            "canonical_url": stable_url,
            "record_type": "gharchive_hour_shard",
            "hour_start": _isoformat_hour(hour_start),
            "hour_end": _isoformat_hour(hour_end),
        }
    )
    return _ControlShard(
        hour_key=hour_key,
        hour_start=hour_start,
        hour_end=hour_end,
        archive_path=archive_path,
        stable_url=stable_url,
        source_record_id=source_record_id,
        control_sha256=control_sha256,
    )


def _bulk_receipt(
    receipt: ShardReceipt,
    shard: _ControlShard,
    *,
    download: _Download | None,
    stats: _ParseStats | None,
) -> GhArchiveBulkReceipt:
    if receipt.upstream_bytes is None:
        raise ValueError("landing receipt is missing upstream byte count")
    return GhArchiveBulkReceipt(
        source=receipt.source,
        dataset=receipt.dataset,
        release=receipt.release,
        shard=receipt.shard,
        hour_key=shard.hour_key,
        control_sha256=receipt.control_sha256,
        upstream_sha256=receipt.upstream_sha256,
        compressed_bytes=receipt.upstream_bytes,
        uncompressed_bytes=stats.uncompressed_bytes if stats is not None else None,
        event_count=stats.events if stats is not None else None,
        row_count=receipt.row_count,
        path=receipt.path,
        already_committed=receipt.already_committed,
        response_etag=download.response_etag if download is not None else None,
    )


def _repository_name(value: Any) -> str:
    name = _bounded_exact_text(value, "repository name", maximum=512)
    if not name.isascii() or any(ord(character) < 33 for character in name):
        raise ValueError("repository name must contain visible ASCII characters")
    components = name.split("/")
    if len(components) != 2 or any(part in {"", ".", ".."} for part in components):
        raise ValueError("repository name must contain exactly one owner/name pair")
    if any("\\" in part or "?" in part or "#" in part for part in components):
        raise ValueError("repository name contains unsafe URL characters")
    return name


def _repository_web_url(name: str) -> str:
    owner, repository = name.split("/", 1)
    return f"https://github.com/{quote(owner, safe='')}/{quote(repository, safe='')}"


def _repository_api_url(value: Any, name: str) -> str:
    owner, repository = name.split("/", 1)
    expected_path = f"/repos/{quote(owner, safe='')}/{quote(repository, safe='')}"
    expected = f"https://api.github.com{expected_path}"
    if value is None:
        return expected
    url = _required_exact_text(value, "repository API URL")
    parts = urlsplit(url)
    if (
        parts.scheme.casefold() != "https"
        or (parts.hostname or "").casefold() != "api.github.com"
        or parts.port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or unquote(parts.path).casefold() != unquote(expected_path).casefold()
    ):
        raise ValueError("repository API URL disagrees with repository name")
    return expected


def _validate_exact_object_url(url: str, *, expected: str, data_url: str) -> None:
    if url != expected or not url.startswith(data_url):
        raise ValueError("GH Archive transfer URL changed from its exact control")
    parts = urlsplit(url)
    base = urlsplit(data_url)
    if (
        parts.scheme != "https"
        or parts.hostname != base.hostname
        or parts.port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError("GH Archive transfer URL is unsafe")
    decoded = unquote(parts.path)
    if decoded != parts.path or "\\" in decoded or "\x00" in decoded:
        raise ValueError("GH Archive transfer URL has an unsafe encoded path")
    if any(segment in {"", ".", ".."} for segment in decoded.split("/")[1:]):
        raise ValueError("GH Archive transfer URL contains path traversal")


def _exact_hour(value: Any, label: str) -> datetime:
    text = _required_exact_text(value, label)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:00:00Z").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"{label} must be an exact UTC-hour timestamp") from error
    if _isoformat_hour(parsed) != text:
        raise ValueError(f"{label} must be canonical")
    return parsed


def _hour_key(value: datetime) -> str:
    return f"{value:%Y-%m-%d}-{value.hour}"


def _isoformat_hour(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:00:00Z")


def _utc_timestamp(value: Any, label: str) -> str:
    text = _required_exact_text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} must identify a UTC instant")
    return text


def _timestamp_value(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _positive_decimal(value: Any, label: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a positive integer")
    text = str(value)
    if not text.isascii() or not text.isdecimal() or text.startswith("0"):
        raise ValueError(f"{label} must be a positive decimal integer")
    return text


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _content_length(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    if not value.isascii() or not value.isdecimal():
        raise ValueError("HTTP Content-Length must contain decimal digits")
    return int(value)


def _base64_digest(value: str, label: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{label} is not valid base64") from error


def _bounded_exact_text(value: Any, label: str, *, maximum: int) -> str:
    text = _required_exact_text(value, label)
    if len(text.encode("utf-8")) > maximum:
        raise ValueError(f"{label} exceeds {maximum} bytes")
    return text


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _required_exact_text(value: Any, label: str) -> str:
    result = _required_text(value, label)
    if result != value:
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return result


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


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


def _require_public_https(url: str) -> None:
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").casefold().rstrip(".")
        if (
            parts.scheme.casefold() != "https"
            or not host
            or parts.username is not None
            or parts.password is not None
            or parts.port not in {None, 443}
            or parts.fragment
            or host == "localhost"
            or host.endswith(".localhost")
        ):
            raise ValueError
        try:
            addresses = (ipaddress.ip_address(host),)
        except ValueError:
            addresses = tuple(
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            )
        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError
    except (OSError, ValueError):
        raise ValueError("transfer target is not a public HTTPS endpoint") from None


__all__ = [
    "GHARCHIVE_DEFAULT_NEW_SHARD_BUDGET",
    "GhArchiveBulkLimits",
    "GhArchiveBulkReceipt",
    "GhArchiveEventBulkLoader",
    "HttpsGhArchiveBulkTransport",
]
