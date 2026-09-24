from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import hmac
import ipaddress
import os
import re
import socket
import tempfile
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit, urlunsplit
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

_DATA_URL = "https://data.commoncrawl.org/"
_COLLECTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
_DIGEST_ALGORITHMS = {
    "md5": "md5",
    "sha1": "sha1",
    "sha-1": "sha1",
    "sha256": "sha256",
    "sha-256": "sha256",
    "sha512": "sha512",
    "sha-512": "sha512",
}


class StreamingResponse(Protocol):
    status: int
    headers: Mapping[str, str]
    url: str

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> StreamingResponse: ...

    def __exit__(self, *args: Any) -> None: ...


class CommonCrawlBulkTransport(Protocol):
    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> AbstractContextManager[StreamingResponse]: ...


class HttpsCommonCrawlBulkTransport:
    """Open bounded HTTPS bodies while validating every redirect target."""

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
        self.user_agent = _required_text(user_agent, "user_agent")
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
            f"Common Crawl WET transfer failed ({type(last_error).__name__})"
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
class CommonCrawlBulkLimits:
    max_compressed_bytes: int = 2 * 1024 * 1024 * 1024
    max_uncompressed_bytes: int = 16 * 1024 * 1024 * 1024
    max_record_bytes: int = 16 * 1024 * 1024
    max_text_chars_per_record: int = 16 * 1024 * 1024
    max_records: int = 1_000_000
    max_header_bytes: int = 256 * 1024
    max_header_line_bytes: int = 64 * 1024
    max_headers_per_record: int = 1_024
    download_chunk_bytes: int = 1024 * 1024
    record_chunk_bytes: int = 1024 * 1024
    parquet_batch_rows: int = 16

    def __post_init__(self) -> None:
        for field in (
            "max_compressed_bytes",
            "max_uncompressed_bytes",
            "max_record_bytes",
            "max_text_chars_per_record",
            "max_records",
            "max_header_bytes",
            "max_header_line_bytes",
            "max_headers_per_record",
            "download_chunk_bytes",
            "record_chunk_bytes",
            "parquet_batch_rows",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.download_chunk_bytes > self.max_compressed_bytes:
            raise ValueError(
                "download_chunk_bytes must not exceed max_compressed_bytes"
            )
        if self.record_chunk_bytes > self.max_record_bytes:
            raise ValueError("record_chunk_bytes must not exceed max_record_bytes")
        if self.max_header_line_bytes > self.max_header_bytes:
            raise ValueError("max_header_line_bytes must not exceed max_header_bytes")


@dataclass(frozen=True, slots=True)
class CommonCrawlBulkReceipt:
    source: str
    dataset: str
    release: str
    shard: str
    collection_id: str
    manifest_index: int
    total_shards: int
    control_sha256: str
    upstream_sha256: str
    compressed_bytes: int
    uncompressed_bytes: int | None
    warc_records: int | None
    row_count: int
    path: Path
    already_committed: bool
    response_etag: str | None = None
    shard_receipt: ShardReceipt | None = None


@dataclass(frozen=True, slots=True)
class _PublishedDigest:
    algorithm: str
    expected: bytes
    encoding: str


@dataclass(frozen=True, slots=True)
class _ControlShard:
    collection_id: str
    collection_from: str
    collection_to: str
    manifest_url: str
    manifest_digest: str
    stable_url: str
    path: str
    manifest_index: int
    total_shards: int
    source_record_id: str
    control_sha256: str
    published_digest: _PublishedDigest | None
    object_etag: str | None

    @property
    def application_order(self) -> ShardApplicationOrder:
        return ShardApplicationOrder.snapshot(self.manifest_index)


@dataclass(frozen=True, slots=True)
class _Download:
    path: Path
    sha256: str
    byte_count: int
    response_etag: str | None


@dataclass(slots=True)
class _ParseStats:
    uncompressed_bytes: int = 0
    warc_records: int = 0
    conversion_records: int = 0


@dataclass(frozen=True, slots=True)
class _WarcHeader:
    name: str
    value: str


@dataclass(frozen=True, slots=True)
class _WarcEnvelope:
    version: str
    headers: tuple[_WarcHeader, ...]

    def values(self, name: str) -> tuple[str, ...]:
        key = name.casefold()
        return tuple(header.value for header in self.headers if header.name == key)

    def one(self, name: str, *, required: bool = False) -> str | None:
        values = self.values(name)
        if len(values) > 1:
            raise ValueError(f"WARC record repeats singleton header {name}")
        if not values:
            if required:
                raise ValueError(f"WARC record is missing mandatory header {name}")
            return None
        return values[0]


class CommonCrawlWetBulkLoader:
    """Stream one exact Common Crawl WET object into immutable Parquet parts."""

    def __init__(
        self,
        lake: ParquetLandingZone,
        *,
        source: str = "commoncrawl",
        dataset: str = "wet",
        data_url: str = _DATA_URL,
        limits: CommonCrawlBulkLimits | None = None,
        transport: CommonCrawlBulkTransport | None = None,
    ) -> None:
        if not isinstance(lake, ParquetLandingZone):
            raise TypeError("lake must be a ParquetLandingZone")
        self.lake = lake
        self.source = _required_text(source, "source")
        self.dataset = _required_text(dataset, "dataset")
        self.data_url = _https_directory_url(data_url, "data_url")
        self.limits = limits or CommonCrawlBulkLimits()
        if not isinstance(self.limits, CommonCrawlBulkLimits):
            raise TypeError("limits must be CommonCrawlBulkLimits")
        self.transport = transport or HttpsCommonCrawlBulkTransport()

    def load(self, control_record: SourceRecord) -> CommonCrawlBulkReceipt:
        shard = _control_shard(control_record, data_url=self.data_url)
        cached = self.lake.lookup_committed_shard(
            source=self.source,
            dataset=self.dataset,
            release=shard.collection_id,
            shard=shard.source_record_id,
            control_sha256=shard.control_sha256,
            upstream_url=shard.stable_url,
            application_order=shard.application_order,
        )
        if cached is not None:
            if (
                shard.published_digest is not None
                and shard.published_digest.algorithm == "sha256"
                and not hmac.compare_digest(
                    cached.upstream_sha256,
                    shard.published_digest.expected.hex(),
                )
            ):
                raise ValueError(
                    "cached WET object does not match its published SHA-256 digest"
                )
            return _bulk_receipt(cached, shard, download=None, stats=None)

        self.lake.initialize()
        with tempfile.TemporaryDirectory(
            prefix="commoncrawl-wet-",
            dir=self.lake.staging_root,
        ) as temporary:
            path = Path(temporary) / "payload.warc.wet.gz"
            download = self._download(shard, path)
            stats = _ParseStats()
            receipt = self.lake.commit_shard(
                source=self.source,
                dataset=self.dataset,
                release=shard.collection_id,
                shard=shard.source_record_id,
                control_sha256=shard.control_sha256,
                upstream_sha256=download.sha256,
                upstream_url=shard.stable_url,
                upstream_bytes=download.byte_count,
                application_order=shard.application_order,
                records=self._records(download.path, shard, stats),
                batch_rows=self.limits.parquet_batch_rows,
            )
        return _bulk_receipt(
            receipt,
            shard,
            download=download,
            stats=stats if not receipt.already_committed else None,
        )

    def plan_shard(self, control_record: SourceRecord) -> BulkShardPlan | None:
        if control_record.raw.get("record_type") != "commoncrawl_wet_shard":
            return None
        shard = _control_shard(control_record, data_url=self.data_url)
        upstream_sha256 = None
        if (
            shard.published_digest is not None
            and shard.published_digest.algorithm == "sha256"
        ):
            upstream_sha256 = shard.published_digest.expected.hex()
        return BulkShardPlan(
            source=self.source,
            dataset=self.dataset,
            release=shard.collection_id,
            shard=shard.source_record_id,
            control_sha256=shard.control_sha256,
            upstream_sha256=upstream_sha256,
        )

    def shard_order(self, control_record: SourceRecord) -> tuple[Any, ...]:
        shard = _control_shard(control_record, data_url=self.data_url)
        return (
            shard.collection_from,
            shard.collection_id,
            shard.manifest_index,
            shard.source_record_id,
        )

    def _download(self, shard: _ControlShard, destination: Path) -> _Download:
        sha256 = hashlib.sha256()
        published_hasher = (
            _new_hasher(shard.published_digest.algorithm)
            if shard.published_digest is not None
            else None
        )
        byte_count = 0
        content_md5: bytes | None = None
        md5 = None
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
                raise ValueError(f"Common Crawl WET object returned HTTP {status}")
            final_url = str(getattr(response, "url", ""))
            _validate_exact_object_url(
                final_url,
                expected=shard.stable_url,
                data_url=self.data_url,
            )
            response_headers = {
                str(key).casefold(): str(value)
                for key, value in response.headers.items()
            }
            encoding = response_headers.get("content-encoding", "").strip().casefold()
            if encoding not in {"", "identity"}:
                raise ValueError(
                    "Common Crawl WET object must not use HTTP content encoding"
                )
            content_length = _content_length(response_headers.get("content-length"))
            if (
                content_length is not None
                and content_length > self.limits.max_compressed_bytes
            ):
                raise ValueError(
                    "Common Crawl WET object exceeds max_compressed_bytes"
                )
            raw_content_md5 = _optional_text(response_headers.get("content-md5"))
            if raw_content_md5:
                content_md5 = _base64_digest(raw_content_md5, "Content-MD5")
                if len(content_md5) != 16:
                    raise ValueError("Content-MD5 has the wrong digest length")
                md5 = hashlib.md5(usedforsecurity=False)

            response_etag = _optional_text(response_headers.get("etag")) or None
            if shard.object_etag is not None and response_etag != shard.object_etag:
                raise ValueError("Common Crawl object ETag changed from its control")

            with destination.open("xb") as output:
                while True:
                    block = response.read(self.limits.download_chunk_bytes)
                    if not isinstance(block, bytes):
                        raise TypeError("streaming transport returned a non-bytes block")
                    if not block:
                        break
                    byte_count += len(block)
                    if byte_count > self.limits.max_compressed_bytes:
                        raise ValueError(
                            "Common Crawl WET object exceeds max_compressed_bytes"
                        )
                    sha256.update(block)
                    if published_hasher is not None:
                        published_hasher.update(block)
                    if md5 is not None:
                        md5.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())

        if byte_count == 0:
            raise ValueError("Common Crawl WET object download was empty")
        if content_length is not None and content_length != byte_count:
            raise ValueError(
                f"Common Crawl WET object length mismatch: expected {content_length}, "
                f"received {byte_count}"
            )
        if content_md5 is not None and md5 is not None and not hmac.compare_digest(
            md5.digest(), content_md5
        ):
            raise ValueError("Common Crawl WET object failed Content-MD5 verification")
        if (
            shard.published_digest is not None
            and published_hasher is not None
            and not hmac.compare_digest(
                published_hasher.digest(), shard.published_digest.expected
            )
        ):
            raise ValueError(
                "Common Crawl WET object failed its published compressed-object digest"
            )
        return _Download(destination, sha256.hexdigest(), byte_count, response_etag)

    def _records(
        self,
        path: Path,
        shard: _ControlShard,
        stats: _ParseStats,
    ) -> Iterator[LakeRecord]:
        try:
            with (
                path.open("rb") as compressed,
                gzip.GzipFile(fileobj=compressed, mode="rb") as decompressed,
            ):
                reader = _BoundedReader(
                    decompressed,
                    limit=self.limits.max_uncompressed_bytes,
                )
                yield from _iter_wet_records(reader, shard, self.limits, stats)
                stats.uncompressed_bytes = reader.byte_count
        except (gzip.BadGzipFile, EOFError, zlib.error, OSError) as error:
            raise ValueError(
                f"invalid or truncated Common Crawl WET gzip: {error}"
            ) from error


class _BoundedReader:
    def __init__(self, stream: BinaryIO, *, limit: int) -> None:
        self.stream = stream
        self.limit = limit
        self.byte_count = 0

    def readline(self, limit: int, *, eof_ok: bool = False) -> bytes:
        line = self.stream.readline(limit + 1)
        if not isinstance(line, bytes):
            raise TypeError("WET gzip reader returned a non-bytes line")
        self._count(len(line))
        if not line and eof_ok:
            return b""
        if not line:
            raise ValueError("WARC header is truncated")
        if len(line) > limit:
            raise ValueError(f"WARC header line exceeds {limit} bytes")
        if not line.endswith(b"\r\n"):
            raise ValueError("WARC header line is truncated or not CRLF-terminated")
        return line

    def read_exact(self, size: int, *, chunk_size: int) -> Iterator[bytes]:
        remaining = size
        while remaining:
            block = self.stream.read(min(remaining, chunk_size))
            if not isinstance(block, bytes):
                raise TypeError("WET gzip reader returned a non-bytes block")
            if not block:
                raise ValueError("WARC content block is truncated")
            self._count(len(block))
            remaining -= len(block)
            yield block

    def _count(self, count: int) -> None:
        self.byte_count += count
        if self.byte_count > self.limit:
            raise ValueError(
                f"WET shard exceeds {self.limit} uncompressed bytes"
            )


def _iter_wet_records(
    reader: _BoundedReader,
    shard: _ControlShard,
    limits: CommonCrawlBulkLimits,
    stats: _ParseStats,
) -> Iterator[LakeRecord]:
    seen_record_ids: set[str] = set()
    while True:
        envelope = _read_envelope(reader, limits)
        if envelope is None:
            break
        stats.warc_records += 1
        if stats.warc_records > limits.max_records:
            raise ValueError(f"WET shard exceeds {limits.max_records} WARC records")

        record_type = _required_header(envelope, "warc-type").casefold()
        record_id = _warc_record_id(_required_header(envelope, "warc-record-id"))
        warc_date = _warc_date(_required_header(envelope, "warc-date"))
        content_length = _decimal_length(_required_header(envelope, "content-length"))
        if content_length > limits.max_record_bytes:
            raise ValueError(
                f"WARC record exceeds {limits.max_record_bytes} content bytes"
            )
        if record_id in seen_record_ids:
            raise ValueError(f"WET shard repeats WARC-Record-ID {record_id!r}")
        seen_record_ids.add(record_id)

        block_digest = _optional_labelled_digest(
            envelope.one("warc-block-digest"), "WARC-Block-Digest"
        )
        payload_digest = _optional_labelled_digest(
            envelope.one("warc-payload-digest"), "WARC-Payload-Digest"
        )
        segment_number = envelope.one("warc-segment-number")
        if record_type == "conversion" and segment_number is not None:
            raise ValueError(
                "segmented WET conversions cannot be emitted as complete text records"
            )
        digest_algorithms = {"sha256"}
        if block_digest is not None:
            digest_algorithms.add(block_digest.algorithm)
        verify_payload = payload_digest is not None and record_type == "conversion"
        if verify_payload:
            digest_algorithms.add(payload_digest.algorithm)
        hashers = {algorithm: _new_hasher(algorithm) for algorithm in digest_algorithms}

        collect = record_type == "conversion"
        content = bytearray() if collect else None
        for block in reader.read_exact(
            content_length,
            chunk_size=limits.record_chunk_bytes,
        ):
            for hasher in hashers.values():
                hasher.update(block)
            if content is not None:
                content.extend(block)

        separator = b"".join(reader.read_exact(4, chunk_size=4))
        if separator != b"\r\n\r\n":
            raise ValueError("WARC record does not end with two CRLF separators")
        _verify_digest(block_digest, hashers, "WARC-Block-Digest")
        if verify_payload:
            _verify_digest(payload_digest, hashers, "WARC-Payload-Digest")

        if not collect or content is None:
            continue
        target_uri = _target_uri(_required_header(envelope, "warc-target-uri"))
        content_type = _required_header(envelope, "content-type")
        if content_type.partition(";")[0].strip().casefold() != "text/plain":
            raise ValueError("WET conversion Content-Type must be text/plain")
        try:
            text = bytes(content).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("WET conversion content is not valid UTF-8") from error
        if len(text) > limits.max_text_chars_per_record:
            raise ValueError(
                "WET conversion text exceeds max_text_chars_per_record"
            )
        stats.conversion_records += 1
        record_key = canonical_control_sha256(
            {
                "collection_id": shard.collection_id,
                "object_url": shard.stable_url,
                "warc_record_id": record_id,
            }
        )
        yield LakeRecord(
            source_record_id=f"commoncrawl:wet-record:{record_key}",
            payload=_conversion_payload(
                envelope=envelope,
                shard=shard,
                record_id=record_id,
                warc_date=warc_date,
                target_uri=target_uri,
                content_type=content_type,
                content=text,
                content_length=content_length,
                content_sha256=hashers["sha256"].hexdigest(),
                record_index=stats.warc_records - 1,
                conversion_index=stats.conversion_records - 1,
                block_digest=block_digest,
                payload_digest=payload_digest if verify_payload else None,
            ),
        )
    if stats.warc_records == 0:
        raise ValueError("WET shard contains no WARC records")
    if stats.conversion_records == 0:
        raise ValueError("WET shard contains no conversion records")


def _read_envelope(
    reader: _BoundedReader,
    limits: CommonCrawlBulkLimits,
) -> _WarcEnvelope | None:
    version_line = reader.readline(limits.max_header_line_bytes, eof_ok=True)
    if not version_line:
        return None
    header_bytes = len(version_line)
    try:
        version = version_line[:-2].decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("WARC version line is not ASCII") from error
    if version not in {"WARC/1.0", "WARC/1.1"}:
        raise ValueError(f"unsupported WARC version line {version!r}")

    raw_headers: list[list[str]] = []
    while True:
        line = reader.readline(limits.max_header_line_bytes)
        header_bytes += len(line)
        if header_bytes > limits.max_header_bytes:
            raise ValueError(
                f"WARC header exceeds {limits.max_header_bytes} bytes"
            )
        raw = line[:-2]
        if not raw:
            break
        if raw[:1] in {b" ", b"\t"}:
            if not raw_headers:
                raise ValueError("WARC header starts with an orphan continuation line")
            try:
                continuation = raw.lstrip().decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ValueError("WARC header continuation is not valid UTF-8") from error
            raw_headers[-1][1] = f"{raw_headers[-1][1]} {continuation}"
            continue
        if len(raw_headers) >= limits.max_headers_per_record:
            raise ValueError(
                f"WARC record exceeds {limits.max_headers_per_record} headers"
            )
        name_bytes, separator, value_bytes = raw.partition(b":")
        if not separator:
            raise ValueError("WARC header line has no colon")
        try:
            name = name_bytes.decode("ascii", errors="strict")
            value = value_bytes.decode("utf-8", errors="strict").strip(" \t")
        except UnicodeDecodeError as error:
            raise ValueError("WARC header is not valid ASCII/UTF-8") from error
        if not _HEADER_NAME_RE.fullmatch(name):
            raise ValueError(f"WARC header has invalid field name {name!r}")
        if not value:
            raise ValueError(f"WARC header {name!r} has an empty value")
        if any(ord(character) < 32 and character != "\t" for character in value):
            raise ValueError(f"WARC header {name!r} contains control characters")
        raw_headers.append([name.casefold(), value])
    return _WarcEnvelope(
        version=version,
        headers=tuple(_WarcHeader(name, value) for name, value in raw_headers),
    )


def _conversion_payload(
    *,
    envelope: _WarcEnvelope,
    shard: _ControlShard,
    record_id: str,
    warc_date: str,
    target_uri: str,
    content_type: str,
    content: str,
    content_length: int,
    content_sha256: str,
    record_index: int,
    conversion_index: int,
    block_digest: _PublishedDigest | None,
    payload_digest: _PublishedDigest | None,
) -> dict[str, Any]:
    warc: dict[str, Any] = {
        "version": envelope.version,
        "record_id": record_id,
        "type": "conversion",
        "headers": [
            {"name": header.name, "value": header.value}
            for header in envelope.headers
        ],
    }
    if refers_to := envelope.one("warc-refers-to"):
        warc["refers_to"] = refers_to
    if truncated := envelope.one("warc-truncated"):
        warc["truncated"] = truncated
    verified: dict[str, Any] = {}
    if block_digest is not None:
        verified["block"] = _digest_evidence(block_digest)
    if payload_digest is not None:
        verified["payload"] = _digest_evidence(payload_digest)
    if verified:
        warc["verified_digests"] = verified
    return {
        "uri": target_uri,
        "date": warc_date,
        "content": content,
        "content_type": content_type,
        "content_bytes": content_length,
        "content_sha256": content_sha256,
        "warc": warc,
        "bulk": {
            "collection_id": shard.collection_id,
            "collection_from": shard.collection_from,
            "collection_to": shard.collection_to,
            "manifest_index": shard.manifest_index,
            "total_shards": shard.total_shards,
            "object_url": shard.stable_url,
            "path": shard.path,
            "record_index": record_index,
            "conversion_index": conversion_index,
        },
    }


def _control_shard(record: SourceRecord, *, data_url: str) -> _ControlShard:
    if not isinstance(record, SourceRecord):
        raise TypeError("control_record must be a SourceRecord")
    if record.kind is not ArtifactKind.CATALOG_RECORD:
        raise ValueError("Common Crawl control must be a catalog record")
    raw = record.raw
    if not isinstance(raw, Mapping):
        raise TypeError("Common Crawl control raw value must be a mapping")
    if raw.get("record_type") != "commoncrawl_wet_shard":
        raise ValueError("control record is not a Common Crawl WET shard")
    collection_id = _collection_id(raw.get("collection_id"))
    collection_from = _collection_timestamp(
        raw.get("collection_from"), "collection_from"
    )
    collection_to = _collection_timestamp(raw.get("collection_to"), "collection_to")
    if _timestamp_value(collection_to) < _timestamp_value(collection_from):
        raise ValueError("Common Crawl collection ends before it starts")
    manifest_url = _required_text(raw.get("manifest_url"), "manifest_url")
    _validate_manifest_url(manifest_url, collection_id=collection_id, data_url=data_url)
    manifest_digest = _sha256(raw.get("manifest_digest"), "manifest_digest")
    if _required_text(
        raw.get("manifest_digest_algorithm"), "manifest_digest_algorithm"
    ).casefold() != "sha256":
        raise ValueError("manifest digest algorithm must be SHA-256")
    stable_url = _required_text(raw.get("stable_object_url"), "stable_object_url")
    path = _required_text(raw.get("path"), "path")
    _validate_exact_object_url(stable_url, expected=stable_url, data_url=data_url)
    expected_url = f"{data_url}{path}"
    if stable_url != expected_url or record.canonical_url != stable_url:
        raise ValueError("Common Crawl control URL, path, and canonical URL disagree")
    _validate_wet_path(path, collection_id)
    manifest_index = _integer(raw.get("manifest_index"), "manifest_index", minimum=0)
    total_shards = _integer(raw.get("total_shards"), "total_shards", minimum=1)
    if manifest_index >= total_shards:
        raise ValueError("manifest_index must be smaller than total_shards")

    record_key = canonical_control_sha256(
        {"collection_id": collection_id, "stable_object_url": stable_url}
    )
    expected_record_id = f"commoncrawl:wet-shard:{record_key}"
    if record.source_record_id != expected_record_id:
        raise ValueError("Common Crawl control source_record_id is inconsistent")
    published_digest = _object_digest(raw.get("object_digest"))
    object_etag = _optional_exact_text(raw.get("object_etag"), "object_etag")
    return _ControlShard(
        collection_id=collection_id,
        collection_from=collection_from,
        collection_to=collection_to,
        manifest_url=manifest_url,
        manifest_digest=manifest_digest,
        stable_url=stable_url,
        path=path,
        manifest_index=manifest_index,
        total_shards=total_shards,
        source_record_id=record.source_record_id,
        control_sha256=canonical_control_sha256(
            {
                "canonical_url": record.canonical_url,
                "raw": dict(raw),
                "source_record_id": record.source_record_id,
            }
        ),
        published_digest=published_digest,
        object_etag=object_etag,
    )


def _validate_manifest_url(url: str, *, collection_id: str, data_url: str) -> None:
    expected = f"{data_url}crawl-data/{collection_id}/wet.paths.gz"
    if url != expected:
        raise ValueError("Common Crawl manifest URL is not the canonical collection URL")


def _validate_exact_object_url(url: str, *, expected: str, data_url: str) -> None:
    if url != expected or not url.startswith(data_url):
        raise ValueError("Common Crawl transfer URL changed from its exact control")
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
        raise ValueError("Common Crawl transfer URL is unsafe")
    decoded = unquote(parts.path)
    if decoded != parts.path or "\\" in decoded or "\x00" in decoded:
        raise ValueError("Common Crawl transfer URL has an unsafe encoded path")
    if any(segment in {"", ".", ".."} for segment in decoded.split("/")[1:]):
        raise ValueError("Common Crawl transfer URL contains path traversal")


def _validate_wet_path(path: str, collection_id: str) -> None:
    if path != path.strip() or path.startswith("/") or "\\" in path or "//" in path:
        raise ValueError("Common Crawl WET object path is unsafe")
    if unquote(path) != path:
        raise ValueError("Common Crawl WET object path must not be percent-encoded")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Common Crawl WET object path contains traversal")
    if (
        len(parts) < 5
        or parts[:2] != ["crawl-data", collection_id]
        or "wet" not in parts[2:-1]
        or not parts[-1].endswith(".warc.wet.gz")
    ):
        raise ValueError("Common Crawl control path is not a collection WET shard")


def _object_digest(value: Any) -> _PublishedDigest | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("object_digest must be a mapping")
    if _required_text(value.get("scope"), "object_digest scope") != "compressed_object":
        raise ValueError("object_digest scope must be compressed_object")
    algorithm = _digest_algorithm(value.get("algorithm"), "object_digest algorithm")
    encoding = _required_text(value.get("encoding"), "object_digest encoding").casefold()
    expected = _decode_digest_value(
        _required_text(value.get("value"), "object_digest value"),
        algorithm=algorithm,
        encoding=encoding,
        label="object_digest",
    )
    return _PublishedDigest(algorithm, expected, encoding)


def _optional_labelled_digest(value: str | None, label: str) -> _PublishedDigest | None:
    if value is None:
        return None
    algorithm_label, separator, digest_value = value.partition(":")
    if not separator or not algorithm_label or not digest_value:
        raise ValueError(f"{label} is not a labelled digest")
    algorithm = _digest_algorithm(algorithm_label, f"{label} algorithm")
    encoding = _infer_digest_encoding(digest_value, algorithm)
    expected = _decode_digest_value(
        digest_value,
        algorithm=algorithm,
        encoding=encoding,
        label=label,
    )
    return _PublishedDigest(algorithm, expected, encoding)


def _infer_digest_encoding(value: str, algorithm: str) -> str:
    digest_size = _new_hasher(algorithm).digest_size
    if len(value) == digest_size * 2 and all(
        character in "0123456789abcdefABCDEF" for character in value
    ):
        return "hex"
    return "base32"


def _decode_digest_value(
    value: str,
    *,
    algorithm: str,
    encoding: str,
    label: str,
) -> bytes:
    if any(character.isspace() for character in value):
        raise ValueError(f"{label} digest must not contain whitespace")
    try:
        if encoding == "hex":
            result = bytes.fromhex(value)
        elif encoding == "base32":
            padding = "=" * (-len(value) % 8)
            result = base64.b32decode(value + padding, casefold=True)
        elif encoding == "base64":
            result = base64.b64decode(value, validate=True)
        else:
            raise ValueError(f"{label} has unsupported digest encoding {encoding!r}")
    except (binascii.Error, UnicodeEncodeError, ValueError) as error:
        raise ValueError(f"{label} has an invalid {encoding} digest") from error
    expected_size = _new_hasher(algorithm).digest_size
    if len(result) != expected_size:
        raise ValueError(f"{label} has the wrong digest length")
    return result


def _verify_digest(
    digest: _PublishedDigest | None,
    hashers: Mapping[str, Any],
    label: str,
) -> None:
    if digest is None:
        return
    if not hmac.compare_digest(hashers[digest.algorithm].digest(), digest.expected):
        raise ValueError(f"{label} verification failed")


def _digest_evidence(digest: _PublishedDigest) -> dict[str, Any]:
    return {
        "algorithm": digest.algorithm,
        "encoding": digest.encoding,
        "value": _encode_digest(digest.expected, digest.encoding),
        "verified": True,
    }


def _encode_digest(value: bytes, encoding: str) -> str:
    if encoding == "hex":
        return value.hex()
    if encoding == "base32":
        return base64.b32encode(value).decode().rstrip("=")
    return base64.b64encode(value).decode()


def _new_hasher(algorithm: str) -> Any:
    if algorithm == "md5":
        return hashlib.md5(usedforsecurity=False)
    return hashlib.new(algorithm)


def _digest_algorithm(value: Any, label: str) -> str:
    raw = _required_text(value, label).casefold()
    algorithm = _DIGEST_ALGORITHMS.get(raw)
    if algorithm is None:
        raise ValueError(f"{label} is unsupported")
    return algorithm


def _required_header(envelope: _WarcEnvelope, name: str) -> str:
    value = envelope.one(name, required=True)
    if value is None:  # Required headers return or raise.
        raise ValueError(f"WARC record is missing mandatory header {name}")
    return value


def _warc_record_id(value: str) -> str:
    candidate = value[1:-1] if value.startswith("<") and value.endswith(">") else value
    if not candidate or not _URI_SCHEME_RE.match(candidate):
        raise ValueError("WARC-Record-ID must be an absolute URI")
    if any(character.isspace() or ord(character) < 32 for character in candidate):
        raise ValueError("WARC-Record-ID contains whitespace or control characters")
    return candidate


def _target_uri(value: str) -> str:
    if not _URI_SCHEME_RE.match(value):
        raise ValueError("WARC-Target-URI must be an absolute URI")
    if any(ord(character) < 32 or character.isspace() for character in value):
        raise ValueError("WARC-Target-URI contains whitespace or control characters")
    return value


def _warc_date(value: str) -> str:
    _utc_timestamp(value, "WARC-Date")
    return value


def _collection_timestamp(value: Any, label: str) -> str:
    text = _required_exact_text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is not None and parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} must identify a UTC instant")
    return text


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
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result


def _decimal_length(value: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ValueError("WARC Content-Length must contain decimal digits")
    return int(value)


def _bulk_receipt(
    receipt: ShardReceipt,
    shard: _ControlShard,
    *,
    download: _Download | None,
    stats: _ParseStats | None,
) -> CommonCrawlBulkReceipt:
    compressed_bytes = receipt.upstream_bytes
    if compressed_bytes is None:
        raise ValueError("landing receipt is missing upstream byte count")
    return CommonCrawlBulkReceipt(
        source=receipt.source,
        dataset=receipt.dataset,
        release=receipt.release,
        shard=receipt.shard,
        collection_id=shard.collection_id,
        manifest_index=shard.manifest_index,
        total_shards=shard.total_shards,
        control_sha256=receipt.control_sha256,
        upstream_sha256=receipt.upstream_sha256,
        compressed_bytes=compressed_bytes,
        uncompressed_bytes=stats.uncompressed_bytes if stats is not None else None,
        warc_records=stats.warc_records if stats is not None else None,
        row_count=receipt.row_count,
        path=receipt.path,
        already_committed=receipt.already_committed,
        response_etag=download.response_etag if download is not None else None,
        shard_receipt=receipt,
    )


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


def _collection_id(value: Any) -> str:
    result = _required_exact_text(value, "collection_id")
    if not _COLLECTION_ID_RE.fullmatch(result) or result in {".", ".."}:
        raise ValueError("Common Crawl collection_id is malformed")
    return result


def _sha256(value: Any, label: str) -> str:
    result = _required_exact_text(value, label).casefold()
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


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


def _optional_exact_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _required_exact_text(value, label)


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
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
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
    "CommonCrawlBulkLimits",
    "CommonCrawlBulkReceipt",
    "CommonCrawlWetBulkLoader",
    "HttpsCommonCrawlBulkTransport",
]
