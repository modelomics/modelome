from __future__ import annotations

import gzip
import hashlib
import hmac
import ipaddress
import json
import os
import re
import socket
import tarfile
import tempfile
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
from modelome.normalize import content_hash

_DATASET = "graph"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_RECORD_ID_RE = re.compile(r"^[1-9][0-9]*$")
_RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})


class StreamingResponse(Protocol):
    status: int
    headers: Mapping[str, str]
    url: str

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> StreamingResponse: ...

    def __exit__(self, *args: Any) -> None: ...


class OpenAireBulkTransport(Protocol):
    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> AbstractContextManager[StreamingResponse]: ...


class HttpsOpenAireBulkTransport:
    """Open public HTTPS byte streams with retry-safe redirect validation."""

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
            f"OpenAIRE Graph shard transfer failed ({type(last_error).__name__})"
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
class OpenAireBulkLimits:
    """Hard ceilings for one OpenAIRE tar shard and its gzip JSONL members."""

    max_archive_bytes: int = 16 * 1024 * 1024 * 1024
    max_archive_members: int = 100_000
    max_member_name_bytes: int = 4 * 1024
    max_member_compressed_bytes: int = 16 * 1024 * 1024 * 1024
    max_member_uncompressed_bytes: int = 64 * 1024 * 1024 * 1024
    max_total_uncompressed_bytes: int = 256 * 1024 * 1024 * 1024
    max_payload_bytes: int = 32 * 1024 * 1024
    max_rows: int = 50_000_000
    download_chunk_bytes: int = 1024 * 1024
    decompression_chunk_bytes: int = 1024 * 1024
    parquet_batch_rows: int = 16

    def __post_init__(self) -> None:
        for field in (
            "max_archive_bytes",
            "max_archive_members",
            "max_member_name_bytes",
            "max_member_compressed_bytes",
            "max_member_uncompressed_bytes",
            "max_total_uncompressed_bytes",
            "max_payload_bytes",
            "max_rows",
            "download_chunk_bytes",
            "decompression_chunk_bytes",
            "parquet_batch_rows",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.download_chunk_bytes > self.max_archive_bytes:
            raise ValueError("download_chunk_bytes must not exceed max_archive_bytes")
        if self.decompression_chunk_bytes > self.max_member_uncompressed_bytes:
            raise ValueError(
                "decompression_chunk_bytes must not exceed max_member_uncompressed_bytes"
            )
        if self.decompression_chunk_bytes > self.max_total_uncompressed_bytes:
            raise ValueError(
                "decompression_chunk_bytes must not exceed max_total_uncompressed_bytes"
            )
        if self.max_payload_bytes > self.max_member_uncompressed_bytes:
            raise ValueError("max_payload_bytes must not exceed max_member_uncompressed_bytes")


@dataclass(frozen=True, slots=True)
class _ControlShard:
    release: str
    manifest_signature: str
    manifest_index: int
    manifest_count: int
    file_key: str
    entity_partition: str
    stable_url: str
    expected_bytes: int
    expected_md5: str
    source_record_id: str
    control_sha256: str

    @property
    def application_order(self) -> ShardApplicationOrder:
        return ShardApplicationOrder.snapshot(self.manifest_index)


@dataclass(frozen=True, slots=True)
class _Download:
    path: Path
    sha256: str
    byte_count: int


@dataclass(slots=True)
class _ParseStats:
    archive_members: int = 0
    gzip_members: int = 0
    member_compressed_bytes: int = 0
    uncompressed_bytes: int = 0
    rows: int = 0


class OpenAireGraphBulkLoader:
    """Verify and stream one exact OpenAIRE full-graph control into Parquet."""

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        source: str = "openaire-graph",
        limits: OpenAireBulkLimits | None = None,
        transport: OpenAireBulkTransport | None = None,
    ) -> None:
        if not isinstance(landing_zone, ParquetLandingZone):
            raise TypeError("landing_zone must be a ParquetLandingZone")
        self.landing_zone = landing_zone
        self.source = _required_text(source, "source")
        self.dataset = _DATASET
        self.limits = limits or OpenAireBulkLimits()
        if not isinstance(self.limits, OpenAireBulkLimits):
            raise TypeError("limits must be OpenAireBulkLimits")
        self.transport = transport or HttpsOpenAireBulkTransport()

    def load(self, control_record: SourceRecord) -> ShardReceipt:
        """Land one exact control, replaying an already committed control offline."""

        shard = _control_shard(control_record)
        self._validate_control_limits(shard)
        cached = self.landing_zone.lookup_committed_shard(
            source=self.source,
            dataset=self.dataset,
            release=shard.release,
            shard=shard.source_record_id,
            control_sha256=shard.control_sha256,
            upstream_url=shard.stable_url,
            application_order=shard.application_order,
        )
        if cached is not None:
            if cached.upstream_bytes != shard.expected_bytes:
                raise ValueError("cached OpenAIRE shard byte count changed from its exact control")
            return cached

        self.landing_zone.initialize()
        with tempfile.TemporaryDirectory(
            prefix="openaire-graph-",
            dir=self.landing_zone.staging_root,
        ) as temporary:
            path = Path(temporary) / "payload.tar"
            download = self._download(shard, path)
            stats = _ParseStats()
            records = self._records(download.path, shard, stats)
            receipt = self.landing_zone.commit_shard(
                source=self.source,
                dataset=self.dataset,
                release=shard.release,
                shard=shard.source_record_id,
                control_sha256=shard.control_sha256,
                upstream_sha256=download.sha256,
                upstream_url=shard.stable_url,
                upstream_bytes=download.byte_count,
                application_order=shard.application_order,
                records=records,
                batch_rows=self.limits.parquet_batch_rows,
            )
        if receipt.row_count != stats.rows:
            raise ValueError("landed OpenAIRE row count differs from parsed row count")
        return receipt

    def plan_shard(self, control_record: SourceRecord) -> BulkShardPlan | None:
        """Return one release-wide graph plan without opening the network."""

        if control_record.raw.get("record_type") != "dataset_shard":
            return None
        shard = _control_shard(control_record)
        self._validate_control_limits(shard)
        return BulkShardPlan(
            source=self.source,
            dataset=self.dataset,
            release=shard.release,
            shard=shard.source_record_id,
            control_sha256=shard.control_sha256,
        )

    def shard_order(self, control_record: SourceRecord) -> tuple[int, int, str]:
        """Order every partition solely by its frozen manifest position."""

        shard = _control_shard(control_record)
        return (int(shard.release), shard.manifest_index, shard.source_record_id)

    def _validate_control_limits(self, shard: _ControlShard) -> None:
        if shard.expected_bytes > self.limits.max_archive_bytes:
            raise ValueError("OpenAIRE archive exceeds max_archive_bytes")
        if shard.manifest_count > self.limits.max_archive_members:
            # This is a separate, conservative ceiling on release controls. It
            # prevents an untrusted manifest from claiming an unbounded group.
            raise ValueError("OpenAIRE release exceeds the control-count ceiling")

    def _download(self, shard: _ControlShard, destination: Path) -> _Download:
        sha256 = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        byte_count = 0
        headers = {
            # Zenodo's file-content endpoint currently negotiates downloads
            # through the wildcard media range and returns octet-stream.
            "Accept": "*/*",
            "Accept-Encoding": "identity",
        }
        with self.transport.open(
            shard.stable_url,
            headers=headers,
            redirect_validator=lambda url: _validate_exact_url(url, expected=shard.stable_url),
        ) as response:
            status = int(getattr(response, "status", 0))
            if status != 200:
                raise ValueError(f"OpenAIRE archive returned HTTP {status}")
            _validate_exact_url(
                str(getattr(response, "url", "")),
                expected=shard.stable_url,
            )
            response_headers = {
                str(key).casefold(): str(value) for key, value in response.headers.items()
            }
            encoding = response_headers.get("content-encoding", "").strip().casefold()
            if encoding not in {"", "identity"}:
                raise ValueError("OpenAIRE archive must not use HTTP content encoding")
            content_length = _content_length(response_headers.get("content-length"))
            if content_length is None:
                raise ValueError("OpenAIRE archive response is missing Content-Length")
            if content_length != shard.expected_bytes:
                raise ValueError("OpenAIRE archive Content-Length changed from its exact control")
            if content_length > self.limits.max_archive_bytes:
                raise ValueError("OpenAIRE archive exceeds max_archive_bytes")
            response_md5 = _response_md5(response_headers.get("oc-checksum"))
            if response_md5 is not None and not hmac.compare_digest(
                response_md5,
                shard.expected_md5,
            ):
                raise ValueError("OpenAIRE response checksum changed from its exact control")

            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb") as output:
                    descriptor = -1
                    while True:
                        block = response.read(self.limits.download_chunk_bytes)
                        if not isinstance(block, bytes):
                            raise TypeError("OpenAIRE bulk transport returned non-bytes")
                        if not block:
                            break
                        byte_count += len(block)
                        if byte_count > shard.expected_bytes:
                            raise ValueError("OpenAIRE archive exceeded its exact control size")
                        if byte_count > self.limits.max_archive_bytes:
                            raise ValueError("OpenAIRE archive exceeds max_archive_bytes")
                        sha256.update(block)
                        md5.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

        if byte_count != content_length or byte_count != shard.expected_bytes:
            raise ValueError(
                f"OpenAIRE archive length mismatch: expected {shard.expected_bytes}, "
                f"received {byte_count}"
            )
        if not hmac.compare_digest(md5.hexdigest(), shard.expected_md5):
            raise ValueError("OpenAIRE archive failed its published MD5 checksum")
        return _Download(destination, sha256.hexdigest(), byte_count)

    def _records(
        self,
        path: Path,
        shard: _ControlShard,
        stats: _ParseStats,
    ) -> Iterator[LakeRecord]:
        if path.stat().st_size != shard.expected_bytes:
            raise ValueError("downloaded OpenAIRE archive size changed before parsing")
        seen_names: set[str] = set()
        try:
            with (
                path.open("rb") as stream,
                tarfile.open(
                    fileobj=stream,
                    mode="r|",
                ) as archive,
            ):
                for member in archive:
                    stats.archive_members += 1
                    if stats.archive_members > self.limits.max_archive_members:
                        raise ValueError("OpenAIRE archive exceeded max_archive_members")
                    name = _safe_member_name(
                        member.name,
                        maximum_bytes=self.limits.max_member_name_bytes,
                    )
                    folded = name.casefold()
                    if folded in seen_names:
                        raise ValueError(f"OpenAIRE archive repeats member {name!r}")
                    seen_names.add(folded)
                    if not member.isfile() or member.islnk() or member.issym():
                        raise ValueError(f"OpenAIRE archive member {name!r} is not a regular file")
                    if getattr(member, "sparse", None):
                        raise ValueError(f"OpenAIRE archive member {name!r} must not be sparse")
                    if member.size < 1:
                        raise ValueError(f"OpenAIRE archive member {name!r} must not be empty")
                    if member.size > self.limits.max_member_compressed_bytes:
                        raise ValueError(
                            "OpenAIRE archive member exceeds max_member_compressed_bytes"
                        )
                    stats.member_compressed_bytes += member.size
                    if stats.member_compressed_bytes > self.limits.max_archive_bytes:
                        raise ValueError("OpenAIRE archive member bytes exceed max_archive_bytes")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"OpenAIRE archive member {name!r} could not be read")
                    stats.gzip_members += 1
                    yield from _iter_gzip_jsonl(
                        extracted,
                        member_size=member.size,
                        member_name=name,
                        shard=shard,
                        limits=self.limits,
                        stats=stats,
                        source=self.source,
                    )
        except (tarfile.TarError, EOFError, OSError) as error:
            raise ValueError("OpenAIRE shard is an invalid or truncated tar archive") from error
        if stats.gzip_members == 0:
            raise ValueError("OpenAIRE archive contained no gzip JSONL members")
        if stats.rows == 0:
            raise ValueError("OpenAIRE archive contained no JSON objects")


class _CountingReader:
    def __init__(self, stream: BinaryIO, *, limit: int) -> None:
        self.stream = stream
        self.limit = limit
        self.byte_count = 0

    def read(self, size: int = -1) -> bytes:
        block = self.stream.read(size)
        if not isinstance(block, bytes):
            raise TypeError("tar member reader returned non-bytes")
        self.byte_count += len(block)
        if self.byte_count > self.limit:
            raise ValueError("OpenAIRE gzip member exceeded its declared size")
        return block


def _iter_gzip_jsonl(
    stream: BinaryIO,
    *,
    member_size: int,
    member_name: str,
    shard: _ControlShard,
    limits: OpenAireBulkLimits,
    stats: _ParseStats,
    source: str,
) -> Iterator[LakeRecord]:
    compressed = _CountingReader(stream, limit=member_size)
    member_uncompressed = 0
    member_rows = 0
    line_number = 0
    buffer = bytearray()
    try:
        with gzip.GzipFile(fileobj=compressed, mode="rb") as decompressed:
            while True:
                block = decompressed.read1(limits.decompression_chunk_bytes)
                if not isinstance(block, bytes):
                    raise TypeError("OpenAIRE gzip reader returned non-bytes")
                if not block:
                    break
                member_uncompressed += len(block)
                stats.uncompressed_bytes += len(block)
                if member_uncompressed > limits.max_member_uncompressed_bytes:
                    raise ValueError("OpenAIRE gzip member exceeded max_member_uncompressed_bytes")
                if stats.uncompressed_bytes > limits.max_total_uncompressed_bytes:
                    raise ValueError("OpenAIRE archive exceeded max_total_uncompressed_bytes")
                buffer.extend(block)
                while (newline := buffer.find(b"\n")) >= 0:
                    line = bytes(buffer[:newline])
                    del buffer[: newline + 1]
                    line_number += 1
                    stats.rows += 1
                    member_rows += 1
                    if stats.rows > limits.max_rows:
                        raise ValueError("OpenAIRE archive exceeded max_rows")
                    yield _lake_record(
                        line,
                        source=source,
                        shard=shard,
                        member_name=member_name,
                        line_number=line_number,
                        max_payload_bytes=limits.max_payload_bytes,
                    )
                if len(buffer) > limits.max_payload_bytes:
                    raise ValueError(
                        f"OpenAIRE JSONL payload {line_number + 1} exceeded max_payload_bytes"
                    )
    except (gzip.BadGzipFile, EOFError, zlib.error, OSError) as error:
        raise ValueError(
            f"OpenAIRE archive member {member_name!r} is invalid or truncated gzip"
        ) from error
    if compressed.byte_count != member_size:
        raise ValueError(f"OpenAIRE archive member {member_name!r} compressed size was truncated")
    if buffer:
        raise ValueError(
            f"OpenAIRE archive member {member_name!r} has an unterminated JSONL payload"
        )
    if member_rows == 0:
        raise ValueError(f"OpenAIRE archive member {member_name!r} contained no JSON objects")


def _lake_record(
    line: bytes,
    *,
    source: str,
    shard: _ControlShard,
    member_name: str,
    line_number: int,
    max_payload_bytes: int,
) -> LakeRecord:
    if line.endswith(b"\r"):
        line = line[:-1]
    if not line:
        raise ValueError(f"OpenAIRE JSONL payload {line_number} is empty")
    if len(line) > max_payload_bytes:
        raise ValueError(f"OpenAIRE JSONL payload {line_number} exceeded max_payload_bytes")
    try:
        payload = json.loads(
            line,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ValueError(
            f"OpenAIRE JSONL payload {line_number} is not a valid JSON object"
        ) from None
    if not isinstance(payload, Mapping):
        raise ValueError(f"OpenAIRE JSONL payload {line_number} must contain a JSON object")
    locator = content_hash(
        {
            "release": shard.release,
            "file_key": shard.file_key,
            "member_name": member_name,
            "line_number": line_number,
        }
    )
    return LakeRecord(
        source_record_id=f"{source}:row:{locator}",
        payload=dict(payload),
        operation="upsert",
    )


def _control_shard(record: SourceRecord) -> _ControlShard:
    if not isinstance(record, SourceRecord):
        raise TypeError("control_record must be a SourceRecord")
    if record.kind is not ArtifactKind.CATALOG_RECORD:
        raise ValueError("OpenAIRE bulk control must be a catalog record")
    if record.deleted:
        raise ValueError("OpenAIRE bulk control must not be deleted")
    raw = record.raw
    if not isinstance(raw, Mapping):
        raise TypeError("control_record.raw must be a mapping")
    if raw.get("record_type") != "dataset_shard":
        raise ValueError("control record is not an OpenAIRE dataset shard")
    if raw.get("operation") != "snapshot":
        raise ValueError("OpenAIRE dataset shard operation must be snapshot")

    release = _record_id(raw.get("release_id"), "release_id")
    manifest_signature = _digest(
        raw.get("manifest_signature"),
        _SHA256_RE,
        "manifest_signature",
    )
    manifest_index = _nonnegative_integer(raw.get("manifest_index"), "manifest_index")
    manifest_count = _positive_integer(raw.get("manifest_count"), "manifest_count")
    if manifest_index >= manifest_count:
        raise ValueError("manifest_index must be smaller than manifest_count")
    file_key = _safe_file_key(raw.get("file_key"))
    entity_partition = _required_text(raw.get("entity_partition"), "entity_partition")
    expected_bytes = _positive_integer(raw.get("size"), "control size")

    checksum = raw.get("checksum")
    if not isinstance(checksum, Mapping):
        raise ValueError("control checksum must be a mapping")
    algorithm = _required_text(checksum.get("algorithm"), "checksum algorithm").casefold()
    if algorithm != "md5":
        raise ValueError("OpenAIRE control checksum algorithm must be md5")
    expected_md5 = _digest(checksum.get("value"), _MD5_RE, "MD5 checksum")

    stable_url = _stable_https_url(_required_text(raw.get("download_url"), "download_url"))
    _validate_download_path(stable_url, release=release, file_key=file_key)
    if record.canonical_url != stable_url:
        raise ValueError("control canonical URL does not match download_url")
    bulk_links = [link for link in record.links if link.relation == "bulk_payload"]
    if len(bulk_links) != 1 or bulk_links[0].url != stable_url:
        raise ValueError("control must contain exactly one matching bulk_payload link")
    for link in record.links:
        _stable_https_url(link.url)
        if link.crawl:
            raise ValueError("OpenAIRE control links must not enter the crawl frontier")

    source_record_id = _required_text(record.source_record_id, "source_record_id")
    control_sha256 = canonical_control_sha256(
        {
            "canonical_url": record.canonical_url,
            "raw": dict(raw),
            "source_record_id": source_record_id,
        }
    )
    return _ControlShard(
        release=release,
        manifest_signature=manifest_signature,
        manifest_index=manifest_index,
        manifest_count=manifest_count,
        file_key=file_key,
        entity_partition=entity_partition,
        stable_url=stable_url,
        expected_bytes=expected_bytes,
        expected_md5=expected_md5,
        source_record_id=source_record_id,
        control_sha256=control_sha256,
    )


def _validate_download_path(url: str, *, release: str, file_key: str) -> None:
    parts = urlsplit(url)
    components = parts.path.split("/")
    if len(components) != 7 or components[:4] != ["", "api", "records", release]:
        raise ValueError("OpenAIRE download URL is not a canonical record file URL")
    if components[4] != "files" or components[6] != "content":
        raise ValueError("OpenAIRE download URL is not a canonical record file URL")
    if unquote(components[5]) != file_key:
        raise ValueError("OpenAIRE download URL does not match file_key")


def _safe_file_key(value: Any) -> str:
    key = _required_text(value, "file_key")
    if (
        len(key.encode("utf-8")) > 4_096
        or not key.endswith(".tar")
        or key in {".", ".."}
        or "/" in key
        or "\\" in key
        or any(ord(character) < 32 for character in key)
    ):
        raise ValueError("file_key must be a safe .tar filename")
    return key


def _safe_member_name(value: Any, *, maximum_bytes: int) -> str:
    name = _required_text(value, "tar member name")
    try:
        encoded = name.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("tar member name is not valid UTF-8") from None
    path = PurePosixPath(name)
    if (
        len(encoded) > maximum_bytes
        or name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or any(ord(character) < 32 for character in name)
        or any(component in {"", ".", ".."} for component in name.split("/"))
        or not path.name.endswith(".gz")
    ):
        raise ValueError(f"unsafe OpenAIRE tar member name {name!r}")
    return name


def _content_length(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text.isascii() or not text.isdigit():
        raise ValueError("OpenAIRE Content-Length must be a nonnegative integer")
    return int(text)


def _response_md5(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    algorithm, separator, digest = text.partition(":")
    if (
        separator != ":"
        or algorithm.casefold() != "md5"
        or _MD5_RE.fullmatch(digest.casefold()) is None
    ):
        raise ValueError("OpenAIRE OC-Checksum header is malformed")
    return digest.casefold()


def _validate_exact_url(value: str, *, expected: str) -> None:
    actual = _stable_https_url(value)
    if actual != expected:
        raise ValueError("OpenAIRE archive redirect or final URL changed exact object")


def _stable_https_url(value: str) -> str:
    parts = urlsplit(value)
    try:
        port = parts.port
    except ValueError:
        raise ValueError("OpenAIRE URL has an invalid port") from None
    host = (parts.hostname or "").casefold().rstrip(".")
    if (
        parts.scheme.casefold() != "https"
        or not host
        or parts.username is not None
        or parts.password is not None
        or port not in {None, 443}
        or parts.query
        or parts.fragment
        or not parts.path
    ):
        raise ValueError("OpenAIRE URL must be credential-free query-free HTTPS")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("OpenAIRE URL host must be public")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("OpenAIRE URL host must be public")
    stable_host = f"[{host}]" if ":" in host else host
    return urlunsplit(("https", stable_host, parts.path, "", ""))


def _require_public_https(value: str) -> None:
    stable = _stable_https_url(value)
    host = urlsplit(stable).hostname
    if host is None:
        raise ValueError("OpenAIRE URL host must be public")
    try:
        addresses = (ipaddress.ip_address(host),)
    except ValueError:
        try:
            addresses = tuple(
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            )
        except OSError:
            raise ValueError("OpenAIRE URL host could not be resolved safely") from None
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("OpenAIRE URL host must be public")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _record_id(value: Any, field: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive decimal integer")
    text = str(value) if isinstance(value, int) else value
    if not isinstance(text, str) or not _RECORD_ID_RE.fullmatch(text):
        raise ValueError(f"{field} must be a positive decimal integer")
    return text


def _digest(value: Any, pattern: re.Pattern[str], field: str) -> str:
    text = _required_text(value, field).casefold()
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{field} is malformed")
    return text


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


__all__ = [
    "HttpsOpenAireBulkTransport",
    "OpenAireBulkLimits",
    "OpenAireBulkTransport",
    "OpenAireGraphBulkLoader",
]
