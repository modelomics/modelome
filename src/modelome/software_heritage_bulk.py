from __future__ import annotations

import base64
import hashlib
import ipaddress
import os
import re
import socket
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pyarrow as pa
import pyarrow.orc as orc

from modelome.bulk import BulkShardPlan
from modelome.lake import (
    LakeRecord,
    ParquetLandingZone,
    ShardApplicationOrder,
    ShardReceipt,
    canonical_control_sha256,
)
from modelome.models import ArtifactKind, SourceRecord

_BUCKET_URL = "https://softwareheritage.s3.amazonaws.com/"
_GRAPH_PREFIX = "graph/"
_DATASET = "origins"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN_FILE_RE = re.compile(r"^origin-[A-Za-z0-9][A-Za-z0-9._-]*\.orc$")
_RELEASE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
SOFTWARE_HERITAGE_DEFAULT_NEW_SHARD_BUDGET = 4


class StreamingResponse(Protocol):
    status: int
    headers: Mapping[str, str]
    url: str

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> StreamingResponse: ...

    def __exit__(self, *args: Any) -> None: ...


class SoftwareHeritageBulkTransport(Protocol):
    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> AbstractContextManager[StreamingResponse]: ...


class HttpsSoftwareHeritageBulkTransport:
    """Open one public S3 object and validate every redirect target."""

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
            try:
                return opener.open(  # type: ignore[return-value]
                    Request(url, headers=request_headers, method="GET"),
                    timeout=self.timeout,
                )
            except HTTPError as error:
                last_error = error
                if error.code not in _RETRYABLE_HTTP:
                    break
            except (TimeoutError, URLError, OSError, ValueError) as error:
                last_error = error
            if attempt + 1 < self.attempts:
                self._sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"Software Heritage ORC transfer failed ({type(last_error).__name__})"
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
                "user-agent",
                "x-amz-checksum-mode",
            }:
                redirected.remove_header(name)
        return redirected


@dataclass(frozen=True, slots=True)
class SoftwareHeritageOriginBulkLimits:
    max_object_bytes: int = 1024 * 1024 * 1024
    max_rows: int = 250_000_000
    max_stripes: int = 1_000_000
    max_url_bytes: int = 1024 * 1024
    download_chunk_bytes: int = 1024 * 1024
    parquet_batch_rows: int = 16_384

    def __post_init__(self) -> None:
        for field in (
            "max_object_bytes",
            "max_rows",
            "max_stripes",
            "max_url_bytes",
            "download_chunk_bytes",
            "parquet_batch_rows",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.download_chunk_bytes > self.max_object_bytes:
            raise ValueError("download_chunk_bytes must not exceed max_object_bytes")


@dataclass(frozen=True, slots=True)
class SoftwareHeritageOriginBulkReceipt:
    source: str
    dataset: str
    release: str
    shard: str
    manifest_index: int
    manifest_count: int
    control_sha256: str
    upstream_sha256: str
    upstream_bytes: int
    row_count: int
    path: Path
    already_committed: bool
    response_etag: str | None = None
    response_checksum_crc32: str | None = None
    shard_receipt: ShardReceipt | None = None


@dataclass(frozen=True, slots=True)
class _ControlShard:
    release: str
    object_key: str
    stable_url: str
    expected_bytes: int
    object_etag: str
    last_modified: str
    checksum_algorithms: tuple[str, ...]
    checksum_type: str | None
    manifest_index: int
    manifest_count: int
    manifest_signature: str
    release_approval_url: str
    release_approval_document_sha256: str
    export_metadata_url: str
    export_metadata_sha256: str
    export_start: str
    export_end: str
    export_tool_name: str
    export_tool_version: str
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
    response_etag: str
    response_checksum_crc32: str | None


class SoftwareHeritageOriginBulkLoader:
    """Stream every byte-valued origin in one frozen ORC object to Parquet.

    S3 multipart ETags and composite CRC32 values are preserved as continuity
    evidence but are not presented as cryptographic whole-object hashes. MODELOME
    computes SHA-256 over the downloaded bytes before parsing the ORC object.
    """

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        source: str = "software-heritage",
        dataset: str = _DATASET,
        bucket_url: str = _BUCKET_URL,
        graph_prefix: str = _GRAPH_PREFIX,
        default_max_new_shards: int = SOFTWARE_HERITAGE_DEFAULT_NEW_SHARD_BUDGET,
        limits: SoftwareHeritageOriginBulkLimits | None = None,
        transport: SoftwareHeritageBulkTransport | None = None,
    ) -> None:
        if not isinstance(landing_zone, ParquetLandingZone):
            raise TypeError("landing_zone must be a ParquetLandingZone")
        self.landing_zone = landing_zone
        self.source = _required_text(source, "source")
        self.dataset = _required_text(dataset, "dataset")
        self.bucket_url = _https_directory_url(bucket_url, "bucket URL")
        self.graph_prefix = _safe_prefix(graph_prefix)
        self.default_max_new_shards = _positive_integer(
            default_max_new_shards, "default_max_new_shards"
        )
        self.limits = limits or SoftwareHeritageOriginBulkLimits()
        if not isinstance(self.limits, SoftwareHeritageOriginBulkLimits):
            raise TypeError("limits must be SoftwareHeritageOriginBulkLimits")
        self.transport = transport or HttpsSoftwareHeritageBulkTransport()

    def shard_budget(self, explicit: int | None = None) -> int:
        if explicit is None:
            return self.default_max_new_shards
        if isinstance(explicit, bool) or not isinstance(explicit, int):
            raise TypeError("explicit shard budget must be an integer")
        if explicit < 0:
            raise ValueError("explicit shard budget must be nonnegative")
        return explicit

    def load(self, control_record: SourceRecord) -> SoftwareHeritageOriginBulkReceipt:
        shard = _control_shard(
            control_record,
            bucket_url=self.bucket_url,
            graph_prefix=self.graph_prefix,
        )
        if shard.expected_bytes > self.limits.max_object_bytes:
            raise ValueError("Software Heritage ORC object exceeds max_object_bytes")
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
                raise ValueError("cached ORC byte count changed from its exact control")
            return _bulk_receipt(cached, shard, None)

        self.landing_zone.initialize()
        with tempfile.TemporaryDirectory(
            prefix="software-heritage-origin-",
            dir=self.landing_zone.staging_root,
        ) as temporary:
            path = Path(temporary) / "origin.orc"
            download = self._download(shard, path)
            reader = self._reader(path)
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
                expected_rows=int(reader.nrows),
                records=self._records(reader, shard, download),
                batch_rows=self.limits.parquet_batch_rows,
            )
        return _bulk_receipt(receipt, shard, download)

    def plan_shard(self, control_record: SourceRecord) -> BulkShardPlan | None:
        if control_record.raw.get("record_type") != "software_heritage_origin_orc_shard":
            return None
        shard = _control_shard(
            control_record,
            bucket_url=self.bucket_url,
            graph_prefix=self.graph_prefix,
        )
        if shard.expected_bytes > self.limits.max_object_bytes:
            raise ValueError("Software Heritage ORC object exceeds max_object_bytes")
        return BulkShardPlan(
            source=self.source,
            dataset=self.dataset,
            release=shard.release,
            shard=shard.source_record_id,
            control_sha256=shard.control_sha256,
        )

    def shard_order(self, control_record: SourceRecord) -> tuple[str, int, str]:
        shard = _control_shard(
            control_record,
            bucket_url=self.bucket_url,
            graph_prefix=self.graph_prefix,
        )
        return shard.release, shard.manifest_index, shard.source_record_id

    def _download(self, shard: _ControlShard, destination: Path) -> _Download:
        digest = hashlib.sha256()
        byte_count = 0
        headers = {
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "x-amz-checksum-mode": "ENABLED",
        }
        with self.transport.open(
            shard.stable_url,
            headers=headers,
            redirect_validator=lambda url: _validate_exact_url(
                url,
                expected=shard.stable_url,
                bucket_url=self.bucket_url,
            ),
        ) as response:
            if int(getattr(response, "status", 0)) != 200:
                raise ValueError(f"Software Heritage ORC object returned HTTP {response.status}")
            _validate_exact_url(
                str(getattr(response, "url", "")),
                expected=shard.stable_url,
                bucket_url=self.bucket_url,
            )
            response_headers = {
                str(key).casefold(): str(value) for key, value in response.headers.items()
            }
            encoding = response_headers.get("content-encoding", "").strip().casefold()
            if encoding not in {"", "identity"}:
                raise ValueError("Software Heritage ORC response must not be encoded")
            content_length = _content_length(response_headers.get("content-length"))
            if content_length is None:
                raise ValueError("Software Heritage ORC response lacks Content-Length")
            if content_length != shard.expected_bytes:
                raise ValueError("ORC Content-Length changed from its exact control")
            etag = _required_exact_text(response_headers.get("etag"), "response ETag")
            if etag != shard.object_etag:
                raise ValueError("ORC ETag changed from its exact control")
            crc32 = _optional_exact_text(response_headers.get("x-amz-checksum-crc32"))
            with destination.open("xb") as stream:
                while True:
                    chunk = response.read(self.limits.download_chunk_bytes)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise TypeError("Software Heritage transport returned non-byte data")
                    byte_count += len(chunk)
                    if byte_count > shard.expected_bytes:
                        raise ValueError("ORC response exceeds the exact object byte count")
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if byte_count != shard.expected_bytes:
            raise ValueError("ORC response ended before the exact object byte count")
        return _Download(
            path=destination,
            sha256=digest.hexdigest(),
            byte_count=byte_count,
            response_etag=etag,
            response_checksum_crc32=crc32,
        )

    def _reader(self, path: Path) -> orc.ORCFile:
        try:
            reader = orc.ORCFile(path)
        except (OSError, pa.ArrowException) as error:
            raise ValueError("Software Heritage object is not a valid ORC file") from error
        schema = reader.schema
        if schema.names != ["url"] or not pa.types.is_binary(schema.field("url").type):
            raise ValueError("Software Heritage origin ORC schema must contain only binary url")
        if int(reader.nrows) > self.limits.max_rows:
            raise ValueError("Software Heritage origin ORC exceeds max_rows")
        if int(reader.nstripes) > self.limits.max_stripes:
            raise ValueError("Software Heritage origin ORC exceeds max_stripes")
        return reader

    def _records(
        self,
        reader: orc.ORCFile,
        shard: _ControlShard,
        download: _Download,
    ) -> Iterator[LakeRecord]:
        ordinal = 0
        for stripe_index in range(int(reader.nstripes)):
            try:
                stripe = reader.read_stripe(stripe_index, columns=["url"])
            except (OSError, pa.ArrowException) as error:
                raise ValueError("Software Heritage ORC stripe is corrupt") from error
            for offset in range(0, stripe.num_rows, self.limits.parquet_batch_rows):
                batch = stripe.slice(offset, self.limits.parquet_batch_rows)
                urls = batch.column(0)
                for scalar in urls:
                    value = scalar.as_py()
                    if value is None:
                        raise ValueError("Software Heritage origin URL must not be null")
                    if not isinstance(value, bytes):
                        raise TypeError("Software Heritage origin URL must be bytes")
                    if len(value) > self.limits.max_url_bytes:
                        raise ValueError("Software Heritage origin URL exceeds max_url_bytes")
                    digest = hashlib.sha256(value).hexdigest()
                    try:
                        text = value.decode("utf-8")
                    except UnicodeDecodeError:
                        text = None
                    locator = (
                        f"swh-export:{shard.release}:{shard.object_key}:row:{ordinal}:"
                        f"object-sha256:{download.sha256}"
                    )
                    yield LakeRecord(
                        source_record_id=(f"software-heritage:origin-url-sha256:{digest}"),
                        payload={
                            "record_type": "software_heritage_origin",
                            "origin_url": text,
                            "origin_url_base64": base64.b64encode(value).decode("ascii"),
                            "origin_url_sha256": digest,
                            "origin_identifier": {
                                "namespace": "software-heritage:origin-url-sha256",
                                "value": digest,
                            },
                            "evidence": {
                                "release": shard.release,
                                "table": "origin",
                                "object_key": shard.object_key,
                                "object_url": shard.stable_url,
                                "object_etag": shard.object_etag,
                                "object_last_modified": shard.last_modified,
                                "object_checksum_algorithms": list(shard.checksum_algorithms),
                                "object_checksum_type": shard.checksum_type,
                                "response_checksum_crc32": (download.response_checksum_crc32),
                                "object_sha256": download.sha256,
                                "manifest_signature": shard.manifest_signature,
                                "manifest_index": shard.manifest_index,
                                "manifest_count": shard.manifest_count,
                                "release_approval_url": shard.release_approval_url,
                                "release_approval_document_sha256": (
                                    shard.release_approval_document_sha256
                                ),
                                "export_metadata_url": shard.export_metadata_url,
                                "export_metadata_sha256": shard.export_metadata_sha256,
                                "export_start": shard.export_start,
                                "export_end": shard.export_end,
                                "export_tool": {
                                    "name": shard.export_tool_name,
                                    "version": shard.export_tool_version,
                                },
                                "source_row_ordinal": ordinal,
                                "locator": locator,
                            },
                        },
                    )
                    ordinal += 1
        if ordinal != int(reader.nrows):
            raise ValueError("ORC rows read differ from its declared row count")


def _control_shard(
    record: SourceRecord,
    *,
    bucket_url: str,
    graph_prefix: str,
) -> _ControlShard:
    if not isinstance(record, SourceRecord):
        raise TypeError("control_record must be a SourceRecord")
    if record.kind is not ArtifactKind.CATALOG_RECORD:
        raise ValueError("Software Heritage control must be a catalog record")
    raw = record.raw
    if not isinstance(raw, Mapping):
        raise TypeError("Software Heritage control raw value must be a mapping")
    if raw.get("record_type") != "software_heritage_origin_orc_shard":
        raise ValueError("control record is not a Software Heritage origin ORC shard")
    if raw.get("table") != "origin" or raw.get("archive_format") != "orc":
        raise ValueError("Software Heritage control has an unsupported table or format")
    if raw.get("coverage_scope") != "all_software_heritage_origins_in_export":
        raise ValueError("Software Heritage control has unknown coverage semantics")
    if raw.get("github_name_or_keyword_filter") is not False:
        raise ValueError("Software Heritage acquisition must not use a GitHub filter")
    if raw.get("historical_github_census") is not False:
        raise ValueError("Software Heritage control must not claim a GitHub census")
    if raw.get("contains_repository_content") is not False:
        raise ValueError("Software Heritage origin table must not claim code content")

    release = _release(raw.get("release"))
    object_key = _required_exact_text(raw.get("object_key"), "object_key")
    prefix = f"{graph_prefix}{release}/orc/origin/"
    filename = object_key.removeprefix(prefix)
    if (
        not object_key.startswith(prefix)
        or "/" in filename
        or _ORIGIN_FILE_RE.fullmatch(filename) is None
    ):
        raise ValueError("Software Heritage object_key is not canonical")
    stable_url = _required_exact_text(raw.get("stable_object_url"), "stable_object_url")
    expected_url = f"{bucket_url}{quote(object_key, safe='/-._~')}"
    if stable_url != expected_url or record.canonical_url != stable_url:
        raise ValueError("Software Heritage control URLs disagree")
    _validate_exact_url(stable_url, expected=expected_url, bucket_url=bucket_url)
    expected_bytes = _positive_integer(raw.get("expected_bytes"), "expected_bytes")
    object_etag = _required_exact_text(raw.get("object_etag"), "object_etag")
    last_modified = _required_exact_text(raw.get("object_last_modified"), "object_last_modified")
    checksum_raw = raw.get("object_checksum_algorithms")
    if not isinstance(checksum_raw, list) or any(
        not isinstance(value, str) or not value for value in checksum_raw
    ):
        raise ValueError("object_checksum_algorithms must be a list of strings")
    checksum_algorithms = tuple(checksum_raw)
    checksum_type = _optional_exact_text(raw.get("object_checksum_type"))
    manifest_index = _nonnegative_integer(raw.get("manifest_index"), "manifest_index")
    manifest_count = _positive_integer(raw.get("manifest_count"), "manifest_count")
    if manifest_index >= manifest_count:
        raise ValueError("manifest_index is outside manifest_count")
    manifest_signature = _sha256(raw.get("manifest_signature"), "manifest_signature")
    release_approval_url = _https_document_url(
        raw.get("release_approval_url"), "release_approval_url"
    )
    release_approval_document_sha256 = _sha256(
        raw.get("release_approval_document_sha256"),
        "release_approval_document_sha256",
    )
    export_metadata_url = _https_document_url(raw.get("export_metadata_url"), "export_metadata_url")
    expected_metadata_url = f"{bucket_url}{graph_prefix}{release}/meta/export.json"
    if export_metadata_url != expected_metadata_url:
        raise ValueError("export_metadata_url is not canonical")
    export_metadata_sha256 = _sha256(raw.get("export_metadata_sha256"), "export_metadata_sha256")
    if raw.get("export_flavor") != "full":
        raise ValueError("Software Heritage export is not full")
    export_formats = raw.get("export_formats")
    if not isinstance(export_formats, list) or "orc" not in {
        str(value).casefold() for value in export_formats
    }:
        raise ValueError("Software Heritage export does not include ORC")
    export_object_types = raw.get("export_object_types")
    if not isinstance(export_object_types, list) or "origin" not in {
        str(value).casefold() for value in export_object_types
    }:
        raise ValueError("Software Heritage export does not include origin")
    export_start = _required_exact_text(raw.get("export_start"), "export_start")
    export_end = _required_exact_text(raw.get("export_end"), "export_end")
    export_tool = raw.get("export_tool")
    if not isinstance(export_tool, Mapping) or export_tool.get("name") != "swh.export":
        raise ValueError("Software Heritage export tool identity is invalid")
    export_tool_name = "swh.export"
    export_tool_version = _required_exact_text(export_tool.get("version"), "export tool version")
    source_record_id = f"software-heritage:origin-orc:{release}:{filename}"
    if record.source_record_id != source_record_id:
        raise ValueError("Software Heritage source_record_id is inconsistent")
    control_sha256 = canonical_control_sha256(
        {
            "source_record_id": source_record_id,
            "canonical_url": stable_url,
            "release": release,
            "object_key": object_key,
            "expected_bytes": expected_bytes,
            "object_etag": object_etag,
            "manifest_index": manifest_index,
            "manifest_count": manifest_count,
            "manifest_signature": manifest_signature,
            "release_approval_document_sha256": release_approval_document_sha256,
            "export_metadata_sha256": export_metadata_sha256,
        }
    )
    return _ControlShard(
        release=release,
        object_key=object_key,
        stable_url=stable_url,
        expected_bytes=expected_bytes,
        object_etag=object_etag,
        last_modified=last_modified,
        checksum_algorithms=checksum_algorithms,
        checksum_type=checksum_type,
        manifest_index=manifest_index,
        manifest_count=manifest_count,
        manifest_signature=manifest_signature,
        release_approval_url=release_approval_url,
        release_approval_document_sha256=release_approval_document_sha256,
        export_metadata_url=export_metadata_url,
        export_metadata_sha256=export_metadata_sha256,
        export_start=export_start,
        export_end=export_end,
        export_tool_name=export_tool_name,
        export_tool_version=export_tool_version,
        source_record_id=source_record_id,
        control_sha256=control_sha256,
    )


def _bulk_receipt(
    receipt: ShardReceipt,
    shard: _ControlShard,
    download: _Download | None,
) -> SoftwareHeritageOriginBulkReceipt:
    if receipt.upstream_bytes is None:
        raise ValueError("landing receipt is missing upstream byte count")
    return SoftwareHeritageOriginBulkReceipt(
        source=receipt.source,
        dataset=receipt.dataset,
        release=receipt.release,
        shard=receipt.shard,
        manifest_index=shard.manifest_index,
        manifest_count=shard.manifest_count,
        control_sha256=receipt.control_sha256,
        upstream_sha256=receipt.upstream_sha256,
        upstream_bytes=receipt.upstream_bytes,
        row_count=receipt.row_count,
        path=receipt.path,
        already_committed=receipt.already_committed,
        response_etag=download.response_etag if download is not None else None,
        response_checksum_crc32=(
            download.response_checksum_crc32 if download is not None else None
        ),
        shard_receipt=receipt,
    )


def _validate_exact_url(url: str, *, expected: str, bucket_url: str) -> None:
    if url != expected or not url.startswith(bucket_url):
        raise ValueError("Software Heritage transfer URL changed from exact control")
    parts = urlsplit(url)
    base = urlsplit(bucket_url)
    if (
        parts.scheme.casefold() != "https"
        or (parts.hostname or "").casefold() != (base.hostname or "").casefold()
        or parts.port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Software Heritage transfer URL is unsafe")
    decoded = unquote(parts.path)
    if "\\" in decoded or "\x00" in decoded:
        raise ValueError("Software Heritage transfer URL has an unsafe path")
    if any(segment in {"", ".", ".."} for segment in decoded.split("/")[1:]):
        raise ValueError("Software Heritage transfer URL contains path traversal")


def _https_document_url(value: Any, label: str) -> str:
    text = _required_exact_text(value, label)
    parts = urlsplit(text)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"{label} must be a credential-free HTTPS URL")
    return text


def _require_public_https(url: str) -> None:
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").casefold().rstrip(".")
        if (
            parts.scheme.casefold() != "https"
            or not host
            or parts.username is not None
            or parts.password is not None
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
        raise ValueError("Software Heritage URL is not public HTTPS") from None


def _content_length(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    if not value.isascii() or not value.isdecimal():
        raise ValueError("HTTP Content-Length must contain decimal digits")
    return int(value)


def _release(value: Any) -> str:
    text = _required_exact_text(value, "release")
    if _RELEASE_RE.fullmatch(text) is None:
        raise ValueError("release must be an ISO calendar date")
    try:
        date.fromisoformat(text)
    except ValueError as error:
        raise ValueError("release must be an ISO calendar date") from error
    return text


def _sha256(value: Any, label: str) -> str:
    text = _required_exact_text(value, label).casefold()
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _safe_prefix(value: Any) -> str:
    prefix = _required_exact_text(value, "graph prefix")
    if (
        not prefix.endswith("/")
        or prefix.startswith("/")
        or "//" in prefix
        or any(segment in {"", ".", ".."} for segment in prefix[:-1].split("/"))
        or not prefix.isascii()
    ):
        raise ValueError("graph prefix must be a safe relative directory")
    return prefix


def _https_directory_url(value: Any, label: str) -> str:
    text = _required_exact_text(value, label)
    parts = urlsplit(text)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"{label} must be an HTTPS directory URL")
    path = parts.path or "/"
    if not path.endswith("/"):
        path += "/"
    return urlunsplit(("https", parts.netloc.casefold(), path, "", ""))


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _required_exact_text(value: Any, label: str) -> str:
    text = _required_text(value, label)
    if text != value:
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return text


def _optional_exact_text(value: Any) -> str | None:
    if value is None:
        return None
    return _required_exact_text(value, "optional text")


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value
