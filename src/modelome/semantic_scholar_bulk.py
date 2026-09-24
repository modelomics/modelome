from __future__ import annotations

import gzip
import hashlib
import ipaddress
import json
import os
import socket
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from modelome.bulk import BulkShardPlan
from modelome.http import HttpClient, HttpResponse
from modelome.lake import (
    LakeRecord,
    ParquetLandingZone,
    ShardApplicationOrder,
    ShardReceipt,
    canonical_control_sha256,
)
from modelome.models import SourceRecord
from modelome.sources.semantic_scholar import SemanticScholarDatasetSourceAdapter

_DEFAULT_DATASETS = ("papers", "abstracts", "paper-ids")
_SUPPORTED_DATASETS = (*_DEFAULT_DATASETS, "citations")
_SHA1 = frozenset("0123456789abcdef")


class SemanticScholarBulkTransport(Protocol):
    """Bounded metadata requests plus streaming object transfer."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse: ...

    def iter_bytes(self, url: str, *, chunk_size: int) -> Iterator[bytes]: ...


@dataclass(frozen=True, slots=True)
class BulkLoadLimits:
    """Hard ceilings for one compressed Semantic Scholar shard."""

    max_compressed_bytes: int = 8 * 1024 * 1024 * 1024
    max_uncompressed_bytes: int = 64 * 1024 * 1024 * 1024
    max_line_bytes: int = 16 * 1024 * 1024
    max_records: int = 50_000_000
    network_chunk_bytes: int = 1024 * 1024
    decompression_chunk_bytes: int = 1024 * 1024
    parquet_batch_rows: int = 16

    def __post_init__(self) -> None:
        for field in (
            "max_compressed_bytes",
            "max_uncompressed_bytes",
            "max_line_bytes",
            "max_records",
            "network_chunk_bytes",
            "decompression_chunk_bytes",
            "parquet_batch_rows",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.network_chunk_bytes > self.max_compressed_bytes:
            raise ValueError("network_chunk_bytes must not exceed max_compressed_bytes")
        if self.decompression_chunk_bytes > self.max_uncompressed_bytes:
            raise ValueError(
                "decompression_chunk_bytes must not exceed max_uncompressed_bytes"
            )


class UrllibSemanticScholarBulkTransport:
    """Production transport with bounded manifests and public-HTTPS redirects."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        max_manifest_bytes: int = 32 * 1024 * 1024,
        user_agent: str | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("transport timeout must be positive")
        if max_manifest_bytes < 1:
            raise ValueError("max_manifest_bytes must be positive")
        self.timeout = float(timeout)
        self.user_agent = user_agent or "modelome/0.1"
        self._manifest_client = HttpClient(
            timeout=self.timeout,
            max_response_bytes=max_manifest_bytes,
            user_agent=self.user_agent,
        )
        self._download_opener = build_opener(_PublicHttpsRedirectHandler())

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        _require_public_https_url(url)
        origin = _origin(url)

        def validate_redirect(target: str) -> None:
            _require_public_https_url(target)
            if _origin(target) != origin:
                raise ValueError("Semantic Scholar manifest redirect changed origin")

        return self._manifest_client.get(
            url,
            headers=headers,
            redirect_validator=validate_redirect,
        )

    def iter_bytes(self, url: str, *, chunk_size: int) -> Iterator[bytes]:
        if chunk_size < 1:
            raise ValueError("download chunk_size must be positive")
        _require_public_https_url(url)
        request = Request(
            url,
            headers={
                "Accept": "application/octet-stream",
                "User-Agent": self.user_agent,
            },
            method="GET",
        )
        try:
            with self._download_opener.open(request, timeout=self.timeout) as response:  # noqa: S310
                _require_public_https_url(response.url)
                while block := response.read(chunk_size):
                    yield block
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            raise RuntimeError(
                f"Semantic Scholar shard transfer failed ({type(error).__name__})"
            ) from None


class SemanticScholarBulkLoader:
    """Resolve, verify, decode, and atomically land one S2AG shard."""

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        api_key: str | None = None,
        url: str = "https://api.semanticscholar.org/datasets/v1",
        source: str = "semantic-scholar",
        limits: BulkLoadLimits | None = None,
        transport: SemanticScholarBulkTransport | None = None,
        control_adapter: SemanticScholarDatasetSourceAdapter | None = None,
        temp_root: str | Path | None = None,
    ) -> None:
        if not isinstance(landing_zone, ParquetLandingZone):
            raise TypeError("landing_zone must be a ParquetLandingZone")
        self.landing_zone = landing_zone
        self.source = _required_text(source, "source")
        self.limits = limits or BulkLoadLimits()
        self.transport = transport or UrllibSemanticScholarBulkTransport()
        if control_adapter is not None:
            if not isinstance(control_adapter, SemanticScholarDatasetSourceAdapter):
                raise TypeError(
                    "control_adapter must be a SemanticScholarDatasetSourceAdapter"
                )
            unsupported = sorted(set(control_adapter.datasets) - set(_SUPPORTED_DATASETS))
            if unsupported:
                raise ValueError(
                    "unsupported Semantic Scholar bulk datasets: "
                    + ", ".join(unsupported)
                )
            self.control = control_adapter
        else:
            self.control = SemanticScholarDatasetSourceAdapter(
                name=f"{self.source}-bulk-control",
                url=url,
                datasets=_DEFAULT_DATASETS,
                api_key=api_key,
                client=self.transport,
            )
        self._temp_root = _prepare_temp_root(temp_root)

    def load_shard(self, control_record: SourceRecord) -> ShardReceipt:
        """Load one control record without retaining its pre-signed URL."""

        descriptor = _control_descriptor(control_record)
        cached = self.landing_zone.lookup_committed_shard(
            source=self.source,
            dataset=descriptor.dataset,
            release=descriptor.target_release,
            shard=control_record.source_record_id,
            control_sha256=descriptor.control_sha256,
            upstream_url=descriptor.stable_object_url,
            application_order=descriptor.application_order,
        )
        if cached is not None:
            return cached
        temporary_url = self.control.resolve_download_url(control_record)
        _validate_resolved_download(
            temporary_url,
            expected_stable_url=descriptor.stable_object_url,
        )

        with tempfile.TemporaryDirectory(
            prefix="modelome-s2-shard-",
            dir=self._temp_root,
        ) as temporary_directory:
            compressed_path = Path(temporary_directory) / "upstream.jsonl.gz"
            download = self._download(temporary_url, compressed_path)
            records = _iter_lake_records(
                compressed_path,
                dataset=descriptor.dataset,
                operation=descriptor.lake_operation,
                limits=self.limits,
            )
            return self.landing_zone.commit_shard(
                source=self.source,
                dataset=descriptor.dataset,
                release=descriptor.target_release,
                shard=control_record.source_record_id,
                control_sha256=descriptor.control_sha256,
                upstream_sha256=download.sha256,
                records=records,
                upstream_url=descriptor.stable_object_url,
                upstream_bytes=download.byte_count,
                application_order=descriptor.application_order,
                batch_rows=self.limits.parquet_batch_rows,
            )

    def plan_shard(self, control_record: SourceRecord) -> BulkShardPlan | None:
        """Describe one downloadable control without resolving or transferring it."""

        if control_record.raw.get("record_type") != "dataset_shard":
            return None
        descriptor = _control_descriptor(control_record)
        return BulkShardPlan(
            source=self.source,
            dataset=descriptor.dataset,
            release=descriptor.target_release,
            shard=control_record.source_record_id,
            control_sha256=descriptor.control_sha256,
        )

    def shard_order(self, control_record: SourceRecord) -> tuple[Any, ...]:
        """Return the protocol-defined replay order for one selected control."""

        descriptor = _control_descriptor(control_record)
        order = descriptor.application_order
        if order.mode == "snapshot":
            return (
                descriptor.target_release,
                descriptor.dataset,
                0,
                order.manifest_index,
            )
        return (
            descriptor.target_release,
            descriptor.dataset,
            1,
            order.diff_index,
            0 if order.operation == "upsert" else 1,
            order.operation_index,
        )

    def _download(self, temporary_url: str, destination: Path) -> _Download:
        digest = hashlib.sha256()
        byte_count = 0
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                for block in self.transport.iter_bytes(
                    temporary_url,
                    chunk_size=self.limits.network_chunk_bytes,
                ):
                    if not isinstance(block, bytes):
                        raise TypeError("bulk transport must yield bytes")
                    if not block:
                        continue
                    byte_count += len(block)
                    if byte_count > self.limits.max_compressed_bytes:
                        raise ValueError(
                            "Semantic Scholar shard exceeded max_compressed_bytes"
                        )
                    digest.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if byte_count == 0:
            raise ValueError("Semantic Scholar shard download was empty")
        return _Download(sha256=digest.hexdigest(), byte_count=byte_count)


@dataclass(frozen=True, slots=True)
class _Download:
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class _ControlDescriptor:
    dataset: str
    target_release: str
    stable_object_url: str
    lake_operation: str
    application_order: ShardApplicationOrder
    control_sha256: str


def _control_descriptor(record: SourceRecord) -> _ControlDescriptor:
    if not isinstance(record, SourceRecord):
        raise TypeError("control_record must be a SourceRecord")
    raw = record.raw
    if not isinstance(raw, Mapping):
        raise TypeError("control_record.raw must be a mapping")
    if raw.get("record_type") != "dataset_shard":
        raise ValueError("control record is not a Semantic Scholar dataset shard")
    dataset = _required_text(raw.get("dataset"), "control dataset")
    if dataset not in _SUPPORTED_DATASETS:
        raise ValueError(f"unsupported Semantic Scholar bulk dataset: {dataset}")
    target_release = _iso_date(raw.get("target_release"), "target_release")
    stable_object_url = _stable_https_url(
        _required_text(raw.get("stable_object_url"), "stable_object_url")
    )
    if record.canonical_url != stable_object_url:
        raise ValueError("control record canonical URL does not match stable object URL")
    operation = _required_text(raw.get("operation"), "control operation")
    _nonnegative_integer(raw.get("manifest_index"), "manifest_index")
    if operation == "snapshot":
        lake_operation = "upsert"
        application_order = ShardApplicationOrder.snapshot(
            _nonnegative_integer(raw.get("manifest_index"), "manifest_index")
        )
    elif operation in {"upsert", "delete"}:
        lake_operation = operation
        from_release = _iso_date(raw.get("from_release"), "from_release")
        to_release = _iso_date(raw.get("to_release"), "to_release")
        application_order = ShardApplicationOrder.diff(
            diff_index=_nonnegative_integer(raw.get("diff_index"), "diff_index"),
            operation=operation,
            operation_index=_nonnegative_integer(
                raw.get("operation_index"), "operation_index"
            ),
            from_release=from_release,
            to_release=to_release,
        )
    else:
        raise ValueError("control record has an unsupported operation")
    return _ControlDescriptor(
        dataset=dataset,
        target_release=target_release,
        stable_object_url=stable_object_url,
        lake_operation=lake_operation,
        application_order=application_order,
        control_sha256=canonical_control_sha256(
            {
                "canonical_url": record.canonical_url,
                "raw": dict(raw),
                "source_record_id": record.source_record_id,
            }
        ),
    )


def _iter_lake_records(
    path: Path,
    *,
    dataset: str,
    operation: str,
    limits: BulkLoadLimits,
) -> Iterator[LakeRecord]:
    uncompressed_bytes = 0
    record_count = 0
    line_number = 0
    buffer = bytearray()
    try:
        with path.open("rb") as compressed, gzip.GzipFile(fileobj=compressed) as stream:
            while block := stream.read1(limits.decompression_chunk_bytes):
                uncompressed_bytes += len(block)
                if uncompressed_bytes > limits.max_uncompressed_bytes:
                    raise ValueError(
                        "Semantic Scholar shard exceeded max_uncompressed_bytes"
                    )
                buffer.extend(block)
                while (newline := buffer.find(b"\n")) >= 0:
                    line = bytes(buffer[:newline])
                    del buffer[: newline + 1]
                    line_number += 1
                    record_count += 1
                    if record_count > limits.max_records:
                        raise ValueError("Semantic Scholar shard exceeded max_records")
                    yield _lake_record(
                        line,
                        dataset=dataset,
                        operation=operation,
                        line_number=line_number,
                        max_line_bytes=limits.max_line_bytes,
                    )
                if len(buffer) > limits.max_line_bytes:
                    raise ValueError(
                        f"Semantic Scholar JSONL line {line_number + 1} "
                        "exceeded max_line_bytes"
                    )
            if buffer:
                line_number += 1
                record_count += 1
                if record_count > limits.max_records:
                    raise ValueError("Semantic Scholar shard exceeded max_records")
                yield _lake_record(
                    bytes(buffer),
                    dataset=dataset,
                    operation=operation,
                    line_number=line_number,
                    max_line_bytes=limits.max_line_bytes,
                )
            if record_count == 0:
                raise ValueError("Semantic Scholar shard contained no JSONL records")
    except (gzip.BadGzipFile, EOFError) as error:
        raise ValueError("Semantic Scholar shard is not valid gzip data") from error


def _lake_record(
    line: bytes,
    *,
    dataset: str,
    operation: str,
    line_number: int,
    max_line_bytes: int,
) -> LakeRecord:
    if line.endswith(b"\r"):
        line = line[:-1]
    if not line:
        raise ValueError(f"Semantic Scholar JSONL line {line_number} is empty")
    if len(line) > max_line_bytes:
        raise ValueError(
            f"Semantic Scholar JSONL line {line_number} exceeded max_line_bytes"
        )
    try:
        payload = json.loads(line, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError):
        raise ValueError(
            f"Semantic Scholar JSONL line {line_number} is not valid JSON"
        ) from None
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Semantic Scholar JSONL line {line_number} must contain a JSON object"
        )
    source_record_id = _row_identity(dataset, payload, line_number)
    return LakeRecord(
        source_record_id=source_record_id,
        payload=dict(payload),
        operation=operation,
    )


def _row_identity(dataset: str, payload: Mapping[str, Any], line_number: int) -> str:
    if dataset == "citations":
        citing_paper_id, cited_paper_id = citation_edge_ids(payload)
        return f"semantic-scholar:citation:{citing_paper_id}:{cited_paper_id}"

    corpus_id = payload.get("corpusid")
    if isinstance(corpus_id, bool):
        corpus_id = None
    if isinstance(corpus_id, int):
        normalized_corpus_id = str(corpus_id) if corpus_id > 0 else ""
    elif isinstance(corpus_id, str) and corpus_id.isascii() and corpus_id.isdigit():
        normalized_corpus_id = str(int(corpus_id)) if int(corpus_id) > 0 else ""
    else:
        normalized_corpus_id = ""
    if not normalized_corpus_id:
        raise ValueError(
            f"Semantic Scholar {dataset} line {line_number} has no valid corpusid"
        )
    if dataset == "paper-ids":
        sha = payload.get("sha")
        if not isinstance(sha, str):
            raise ValueError(
                f"Semantic Scholar paper-ids line {line_number} has no valid sha"
            )
        normalized_sha = sha.strip().casefold()
        if len(normalized_sha) != 40 or not set(normalized_sha) <= _SHA1:
            raise ValueError(
                f"Semantic Scholar paper-ids line {line_number} has no valid sha"
            )
        return f"semantic-scholar:paper-id:{normalized_sha}"
    return f"semantic-scholar:corpus:{normalized_corpus_id}"


def citation_edge_ids(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Extract the exact S2 corpus IDs at both ends of one citation edge.

    The Datasets API's ``citations`` records refer to papers by
    ``citingPaperId`` and ``citedPaperId``. Both are corpus IDs. Canonical
    decimal strings preserve those IDs without attempting title-based joins.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("citation row must be a mapping")
    return (
        _citation_corpus_id(payload.get("citingPaperId"), "citingPaperId"),
        _citation_corpus_id(payload.get("citedPaperId"), "citedPaperId"),
    )


def _citation_corpus_id(value: Any, field: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"citation {field} must be a positive corpus ID")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        parsed = int(value)
    else:
        parsed = 0
    if parsed < 1:
        raise ValueError(f"citation {field} must be a positive corpus ID")
    return str(parsed)


def _validate_resolved_download(url: str, *, expected_stable_url: str) -> None:
    stable = _stable_https_url(url)
    if stable != expected_stable_url:
        raise ValueError("resolved shard URL does not match its stable object selector")


def _stable_https_url(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise ValueError("shard URL must be credential-free public HTTPS")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("shard URL has an invalid port") from None
    if port not in {None, 443}:
        raise ValueError("shard URL must use the default HTTPS port")
    host = parts.hostname.casefold().rstrip(".")
    if not host or host == "localhost" or host.endswith(".localhost"):
        raise ValueError("shard URL host is not public")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        stable_host = host
    else:
        if not address.is_global:
            raise ValueError("shard URL host is not public")
        stable_host = f"[{host}]" if address.version == 6 else host
    if not parts.path or parts.path == "/":
        raise ValueError("shard URL is missing an object path")
    return urlunsplit(("https", stable_host, parts.path, "", ""))


def _require_public_https_url(url: str) -> None:
    stable = _stable_https_url(url)
    host = urlsplit(stable).hostname
    if host is None:
        raise ValueError("shard URL host is not public")
    try:
        addresses = (ipaddress.ip_address(host),)
    except ValueError:
        try:
            addresses = tuple(
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            )
        except OSError:
            raise ValueError("shard URL host could not be resolved safely") from None
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("shard URL host is not public")


def _origin(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    return (
        parts.scheme.casefold(),
        (parts.hostname or "").casefold().rstrip("."),
        parts.port or 443,
    )


class _PublicHttpsRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> Request | None:
        _require_public_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _prepare_temp_root(value: str | Path | None) -> str | None:
    if value is None:
        return None
    raw = Path(value).expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"temporary root must not be a symlink: {raw}")
    raw.mkdir(parents=True, exist_ok=True)
    if not raw.is_dir():
        raise ValueError(f"temporary root is not a directory: {raw}")
    return str(raw.resolve())


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _iso_date(value: Any, field: str) -> str:
    text = _required_text(value, field)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{field} must be an ISO calendar date") from None
    if parsed.isoformat() != text:
        raise ValueError(f"{field} must use YYYY-MM-DD format")
    return text


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value.strip()


__all__ = [
    "BulkLoadLimits",
    "SemanticScholarBulkLoader",
    "SemanticScholarBulkTransport",
    "UrllibSemanticScholarBulkTransport",
]
