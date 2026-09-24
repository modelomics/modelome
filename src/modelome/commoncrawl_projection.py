from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.extract import Extractor, IntroductionCueExtractor
from modelome.lake import (
    ParquetLandingZone,
    ReleaseReceipt,
    ShardReceipt,
    canonical_control_sha256,
)
from modelome.models import ArtifactKind, Identifier, ModelHint, ModelStatus, SourceRecord
from modelome.normalize import (
    extract_url_mentions,
    identifier_from_url,
    infer_url_relation,
    normalize_name,
)

_FORMAT = "modelome-commoncrawl-wet-discovery-projection-v1"
_ALGORITHM = "sealed-wet-evidence-stream-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_SOURCE_RECORD_ID = re.compile(r"^commoncrawl:wet-record:([0-9a-f]{64})$")
_MODEL_LOCATOR = re.compile(r"^(title|text):(\d+):(\d+)$")
_URL_LOCATOR = re.compile(r"^text:(\d+)-(\d+)$")
_INPUT_COLUMNS = (
    "source_record_id",
    "operation",
    "payload_json",
    "content_sha256",
)

DOCUMENT_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("release", pa.string(), nullable=False),
        pa.field("source_url", pa.large_string(), nullable=False),
        pa.field("source_identifier_namespace", pa.string()),
        pa.field("source_identifier_value", pa.large_string()),
        pa.field("source_repository_url", pa.large_string()),
        pa.field("crawled_at", pa.string(), nullable=False),
        pa.field("content_type", pa.string(), nullable=False),
        pa.field("content_bytes", pa.int64(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("warc_record_id", pa.large_string(), nullable=False),
        pa.field("crawl_object_url", pa.large_string(), nullable=False),
        pa.field("crawl_path", pa.large_string(), nullable=False),
        pa.field("crawl_manifest_index", pa.int64(), nullable=False),
        pa.field("crawl_record_index", pa.int64(), nullable=False),
        pa.field("crawl_conversion_index", pa.int64(), nullable=False),
        pa.field("crawl_locator_json", pa.large_string(), nullable=False),
        pa.field("landing_payload_sha256", pa.string(), nullable=False),
        pa.field("source_input_kind", pa.string(), nullable=False),
        pa.field("source_input_sha256", pa.string(), nullable=False),
        pa.field("source_shard", pa.large_string()),
        pa.field("source_row_ordinal", pa.int64(), nullable=False),
        pa.field("projection_artifact_id", pa.string(), nullable=False),
    ]
)

URL_RELATION_SCHEMA = pa.schema(
    [
        pa.field("relation_assertion_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("release", pa.string(), nullable=False),
        pa.field("source_url", pa.large_string(), nullable=False),
        pa.field("predicate", pa.string(), nullable=False),
        pa.field("target_url", pa.large_string(), nullable=False),
        pa.field("target_identifier_namespace", pa.string()),
        pa.field("target_identifier_value", pa.large_string()),
        pa.field("target_repository_url", pa.large_string()),
        pa.field("locator", pa.string(), nullable=False),
        pa.field("evidence_start", pa.int64(), nullable=False),
        pa.field("evidence_end", pa.int64(), nullable=False),
        pa.field("evidence_text", pa.large_string(), nullable=False),
        pa.field("context_start", pa.int64(), nullable=False),
        pa.field("context_end", pa.int64(), nullable=False),
        pa.field("context_text", pa.large_string(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("crawl_locator_json", pa.large_string(), nullable=False),
        pa.field("source_input_kind", pa.string(), nullable=False),
        pa.field("source_input_sha256", pa.string(), nullable=False),
        pa.field("source_shard", pa.large_string()),
        pa.field("source_row_ordinal", pa.int64(), nullable=False),
        pa.field("projection_artifact_id", pa.string(), nullable=False),
    ]
)

MODEL_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("candidate_assertion_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("release", pa.string(), nullable=False),
        pa.field("source_url", pa.large_string(), nullable=False),
        pa.field("source_identifier_namespace", pa.string()),
        pa.field("source_identifier_value", pa.large_string()),
        pa.field("source_repository_url", pa.large_string()),
        pa.field("name", pa.large_string(), nullable=False),
        pa.field("normalized_name", pa.large_string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("extractor", pa.string(), nullable=False),
        pa.field("extractor_local_id", pa.string(), nullable=False),
        pa.field("locator", pa.string(), nullable=False),
        pa.field("evidence_field", pa.string(), nullable=False),
        pa.field("evidence_start", pa.int64(), nullable=False),
        pa.field("evidence_end", pa.int64(), nullable=False),
        pa.field("evidence_text", pa.large_string(), nullable=False),
        pa.field("context_start", pa.int64(), nullable=False),
        pa.field("context_end", pa.int64(), nullable=False),
        pa.field("context_text", pa.large_string(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("crawl_locator_json", pa.large_string(), nullable=False),
        pa.field("source_input_kind", pa.string(), nullable=False),
        pa.field("source_input_sha256", pa.string(), nullable=False),
        pa.field("source_shard", pa.large_string()),
        pa.field("source_row_ordinal", pa.int64(), nullable=False),
        pa.field("projection_artifact_id", pa.string(), nullable=False),
        pa.field("derivation_json", pa.large_string(), nullable=False),
    ]
)

_TABLE_SCHEMAS = {
    "documents": DOCUMENT_SCHEMA,
    "model_candidates": MODEL_CANDIDATE_SCHEMA,
    "url_relations": URL_RELATION_SCHEMA,
}


@dataclass(frozen=True, slots=True)
class CommonCrawlDiscoveryLimits:
    """Hard memory and cardinality bounds for one WET release projection."""

    input_batch_rows: int = 8
    max_input_batch_bytes: int = 160 * 1024 * 1024
    max_document_text_bytes: int = 16 * 1024 * 1024
    max_url_mentions_per_document: int = 100_000
    max_candidates_per_document: int = 1_024
    evidence_context_chars: int = 4_096
    output_part_rows: int = 2_048
    max_output_buffer_bytes: int = 128 * 1024 * 1024
    max_output_row_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_output_row_bytes > self.max_output_buffer_bytes:
            raise ValueError(
                "max_output_row_bytes must not exceed max_output_buffer_bytes"
            )


@dataclass(frozen=True, slots=True)
class CommonCrawlDiscoveryReceipt:
    source: str
    dataset: str
    release: str
    artifact_id: str
    source_input_kind: str
    source_input_sha256: str
    source_shard: str | None
    document_count: int
    url_relation_count: int
    candidate_count: int
    table_part_counts: tuple[tuple[str, int], ...]
    path: Path
    already_materialized: bool = False


@dataclass(frozen=True, slots=True)
class _Document:
    row: dict[str, Any]
    source_record: SourceRecord
    source_identifier: Identifier | None


@dataclass(frozen=True, slots=True)
class _SourceInput:
    kind: str
    receipt: ReleaseReceipt | ShardReceipt
    sha256: str

    @property
    def source(self) -> str:
        return self.receipt.source

    @property
    def dataset(self) -> str:
        return self.receipt.dataset

    @property
    def release(self) -> str:
        return self.receipt.release

    @property
    def row_count(self) -> int:
        return self.receipt.row_count

    @property
    def shard(self) -> str | None:
        return self.receipt.shard if isinstance(self.receipt, ShardReceipt) else None


class _TableSink:
    def __init__(
        self,
        root: Path,
        table_name: str,
        schema: pa.Schema,
        limits: CommonCrawlDiscoveryLimits,
    ) -> None:
        self.root = root
        self.table_name = table_name
        self.schema = schema
        self.limits = limits
        self.root.mkdir()
        self.rows: list[dict[str, Any]] = []
        self.buffer_bytes = 0
        self.parts: list[dict[str, Any]] = []
        self.row_count = 0

    def add(self, row: dict[str, Any]) -> None:
        row_bytes = _row_bytes(row)
        if row_bytes > self.limits.max_output_row_bytes:
            raise ValueError(
                f"{self.table_name} row exceeded max_output_row_bytes"
            )
        if self.rows and (
            len(self.rows) >= self.limits.output_part_rows
            or self.buffer_bytes + row_bytes > self.limits.max_output_buffer_bytes
        ):
            self.flush()
        self.rows.append(row)
        self.buffer_bytes += row_bytes
        self.row_count += 1

    def flush(self) -> None:
        if not self.rows:
            return
        index = len(self.parts)
        name = f"part-{index:06d}.parquet"
        path = self.root / name
        pq.write_table(
            pa.Table.from_pylist(self.rows, schema=self.schema),
            path,
            compression="zstd",
            write_statistics=True,
        )
        _fsync_file(path)
        self.parts.append(
            {
                "index": index,
                "name": name,
                "path": f"{self.table_name}/{name}",
                "row_count": len(self.rows),
                "byte_count": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
        self.rows = []
        self.buffer_bytes = 0


class CommonCrawlWetDiscoveryProjector:
    """Turn a sealed Common Crawl WET release into actionable evidence tables.

    Every landed conversion document is retained. URL and model claims are
    separate, source-backed assertions; neither table filters the input corpus by
    host, venue, discipline, or a vocabulary of known model names.
    """

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        source: str = "commoncrawl",
        dataset: str = "wet",
        output_root: str | Path | None = None,
        extractor: Extractor | None = None,
        limits: CommonCrawlDiscoveryLimits | None = None,
    ) -> None:
        if not isinstance(landing_zone, ParquetLandingZone):
            raise TypeError("landing_zone must be a ParquetLandingZone")
        self.landing_zone = landing_zone
        self.source = _required_text(source, "source")
        self.dataset = _required_text(dataset, "dataset")
        self.extractor = extractor or IntroductionCueExtractor()
        self.extractor_name = _required_text(
            getattr(self.extractor, "name", None),
            "extractor name",
        )
        if not callable(getattr(self.extractor, "extract", None)):
            raise TypeError("extractor must define extract(record)")
        self.limits = limits or CommonCrawlDiscoveryLimits()
        self.output_root = (
            landing_zone.root / "projections" / "commoncrawl-wet-discovery"
            if output_root is None
            else Path(output_root).expanduser().absolute()
        )

    def materialize(
        self,
        source_receipt: ReleaseReceipt | ShardReceipt,
    ) -> CommonCrawlDiscoveryReceipt:
        source = self._verify_source_receipt(source_receipt)
        self._initialize_output()
        artifact_id = self._artifact_id(source)
        final_dir = self._final_path(source.release, artifact_id)
        if final_dir.exists() or final_dir.is_symlink():
            manifest = self._verify_projection(
                final_dir,
                source_input=source,
                artifact_id=artifact_id,
            )
            return _receipt(final_dir, manifest, already_materialized=True)

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        _require_plain_directory(final_dir.parent.parent, "projection source directory")
        _require_plain_directory(final_dir.parent, "projection release directory")
        stage = Path(
            tempfile.mkdtemp(
                prefix="commoncrawl-discovery-",
                dir=self.output_root / ".staging",
            )
        )
        try:
            sinks = {
                name: _TableSink(stage / name, name, schema, self.limits)
                for name, schema in _TABLE_SCHEMAS.items()
            }
            stats = self._project(
                source,
                artifact_id=artifact_id,
                sinks=sinks,
            )
            reverified_source = self._verify_source_receipt(source.receipt)
            if _source_identity(reverified_source) != _source_identity(source):
                raise ValueError("source input changed during discovery projection")
            for sink in sinks.values():
                sink.flush()
            created_at = datetime.now(UTC).isoformat()
            tables = {
                name: {
                    "schema": str(sink.schema),
                    "row_count": sink.row_count,
                    "part_count": len(sink.parts),
                    "parts": sink.parts,
                }
                for name, sink in sorted(sinks.items())
            }
            manifest: dict[str, Any] = {
                "format": _FORMAT,
                "algorithm": _ALGORITHM,
                "artifact_id": artifact_id,
                "source_input": _source_identity(source),
                "extractor": {"name": self.extractor_name},
                "layout": self._layout(),
                "stats": stats,
                "tables": tables,
                "created_at": created_at,
            }
            manifest["manifest_sha256"] = _json_sha256(manifest)
            _write_json_exclusive(stage / "manifest.json", manifest)
            seal: dict[str, Any] = {
                "format": _FORMAT,
                "artifact_id": artifact_id,
                "manifest_sha256": manifest["manifest_sha256"],
                "data_sha256": _json_sha256({"tables": tables}),
                "sealed_at": created_at,
            }
            seal["seal_sha256"] = _json_sha256(seal)
            _write_json_exclusive(stage / "SEAL.json", seal)
            _fsync_tree(stage)
            try:
                os.replace(stage, final_dir)
            except OSError:
                if not final_dir.exists():
                    raise
                manifest = self._verify_projection(
                    final_dir,
                    source_input=source,
                    artifact_id=artifact_id,
                )
                return _receipt(final_dir, manifest, already_materialized=True)
            _fsync_directory(final_dir.parent)
            manifest = self._verify_projection(
                final_dir,
                source_input=source,
                artifact_id=artifact_id,
            )
            return _receipt(final_dir, manifest, already_materialized=False)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def open_projection(
        self,
        source_receipt: ReleaseReceipt | ShardReceipt,
    ) -> CommonCrawlDiscoveryReceipt:
        source = self._verify_source_receipt(source_receipt)
        self._initialize_output()
        artifact_id = self._artifact_id(source)
        path = self._final_path(source.release, artifact_id)
        manifest = self._verify_projection(
            path,
            source_input=source,
            artifact_id=artifact_id,
        )
        return _receipt(path, manifest, already_materialized=True)

    def iter_batches(
        self,
        receipt: CommonCrawlDiscoveryReceipt,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        if not isinstance(receipt, CommonCrawlDiscoveryReceipt):
            raise TypeError("receipt must be a CommonCrawlDiscoveryReceipt")
        if table not in _TABLE_SCHEMAS:
            raise ValueError(f"unknown discovery table {table!r}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        manifest = self._verify_projection(receipt.path, artifact_id=receipt.artifact_id)
        expected = _receipt(receipt.path, manifest, already_materialized=True)
        if _receipt_identity(receipt) != _receipt_identity(expected):
            raise ValueError("discovery projection receipt does not match its artifact")
        for part in manifest["tables"][table]["parts"]:
            yield from pq.ParquetFile(receipt.path / str(part["path"])).iter_batches(
                columns=columns,
                batch_size=batch_size,
            )

    def _verify_source_receipt(
        self,
        receipt: ReleaseReceipt | ShardReceipt,
    ) -> _SourceInput:
        if not isinstance(receipt, (ReleaseReceipt, ShardReceipt)):
            raise TypeError("source_receipt must be a ReleaseReceipt or ShardReceipt")
        if receipt.source != self.source or receipt.dataset != self.dataset:
            raise ValueError("source receipt does not identify this projector's dataset")
        if isinstance(receipt, ShardReceipt):
            verified_shard = self.landing_zone.lookup_committed_shard(
                source=receipt.source,
                dataset=receipt.dataset,
                release=receipt.release,
                shard=receipt.shard,
                control_sha256=receipt.control_sha256,
                upstream_url=receipt.upstream_url,
                application_order=receipt.application_order,
            )
            if verified_shard is None:
                raise ValueError("source shard is not committed")
            if _shard_receipt_identity(receipt) != _shard_receipt_identity(verified_shard):
                raise ValueError("source shard receipt does not match its committed shard")
            manifest = _read_json(verified_shard.path / "manifest.json", "source shard")
            input_sha256 = manifest.get("manifest_sha256")
            if not _is_sha256(input_sha256):
                raise ValueError("source shard checksum is invalid")
            return _SourceInput("shard", verified_shard, str(input_sha256))

        matches = tuple(
            item
            for item in self.landing_zone.list_releases(
                source=self.source,
                dataset=self.dataset,
                verify_shards=True,
            )
            if item.release == receipt.release
        )
        if len(matches) != 1:
            raise ValueError("source release is absent or ambiguous")
        verified = matches[0]
        if (
            receipt.source,
            receipt.dataset,
            receipt.release,
            receipt.shard_count,
            receipt.row_count,
            receipt.application_mode,
            receipt.path,
        ) != (
            verified.source,
            verified.dataset,
            verified.release,
            verified.shard_count,
            verified.row_count,
            verified.application_mode,
            verified.path,
        ):
            raise ValueError("source release receipt does not match its sealed release")
        seal = _read_json(verified.path, "source release")
        release_sha256 = seal.get("release_sha256")
        if not _is_sha256(release_sha256):
            raise ValueError("source release checksum is invalid")
        return _SourceInput("release", verified, str(release_sha256))

    def _initialize_output(self) -> None:
        if self.output_root.exists() and not self.output_root.is_dir():
            raise ValueError(f"projection root is not a directory: {self.output_root}")
        if self.output_root.is_symlink():
            raise ValueError(f"projection root must not be a symlink: {self.output_root}")
        protected = self.landing_zone.shards_root
        if (
            self.output_root == protected
            or self.output_root in protected.parents
            or protected in self.output_root.parents
        ):
            raise ValueError("projection root must be separate from landing-zone shards")
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.output_root.resolve() != self.output_root:
            raise ValueError("projection root ancestors must not redirect through symlinks")
        staging = self.output_root / ".staging"
        staging.mkdir(exist_ok=True)
        _require_plain_directory(self.output_root, "projection root")
        _require_plain_directory(staging, "projection staging directory")

    def _artifact_id(self, source: _SourceInput) -> str:
        return _json_sha256(
            {
                "format": _FORMAT,
                "algorithm": _ALGORITHM,
                "source_input": _source_identity(source),
                "extractor": self.extractor_name,
                "layout": self._layout(),
                "schemas": {
                    name: str(schema) for name, schema in sorted(_TABLE_SCHEMAS.items())
                },
            }
        )

    def _layout(self) -> dict[str, int]:
        return {
            "evidence_context_chars": self.limits.evidence_context_chars,
            "max_output_buffer_bytes": self.limits.max_output_buffer_bytes,
            "output_part_rows": self.limits.output_part_rows,
        }

    def _final_path(self, release: str, artifact_id: str) -> Path:
        return (
            self.output_root
            / _path_component(self.source)
            / _path_component(release)
            / artifact_id
        )

    def _project(
        self,
        source: _SourceInput,
        *,
        artifact_id: str,
        sinks: Mapping[str, _TableSink],
    ) -> dict[str, int]:
        input_rows = 0
        documents_with_urls = 0
        documents_with_candidates = 0
        for batch in self._iter_input_batches(source):
            if batch.nbytes > self.limits.max_input_batch_bytes:
                raise ValueError("WET input batch exceeded max_input_batch_bytes")
            if tuple(batch.schema.names) != _INPUT_COLUMNS:
                raise ValueError("WET release returned unexpected input columns")
            for values in batch.to_pylist():
                row_ordinal = input_rows
                input_rows += 1
                document = self._decode_document(
                    values,
                    source=source,
                    artifact_id=artifact_id,
                    row_ordinal=row_ordinal,
                )
                sinks["documents"].add(document.row)

                mentions = extract_url_mentions(document.source_record.text)
                if len(mentions) > self.limits.max_url_mentions_per_document:
                    raise ValueError(
                        f"source record {document.source_record.source_record_id} "
                        "exceeded max_url_mentions_per_document"
                    )
                if mentions:
                    documents_with_urls += 1
                for target_url, locator in mentions:
                    sinks["url_relations"].add(
                        self._url_relation_row(
                            document,
                            target_url=target_url,
                            locator=locator,
                        )
                    )

                hints = tuple(self.extractor.extract(document.source_record))
                if len(hints) > self.limits.max_candidates_per_document:
                    raise ValueError(
                        f"source record {document.source_record.source_record_id} "
                        "exceeded max_candidates_per_document"
                    )
                if hints:
                    documents_with_candidates += 1
                assertion_ids: set[str] = set()
                for hint in hints:
                    candidate = self._candidate_row(document, hint)
                    assertion_id = str(candidate["candidate_assertion_id"])
                    if assertion_id in assertion_ids:
                        raise ValueError(
                            f"extractor emitted duplicate candidate {assertion_id}"
                        )
                    assertion_ids.add(assertion_id)
                    sinks["model_candidates"].add(candidate)
        if input_rows != source.row_count:
            raise ValueError(
                f"source release row count drift: expected {source.row_count}, "
                f"read {input_rows}"
            )
        return {
            "input_row_count": input_rows,
            "document_row_count": sinks["documents"].row_count,
            "documents_with_urls": documents_with_urls,
            "documents_without_urls": input_rows - documents_with_urls,
            "url_relation_row_count": sinks["url_relations"].row_count,
            "documents_with_candidates": documents_with_candidates,
            "documents_without_candidates": input_rows - documents_with_candidates,
            "model_candidate_row_count": sinks["model_candidates"].row_count,
        }

    def _iter_input_batches(self, source: _SourceInput) -> Iterator[pa.RecordBatch]:
        if isinstance(source.receipt, ReleaseReceipt):
            yield from self.landing_zone.iter_release_batches(
                source=source.source,
                dataset=source.dataset,
                release=source.release,
                columns=_INPUT_COLUMNS,
                batch_size=self.limits.input_batch_rows,
            )
            return
        manifest = _read_json(source.receipt.path / "manifest.json", "source shard")
        for part in manifest["parts"]:
            path = source.receipt.path / "parts" / str(part["name"])
            yield from pq.ParquetFile(path).iter_batches(
                columns=_INPUT_COLUMNS,
                batch_size=self.limits.input_batch_rows,
            )

    def _decode_document(
        self,
        values: Mapping[str, Any],
        *,
        source: _SourceInput,
        artifact_id: str,
        row_ordinal: int,
    ) -> _Document:
        source_record_id = _required_text(
            values.get("source_record_id"), "source record ID"
        )
        match = _SOURCE_RECORD_ID.fullmatch(source_record_id)
        if match is None:
            raise ValueError("WET source record ID is noncanonical")
        if values.get("operation") != "upsert":
            raise ValueError("WET discovery accepts only conversion upserts")
        payload_json = _required_json_text(values.get("payload_json"), "WET payload")
        landing_payload_sha256 = _sha256(
            values.get("content_sha256"), "landing payload checksum"
        )
        if hashlib.sha256(payload_json.encode()).hexdigest() != landing_payload_sha256:
            raise ValueError("WET landing payload checksum mismatch")
        payload = json.loads(payload_json)
        if not isinstance(payload, Mapping):
            raise ValueError("WET payload must be an object")
        if _canonical_json(payload) != payload_json:
            raise ValueError("WET payload JSON must be canonical")

        source_url = _exact_text(payload.get("uri"), "WET source URL")
        crawled_at = _exact_text(payload.get("date"), "WET crawl date")
        content = _text(payload.get("content"), "WET content")
        if len(content.encode()) > self.limits.max_document_text_bytes:
            raise ValueError(
                f"source record {source_record_id} exceeded max_document_text_bytes"
            )
        content_type = _exact_text(payload.get("content_type"), "WET content type")
        content_bytes = _nonnegative_integer(
            payload.get("content_bytes"), "WET content bytes"
        )
        if content_bytes != len(content.encode()):
            raise ValueError("WET content byte count mismatch")
        content_sha256 = _sha256(payload.get("content_sha256"), "WET content checksum")
        if hashlib.sha256(content.encode()).hexdigest() != content_sha256:
            raise ValueError("WET content checksum mismatch")

        warc = _mapping(payload.get("warc"), "WET WARC provenance")
        if warc.get("type") != "conversion":
            raise ValueError("WET payload is not a conversion record")
        warc_record_id = _exact_text(warc.get("record_id"), "WARC record ID")
        bulk = _mapping(payload.get("bulk"), "WET bulk provenance")
        collection_id = _exact_text(bulk.get("collection_id"), "collection ID")
        if collection_id != source.release:
            raise ValueError("WET collection conflicts with its sealed release")
        object_url = _exact_text(bulk.get("object_url"), "crawl object URL")
        crawl_path = _exact_text(bulk.get("path"), "crawl object path")
        manifest_index = _nonnegative_integer(
            bulk.get("manifest_index"), "crawl manifest index"
        )
        record_index = _nonnegative_integer(
            bulk.get("record_index"), "crawl record index"
        )
        conversion_index = _nonnegative_integer(
            bulk.get("conversion_index"), "crawl conversion index"
        )
        expected_key = canonical_control_sha256(
            {
                "collection_id": collection_id,
                "object_url": object_url,
                "warc_record_id": warc_record_id,
            }
        )
        if match.group(1) != expected_key:
            raise ValueError("WET source record ID conflicts with its crawl provenance")

        crawl_locator = {
            "collection_id": collection_id,
            "conversion_index": conversion_index,
            "manifest_index": manifest_index,
            "object_url": object_url,
            "path": crawl_path,
            "record_index": record_index,
            "warc_record_id": warc_record_id,
        }
        crawl_locator_json = _canonical_json(crawl_locator)
        document_id = _json_sha256(
            {
                "source": source.source,
                "source_record_id": source_record_id,
                "source_url": source_url,
                "content_sha256": content_sha256,
                "crawl_locator": crawl_locator,
            }
        )
        source_identifier = _identifier_or_none(source_url)
        source_repository_url = _repository_url(source_identifier)
        row = {
            "document_id": document_id,
            "source_record_id": source_record_id,
            "source": source.source,
            "dataset": source.dataset,
            "release": source.release,
            "source_url": source_url,
            "source_identifier_namespace": (
                source_identifier.namespace if source_identifier is not None else None
            ),
            "source_identifier_value": (
                source_identifier.value if source_identifier is not None else None
            ),
            "source_repository_url": source_repository_url,
            "crawled_at": crawled_at,
            "content_type": content_type,
            "content_bytes": content_bytes,
            "content_sha256": content_sha256,
            "text": content,
            "warc_record_id": warc_record_id,
            "crawl_object_url": object_url,
            "crawl_path": crawl_path,
            "crawl_manifest_index": manifest_index,
            "crawl_record_index": record_index,
            "crawl_conversion_index": conversion_index,
            "crawl_locator_json": crawl_locator_json,
            "landing_payload_sha256": landing_payload_sha256,
            "source_input_kind": source.kind,
            "source_input_sha256": source.sha256,
            "source_shard": source.shard,
            "source_row_ordinal": row_ordinal,
            "projection_artifact_id": artifact_id,
        }
        record = SourceRecord(
            source_record_id=source_record_id,
            kind=ArtifactKind.WEB_PAGE,
            canonical_url=source_url,
            title="",
            text=content,
            raw={"crawl_locator": crawl_locator},
            published_at=crawled_at,
            identifiers=(source_identifier,) if source_identifier is not None else (),
        )
        return _Document(row=row, source_record=record, source_identifier=source_identifier)

    def _url_relation_row(
        self,
        document: _Document,
        *,
        target_url: str,
        locator: str,
    ) -> dict[str, Any]:
        match = _URL_LOCATOR.fullmatch(locator)
        if match is None:
            raise ValueError("URL locator is noncanonical")
        start, end = (int(value) for value in match.groups())
        text = document.source_record.text
        if start < 0 or start >= end or end > len(text):
            raise ValueError("URL locator is outside its source document")
        predicate = infer_url_relation(text, locator)
        target_identifier = _identifier_or_none(target_url)
        context_start, context_end = _bounded_context(
            text,
            start,
            end,
            self.limits.evidence_context_chars,
        )
        identity = {
            "document_id": document.row["document_id"],
            "predicate": predicate,
            "target_url": target_url,
            "locator": locator,
            "content_sha256": document.row["content_sha256"],
        }
        return {
            "relation_assertion_id": _json_sha256(identity),
            "document_id": document.row["document_id"],
            "source_record_id": document.row["source_record_id"],
            "source": document.row["source"],
            "dataset": document.row["dataset"],
            "release": document.row["release"],
            "source_url": document.row["source_url"],
            "predicate": predicate,
            "target_url": target_url,
            "target_identifier_namespace": (
                target_identifier.namespace if target_identifier is not None else None
            ),
            "target_identifier_value": (
                target_identifier.value if target_identifier is not None else None
            ),
            "target_repository_url": _repository_url(target_identifier),
            "locator": locator,
            "evidence_start": start,
            "evidence_end": end,
            "evidence_text": text[start:end],
            "context_start": context_start,
            "context_end": context_end,
            "context_text": text[context_start:context_end],
            "content_sha256": document.row["content_sha256"],
            "crawl_locator_json": document.row["crawl_locator_json"],
            "source_input_kind": document.row["source_input_kind"],
            "source_input_sha256": document.row["source_input_sha256"],
            "source_shard": document.row["source_shard"],
            "source_row_ordinal": document.row["source_row_ordinal"],
            "projection_artifact_id": document.row["projection_artifact_id"],
        }

    def _candidate_row(
        self,
        document: _Document,
        hint: ModelHint,
    ) -> dict[str, Any]:
        if not isinstance(hint, ModelHint):
            raise TypeError("extractor returned a value that is not a ModelHint")
        if hint.status is not ModelStatus.CANDIDATE:
            raise ValueError("discovery projector accepts only candidate model hints")
        name = _required_text(hint.name, "candidate name")
        normalized_name = normalize_name(name)
        if not normalized_name:
            raise ValueError("candidate name does not have a normalized form")
        confidence = hint.confidence
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("candidate confidence must be finite and between zero and one")
        local_id = _required_text(hint.local_id, "extractor local ID")
        locator = _required_text(hint.locator, "candidate locator")
        match = _MODEL_LOCATOR.fullmatch(locator)
        if match is None:
            raise ValueError("candidate locator must identify an exact title or text span")
        field = match.group(1)
        start, end = int(match.group(2)), int(match.group(3))
        evidence_source = (
            document.source_record.title
            if field == "title"
            else document.source_record.text
        )
        if start < 0 or start >= end or end > len(evidence_source):
            raise ValueError("candidate locator is outside its evidence field")
        evidence_text = evidence_source[start:end]
        if normalize_name(evidence_text) != normalized_name:
            raise ValueError("candidate locator does not identify the candidate name")
        context_start, context_end = _bounded_context(
            evidence_source,
            start,
            end,
            self.limits.evidence_context_chars,
        )
        assertion_identity = {
            "document_id": document.row["document_id"],
            "extractor": self.extractor_name,
            "extractor_local_id": local_id,
            "locator": locator,
            "normalized_name": normalized_name,
            "content_sha256": document.row["content_sha256"],
        }
        derivation = {
            "algorithm": _ALGORITHM,
            "assertion_identity": assertion_identity,
            "crawl_locator": json.loads(document.row["crawl_locator_json"]),
            "evidence": {
                "field": field,
                "start": start,
                "end": end,
                "text": evidence_text,
                "context_start": context_start,
                "context_end": context_end,
                "context_text": evidence_source[context_start:context_end],
            },
            "projection_artifact_id": document.row["projection_artifact_id"],
            "source_input": {
                "kind": document.row["source_input_kind"],
                "sha256": document.row["source_input_sha256"],
                "shard": document.row["source_shard"],
            },
            "source_row_ordinal": document.row["source_row_ordinal"],
        }
        return {
            "candidate_assertion_id": _json_sha256(assertion_identity),
            "document_id": document.row["document_id"],
            "source_record_id": document.row["source_record_id"],
            "source": document.row["source"],
            "dataset": document.row["dataset"],
            "release": document.row["release"],
            "source_url": document.row["source_url"],
            "source_identifier_namespace": document.row["source_identifier_namespace"],
            "source_identifier_value": document.row["source_identifier_value"],
            "source_repository_url": document.row["source_repository_url"],
            "name": name,
            "normalized_name": normalized_name,
            "status": ModelStatus.CANDIDATE.value,
            "confidence": float(confidence),
            "extractor": self.extractor_name,
            "extractor_local_id": local_id,
            "locator": locator,
            "evidence_field": field,
            "evidence_start": start,
            "evidence_end": end,
            "evidence_text": evidence_text,
            "context_start": context_start,
            "context_end": context_end,
            "context_text": evidence_source[context_start:context_end],
            "content_sha256": document.row["content_sha256"],
            "crawl_locator_json": document.row["crawl_locator_json"],
            "source_input_kind": document.row["source_input_kind"],
            "source_input_sha256": document.row["source_input_sha256"],
            "source_shard": document.row["source_shard"],
            "source_row_ordinal": document.row["source_row_ordinal"],
            "projection_artifact_id": document.row["projection_artifact_id"],
            "derivation_json": _canonical_json(derivation),
        }

    def _verify_projection(
        self,
        path: Path,
        *,
        source_input: _SourceInput | None = None,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"discovery projection is not a plain directory: {path}")
        manifest = _read_json(path / "manifest.json", "projection manifest")
        seal = _read_json(path / "SEAL.json", "projection seal")
        if manifest.get("format") != _FORMAT or manifest.get("algorithm") != _ALGORITHM:
            raise ValueError("discovery projection format is incompatible")
        manifest_sha256 = manifest.get("manifest_sha256")
        if not _is_sha256(manifest_sha256):
            raise ValueError("discovery projection manifest checksum is invalid")
        unsigned_manifest = dict(manifest)
        unsigned_manifest.pop("manifest_sha256", None)
        if _json_sha256(unsigned_manifest) != manifest_sha256:
            raise ValueError("discovery projection manifest checksum mismatch")
        actual_artifact_id = manifest.get("artifact_id")
        if not _is_sha256(actual_artifact_id):
            raise ValueError("discovery projection artifact ID is invalid")
        if artifact_id is not None and actual_artifact_id != artifact_id:
            raise ValueError("discovery projection artifact ID mismatch")
        if path.name != actual_artifact_id:
            raise ValueError("discovery projection path is noncanonical")
        if manifest.get("extractor") != {"name": self.extractor_name}:
            raise ValueError("discovery projection extractor identity is invalid")
        if manifest.get("layout") != self._layout():
            raise ValueError("discovery projection layout is incompatible")

        if set(manifest) != {
            "format",
            "algorithm",
            "artifact_id",
            "source_input",
            "extractor",
            "layout",
            "stats",
            "tables",
            "created_at",
            "manifest_sha256",
        }:
            raise ValueError("discovery projection manifest fields are invalid")
        source_identity = manifest.get("source_input")
        if not isinstance(source_identity, Mapping):
            raise ValueError("discovery projection source identity is invalid")
        if source_input is not None:
            expected_source = _source_identity(source_input)
            if (
                dict(source_identity) != expected_source
                or actual_artifact_id
                != self._artifact_id(source_input)
            ):
                raise ValueError("discovery projection does not match its source")
        _validate_source_identity(source_identity)

        stats = _validate_stats(manifest.get("stats"))
        tables = manifest.get("tables")
        if not isinstance(tables, Mapping) or set(tables) != set(_TABLE_SCHEMAS):
            raise ValueError("discovery projection table inventory is invalid")
        expected_top_level = {
            "SEAL.json",
            "manifest.json",
            *_TABLE_SCHEMAS,
        }
        if {entry.name for entry in path.iterdir()} != expected_top_level:
            raise ValueError("discovery projection directory has unexpected entries")
        for table_name, schema in _TABLE_SCHEMAS.items():
            table = tables.get(table_name)
            if not isinstance(table, Mapping):
                raise ValueError(f"{table_name} projection table is invalid")
            if set(table) != {"schema", "row_count", "part_count", "parts"}:
                raise ValueError(f"{table_name} projection fields are invalid")
            if table.get("schema") != str(schema):
                raise ValueError(f"{table_name} projection schema is incompatible")
            row_count = _nonnegative_integer(
                table.get("row_count"), f"{table_name} row count"
            )
            part_count = _nonnegative_integer(
                table.get("part_count"), f"{table_name} part count"
            )
            parts = table.get("parts")
            if not isinstance(parts, list) or len(parts) != part_count:
                raise ValueError(f"{table_name} part inventory is invalid")
            table_root = path / table_name
            _require_plain_directory(table_root, f"{table_name} table directory")
            total_rows = 0
            expected_names: set[str] = set()
            for index, part in enumerate(parts):
                if not isinstance(part, Mapping):
                    raise ValueError(f"{table_name} part entry is invalid")
                name = f"part-{index:06d}.parquet"
                relative = f"{table_name}/{name}"
                if set(part) != {
                    "index",
                    "name",
                    "path",
                    "row_count",
                    "byte_count",
                    "sha256",
                }:
                    raise ValueError(f"{table_name} part fields are invalid")
                if (
                    part.get("index") != index
                    or part.get("name") != name
                    or part.get("path") != relative
                ):
                    raise ValueError(f"{table_name} part path is noncanonical")
                part_rows = _positive_integer(
                    part.get("row_count"), f"{table_name} part rows"
                )
                byte_count = _positive_integer(
                    part.get("byte_count"), f"{table_name} part bytes"
                )
                part_sha256 = _sha256(part.get("sha256"), f"{table_name} part checksum")
                part_path = path / relative
                if part_path.is_symlink() or not part_path.is_file():
                    raise ValueError(f"{table_name} projection part is missing")
                if (
                    part_path.stat().st_size != byte_count
                    or _file_sha256(part_path) != part_sha256
                ):
                    raise ValueError(f"{table_name} projection part checksum mismatch")
                metadata = pq.read_metadata(part_path)
                if metadata.num_rows != part_rows:
                    raise ValueError(f"{table_name} projection part row count mismatch")
                if metadata.schema.to_arrow_schema() != schema:
                    raise ValueError(f"{table_name} projection part schema mismatch")
                total_rows += part_rows
                expected_names.add(name)
            if total_rows != row_count:
                raise ValueError(f"{table_name} projection row count mismatch")
            if {entry.name for entry in table_root.iterdir()} != expected_names:
                raise ValueError(f"{table_name} table directory has unexpected entries")

        if stats["document_row_count"] != tables["documents"]["row_count"]:
            raise ValueError("document projection stats are inconsistent")
        if stats["url_relation_row_count"] != tables["url_relations"]["row_count"]:
            raise ValueError("URL projection stats are inconsistent")
        if (
            stats["model_candidate_row_count"]
            != tables["model_candidates"]["row_count"]
        ):
            raise ValueError("candidate projection stats are inconsistent")
        if stats["input_row_count"] != source_identity["row_count"]:
            raise ValueError("projection input count conflicts with its source input")

        if set(seal) != {
            "format",
            "artifact_id",
            "manifest_sha256",
            "data_sha256",
            "sealed_at",
            "seal_sha256",
        }:
            raise ValueError("discovery projection seal fields are invalid")
        seal_sha256 = seal.get("seal_sha256")
        if not _is_sha256(seal_sha256):
            raise ValueError("discovery projection seal checksum is invalid")
        unsigned_seal = dict(seal)
        unsigned_seal.pop("seal_sha256", None)
        if _json_sha256(unsigned_seal) != seal_sha256:
            raise ValueError("discovery projection seal checksum mismatch")
        if (
            seal.get("format") != _FORMAT
            or seal.get("artifact_id") != actual_artifact_id
            or seal.get("manifest_sha256") != manifest_sha256
            or seal.get("data_sha256") != _json_sha256({"tables": tables})
        ):
            raise ValueError("discovery projection seal does not match its manifest")
        return manifest


def _source_identity(source: _SourceInput) -> dict[str, Any]:
    receipt = source.receipt
    if isinstance(receipt, ReleaseReceipt):
        details: dict[str, Any] = {
            "application_mode": receipt.application_mode,
            "shard_count": receipt.shard_count,
        }
    else:
        details = {
            "application_order": receipt.application_order.as_dict(),
            "control_sha256": receipt.control_sha256,
            "part_count": receipt.part_count,
            "shard": receipt.shard,
            "upstream_bytes": receipt.upstream_bytes,
            "upstream_sha256": receipt.upstream_sha256,
            "upstream_url": receipt.upstream_url,
        }
    return {
        "kind": source.kind,
        "source": receipt.source,
        "dataset": receipt.dataset,
        "release": receipt.release,
        "input_sha256": _sha256(source.sha256, "source input checksum"),
        "row_count": receipt.row_count,
        "details": details,
    }


def _shard_receipt_identity(receipt: ShardReceipt) -> tuple[Any, ...]:
    return (
        receipt.source,
        receipt.dataset,
        receipt.release,
        receipt.shard,
        receipt.control_sha256,
        receipt.upstream_sha256,
        receipt.upstream_url,
        receipt.upstream_bytes,
        receipt.row_count,
        receipt.part_count,
        receipt.application_order,
        receipt.path,
    )


def _validate_source_identity(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "kind",
        "source",
        "dataset",
        "release",
        "input_sha256",
        "row_count",
        "details",
    }:
        raise ValueError("discovery projection source fields are invalid")
    kind = _required_text(value.get("kind"), "source input kind")
    if kind not in {"release", "shard"}:
        raise ValueError("source input kind must be release or shard")
    _required_text(value.get("source"), "source")
    _required_text(value.get("dataset"), "dataset")
    _required_text(value.get("release"), "release")
    _sha256(value.get("input_sha256"), "source input checksum")
    _nonnegative_integer(value.get("row_count"), "source row count")
    details = _mapping(value.get("details"), "source input details")
    if kind == "release":
        if set(details) != {"application_mode", "shard_count"}:
            raise ValueError("source release detail fields are invalid")
        _required_text(details.get("application_mode"), "application mode")
        _positive_integer(details.get("shard_count"), "source shard count")
        return
    if set(details) != {
        "application_order",
        "control_sha256",
        "part_count",
        "shard",
        "upstream_bytes",
        "upstream_sha256",
        "upstream_url",
    }:
        raise ValueError("source shard detail fields are invalid")
    if not isinstance(details.get("application_order"), Mapping):
        raise ValueError("source shard application order is invalid")
    _sha256(details.get("control_sha256"), "source control checksum")
    _nonnegative_integer(details.get("part_count"), "source part count")
    _required_text(details.get("shard"), "source shard")
    upstream_bytes = details.get("upstream_bytes")
    if upstream_bytes is not None:
        _nonnegative_integer(upstream_bytes, "source upstream bytes")
    _sha256(details.get("upstream_sha256"), "source upstream checksum")
    upstream_url = details.get("upstream_url")
    if upstream_url is not None:
        _required_text(upstream_url, "source upstream URL")


def _validate_stats(value: Any) -> dict[str, int]:
    fields = {
        "input_row_count",
        "document_row_count",
        "documents_with_urls",
        "documents_without_urls",
        "url_relation_row_count",
        "documents_with_candidates",
        "documents_without_candidates",
        "model_candidate_row_count",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("discovery projection stats fields are invalid")
    stats = {name: _nonnegative_integer(value.get(name), name) for name in fields}
    if (
        stats["input_row_count"] != stats["document_row_count"]
        or stats["documents_with_urls"] + stats["documents_without_urls"]
        != stats["input_row_count"]
        or stats["documents_with_candidates"]
        + stats["documents_without_candidates"]
        != stats["input_row_count"]
    ):
        raise ValueError("discovery projection stats are inconsistent")
    return stats


def _receipt(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    already_materialized: bool,
) -> CommonCrawlDiscoveryReceipt:
    source = manifest["source_input"]
    details = source["details"]
    tables = manifest["tables"]
    return CommonCrawlDiscoveryReceipt(
        source=str(source["source"]),
        dataset=str(source["dataset"]),
        release=str(source["release"]),
        artifact_id=str(manifest["artifact_id"]),
        source_input_kind=str(source["kind"]),
        source_input_sha256=str(source["input_sha256"]),
        source_shard=(str(details["shard"]) if source["kind"] == "shard" else None),
        document_count=int(tables["documents"]["row_count"]),
        url_relation_count=int(tables["url_relations"]["row_count"]),
        candidate_count=int(tables["model_candidates"]["row_count"]),
        table_part_counts=tuple(
            (name, int(table["part_count"]))
            for name, table in sorted(tables.items())
        ),
        path=path,
        already_materialized=already_materialized,
    )


def _receipt_identity(receipt: CommonCrawlDiscoveryReceipt) -> tuple[Any, ...]:
    return (
        receipt.source,
        receipt.dataset,
        receipt.release,
        receipt.artifact_id,
        receipt.source_input_kind,
        receipt.source_input_sha256,
        receipt.source_shard,
        receipt.document_count,
        receipt.url_relation_count,
        receipt.candidate_count,
        receipt.table_part_counts,
        receipt.path,
    )


def _identifier_or_none(url: str) -> Identifier | None:
    try:
        return identifier_from_url(url)
    except ValueError:
        return None


def _repository_url(identifier: Identifier | None) -> str | None:
    if identifier is None or identifier.namespace != "github:repository":
        return None
    return f"https://github.com/{identifier.value}"


def _bounded_context(
    text: str,
    start: int,
    end: int,
    limit: int,
) -> tuple[int, int]:
    if end - start >= limit:
        return start, end
    remaining = limit - (end - start)
    left = min(start, remaining * 2 // 3)
    right = min(len(text) - end, remaining - left)
    if left + right < remaining:
        left += min(start - left, remaining - left - right)
    return start - left, end + right


def _required_json_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be JSON text")
    try:
        json.loads(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} is not valid JSON") from None
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    return value


def _exact_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be nonempty exact text")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        raise ValueError("value contains a non-JSON value") from None


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(value)).encode()).hexdigest()


def _row_bytes(row: Mapping[str, Any]) -> int:
    return len(_canonical_json(dict(row)).encode())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir():
            _fsync_directory(path)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = (_canonical_json(dict(value)) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            path.unlink()
        raise


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} metadata is missing: {path}")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"{label} metadata is too large: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"{label} metadata is invalid: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label} metadata is not an object: {path}")
    return value


def _require_plain_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a plain directory: {path}")


def _path_component(value: str) -> str:
    raw = _required_text(value, "path component")
    readable = _COMPONENT.sub("-", raw).strip(".-_")[:48] or "item"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{readable}-{digest}"


__all__ = [
    "DOCUMENT_SCHEMA",
    "MODEL_CANDIDATE_SCHEMA",
    "URL_RELATION_SCHEMA",
    "CommonCrawlDiscoveryLimits",
    "CommonCrawlDiscoveryReceipt",
    "CommonCrawlWetDiscoveryProjector",
]
