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
from modelome.models import ArtifactKind, ModelHint, ModelStatus, SourceRecord
from modelome.normalize import normalize_name
from modelome.semantic_scholar_materialize import (
    ProjectionReceipt,
    SemanticScholarProjectionMaterializer,
)

_FORMAT = "modelome-model-candidate-projection-v1"
_ALGORITHM = "bounded-source-backed-extraction-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_LOCATOR = re.compile(r"^(title|text):(\d+):(\d+)$")

_INPUT_COLUMNS = (
    "source_record_id",
    "source",
    "release",
    "operation",
    "tombstone",
    "corpus_id",
    "title",
    "text",
    "external_ids_json",
    "publication_date",
    "canonical_url",
    "urls_json",
    "sha_aliases",
    "evidence_json",
)

MODEL_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("candidate_assertion_id", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("release", pa.string(), nullable=False),
        pa.field("corpus_id", pa.string(), nullable=False),
        pa.field("operation", pa.string(), nullable=False),
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
        pa.field("supporting_text", pa.large_string(), nullable=False),
        pa.field("document_title", pa.large_string()),
        pa.field("document_url", pa.large_string()),
        pa.field("publication_date", pa.string()),
        pa.field("external_ids_json", pa.large_string(), nullable=False),
        pa.field("urls_json", pa.large_string(), nullable=False),
        pa.field("sha_aliases", pa.list_(pa.string()), nullable=False),
        pa.field("source_projection_artifact_id", pa.string(), nullable=False),
        pa.field("source_projection_row_ordinal", pa.int64(), nullable=False),
        pa.field("source_evidence_json", pa.large_string(), nullable=False),
        pa.field("derivation_json", pa.large_string(), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class CandidateProjectionLimits:
    """Memory and row ceilings for a streaming candidate projection."""

    input_batch_rows: int = 4_096
    max_input_batch_bytes: int = 128 * 1024 * 1024
    max_document_text_bytes: int = 16 * 1024 * 1024
    max_candidates_per_document: int = 1_024
    output_part_rows: int = 25_000
    max_output_buffer_bytes: int = 128 * 1024 * 1024
    max_output_row_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        for field in (
            "input_batch_rows",
            "max_input_batch_bytes",
            "max_document_text_bytes",
            "max_candidates_per_document",
            "output_part_rows",
            "max_output_buffer_bytes",
            "max_output_row_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.max_output_row_bytes > self.max_output_buffer_bytes:
            raise ValueError(
                "max_output_row_bytes must not exceed max_output_buffer_bytes"
            )


@dataclass(frozen=True, slots=True)
class CandidateProjectionReceipt:
    source: str
    release: str
    artifact_id: str
    source_projection_artifact_id: str
    row_count: int
    part_count: int
    document_count: int
    path: Path
    already_materialized: bool = False


class SemanticScholarModelCandidateProjector:
    """Derive evidence-bearing neural-model assertions from a sealed S2 corpus.

    Input and output are both immutable Parquet projections. The projector scans
    bounded record batches and publishes only after every output part and its
    checksums are durable. Repeating a completed run verifies and reopens the same
    content-addressed artifact; an interrupted staging directory is never visible.
    """

    def __init__(
        self,
        source_materializer: SemanticScholarProjectionMaterializer,
        *,
        output_root: str | Path | None = None,
        extractor: Extractor | None = None,
        limits: CandidateProjectionLimits | None = None,
    ) -> None:
        if not isinstance(
            source_materializer,
            SemanticScholarProjectionMaterializer,
        ):
            raise TypeError(
                "source_materializer must be a SemanticScholarProjectionMaterializer"
            )
        self.source_materializer = source_materializer
        self.extractor = extractor or IntroductionCueExtractor()
        self.extractor_name = _required_text(
            getattr(self.extractor, "name", None),
            "extractor name",
        )
        if not callable(getattr(self.extractor, "extract", None)):
            raise TypeError("extractor must define extract(record)")
        self.limits = limits or CandidateProjectionLimits()
        if output_root is None:
            root = source_materializer.landing_zone.root / "model-candidate-projections"
        else:
            root = Path(output_root).expanduser().absolute()
        self.output_root = root

    def materialize(
        self,
        source_receipt: ProjectionReceipt,
    ) -> CandidateProjectionReceipt:
        """Stream one verified source projection into an atomic candidate set."""

        source = self._verify_source_receipt(source_receipt)
        self._initialize_output()
        artifact_id = self._artifact_id(source)
        final_dir = self._final_path(source, artifact_id)
        if final_dir.exists() or final_dir.is_symlink():
            manifest = self._verify_projection(
                final_dir,
                source_receipt=source,
                artifact_id=artifact_id,
            )
            return _receipt(final_dir, manifest, already_materialized=True)

        final_parent = final_dir.parent
        final_parent.mkdir(parents=True, exist_ok=True)
        _require_plain_directory(final_parent.parent, "candidate source directory")
        _require_plain_directory(final_parent, "candidate release directory")
        stage = Path(
            tempfile.mkdtemp(
                prefix="model-candidates-",
                dir=self.output_root / ".staging",
            )
        )
        try:
            parts_root = stage / "parts"
            parts_root.mkdir()
            parts, stats = self._project(source, parts_root)
            created_at = datetime.now(UTC).isoformat()
            manifest: dict[str, Any] = {
                "format": _FORMAT,
                "algorithm": _ALGORITHM,
                "artifact_id": artifact_id,
                "source": source.source,
                "release": source.release,
                "source_projection": {
                    "artifact_id": source.artifact_id,
                    "row_count": source.row_count,
                    "part_count": source.part_count,
                    "bucket_count": source.bucket_count,
                },
                "extractor": {"name": self.extractor_name},
                "layout": self._layout(),
                "schema": str(MODEL_CANDIDATE_SCHEMA),
                "stats": stats,
                "row_count": stats["candidate_row_count"],
                "part_count": len(parts),
                "parts": parts,
                "created_at": created_at,
            }
            manifest["manifest_sha256"] = _json_sha256(manifest)
            _write_json_exclusive(stage / "manifest.json", manifest)
            seal: dict[str, Any] = {
                "format": _FORMAT,
                "artifact_id": artifact_id,
                "manifest_sha256": manifest["manifest_sha256"],
                "data_sha256": _json_sha256({"parts": parts}),
                "row_count": stats["candidate_row_count"],
                "part_count": len(parts),
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
                    source_receipt=source,
                    artifact_id=artifact_id,
                )
                return _receipt(final_dir, manifest, already_materialized=True)
            _fsync_directory(final_parent)
            manifest = self._verify_projection(
                final_dir,
                source_receipt=source,
                artifact_id=artifact_id,
            )
            return _receipt(final_dir, manifest, already_materialized=False)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def open_projection(
        self,
        source_receipt: ProjectionReceipt,
    ) -> CandidateProjectionReceipt:
        """Open the deterministic candidate artifact for one source projection."""

        source = self._verify_source_receipt(source_receipt)
        self._initialize_output()
        artifact_id = self._artifact_id(source)
        path = self._final_path(source, artifact_id)
        manifest = self._verify_projection(
            path,
            source_receipt=source,
            artifact_id=artifact_id,
        )
        return _receipt(path, manifest, already_materialized=True)

    def iter_batches(
        self,
        receipt: CandidateProjectionReceipt,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Verify and lazily scan a sealed candidate projection."""

        if not isinstance(receipt, CandidateProjectionReceipt):
            raise TypeError("receipt must be a CandidateProjectionReceipt")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        manifest = self._verify_projection(
            receipt.path,
            artifact_id=receipt.artifact_id,
        )
        expected = _receipt(receipt.path, manifest, already_materialized=True)
        if (
            receipt.source,
            receipt.release,
            receipt.artifact_id,
            receipt.source_projection_artifact_id,
            receipt.row_count,
            receipt.part_count,
            receipt.document_count,
        ) != (
            expected.source,
            expected.release,
            expected.artifact_id,
            expected.source_projection_artifact_id,
            expected.row_count,
            expected.part_count,
            expected.document_count,
        ):
            raise ValueError("candidate projection receipt does not match its artifact")
        for part in manifest["parts"]:
            yield from pq.ParquetFile(receipt.path / str(part["path"])).iter_batches(
                columns=columns,
                batch_size=batch_size,
            )

    def _verify_source_receipt(self, receipt: ProjectionReceipt) -> ProjectionReceipt:
        if not isinstance(receipt, ProjectionReceipt):
            raise TypeError("source_receipt must be a ProjectionReceipt")
        verified = self.source_materializer.open_projection(
            receipt.release,
            artifact_id=receipt.artifact_id,
        )
        if (
            receipt.source,
            receipt.release,
            receipt.artifact_id,
            receipt.row_count,
            receipt.part_count,
            receipt.bucket_count,
            receipt.path,
        ) != (
            verified.source,
            verified.release,
            verified.artifact_id,
            verified.row_count,
            verified.part_count,
            verified.bucket_count,
            verified.path,
        ):
            raise ValueError("source projection receipt does not match its artifact")
        return verified

    def _initialize_output(self) -> None:
        if self.output_root.exists() and not self.output_root.is_dir():
            raise ValueError(
                f"candidate projection root is not a directory: {self.output_root}"
            )
        if self.output_root.is_symlink():
            raise ValueError(
                f"candidate projection root must not be a symlink: {self.output_root}"
            )
        lake_root = self.source_materializer.landing_zone.root
        protected_roots = (
            self.source_materializer.landing_zone.shards_root,
            self.source_materializer.output_root,
        )
        if (
            self.output_root == lake_root
            or self.output_root in lake_root.parents
            or any(
                self.output_root == root
                or root in self.output_root.parents
                or self.output_root in root.parents
                for root in protected_roots
            )
        ):
            raise ValueError(
                "candidate projection root must be separate from lake and source roots"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.output_root.resolve() != self.output_root:
            raise ValueError(
                "candidate projection root ancestors must not redirect through symlinks"
            )
        staging = self.output_root / ".staging"
        staging.mkdir(exist_ok=True)
        _require_plain_directory(self.output_root, "candidate projection root")
        _require_plain_directory(staging, "candidate staging directory")

    def _artifact_id(self, source: ProjectionReceipt) -> str:
        return _json_sha256(
            {
                "format": _FORMAT,
                "algorithm": _ALGORITHM,
                "source": source.source,
                "release": source.release,
                "source_projection_artifact_id": source.artifact_id,
                "extractor": self.extractor_name,
                "layout": self._layout(),
                "schema": str(MODEL_CANDIDATE_SCHEMA),
            }
        )

    def _layout(self) -> dict[str, int]:
        return {
            "output_part_rows": self.limits.output_part_rows,
            "max_output_buffer_bytes": self.limits.max_output_buffer_bytes,
        }

    def _final_path(self, source: ProjectionReceipt, artifact_id: str) -> Path:
        return (
            self.output_root
            / _path_component(source.source)
            / _path_component(source.release)
            / artifact_id
        )

    def _project(
        self,
        source: ProjectionReceipt,
        parts_root: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        parts: list[dict[str, Any]] = []
        buffer: list[dict[str, Any]] = []
        buffer_bytes = 0
        input_rows = 0
        active_documents = 0
        tombstone_documents = 0
        candidate_documents = 0
        candidate_rows = 0
        part_index = 0
        for batch in self.source_materializer.iter_batches(
            source,
            columns=_INPUT_COLUMNS,
            batch_size=self.limits.input_batch_rows,
        ):
            if batch.nbytes > self.limits.max_input_batch_bytes:
                raise ValueError("source projection batch exceeded max_input_batch_bytes")
            if tuple(batch.schema.names) != _INPUT_COLUMNS:
                raise ValueError("source projection returned unexpected columns")
            for values in batch.to_pylist():
                row_ordinal = input_rows
                input_rows += 1
                document = self._decode_document(values, source)
                if document is None:
                    tombstone_documents += 1
                    continue
                active_documents += 1
                record, metadata = document
                hints = tuple(self.extractor.extract(record))
                if len(hints) > self.limits.max_candidates_per_document:
                    raise ValueError(
                        f"source record {record.source_record_id} exceeded "
                        "max_candidates_per_document"
                    )
                if hints:
                    candidate_documents += 1
                document_assertions: set[str] = set()
                for hint in hints:
                    candidate = self._candidate_row(
                        source=source,
                        row_ordinal=row_ordinal,
                        record=record,
                        metadata=metadata,
                        hint=hint,
                    )
                    assertion_id = str(candidate["candidate_assertion_id"])
                    if assertion_id in document_assertions:
                        raise ValueError(
                            f"extractor emitted duplicate candidate {assertion_id}"
                        )
                    document_assertions.add(assertion_id)
                    row_bytes = _row_bytes(candidate)
                    if row_bytes > self.limits.max_output_row_bytes:
                        raise ValueError(
                            f"candidate {candidate['candidate_assertion_id']} exceeded "
                            "max_output_row_bytes"
                        )
                    if buffer and (
                        len(buffer) >= self.limits.output_part_rows
                        or buffer_bytes + row_bytes
                        > self.limits.max_output_buffer_bytes
                    ):
                        parts.append(
                            _write_part(
                                parts_root,
                                part_index=part_index,
                                rows=buffer,
                            )
                        )
                        part_index += 1
                        buffer = []
                        buffer_bytes = 0
                    buffer.append(candidate)
                    buffer_bytes += row_bytes
                    candidate_rows += 1
        if input_rows != source.row_count:
            raise ValueError(
                "source projection row count drift: "
                f"expected {source.row_count}, read {input_rows}"
            )
        if buffer:
            parts.append(
                _write_part(parts_root, part_index=part_index, rows=buffer)
            )
        return parts, {
            "input_row_count": input_rows,
            "active_document_count": active_documents,
            "tombstone_document_count": tombstone_documents,
            "candidate_document_count": candidate_documents,
            "documents_without_candidates": active_documents - candidate_documents,
            "candidate_row_count": candidate_rows,
        }

    def _decode_document(
        self,
        values: Mapping[str, Any],
        source: ProjectionReceipt,
    ) -> tuple[SourceRecord, dict[str, Any]] | None:
        source_record_id = _required_text(
            values.get("source_record_id"),
            "source record ID",
        )
        if values.get("source") != source.source or values.get("release") != source.release:
            raise ValueError("source projection row has conflicting source identity")
        corpus_id = _corpus_id(values.get("corpus_id"))
        if source_record_id != f"semantic-scholar:corpus:{corpus_id}":
            raise ValueError("source projection row has a conflicting corpus identity")
        operation = _required_text(values.get("operation"), "operation")
        tombstone = values.get("tombstone")
        if not isinstance(tombstone, bool):
            raise ValueError("source projection tombstone must be boolean")
        if tombstone:
            if operation != "delete":
                raise ValueError("source projection tombstone must be a delete")
            return None
        if operation != "upsert":
            raise ValueError("active source projection row must be an upsert")

        title = _optional_text(values.get("title"), "document title") or ""
        text = _optional_text(values.get("text"), "document text") or ""
        if len(title.encode()) + len(text.encode()) > self.limits.max_document_text_bytes:
            raise ValueError(
                f"source record {source_record_id} exceeded max_document_text_bytes"
            )
        canonical_url = _optional_text(values.get("canonical_url"), "document URL")
        publication_date = _optional_text(
            values.get("publication_date"),
            "publication date",
        )
        external_ids_json = _canonical_input_json(
            values.get("external_ids_json"),
            "external IDs",
            Mapping,
        )
        urls_json = _canonical_input_json(values.get("urls_json"), "URLs", list)
        aliases = values.get("sha_aliases")
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias for alias in aliases
        ):
            raise ValueError("source projection SHA aliases must be an array of strings")
        evidence_json = _canonical_input_json(
            values.get("evidence_json"),
            "source evidence",
            Mapping,
        )
        evidence = json.loads(evidence_json)
        if (
            evidence.get("source") != source.source
            or evidence.get("release") != source.release
            or evidence.get("projection_artifact_id") != source.artifact_id
        ):
            raise ValueError("source evidence does not identify its sealed projection")
        record = SourceRecord(
            source_record_id=source_record_id,
            kind=ArtifactKind.PAPER,
            canonical_url=canonical_url or "",
            title=title,
            text=text,
            raw={},
            published_at=publication_date,
        )
        return record, {
            "corpus_id": corpus_id,
            "canonical_url": canonical_url,
            "publication_date": publication_date,
            "external_ids_json": external_ids_json,
            "urls_json": urls_json,
            "sha_aliases": aliases,
            "evidence_json": evidence_json,
        }

    def _candidate_row(
        self,
        *,
        source: ProjectionReceipt,
        row_ordinal: int,
        record: SourceRecord,
        metadata: Mapping[str, Any],
        hint: ModelHint,
    ) -> dict[str, Any]:
        if not isinstance(hint, ModelHint):
            raise TypeError("extractor returned a value that is not a ModelHint")
        if hint.status is not ModelStatus.CANDIDATE:
            raise ValueError("candidate projector accepts only candidate model hints")
        name = _required_text(hint.name, "candidate name")
        normalized_name = normalize_name(name)
        if not normalized_name:
            raise ValueError("candidate name does not have a normalized form")
        local_id = _required_text(hint.local_id, "extractor local ID")
        locator = _required_text(hint.locator, "candidate locator")
        confidence = hint.confidence
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("candidate confidence must be finite and between zero and one")
        evidence_field, start, end, evidence_text, supporting_text = _locate_evidence(
            record,
            locator,
        )
        if normalize_name(evidence_text) != normalized_name:
            raise ValueError("candidate locator does not identify the candidate name")
        assertion_identity = {
            "source": source.source,
            "source_record_id": record.source_record_id,
            "extractor": self.extractor_name,
            "extractor_local_id": local_id,
            "locator": locator,
            "normalized_name": normalized_name,
        }
        assertion_id = _json_sha256(assertion_identity)
        derivation = {
            "algorithm": _ALGORITHM,
            "assertion_identity": assertion_identity,
            "source_projection": {
                "artifact_id": source.artifact_id,
                "row_ordinal": row_ordinal,
            },
            "evidence": {
                "field": evidence_field,
                "start": start,
                "end": end,
                "text": evidence_text,
                "supporting_text": supporting_text,
            },
        }
        return {
            "candidate_assertion_id": assertion_id,
            "source_record_id": record.source_record_id,
            "source": source.source,
            "release": source.release,
            "corpus_id": metadata["corpus_id"],
            "operation": "upsert",
            "name": name,
            "normalized_name": normalized_name,
            "status": ModelStatus.CANDIDATE.value,
            "confidence": float(confidence),
            "extractor": self.extractor_name,
            "extractor_local_id": local_id,
            "locator": locator,
            "evidence_field": evidence_field,
            "evidence_start": start,
            "evidence_end": end,
            "evidence_text": evidence_text,
            "supporting_text": supporting_text,
            "document_title": record.title or None,
            "document_url": metadata["canonical_url"],
            "publication_date": metadata["publication_date"],
            "external_ids_json": metadata["external_ids_json"],
            "urls_json": metadata["urls_json"],
            "sha_aliases": metadata["sha_aliases"],
            "source_projection_artifact_id": source.artifact_id,
            "source_projection_row_ordinal": row_ordinal,
            "source_evidence_json": metadata["evidence_json"],
            "derivation_json": _canonical_json(derivation),
        }

    def _verify_projection(
        self,
        path: Path,
        *,
        source_receipt: ProjectionReceipt | None = None,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"candidate projection is not a plain directory: {path}")
        manifest = _read_json(path / "manifest.json")
        seal = _read_json(path / "SEAL.json")
        if manifest.get("format") != _FORMAT or manifest.get("algorithm") != _ALGORITHM:
            raise ValueError("candidate projection format is incompatible")
        manifest_digest = manifest.get("manifest_sha256")
        if not _is_sha256(manifest_digest):
            raise ValueError("candidate projection manifest checksum is invalid")
        unsigned_manifest = dict(manifest)
        unsigned_manifest.pop("manifest_sha256", None)
        if _json_sha256(unsigned_manifest) != manifest_digest:
            raise ValueError("candidate projection manifest checksum mismatch")
        actual_artifact = manifest.get("artifact_id")
        if not _is_sha256(actual_artifact):
            raise ValueError("candidate projection artifact ID is invalid")
        if artifact_id is not None and actual_artifact != artifact_id:
            raise ValueError("candidate projection artifact ID mismatch")
        if path.name != actual_artifact:
            raise ValueError("candidate projection path is noncanonical")
        source_projection = manifest.get("source_projection")
        if not isinstance(source_projection, Mapping):
            raise ValueError("candidate projection source identity is invalid")
        if source_receipt is not None:
            expected_source = {
                "artifact_id": source_receipt.artifact_id,
                "row_count": source_receipt.row_count,
                "part_count": source_receipt.part_count,
                "bucket_count": source_receipt.bucket_count,
            }
            if (
                manifest.get("source") != source_receipt.source
                or manifest.get("release") != source_receipt.release
                or dict(source_projection) != expected_source
                or actual_artifact != self._artifact_id(source_receipt)
            ):
                raise ValueError("candidate projection does not match its source")
        if manifest.get("extractor") != {"name": self.extractor_name}:
            raise ValueError("candidate projection extractor identity is invalid")
        if manifest.get("layout") != self._layout():
            raise ValueError("candidate projection layout is incompatible")
        if manifest.get("schema") != str(MODEL_CANDIDATE_SCHEMA):
            raise ValueError("candidate projection schema is incompatible")
        parts = manifest.get("parts")
        row_count = _nonnegative_integer(manifest.get("row_count"), "row_count")
        part_count = _nonnegative_integer(manifest.get("part_count"), "part_count")
        if not isinstance(parts, list) or len(parts) != part_count:
            raise ValueError("candidate projection part inventory is invalid")
        stats = manifest.get("stats")
        if not isinstance(stats, Mapping):
            raise ValueError("candidate projection stats are invalid")
        required_stats = {
            "input_row_count",
            "active_document_count",
            "tombstone_document_count",
            "candidate_document_count",
            "documents_without_candidates",
            "candidate_row_count",
        }
        if set(stats) != required_stats:
            raise ValueError("candidate projection stats fields are invalid")
        decoded_stats = {
            key: _nonnegative_integer(stats.get(key), key) for key in required_stats
        }
        if (
            decoded_stats["candidate_row_count"] != row_count
            or decoded_stats["active_document_count"]
            + decoded_stats["tombstone_document_count"]
            != decoded_stats["input_row_count"]
            or decoded_stats["candidate_document_count"]
            + decoded_stats["documents_without_candidates"]
            != decoded_stats["active_document_count"]
        ):
            raise ValueError("candidate projection stats are inconsistent")
        if (
            source_receipt is not None
            and decoded_stats["input_row_count"] != source_receipt.row_count
        ):
            raise ValueError("candidate projection input row count is inconsistent")

        parts_root = path / "parts"
        _require_plain_directory(parts_root, "candidate parts directory")
        total_rows = 0
        expected_names: set[str] = set()
        for index, part in enumerate(parts):
            if not isinstance(part, Mapping):
                raise ValueError("candidate projection part entry is invalid")
            name = f"part-{index:06d}.parquet"
            relative = f"parts/{name}"
            if part.get("index") != index or part.get("name") != name:
                raise ValueError("candidate projection part order is invalid")
            if part.get("path") != relative:
                raise ValueError("candidate projection part path is noncanonical")
            part_rows = _positive_integer(part.get("row_count"), "part row_count")
            byte_count = _positive_integer(part.get("byte_count"), "part byte_count")
            digest = part.get("sha256")
            if not _is_sha256(digest):
                raise ValueError("candidate projection part checksum is invalid")
            part_path = path / relative
            if part_path.is_symlink() or not part_path.is_file():
                raise ValueError("candidate projection part is missing")
            if part_path.stat().st_size != byte_count or _file_sha256(part_path) != digest:
                raise ValueError("candidate projection part checksum mismatch")
            metadata = pq.read_metadata(part_path)
            if metadata.num_rows != part_rows:
                raise ValueError("candidate projection part row count mismatch")
            if metadata.schema.to_arrow_schema() != MODEL_CANDIDATE_SCHEMA:
                raise ValueError("candidate projection part schema mismatch")
            total_rows += part_rows
            expected_names.add(name)
        if total_rows != row_count:
            raise ValueError("candidate projection row count mismatch")
        actual_names = {entry.name for entry in parts_root.iterdir()}
        if actual_names != expected_names:
            raise ValueError("candidate projection parts directory has unexpected entries")
        if {entry.name for entry in path.iterdir()} != {
            "parts",
            "manifest.json",
            "SEAL.json",
        }:
            raise ValueError("candidate projection directory has unexpected entries")

        if seal.get("format") != _FORMAT or seal.get("artifact_id") != actual_artifact:
            raise ValueError("candidate projection seal identity is invalid")
        seal_digest = seal.get("seal_sha256")
        if not _is_sha256(seal_digest):
            raise ValueError("candidate projection seal checksum is invalid")
        unsigned_seal = dict(seal)
        unsigned_seal.pop("seal_sha256", None)
        if _json_sha256(unsigned_seal) != seal_digest:
            raise ValueError("candidate projection seal checksum mismatch")
        if (
            seal.get("manifest_sha256") != manifest_digest
            or seal.get("data_sha256") != _json_sha256({"parts": parts})
            or seal.get("row_count") != row_count
            or seal.get("part_count") != part_count
        ):
            raise ValueError("candidate projection seal does not match its manifest")
        return manifest


def _locate_evidence(
    record: SourceRecord,
    locator: str,
) -> tuple[str, int, int, str, str]:
    match = _LOCATOR.fullmatch(locator)
    if match is None:
        raise ValueError("candidate locator must contain an exact title or text span")
    field = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    value = record.title if field == "title" else record.text
    if start >= end or end > len(value):
        raise ValueError("candidate locator is outside its evidence field")
    evidence_text = value[start:end]
    if field == "title":
        supporting_text = f"{record.title}\n{_first_sentence(record.text)}"
    else:
        supporting_text = _containing_sentence(record.text, start, end)
    return field, start, end, evidence_text, supporting_text


def _first_sentence(value: str) -> str:
    boundary = re.search(r"[.!?\n]", value)
    return value if boundary is None else value[: boundary.end()]


def _containing_sentence(value: str, start: int, end: int) -> str:
    left_boundaries = tuple(re.finditer(r"[.!?\n]", value[:start]))
    left = left_boundaries[-1].end() if left_boundaries else 0
    right_boundary = re.search(r"[.!?\n]", value[end:])
    right = end + right_boundary.end() if right_boundary else len(value)
    return value[left:right]


def _write_part(
    parts_root: Path,
    *,
    part_index: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    name = f"part-{part_index:06d}.parquet"
    path = parts_root / name
    table = pa.Table.from_pylist(rows, schema=MODEL_CANDIDATE_SCHEMA)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return {
        "index": part_index,
        "name": name,
        "path": f"parts/{name}",
        "row_count": len(rows),
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _receipt(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    already_materialized: bool,
) -> CandidateProjectionReceipt:
    source_projection = manifest["source_projection"]
    stats = manifest["stats"]
    return CandidateProjectionReceipt(
        source=str(manifest["source"]),
        release=str(manifest["release"]),
        artifact_id=str(manifest["artifact_id"]),
        source_projection_artifact_id=str(source_projection["artifact_id"]),
        row_count=int(manifest["row_count"]),
        part_count=int(manifest["part_count"]),
        document_count=int(stats["input_row_count"]),
        path=path,
        already_materialized=already_materialized,
    )


def _canonical_input_json(value: Any, label: str, expected_type: type) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be JSON text")
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} is not valid JSON") from None
    if not isinstance(decoded, expected_type):
        raise ValueError(f"{label} has an invalid JSON shape")
    canonical = _canonical_json(decoded)
    if value != canonical:
        raise ValueError(f"{label} JSON is not canonical")
    return value


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"candidate projection metadata is missing: {path}")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"candidate projection metadata is too large: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"candidate projection metadata is invalid: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"candidate projection metadata is not an object: {path}")
    return value


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        elif path.is_dir():
            _fsync_directory(path)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_plain_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a plain directory: {path}")


def _path_component(value: str) -> str:
    raw = _required_text(value, "path component")
    readable = _COMPONENT.sub("-", raw).strip(".-_")[:48] or "item"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{readable}-{digest}"


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    return value


def _corpus_id(value: Any) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError("source projection corpus ID must be decimal text")
    normalized = str(int(value))
    if normalized == "0" or normalized != value:
        raise ValueError("source projection corpus ID is noncanonical")
    return value


def _row_bytes(row: Mapping[str, Any]) -> int:
    return len(_canonical_json(dict(row)).encode())


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


__all__ = [
    "MODEL_CANDIDATE_SCHEMA",
    "CandidateProjectionLimits",
    "CandidateProjectionReceipt",
    "SemanticScholarModelCandidateProjector",
]
