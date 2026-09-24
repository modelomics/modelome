from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from modelome.lake import ParquetLandingZone
from modelome.semantic_scholar_citation_materialize import (
    CitationProjectionLimits,
    CitationProjectionReceipt,
    SemanticScholarCitationMaterializer,
)

_FORMAT = "modelome-semantic-scholar-citation-state-v1"
_ALGORITHM = "bucketed-citation-event-replay-v1"
_STATE_SCHEMA = pa.schema(
    [
        pa.field("citing_paper_id", pa.string(), nullable=False),
        pa.field("cited_paper_id", pa.string(), nullable=False),
        pa.field("tombstone", pa.bool_(), nullable=False),
        pa.field("last_operation", pa.string(), nullable=False),
        pa.field("event_ledger_artifact_id", pa.string(), nullable=False),
        pa.field("event_index", pa.int64(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("source_shard", pa.string(), nullable=False),
        pa.field("source_row_index", pa.int64(), nullable=False),
        pa.field("application_order_json", pa.large_string(), nullable=False),
        pa.field("source_content_sha256", pa.string(), nullable=False),
        pa.field("payload_json", pa.large_string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
_EVENT_SCHEMA = pa.schema(
    [
        pa.field("event_index", pa.int64(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("operation", pa.string(), nullable=False),
        pa.field("citing_paper_id", pa.string(), nullable=False),
        pa.field("cited_paper_id", pa.string(), nullable=False),
        pa.field("source_shard", pa.string(), nullable=False),
        pa.field("source_row_index", pa.int64(), nullable=False),
        pa.field("application_order_json", pa.large_string(), nullable=False),
        pa.field("source_content_sha256", pa.string(), nullable=False),
        pa.field("payload_json", pa.large_string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)
_PART_PATH = re.compile(r"bucket-(\d{5})/part-(\d{6})\.parquet$")


@dataclass(frozen=True, slots=True)
class CitationStateLimits:
    bucket_count: int = 32_768
    event_partition_rows: int = 8_192
    event_partition_bytes: int = 64 * 1024 * 1024
    scan_batch_rows: int = 2_048
    max_scan_batch_bytes: int = 64 * 1024 * 1024
    max_bucket_rows: int = 250_000
    max_bucket_bytes: int = 256 * 1024 * 1024
    output_part_rows: int = 10_000
    max_output_row_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "bucket_count",
            "event_partition_rows",
            "event_partition_bytes",
            "scan_batch_rows",
            "max_scan_batch_bytes",
            "max_bucket_rows",
            "max_bucket_bytes",
            "output_part_rows",
            "max_output_row_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.bucket_count > 65_536:
            raise ValueError("bucket_count must not exceed 65536")


@dataclass(frozen=True, slots=True)
class CitationStateReceipt:
    source: str
    release: str
    artifact_id: str
    edge_count: int
    active_count: int
    deleted_count: int
    part_count: int
    bucket_count: int
    path: Path
    already_materialized: bool = False


class SemanticScholarCitationStateMaterializer:
    """Build latest citation state from immutable citation event projections.

    State is hash partitioned on the exact ordered paper-ID pair. Each bucket
    has a hard row and byte ceiling. Deleted edges remain as tombstones with the
    deleting event's provenance; the linked event-ledger chain retains the full
    history, including prior deletes and reactivations.
    """

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        output_root: str | Path,
        source: str = "semantic-scholar",
        limits: CitationStateLimits | None = None,
    ) -> None:
        if not isinstance(landing_zone, ParquetLandingZone):
            raise TypeError("landing_zone must be a ParquetLandingZone")
        if not isinstance(output_root, (str, Path)):
            raise TypeError("output_root must be a path")
        self.landing_zone = landing_zone
        self.output_root = Path(output_root).expanduser().absolute()
        self.source = _required_text(source, "source")
        self.limits = limits or CitationStateLimits()
        self.events = SemanticScholarCitationMaterializer(
            landing_zone, source=self.source
        )

    def materialize(
        self,
        event_projection: CitationProjectionReceipt,
        *,
        base: CitationStateReceipt | None = None,
    ) -> CitationStateReceipt:
        event_root = event_projection.path.parents[2]
        event_manifest = _read_json(event_projection.path / "manifest.json")
        event_layout = event_manifest.get("layout")
        if not isinstance(event_layout, Mapping) or set(event_layout) != {
            "scan_batch_rows",
            "max_scan_batch_bytes",
            "output_part_rows",
            "max_output_buffer_bytes",
            "max_output_row_bytes",
        }:
            raise ValueError("citation event projection layout is invalid")
        event_limits = CitationProjectionLimits(
            scan_batch_rows=event_layout["scan_batch_rows"],
            max_scan_batch_bytes=event_layout["max_scan_batch_bytes"],
            output_part_rows=event_layout["output_part_rows"],
            max_output_buffer_bytes=event_layout["max_output_buffer_bytes"],
            max_output_row_bytes=event_layout["max_output_row_bytes"],
        )
        self.events = SemanticScholarCitationMaterializer(
            self.landing_zone,
            output_root=event_root,
            source=self.source,
            limits=event_limits,
        )
        event_receipt = self.events.open_projection(event_projection.release)
        if event_receipt.artifact_id != event_projection.artifact_id:
            raise ValueError("citation event projection receipt does not match release")
        event_manifest = _read_json(event_receipt.path / "manifest.json")
        event_lineage = _event_lineage(event_manifest)
        mode = str(event_manifest["input"]["application_mode"])
        transitions = event_manifest["transitions"]
        if mode == "snapshot":
            if base is not None:
                raise ValueError("citation snapshot must not specify a base state")
        elif mode == "diff":
            if base is None:
                raise ValueError("citation diff state requires a base state")
            base = self._verify_state(base)
            if not transitions or transitions[0]["from_release"] != base.release:
                raise ValueError("citation diff base release does not match transition chain")
            prior_release = base.release
            for item in transitions:
                if item["from_release"] != prior_release:
                    raise ValueError("citation diff transitions are not contiguous")
                prior_release = item["to_release"]
            if prior_release != event_receipt.release:
                raise ValueError("citation diff chain does not end at target release")
        else:
            raise ValueError("citation event projection has an unknown application mode")

        artifact_id = _sha256_json(
            {
                "format": _FORMAT,
                "algorithm": _ALGORITHM,
                "source": self.source,
                "release": event_receipt.release,
                "event_ledger_artifact_id": event_receipt.artifact_id,
                "base_artifact_id": None if base is None else base.artifact_id,
                "layout": self._layout(),
                "schema": str(_STATE_SCHEMA),
            }
        )
        final_path = self._final_path(event_receipt.release, artifact_id)
        if final_path.exists() or final_path.is_symlink():
            return self._verify_state_path(
                final_path,
                release=event_receipt.release,
                artifact_id=artifact_id,
                event_manifest=event_lineage,
                base=base,
                already_materialized=True,
            )

        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.", dir=final_path.parent))
        try:
            work_root = temporary / "work"
            work_root.mkdir()
            event_parts = self._partition_events(event_receipt, work_root / "events")
            parts, edge_count, active_count = self._build_buckets(
                temporary,
                event_parts,
                event_receipt,
                base,
            )
            shutil.rmtree(work_root)
            manifest: dict[str, Any] = {
                "algorithm": _ALGORITHM,
                "artifact_id": artifact_id,
                "base": None
                if base is None
                else {
                    "artifact_id": base.artifact_id,
                    "release": base.release,
                },
                "created_at": datetime.now(UTC).isoformat(),
                "event_ledger": {
                    "artifact_id": event_receipt.artifact_id,
                    "release": event_receipt.release,
                    "transitions": transitions,
                },
                "format": _FORMAT,
                "layout": self._layout(),
                "parts": parts,
                "part_count": len(parts),
                "release": event_receipt.release,
                "row_count": edge_count,
                "active_count": active_count,
                "deleted_count": edge_count - active_count,
                "schema": str(_STATE_SCHEMA),
                "source": self.source,
            }
            manifest["manifest_sha256"] = _sha256_json(manifest)
            _write_json(temporary / "manifest.json", manifest)
            os.replace(temporary, final_path)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        return self._verify_state_path(
            final_path,
            release=event_receipt.release,
            artifact_id=artifact_id,
            event_manifest=event_lineage,
            base=base,
            already_materialized=False,
        )

    def open_projection(self, release: str, artifact_id: str) -> CitationStateReceipt:
        """Open one exact persisted citation state after process restart."""

        release = _required_text(release, "release")
        artifact_id = _required_text(artifact_id, "artifact_id")
        if len(artifact_id) != 64 or any(
            char not in "0123456789abcdef" for char in artifact_id
        ):
            raise ValueError("artifact_id must be a lowercase SHA-256 digest")
        path = self._final_path(release, artifact_id)
        manifest = _read_json(path / "manifest.json")
        event_lineage = manifest.get("event_ledger")
        if not isinstance(event_lineage, Mapping):
            raise ValueError("citation state event lineage is invalid")
        return self._verify_state_path(
            path,
            release=release,
            artifact_id=artifact_id,
            event_manifest=event_lineage,
            base=None,
            already_materialized=True,
            allow_unchecked_base=True,
        )

    def iter_batches(
        self,
        receipt: CitationStateReceipt,
        *,
        include_tombstones: bool = False,
    ) -> Iterator[pa.RecordBatch]:
        self._verify_state(receipt)
        manifest = _read_json(receipt.path / "manifest.json")
        for part in manifest["parts"]:
            part_path = receipt.path / "parts" / str(part["path"])
            for batch in pq.ParquetFile(part_path).iter_batches(
                batch_size=self.limits.scan_batch_rows
            ):
                if batch.nbytes > self.limits.max_scan_batch_bytes:
                    raise ValueError("citation state scan batch exceeded byte limit")
                if include_tombstones:
                    yield batch
                else:
                    mask = pc.invert(
                        batch.column(batch.schema.get_field_index("tombstone"))
                    )
                    yield batch.filter(mask)

    def _partition_events(
        self,
        receipt: CitationProjectionReceipt,
        work_root: Path,
    ) -> dict[int, list[Path]]:
        work_root.mkdir()
        buffers: dict[int, list[dict[str, Any]]] = {}
        indexes: dict[int, int] = {}
        output: dict[int, list[Path]] = {}
        buffered_rows = 0
        buffered_bytes = 0

        def flush() -> None:
            nonlocal buffers, indexes, buffered_rows, buffered_bytes
            for bucket in sorted(buffers):
                bucket_root = work_root / f"bucket-{bucket:05d}"
                bucket_root.mkdir(exist_ok=True)
                index = indexes.get(bucket, 0)
                path = bucket_root / f"part-{index:06d}.parquet"
                pq.write_table(
                    pa.Table.from_pylist(buffers[bucket], schema=_EVENT_SCHEMA),
                    path,
                    compression="zstd",
                )
                output.setdefault(bucket, []).append(path)
                indexes[bucket] = index + 1
            buffers = {}
            buffered_rows = 0
            buffered_bytes = 0

        manifest = _read_json(receipt.path / "manifest.json")
        for part in manifest["parts"]:
            part_path = receipt.path / "parts" / str(part["name"])
            for batch in pq.ParquetFile(part_path).iter_batches(
                batch_size=self.limits.scan_batch_rows
            ):
                if batch.nbytes > self.limits.max_scan_batch_bytes:
                    raise ValueError("citation event input batch exceeded byte limit")
                for row in batch.to_pylist():
                    bucket = _bucket(
                        row["citing_paper_id"],
                        row["cited_paper_id"],
                        self.limits.bucket_count,
                    )
                    size = _row_size(row)
                    if size > self.limits.event_partition_bytes:
                        raise ValueError("one citation event exceeded partition byte limit")
                    if buffered_rows and (
                        buffered_rows >= self.limits.event_partition_rows
                        or buffered_bytes + size > self.limits.event_partition_bytes
                    ):
                        flush()
                    buffers.setdefault(bucket, []).append(row)
                    buffered_rows += 1
                    buffered_bytes += size
        if buffers:
            flush()
        return output

    def _build_buckets(
        self,
        temporary: Path,
        event_parts: Mapping[int, list[Path]],
        event_receipt: CitationProjectionReceipt,
        base: CitationStateReceipt | None,
    ) -> tuple[list[dict[str, Any]], int, int]:
        output_root = temporary / "parts"
        output_root.mkdir()
        base_manifest = None if base is None else _read_json(base.path / "manifest.json")
        base_parts: dict[int, list[Mapping[str, Any]]] = {}
        if base_manifest is not None:
            for part in base_manifest["parts"]:
                base_parts.setdefault(int(part["bucket"]), []).append(part)
        output_parts: list[dict[str, Any]] = []
        total_states = 0
        total_active = 0
        for bucket in range(self.limits.bucket_count):
            states: dict[tuple[str, str], dict[str, Any]] = {}
            state_bytes = 0
            for part in base_parts.get(bucket, []):
                part_path = base.path / "parts" / str(part["path"])
                for batch in pq.ParquetFile(part_path).iter_batches(
                    batch_size=self.limits.scan_batch_rows
                ):
                    if batch.nbytes > self.limits.max_scan_batch_bytes:
                        raise ValueError("citation base bucket batch exceeded byte limit")
                    for row in batch.to_pylist():
                        key = (row["citing_paper_id"], row["cited_paper_id"])
                        if key in states:
                            raise ValueError("citation base state has duplicate edge IDs")
                        states[key] = row
                        state_bytes += _memory_size(row)
                        if (
                            len(states) > self.limits.max_bucket_rows
                            or state_bytes > self.limits.max_bucket_bytes
                        ):
                            raise ValueError(
                                f"citation base bucket {bucket} exceeded configured limits"
                            )

            last_event = -1
            for part_path in event_parts.get(bucket, []):
                for batch in pq.ParquetFile(part_path).iter_batches(
                    batch_size=self.limits.scan_batch_rows
                ):
                    for event in batch.to_pylist():
                        if event["event_index"] <= last_event:
                            raise ValueError("citation event bucket order is invalid")
                        last_event = event["event_index"]
                        key = (event["citing_paper_id"], event["cited_paper_id"])
                        prior = states.get(key)
                        if event["operation"] == "delete":
                            if prior is None or prior["tombstone"]:
                                raise ValueError("citation delete references no active edge")
                        elif event["operation"] != "upsert":
                            raise ValueError("citation event operation is invalid")
                        row = _state_row(event, event_receipt.artifact_id)
                        prior_size = 0 if prior is None else _memory_size(prior)
                        state_bytes += _memory_size(row) - prior_size
                        states[key] = row
                        if len(states) > self.limits.max_bucket_rows:
                            raise ValueError(
                                f"citation state bucket {bucket} exceeded row limit; "
                                "increase bucket_count"
                            )
                        if state_bytes > self.limits.max_bucket_bytes:
                            raise ValueError(
                                f"citation state bucket {bucket} exceeded byte limit; "
                                "increase bucket_count"
                            )

            if not states:
                continue
            ordered = sorted(states.values(), key=_edge_sort_key)
            for start in range(0, len(ordered), self.limits.output_part_rows):
                rows = ordered[start : start + self.limits.output_part_rows]
                output_parts.append(
                    _write_state_part(
                        output_root,
                        bucket=bucket,
                        index=start // self.limits.output_part_rows,
                        rows=rows,
                    )
                )
            total_states += len(states)
            total_active += sum(not row["tombstone"] for row in states.values())
        return output_parts, total_states, total_active

    def _verify_state(self, receipt: CitationStateReceipt) -> CitationStateReceipt:
        if not isinstance(receipt, CitationStateReceipt) or receipt.source != self.source:
            raise ValueError("base citation state receipt is invalid")
        path = self._final_path(receipt.release, receipt.artifact_id)
        if receipt.path != path:
            raise ValueError("citation state receipt path is noncanonical")
        event_lineage = _read_json(path / "manifest.json").get("event_ledger")
        if not isinstance(event_lineage, Mapping):
            raise ValueError("citation state event lineage is invalid")
        verified = self._verify_state_path(
            path,
            release=receipt.release,
            artifact_id=receipt.artifact_id,
            event_manifest=event_lineage,
            base=None,
            already_materialized=True,
            allow_unchecked_base=True,
        )
        if (
            receipt.edge_count != verified.edge_count
            or receipt.active_count != verified.active_count
            or receipt.deleted_count != verified.deleted_count
            or receipt.part_count != verified.part_count
            or receipt.bucket_count != verified.bucket_count
        ):
            raise ValueError("citation state receipt counts do not match manifest")
        return verified

    def _verify_state_path(
        self,
        path: Path,
        *,
        release: str,
        artifact_id: str,
        event_manifest: Mapping[str, Any],
        base: CitationStateReceipt | None,
        already_materialized: bool,
        allow_unchecked_base: bool = False,
    ) -> CitationStateReceipt:
        if path.is_symlink() or not path.is_dir() or path != self._final_path(release, artifact_id):
            raise ValueError("citation state projection path is invalid")
        manifest = _read_json(path / "manifest.json")
        digest_values = dict(manifest)
        if digest_values.pop("manifest_sha256", None) != _sha256_json(digest_values):
            raise ValueError("citation state manifest checksum mismatch")
        expected_base = (
            None
            if base is None
            else {"artifact_id": base.artifact_id, "release": base.release}
        )
        manifest_base = manifest.get("base")
        if allow_unchecked_base:
            if manifest_base is not None and (
                not isinstance(manifest_base, Mapping)
                or set(manifest_base) != {"artifact_id", "release"}
            ):
                raise ValueError("citation state base lineage is invalid")
            expected_base = manifest_base
        elif manifest_base != expected_base:
            raise ValueError("citation state base lineage mismatch")
        if manifest.get("event_ledger") != dict(event_manifest):
            raise ValueError("citation state event ledger lineage mismatch")
        identity = {
            "algorithm": _ALGORITHM,
            "artifact_id": artifact_id,
            "format": _FORMAT,
            "release": release,
            "schema": str(_STATE_SCHEMA),
            "source": self.source,
            "layout": self._layout(),
        }
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError("citation state projection identity mismatch")
        expected_artifact = _sha256_json(
            {
                "format": _FORMAT,
                "algorithm": _ALGORITHM,
                "source": self.source,
                "release": release,
                "event_ledger_artifact_id": event_manifest["artifact_id"],
                "base_artifact_id": (
                    None if expected_base is None else expected_base["artifact_id"]
                ),
                "layout": self._layout(),
                "schema": str(_STATE_SCHEMA),
            }
        )
        if expected_artifact != artifact_id:
            raise ValueError("citation state artifact ID does not bind its inputs")
        parts = manifest.get("parts")
        if not isinstance(parts, list) or manifest.get("part_count") != len(parts):
            raise ValueError("citation state part count mismatch")
        rows_seen = active_seen = 0
        prior_position: tuple[int, int] | None = None
        for part in parts:
            if not isinstance(part, Mapping) or set(part) != {
                "bucket",
                "path",
                "row_count",
                "byte_count",
                "sha256",
            }:
                raise ValueError("citation state part entry is invalid")
            relative = part.get("path")
            match = _PART_PATH.fullmatch(relative) if isinstance(relative, str) else None
            bucket = part.get("bucket")
            if (
                match is None
                or isinstance(bucket, bool)
                or not isinstance(bucket, int)
                or bucket < 0
                or bucket >= int(manifest["layout"]["bucket_count"])
                or relative != f"bucket-{bucket:05d}/part-{int(match.group(2)):06d}.parquet"
            ):
                raise ValueError("citation state part path is noncanonical")
            position = (bucket, int(match.group(2)))
            if prior_position is not None and position <= prior_position:
                raise ValueError("citation state part order is invalid")
            prior_position = position
            part_path = path / "parts" / relative
            if (
                part_path.is_symlink()
                or not part_path.is_file()
                or part_path.stat().st_size != part.get("byte_count")
                or _file_sha256(part_path) != part.get("sha256")
            ):
                raise ValueError("citation state part checksum mismatch")
            metadata = pq.read_metadata(part_path)
            if (
                metadata.schema.to_arrow_schema() != _STATE_SCHEMA
                or metadata.num_rows != part.get("row_count")
                or metadata.num_rows < 1
            ):
                raise ValueError("citation state part schema or row count mismatch")
            for batch in pq.ParquetFile(part_path).iter_batches(
                columns=("citing_paper_id", "cited_paper_id", "tombstone", "source_record_id"),
                batch_size=self.limits.scan_batch_rows,
            ):
                if batch.nbytes > self.limits.max_scan_batch_bytes:
                    raise ValueError("citation state verification batch exceeded byte limit")
                for row in batch.to_pylist():
                    if row["source_record_id"] != (
                        f"semantic-scholar:citation:{row['citing_paper_id']}:{row['cited_paper_id']}"
                    ):
                        raise ValueError("citation state exact ID pair is inconsistent")
                    rows_seen += 1
                    active_seen += not row["tombstone"]
        deleted_seen = rows_seen - active_seen
        if (
            rows_seen != manifest.get("row_count")
            or active_seen != manifest.get("active_count")
            or deleted_seen != manifest.get("deleted_count")
        ):
            raise ValueError("citation state summary counts mismatch")
        receipt = CitationStateReceipt(
            source=self.source,
            release=release,
            artifact_id=artifact_id,
            edge_count=rows_seen,
            active_count=active_seen,
            deleted_count=deleted_seen,
            part_count=len(parts),
            bucket_count=int(manifest["layout"]["bucket_count"]),
            path=path,
            already_materialized=already_materialized,
        )
        return receipt

    def _final_path(self, release: str, artifact_id: str) -> Path:
        return (
            self.output_root
            / _path_component(self.source)
            / _path_component(release)
            / artifact_id
        )

    def _layout(self) -> dict[str, int]:
        return {
            "bucket_count": self.limits.bucket_count,
            "event_partition_rows": self.limits.event_partition_rows,
            "event_partition_bytes": self.limits.event_partition_bytes,
            "scan_batch_rows": self.limits.scan_batch_rows,
            "max_scan_batch_bytes": self.limits.max_scan_batch_bytes,
            "max_bucket_rows": self.limits.max_bucket_rows,
            "max_bucket_bytes": self.limits.max_bucket_bytes,
            "output_part_rows": self.limits.output_part_rows,
            "max_output_row_bytes": self.limits.max_output_row_bytes,
        }


def _state_row(event: Mapping[str, Any], event_artifact_id: str) -> dict[str, Any]:
    return {
        "citing_paper_id": event["citing_paper_id"],
        "cited_paper_id": event["cited_paper_id"],
        "tombstone": event["operation"] == "delete",
        "last_operation": event["operation"],
        "event_ledger_artifact_id": event_artifact_id,
        "event_index": event["event_index"],
        "source_record_id": event["source_record_id"],
        "source_shard": event["source_shard"],
        "source_row_index": event["source_row_index"],
        "application_order_json": event["application_order_json"],
        "source_content_sha256": event["source_content_sha256"],
        "payload_json": event["payload_json"],
        "ingested_at": event["ingested_at"],
    }


def _write_state_part(
    parts_root: Path,
    *,
    bucket: int,
    index: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    bucket_root = parts_root / f"bucket-{bucket:05d}"
    bucket_root.mkdir(exist_ok=True)
    name = f"part-{index:06d}.parquet"
    path = bucket_root / name
    pq.write_table(
        pa.Table.from_pylist(rows, schema=_STATE_SCHEMA),
        path,
        compression="zstd",
        write_statistics=True,
    )
    return {
        "bucket": bucket,
        "path": f"{bucket_root.name}/{name}",
        "row_count": len(rows),
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _bucket(citing_id: str, cited_id: str, bucket_count: int) -> int:
    digest = hashlib.sha256(f"{citing_id}\0{cited_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % bucket_count


def _edge_sort_key(row: Mapping[str, Any]) -> tuple[int, str, int, str]:
    citing = str(row["citing_paper_id"])
    cited = str(row["cited_paper_id"])
    return len(citing), citing, len(cited), cited


def _memory_size(row: Mapping[str, Any]) -> int:
    payload = row.get("payload_json", "")
    return len(payload.encode("utf-8")) + 512


def _row_size(row: Mapping[str, Any]) -> int:
    return _memory_size(row)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def _path_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not component or component in {".", ".."}:
        raise ValueError("path component is invalid")
    return component[:160]


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("citation projection manifest is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("citation projection manifest must be an object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _event_lineage(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_id": manifest["artifact_id"],
        "release": manifest["release"],
        "transitions": manifest["transitions"],
    }
