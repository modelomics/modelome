from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.lake import ParquetLandingZone
from modelome.semantic_scholar_bulk import citation_edge_ids

_FORMAT = "modelome-semantic-scholar-citation-events-v2"
_ALGORITHM = "exact-corpus-id-citation-event-stream-v2"
_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_EDGE_SCHEMA = pa.schema(
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


@dataclass(frozen=True, slots=True)
class CitationProjectionLimits:
    scan_batch_rows: int = 4_096
    max_scan_batch_bytes: int = 128 * 1024 * 1024
    output_part_rows: int = 25_000
    max_output_buffer_bytes: int = 128 * 1024 * 1024
    max_output_row_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "scan_batch_rows",
            "max_scan_batch_bytes",
            "output_part_rows",
            "max_output_buffer_bytes",
            "max_output_row_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_output_row_bytes > self.max_output_buffer_bytes:
            raise ValueError("max_output_row_bytes must not exceed buffer bytes")


@dataclass(frozen=True, slots=True)
class CitationProjectionReceipt:
    source: str
    release: str
    artifact_id: str
    event_count: int
    part_count: int
    path: Path
    already_materialized: bool = False


class SemanticScholarCitationMaterializer:
    """Project sealed citation shards to a bounded exact-ID event stream.

    The output is an ordered event ledger, not an active graph snapshot. It
    retains every upsert/delete with the source shard, row position, and lake
    application order, so downstream consumers can replay diffs without
    treating citations as paper or model identity.
    """

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        output_root: str | Path | None = None,
        source: str = "semantic-scholar",
        limits: CitationProjectionLimits | None = None,
    ) -> None:
        if not isinstance(landing_zone, ParquetLandingZone):
            raise TypeError("landing_zone must be a ParquetLandingZone")
        self.landing_zone = landing_zone
        self.source = _required_text(source, "source")
        self.limits = limits or CitationProjectionLimits()
        self.output_root = (
            landing_zone.root / "citation-projections"
            if output_root is None
            else Path(output_root).expanduser().absolute()
        )

    def materialize(self, release: str) -> CitationProjectionReceipt:
        release = _required_text(release, "release")
        self.landing_zone.initialize()
        seal_path = self.landing_zone._release_path(
            self.source, "citations", release
        ) / "RELEASE.json"
        seal = self.landing_zone._verify_release(
            seal_path, self.source, "citations", release
        )
        verified_shards = self.landing_zone._verify_sealed_release_shards(seal)
        artifact_id = self._artifact_id(release, seal)
        final_path = self._final_path(release, artifact_id)
        if final_path.exists() or final_path.is_symlink():
            return self._verify_projection(
                final_path,
                release=release,
                artifact_id=artifact_id,
                seal=seal,
                already_materialized=True,
            )

        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{artifact_id}.", dir=final_path.parent)
        )
        try:
            parts_dir = temporary / "parts"
            parts_dir.mkdir()
            parts: list[dict[str, Any]] = []
            buffer: list[dict[str, Any]] = []
            buffer_bytes = 0
            event_index = 0
            part_index = 0
            for shard_entry, shard_path, shard_manifest in verified_shards:
                application_order = shard_entry["application_order"]
                if not isinstance(application_order, Mapping):
                    raise ValueError("citation shard application order is invalid")
                shard_name = str(shard_entry["shard"])
                shard_row_index = 0
                for part_entry in shard_manifest["parts"]:
                    part_path = shard_path / "parts" / str(part_entry["name"])
                    for batch in pq.ParquetFile(part_path).iter_batches(
                        columns=(
                            "source_record_id",
                            "operation",
                            "payload_json",
                            "content_sha256",
                            "ingested_at",
                        ),
                        batch_size=self.limits.scan_batch_rows,
                    ):
                        if batch.nbytes > self.limits.max_scan_batch_bytes:
                            raise ValueError("citation input batch exceeded byte limit")
                        for raw in batch.to_pylist():
                            row = self._event_row(
                                raw,
                                event_index=event_index,
                                shard=shard_name,
                                shard_row_index=shard_row_index,
                                application_order=application_order,
                            )
                            row_bytes = _row_size(row)
                            if row_bytes > self.limits.max_output_row_bytes:
                                raise ValueError("citation event exceeded row byte limit")
                            if buffer and (
                                len(buffer) >= self.limits.output_part_rows
                                or buffer_bytes + row_bytes
                                > self.limits.max_output_buffer_bytes
                            ):
                                parts.append(_write_part(parts_dir, part_index, buffer))
                                part_index += 1
                                buffer = []
                                buffer_bytes = 0
                            buffer.append(row)
                            buffer_bytes += row_bytes
                            event_index += 1
                            shard_row_index += 1
            if buffer:
                parts.append(_write_part(parts_dir, part_index, buffer))
            if event_index != int(seal["row_count"]):
                raise ValueError("citation release row count drift")

            manifest: dict[str, Any] = {
                "algorithm": _ALGORITHM,
                "artifact_id": artifact_id,
                "created_at": datetime.now(UTC).isoformat(),
                "format": _FORMAT,
                "input": {
                    "application_mode": seal["application_mode"],
                    "dataset": "citations",
                    "release": release,
                    "release_sha256": seal["release_sha256"],
                    "shard_count": seal["shard_count"],
                    "sealed_row_count": seal["row_count"],
                },
                "layout": self._layout(),
                "part_count": len(parts),
                "parts": parts,
                "release": release,
                "row_count": event_index,
                "schema": str(_EDGE_SCHEMA),
                "source": self.source,
                "transitions": _transitions_from_seal(seal),
            }
            manifest["manifest_sha256"] = _sha256_json(manifest)
            _write_json(temporary / "manifest.json", manifest)
            os.replace(temporary, final_path)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        return self._verify_projection(
            final_path,
            release=release,
            artifact_id=artifact_id,
            seal=seal,
            already_materialized=False,
        )

    def _event_row(
        self,
        raw: Mapping[str, Any],
        *,
        event_index: int,
        shard: str,
        shard_row_index: int,
        application_order: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError("citation lake row must be an object")
        payload_json = raw.get("payload_json")
        if not isinstance(payload_json, str):
            raise ValueError("citation lake payload must be JSON text")
        payload = _json_mapping(payload_json)
        citing_id, cited_id = citation_edge_ids(payload)
        expected_id = f"semantic-scholar:citation:{citing_id}:{cited_id}"
        source_record_id = _required_text(raw.get("source_record_id"), "source_record_id")
        if source_record_id != expected_id:
            raise ValueError("citation source record ID does not match its edge IDs")
        operation = raw.get("operation")
        if operation not in {"upsert", "delete"}:
            raise ValueError("citation operation must be upsert or delete")
        content_sha = _required_text(raw.get("content_sha256"), "content_sha256")
        if content_sha != hashlib.sha256(payload_json.encode("utf-8")).hexdigest():
            raise ValueError("citation lake row content checksum mismatch")
        ingested_at = raw.get("ingested_at")
        if not isinstance(ingested_at, datetime):
            raise ValueError("citation lake row has no ingestion timestamp")
        return {
            "event_index": event_index,
            "source_record_id": source_record_id,
            "operation": operation,
            "citing_paper_id": citing_id,
            "cited_paper_id": cited_id,
            "source_shard": shard,
            "source_row_index": shard_row_index,
            "application_order_json": json.dumps(
                dict(application_order), sort_keys=True, separators=(",", ":")
            ),
            "source_content_sha256": content_sha,
            "payload_json": payload_json,
            "ingested_at": ingested_at,
        }

    def _verify_projection(
        self,
        path: Path,
        *,
        release: str,
        artifact_id: str,
        seal: Mapping[str, Any],
        already_materialized: bool,
    ) -> CitationProjectionReceipt:
        if path.is_symlink() or not path.is_dir() or path != self._final_path(release, artifact_id):
            raise ValueError("citation projection path is invalid")
        manifest = _read_json(path / "manifest.json")
        digest_value = dict(manifest)
        digest = digest_value.pop("manifest_sha256", None)
        if digest != _sha256_json(digest_value):
            raise ValueError("citation projection manifest checksum mismatch")
        expected = {
            "algorithm": _ALGORITHM,
            "artifact_id": artifact_id,
            "format": _FORMAT,
            "release": release,
            "schema": str(_EDGE_SCHEMA),
            "source": self.source,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("citation projection identity mismatch")
        expected_layout = self._layout()
        if manifest.get("layout") != expected_layout:
            raise ValueError("citation projection layout mismatch")
        if self._artifact_id(release, seal) != artifact_id:
            raise ValueError("citation projection artifact ID does not bind its input")
        expected_input = {
            "application_mode": seal["application_mode"],
            "dataset": "citations",
            "release": release,
            "release_sha256": seal["release_sha256"],
            "shard_count": seal["shard_count"],
            "sealed_row_count": seal["row_count"],
        }
        if manifest.get("input") != expected_input:
            raise ValueError("citation projection input identity mismatch")
        if manifest.get("transitions") != _transitions_from_seal(seal):
            raise ValueError("citation projection transition lineage mismatch")
        parts = manifest.get("parts")
        if not isinstance(parts, list) or manifest.get("part_count") != len(parts):
            raise ValueError("citation projection part count mismatch")
        rows_seen = 0
        for index, part in enumerate(parts):
            if not isinstance(part, Mapping) or part.get("name") != f"part-{index:06d}.parquet":
                raise ValueError("citation projection part ordering is invalid")
            part_path = path / "parts" / str(part["name"])
            if (
                part_path.is_symlink()
                or not part_path.is_file()
                or part_path.stat().st_size != part.get("byte_count")
                or _file_sha256(part_path) != part.get("sha256")
            ):
                raise ValueError("citation projection part checksum mismatch")
            metadata = pq.read_metadata(part_path)
            if (
                metadata.schema.to_arrow_schema() != _EDGE_SCHEMA
                or metadata.num_rows != part.get("row_count")
            ):
                raise ValueError("citation projection part schema or row count mismatch")
            for batch in pq.ParquetFile(part_path).iter_batches(
                columns=(
                    "event_index",
                    "source_record_id",
                    "operation",
                    "citing_paper_id",
                    "cited_paper_id",
                ),
                batch_size=self.limits.scan_batch_rows,
            ):
                for row in batch.to_pylist():
                    if row["event_index"] != rows_seen:
                        raise ValueError("citation projection event order is invalid")
                    if row["operation"] not in {"upsert", "delete"}:
                        raise ValueError("citation projection operation is invalid")
                    expected_record_id = (
                        f"semantic-scholar:citation:{row['citing_paper_id']}:{row['cited_paper_id']}"
                    )
                    if row["source_record_id"] != expected_record_id:
                        raise ValueError("citation projection exact IDs do not match")
                    rows_seen += 1
        if rows_seen != manifest.get("row_count") or rows_seen != seal["row_count"]:
            raise ValueError("citation projection row count mismatch")
        return CitationProjectionReceipt(
            source=self.source,
            release=release,
            artifact_id=artifact_id,
            event_count=rows_seen,
            part_count=len(parts),
            path=path,
            already_materialized=already_materialized,
        )

    def open_projection(self, release: str) -> CitationProjectionReceipt:
        """Open the deterministic event ledger for one sealed citation release."""

        release = _required_text(release, "release")
        seal_path = self.landing_zone._release_path(
            self.source, "citations", release
        ) / "RELEASE.json"
        seal = self.landing_zone._verify_release(
            seal_path, self.source, "citations", release
        )
        artifact_id = self._artifact_id(release, seal)
        return self._verify_projection(
            self._final_path(release, artifact_id),
            release=release,
            artifact_id=artifact_id,
            seal=seal,
            already_materialized=True,
        )

    def _final_path(self, release: str, artifact_id: str) -> Path:
        source = _component(self.source)
        release_component = _component(release)
        return self.output_root / source / release_component / artifact_id

    def _layout(self) -> dict[str, int]:
        return {
            "scan_batch_rows": self.limits.scan_batch_rows,
            "max_scan_batch_bytes": self.limits.max_scan_batch_bytes,
            "output_part_rows": self.limits.output_part_rows,
            "max_output_buffer_bytes": self.limits.max_output_buffer_bytes,
            "max_output_row_bytes": self.limits.max_output_row_bytes,
        }

    def _artifact_id(self, release: str, seal: Mapping[str, Any]) -> str:
        return _sha256_json(
            {
                "format": _FORMAT,
                "algorithm": _ALGORITHM,
                "source": self.source,
                "release": release,
                "release_sha256": seal["release_sha256"],
                "schema": str(_EDGE_SCHEMA),
                "layout": self._layout(),
            }
        )


def _write_part(parts_dir: Path, index: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    name = f"part-{index:06d}.parquet"
    path = parts_dir / name
    pq.write_table(
        pa.Table.from_pylist(rows, schema=_EDGE_SCHEMA),
        path,
        compression="zstd",
        write_statistics=True,
    )
    return {
        "name": name,
        "row_count": len(rows),
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _transitions_from_seal(seal: Mapping[str, Any]) -> list[dict[str, Any]]:
    if seal.get("application_mode") == "snapshot":
        return []
    transitions: dict[int, dict[str, Any]] = {}
    for shard in seal["shards"]:
        order = shard["application_order"]
        index = int(order["diff_index"])
        transition = {
            "diff_index": index,
            "from_release": order["from_release"],
            "to_release": order["to_release"],
        }
        prior = transitions.get(index)
        if prior is not None and prior != transition:
            raise ValueError("citation release has contradictory diff transitions")
        transitions[index] = transition
    return [transitions[index] for index in sorted(transitions)]


def _row_size(row: Mapping[str, Any]) -> int:
    return len(
        json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )


def _json_mapping(value: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        raise ValueError("citation payload is not valid JSON") from None
    if not isinstance(parsed, Mapping):
        raise ValueError("citation payload must be an object")
    return parsed


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def _component(value: str) -> str:
    component = _COMPONENT.sub("-", value).strip(".-")
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("citation projection manifest is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise ValueError("citation projection manifest is invalid") from None
    if not isinstance(value, dict):
        raise ValueError("citation projection manifest must be an object")
    return value
