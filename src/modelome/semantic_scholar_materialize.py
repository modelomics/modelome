from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.lake import ParquetLandingZone

_FORMAT = "modelome-semantic-scholar-projection-v2"
_ALGORITHM = "sha256-corpus-bucket-stateful-join-v3"
_LEGACY_ALGORITHMS = frozenset({"sha256-corpus-bucket-stateful-join-v2"})
_DATASETS = ("papers", "abstracts", "paper-ids")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")

_WORK_SCHEMA = pa.schema(
    [
        pa.field("corpus_id", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("operation", pa.string(), nullable=False),
        pa.field("payload_json", pa.large_string(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("ingested_at", pa.string(), nullable=False),
        pa.field("sha_alias", pa.string()),
        pa.field("application_order_json", pa.large_string(), nullable=False),
        pa.field("lake_locator", pa.large_string(), nullable=False),
    ]
)

_ALIAS_SCHEMA = pa.schema(
    [
        pa.field("sha_alias", pa.string(), nullable=False),
        pa.field("corpus_id", pa.string(), nullable=False),
    ]
)

PROJECTION_SCHEMA = pa.schema(
    [
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("release", pa.string(), nullable=False),
        pa.field("operation", pa.string(), nullable=False),
        pa.field("tombstone", pa.bool_(), nullable=False),
        pa.field("corpus_id", pa.string(), nullable=False),
        pa.field("title", pa.large_string()),
        pa.field("abstract", pa.large_string()),
        pa.field("text", pa.large_string()),
        pa.field("external_ids_json", pa.large_string(), nullable=False),
        pa.field("authors_json", pa.large_string(), nullable=False),
        pa.field("venue", pa.large_string()),
        pa.field("year", pa.int32()),
        pa.field("publication_date", pa.string()),
        pa.field("canonical_url", pa.large_string()),
        pa.field("urls_json", pa.large_string(), nullable=False),
        pa.field("sha_aliases", pa.list_(pa.string()), nullable=False),
        pa.field("raw_payload_json", pa.large_string(), nullable=False),
        pa.field("evidence_json", pa.large_string(), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class MaterializationLimits:
    """Explicit row and serialized-byte ceilings for bounded materialization."""

    bucket_count: int = 256
    scan_batch_rows: int = 4_096
    max_scan_batch_bytes: int = 128 * 1024 * 1024
    partition_buffer_rows: int = 4_096
    partition_buffer_bytes: int = 128 * 1024 * 1024
    join_batch_rows: int = 4_096
    max_join_batch_bytes: int = 128 * 1024 * 1024
    max_bucket_rows: int = 2_000_000
    max_bucket_payload_bytes: int = 512 * 1024 * 1024
    max_payload_json_bytes: int = 16 * 1024 * 1024
    output_part_rows: int = 25_000
    max_output_buffer_bytes: int = 128 * 1024 * 1024
    max_output_row_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for field in (
            "bucket_count",
            "scan_batch_rows",
            "max_scan_batch_bytes",
            "partition_buffer_rows",
            "partition_buffer_bytes",
            "join_batch_rows",
            "max_join_batch_bytes",
            "max_bucket_rows",
            "max_bucket_payload_bytes",
            "max_payload_json_bytes",
            "output_part_rows",
            "max_output_buffer_bytes",
            "max_output_row_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.bucket_count > 65_536:
            raise ValueError("bucket_count must not exceed 65536")
        if self.max_payload_json_bytes > self.partition_buffer_bytes:
            raise ValueError(
                "max_payload_json_bytes must not exceed partition_buffer_bytes"
            )
        if self.max_output_row_bytes > self.max_output_buffer_bytes:
            raise ValueError(
                "max_output_row_bytes must not exceed max_output_buffer_bytes"
            )


@dataclass(frozen=True, slots=True)
class ProjectionReceipt:
    source: str
    release: str
    artifact_id: str
    row_count: int
    part_count: int
    bucket_count: int
    path: Path
    already_materialized: bool = False


@dataclass(frozen=True, slots=True)
class ProjectionPlan:
    source: str
    release: str
    mode: str
    base_release: str | None
    transitions: tuple[tuple[int, str, str], ...]
    input_release_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _InputRow:
    corpus_id: str
    source_record_id: str
    operation: str
    payload_json: str
    payload: Mapping[str, Any]
    content_sha256: str
    ingested_at: str
    sha_alias: str | None = None
    application_order: Mapping[str, Any] | None = None
    lake_locator: str = ""

    def work_row(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "source_record_id": self.source_record_id,
            "operation": self.operation,
            "payload_json": self.payload_json,
            "content_sha256": self.content_sha256,
            "ingested_at": self.ingested_at,
            "sha_alias": self.sha_alias,
            "application_order_json": _canonical_json(
                dict(self.application_order or {})
            ),
            "lake_locator": self.lake_locator,
        }


@dataclass(frozen=True, slots=True)
class _WorkingPart:
    path: Path
    row_count: int
    byte_count: int
    sha256: str


@dataclass(slots=True)
class _PaperState:
    corpus_id: str
    paper: _InputRow | None = None
    abstract: _InputRow | None = None
    aliases: dict[str, _InputRow] | None = None
    tombstone: _InputRow | None = None
    applied_events: list[tuple[str, _InputRow]] | None = None

    def __post_init__(self) -> None:
        if self.aliases is None:
            self.aliases = {}
        if self.applied_events is None:
            self.applied_events = []


@dataclass(frozen=True, slots=True)
class _ScannedBatch:
    batch: pa.RecordBatch
    application_order: Mapping[str, Any]
    locator_prefix: str
    first_row: int


class SemanticScholarProjectionMaterializer:
    """External-memory stateful join for the three core S2AG datasets.

    The input scanner first hashes rows into deterministic corpus-ID buckets on
    disk. Only one bounded bucket is then held in Python while papers, abstracts,
    and SHA aliases are joined. Incremental releases require an explicit prior
    projection and replay their sealed transition chain into immutable new state.
    """

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        output_root: str | Path | None = None,
        source: str = "semantic-scholar",
        limits: MaterializationLimits | None = None,
    ) -> None:
        if not isinstance(landing_zone, ParquetLandingZone):
            raise TypeError("landing_zone must be a ParquetLandingZone")
        self.landing_zone = landing_zone
        self.source = _required_text(source, "source")
        self.limits = limits or MaterializationLimits()
        root = (
            landing_zone.root / "projections"
            if output_root is None
            else Path(output_root).expanduser().absolute()
        )
        self.output_root = root

    def materialize(
        self,
        release: str,
        *,
        base: ProjectionReceipt | None = None,
    ) -> ProjectionReceipt:
        """Materialize one complete snapshot or replay one sealed diff atomically."""

        release = _required_text(release, "release")
        self.landing_zone.initialize()
        self._initialize_output()
        input_seals = self._verified_input_seals(release, verify_shards=False)
        mode = _input_mode(input_seals)
        base_manifest: Mapping[str, Any] | None = None
        if mode == "snapshot":
            if base is not None:
                raise ValueError("snapshot materialization must not specify a base")
            lineage = _snapshot_lineage(release)
        else:
            if base is None:
                raise ValueError("diff materialization requires a base projection")
            base_manifest = self._verify_receipt(base)
            transitions = _shared_diff_transitions(input_seals, release)
            if base.release != transitions[0]["from_release"]:
                raise ValueError(
                    "diff base projection release does not match the first transition"
                )
            lineage = _diff_lineage(release, base, transitions)
        artifact_id = self._artifact_id(release, input_seals, mode, lineage)
        final_dir = self._final_path(release, artifact_id)
        if final_dir.exists() or final_dir.is_symlink():
            self._verify_input_shards(release, input_seals)
            manifest = self._verify_projection(
                final_dir,
                release=release,
                artifact_id=artifact_id,
                input_seals=input_seals,
                expected_lineage=lineage,
            )
            return _receipt(final_dir, manifest, already_materialized=True)

        final_parent = final_dir.parent
        final_parent.mkdir(parents=True, exist_ok=True)
        _require_plain_directory(
            final_parent.parent,
            "projection source directory",
        )
        _require_plain_directory(final_parent, "projection release directory")
        stage = Path(
            tempfile.mkdtemp(
                prefix="semantic-scholar-",
                dir=self.output_root / ".staging",
            )
        )
        try:
            work_root = stage / "work"
            parts_root = stage / "parts"
            work_root.mkdir()
            parts_root.mkdir()
            working, input_stats = self._partition_inputs(
                release=release,
                work_root=work_root,
                input_seals=input_seals,
                mode=mode,
            )
            parts, row_count = self._join_buckets(
                release=release,
                artifact_id=artifact_id,
                working=working,
                input_seals=input_seals,
                parts_root=parts_root,
                mode=mode,
                lineage=lineage,
                base=None if base is None else (base, base_manifest),
            )
            alias_root = stage / "alias-validation"
            self._validate_global_alias_ownership(
                projection_root=stage,
                parts=parts,
                work_root=alias_root,
            )
            shutil.rmtree(work_root)
            shutil.rmtree(alias_root)
            created_at = datetime.now(UTC).isoformat()
            manifest = {
                "format": _FORMAT,
                "algorithm": _ALGORITHM,
                "artifact_id": artifact_id,
                "source": self.source,
                "release": release,
                "projection_mode": mode,
                "lineage": lineage,
                "layout": self._layout(),
                "schema": str(PROJECTION_SCHEMA),
                "inputs": input_stats,
                "bucket_count": self.limits.bucket_count,
                "row_count": row_count,
                "part_count": len(parts),
                "parts": parts,
                "created_at": created_at,
            }
            manifest["manifest_sha256"] = _json_sha256(manifest)
            _write_json_exclusive(stage / "manifest.json", manifest)
            seal = {
                "format": _FORMAT,
                "artifact_id": artifact_id,
                "manifest_sha256": manifest["manifest_sha256"],
                "data_sha256": _json_sha256({"parts": parts}),
                "row_count": row_count,
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
                existing = self._verify_projection(
                    final_dir,
                    release=release,
                    artifact_id=artifact_id,
                    input_seals=input_seals,
                    expected_lineage=lineage,
                )
                return _receipt(final_dir, existing, already_materialized=True)
            _fsync_directory(final_parent)
            verified = self._verify_projection(
                final_dir,
                release=release,
                artifact_id=artifact_id,
                input_seals=input_seals,
                expected_lineage=lineage,
            )
            return _receipt(final_dir, verified, already_materialized=False)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def plan(self, release: str) -> ProjectionPlan:
        """Verify and describe the sealed lake inputs needed for one projection."""

        release = _required_text(release, "release")
        self.landing_zone.initialize()
        input_seals = self._verified_input_seals(release, verify_shards=True)
        mode = _input_mode(input_seals)
        if mode == "snapshot":
            base_release = None
            transitions: tuple[tuple[int, str, str], ...] = ()
        else:
            values = _shared_diff_transitions(input_seals, release)
            base_release = str(values[0]["from_release"])
            transitions = tuple(
                (
                    int(value["diff_index"]),
                    str(value["from_release"]),
                    str(value["to_release"]),
                )
                for value in values
            )
        return ProjectionPlan(
            source=self.source,
            release=release,
            mode=mode,
            base_release=base_release,
            transitions=transitions,
            input_release_sha256=tuple(
                (
                    dataset,
                    str(input_seals[dataset]["release_sha256"]),
                )
                for dataset in _DATASETS
            ),
        )

    def list_ready_releases(self) -> tuple[str, ...]:
        """List releases with valid seals for all three required lake datasets."""

        self.landing_zone.initialize()
        by_dataset: dict[str, dict[str, Mapping[str, Any]]] = {}
        for dataset in _DATASETS:
            root = (
                self.landing_zone.shards_root
                / _path_component(self.source)
                / _path_component(dataset)
            )
            if not root.exists():
                return ()
            _require_plain_directory(root, f"{dataset} lake release directory")
            releases: dict[str, Mapping[str, Any]] = {}
            for path in sorted(root.iterdir(), key=lambda item: item.name):
                if path.is_symlink() or not path.is_dir():
                    raise ValueError(f"{dataset} lake release directory is tampered")
                seal_path = path / "RELEASE.json"
                if not seal_path.exists():
                    continue
                preview = _read_json(seal_path)
                release = _required_release_date(
                    preview.get("release"),
                    "lake release",
                )
                if path != self.landing_zone._release_path(
                    self.source,
                    dataset,
                    release,
                ):
                    raise ValueError(f"{dataset} lake release path is noncanonical")
                if release in releases:
                    raise ValueError(f"{dataset} lake contains a duplicate release")
                releases[release] = self.landing_zone._verify_release(
                    seal_path,
                    self.source,
                    dataset,
                    release,
                )
            by_dataset[dataset] = releases
        common = set.intersection(
            *(set(by_dataset[dataset]) for dataset in _DATASETS)
        )
        for release in common:
            seals = {
                dataset: by_dataset[dataset][release] for dataset in _DATASETS
            }
            mode = _input_mode(seals)
            if mode == "diff":
                _shared_diff_transitions(seals, release)
        return tuple(sorted(common))

    def latest_ready_release(self) -> str | None:
        """Return the newest fully sealed three-dataset release, if one exists."""

        releases = self.list_ready_releases()
        return releases[-1] if releases else None

    def list_projections(self, release: str) -> tuple[ProjectionReceipt, ...]:
        """Reopen every compatible sealed projection for a release from disk."""

        release = _required_text(release, "release")
        self.landing_zone.initialize()
        self._initialize_output()
        parent = self._final_path(release, "0" * 64).parent
        if not parent.exists():
            return ()
        _require_plain_directory(parent.parent, "projection source directory")
        _require_plain_directory(parent, "projection release directory")
        input_seals = self._verified_input_seals(release, verify_shards=True)
        receipts: list[ProjectionReceipt] = []
        for path in sorted(parent.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_dir() or _SHA256.fullmatch(path.name) is None:
                raise ValueError(f"projection release directory is tampered: {path}")
            preview = _read_json(path / "manifest.json")
            algorithm = preview.get("algorithm")
            if algorithm in _LEGACY_ALGORITHMS:
                # Validate retired artifacts fully, but do not expose them as
                # current projections: their URL derivation predates v3.
                self._verify_projection(
                    path,
                    release=release,
                    artifact_id=path.name,
                    input_seals=input_seals,
                    algorithm=algorithm,
                )
                continue
            manifest = self._verify_projection(
                path,
                release=release,
                artifact_id=path.name,
                input_seals=input_seals,
            )
            receipts.append(_receipt(path, manifest, already_materialized=True))
        return tuple(receipts)

    def open_projection(
        self,
        release: str,
        *,
        artifact_id: str | None = None,
    ) -> ProjectionReceipt:
        """Reopen one sealed projection, failing on absence or ambiguous lineage."""

        release = _required_text(release, "release")
        if artifact_id is not None:
            artifact = _digest(artifact_id, "artifact_id")
            self.landing_zone.initialize()
            self._initialize_output()
            input_seals = self._verified_input_seals(release, verify_shards=True)
            path = self._final_path(release, artifact)
            manifest = self._verify_projection(
                path,
                release=release,
                artifact_id=artifact,
                input_seals=input_seals,
            )
            return _receipt(path, manifest, already_materialized=True)
        receipts = self.list_projections(release)
        if not receipts:
            raise ValueError(f"no sealed projection exists for release {release}")
        if len(receipts) != 1:
            raise ValueError(
                f"release {release} has multiple sealed projections; "
                "artifact_id is required"
            )
        return receipts[0]

    def iter_batches(
        self,
        receipt: ProjectionReceipt,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Verify and lazily scan one materialized projection."""

        if not isinstance(receipt, ProjectionReceipt):
            raise TypeError("receipt must be a ProjectionReceipt")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        input_seals = self._verified_input_seals(
            receipt.release,
            verify_shards=True,
        )
        manifest = self._verify_projection(
            receipt.path,
            release=receipt.release,
            artifact_id=receipt.artifact_id,
            input_seals=input_seals,
        )
        if (
            receipt.source != self.source
            or receipt.row_count != manifest["row_count"]
            or receipt.part_count != manifest["part_count"]
            or receipt.bucket_count != manifest["bucket_count"]
        ):
            raise ValueError("projection receipt does not match its sealed artifact")
        for part in manifest["parts"]:
            path = receipt.path / str(part["path"])
            yield from pq.ParquetFile(path).iter_batches(
                columns=columns,
                batch_size=batch_size,
            )

    def _initialize_output(self) -> None:
        if self.output_root.exists() and not self.output_root.is_dir():
            raise ValueError(f"projection root is not a directory: {self.output_root}")
        if self.output_root.is_symlink():
            raise ValueError(f"projection root must not be a symlink: {self.output_root}")
        shards_root = self.landing_zone.shards_root
        try:
            inside_shards = (
                self.output_root == shards_root
                or shards_root in self.output_root.parents
            )
        except RuntimeError:
            inside_shards = True
        if inside_shards:
            raise ValueError("projection root must not be inside landing-zone shards")
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.output_root.resolve() != self.output_root:
            raise ValueError(
                f"projection root ancestors must not redirect through symlinks: "
                f"{self.output_root}"
            )
        staging = self.output_root / ".staging"
        staging.mkdir(exist_ok=True)
        _require_plain_directory(self.output_root, "projection root")
        _require_plain_directory(staging, "projection staging directory")

    def _verified_input_seals(
        self,
        release: str,
        *,
        verify_shards: bool,
    ) -> dict[str, dict[str, Any]]:
        seals: dict[str, dict[str, Any]] = {}
        for dataset in _DATASETS:
            path = self.landing_zone._release_path(self.source, dataset, release)
            seal = self.landing_zone._verify_release(
                path / "RELEASE.json",
                self.source,
                dataset,
                release,
            )
            seals[dataset] = seal
        _input_mode(seals)
        if verify_shards:
            self._verify_input_shards(release, seals)
        return seals

    def _verify_receipt(self, receipt: ProjectionReceipt) -> Mapping[str, Any]:
        if not isinstance(receipt, ProjectionReceipt):
            raise TypeError("base must be a ProjectionReceipt")
        if receipt.source != self.source:
            raise ValueError("base projection source does not match materializer source")
        seals = self._verified_input_seals(receipt.release, verify_shards=True)
        manifest = self._verify_projection(
            receipt.path,
            release=receipt.release,
            artifact_id=receipt.artifact_id,
            input_seals=seals,
        )
        if (
            receipt.row_count != manifest["row_count"]
            or receipt.part_count != manifest["part_count"]
            or receipt.bucket_count != manifest["bucket_count"]
        ):
            raise ValueError("base projection receipt does not match its artifact")
        return manifest

    def _verify_input_shards(
        self,
        release: str,
        seals: Mapping[str, Mapping[str, Any]],
    ) -> None:
        for dataset in _DATASETS:
            for item in seals[dataset]["shards"]:
                identity = {
                    "source": self.source,
                    "dataset": dataset,
                    "release": release,
                    "shard": str(item["shard"]),
                    "control_sha256": str(item["control_sha256"]),
                    "upstream_sha256": str(item["upstream_sha256"]),
                }
                shard_path = self.landing_zone.root / str(item["path"])
                shard = self.landing_zone._verify_shard(shard_path, identity)
                for field in (
                    "application_order",
                    "upstream_url",
                    "upstream_bytes",
                    "row_count",
                ):
                    if shard[field] != item[field]:
                        raise ValueError(
                            f"release seal does not match {dataset} shard manifest"
                        )

    def _iter_input_batches(
        self,
        *,
        dataset: str,
        release: str,
        seal: Mapping[str, Any],
    ) -> Iterator[_ScannedBatch]:
        columns = (
            "source_record_id",
            "operation",
            "payload_json",
            "content_sha256",
            "ingested_at",
        )
        for item in seal["shards"]:
            identity = {
                "source": self.source,
                "dataset": dataset,
                "release": release,
                "shard": str(item["shard"]),
                "control_sha256": str(item["control_sha256"]),
                "upstream_sha256": str(item["upstream_sha256"]),
            }
            shard_path = self.landing_zone.root / str(item["path"])
            manifest = self.landing_zone._verify_shard(shard_path, identity)
            for field in (
                "application_order",
                "upstream_url",
                "upstream_bytes",
                "row_count",
            ):
                if manifest[field] != item[field]:
                    raise ValueError(
                        f"release seal does not match {dataset} shard manifest"
                    )
            order = item["application_order"]
            for part in manifest["parts"]:
                part_path = shard_path / "parts" / str(part["name"])
                row_offset = 0
                for batch in pq.ParquetFile(part_path).iter_batches(
                    columns=columns,
                    batch_size=self.limits.scan_batch_rows,
                ):
                    yield _ScannedBatch(
                        batch=batch,
                        application_order=order,
                        locator_prefix=(
                            f"{item['path']}/parts/{part['name']}"
                        ),
                        first_row=row_offset,
                    )
                    row_offset += batch.num_rows

    def _artifact_id(
        self,
        release: str,
        input_seals: Mapping[str, Mapping[str, Any]],
        mode: str,
        lineage: Mapping[str, Any],
        *,
        algorithm: str | None = None,
    ) -> str:
        algorithm = _ALGORITHM if algorithm is None else algorithm
        return _json_sha256(
            {
                "format": _FORMAT,
                "algorithm": algorithm,
                "source": self.source,
                "release": release,
                "projection_mode": mode,
                "lineage": dict(lineage),
                "inputs": {
                    dataset: input_seals[dataset]["release_sha256"]
                    for dataset in _DATASETS
                },
                "layout": self._layout(),
                "schema": str(PROJECTION_SCHEMA),
            }
        )

    def _layout(self) -> dict[str, int]:
        return {
            "bucket_count": self.limits.bucket_count,
            "output_part_rows": self.limits.output_part_rows,
            "max_output_buffer_bytes": self.limits.max_output_buffer_bytes,
        }

    def _final_path(self, release: str, artifact_id: str) -> Path:
        return (
            self.output_root
            / _path_component(self.source)
            / _path_component(release)
            / artifact_id
        )

    def _partition_inputs(
        self,
        *,
        release: str,
        work_root: Path,
        input_seals: Mapping[str, Mapping[str, Any]],
        mode: str,
    ) -> tuple[dict[str, dict[int, list[_WorkingPart]]], dict[str, dict[str, Any]]]:
        working: dict[str, dict[int, list[_WorkingPart]]] = {
            dataset: {} for dataset in _DATASETS
        }
        stats: dict[str, dict[str, Any]] = {}
        for dataset in _DATASETS:
            dataset_root = work_root / dataset
            dataset_root.mkdir()
            buffers: dict[int, list[_InputRow]] = {}
            buffered_rows = 0
            buffered_bytes = 0
            part_indexes: dict[int, int] = {}
            row_count = 0
            content_digest = hashlib.sha256()
            for scanned in self._iter_input_batches(
                dataset=dataset,
                release=release,
                seal=input_seals[dataset],
            ):
                batch = scanned.batch
                if batch.nbytes > self.limits.max_scan_batch_bytes:
                    raise ValueError(
                        f"{dataset} input batch exceeded max_scan_batch_bytes"
                    )
                for index in range(batch.num_rows):
                    values = {
                        name: batch.column(position)[index].as_py()
                        for position, name in enumerate(batch.schema.names)
                    }
                    row = _decode_input_row(
                        dataset,
                        values,
                        max_json_bytes=self.limits.max_payload_json_bytes,
                        mode=mode,
                        application_order=scanned.application_order,
                        lake_locator=(
                            f"{scanned.locator_prefix}#row="
                            f"{scanned.first_row + index}"
                        ),
                    )
                    row_bytes = _input_row_bytes(row)
                    if row_bytes > self.limits.partition_buffer_bytes:
                        raise ValueError(
                            f"{dataset} row exceeded partition_buffer_bytes"
                        )
                    if buffers and (
                        buffered_rows >= self.limits.partition_buffer_rows
                        or buffered_bytes + row_bytes
                        > self.limits.partition_buffer_bytes
                    ):
                        self._flush_partition_buffers(
                            dataset_root,
                            buffers,
                            working[dataset],
                            part_indexes,
                        )
                        buffers = {}
                        buffered_rows = 0
                        buffered_bytes = 0
                    bucket = _bucket(row.corpus_id, self.limits.bucket_count)
                    buffers.setdefault(bucket, []).append(row)
                    buffered_rows += 1
                    buffered_bytes += row_bytes
                    row_count += 1
                    _update_framed_digest(content_digest, _input_digest_value(row))
            if buffers:
                self._flush_partition_buffers(
                    dataset_root,
                    buffers,
                    working[dataset],
                    part_indexes,
                )
            expected_rows = int(input_seals[dataset]["row_count"])
            if row_count != expected_rows:
                raise ValueError(
                    f"{dataset} release row count drift: expected "
                    f"{expected_rows}, read {row_count}"
                )
            stats[dataset] = {
                "dataset": dataset,
                "release": release,
                "application_mode": mode,
                "release_sha256": input_seals[dataset]["release_sha256"],
                "shard_count": input_seals[dataset]["shard_count"],
                "sealed_row_count": expected_rows,
                "validated_row_count": row_count,
                "validated_content_sha256": content_digest.hexdigest(),
            }
        return working, stats

    def _flush_partition_buffers(
        self,
        dataset_root: Path,
        buffers: Mapping[int, list[_InputRow]],
        parts_by_bucket: dict[int, list[_WorkingPart]],
        indexes: dict[int, int],
    ) -> None:
        for bucket in sorted(buffers):
            rows = buffers[bucket]
            if not rows:
                continue
            bucket_root = dataset_root / f"bucket-{bucket:06d}"
            bucket_root.mkdir(exist_ok=True)
            index = indexes.get(bucket, 0)
            path = bucket_root / f"part-{index:06d}.parquet"
            table = pa.Table.from_pylist(
                [row.work_row() for row in rows],
                schema=_WORK_SCHEMA,
            )
            pq.write_table(table, path, compression="zstd", write_statistics=True)
            part = _WorkingPart(
                path=path,
                row_count=len(rows),
                byte_count=path.stat().st_size,
                sha256=_file_sha256(path),
            )
            parts_by_bucket.setdefault(bucket, []).append(part)
            indexes[bucket] = index + 1

    def _join_buckets(
        self,
        *,
        release: str,
        artifact_id: str,
        working: Mapping[str, Mapping[int, list[_WorkingPart]]],
        input_seals: Mapping[str, Mapping[str, Any]],
        parts_root: Path,
        mode: str,
        lineage: Mapping[str, Any],
        base: tuple[ProjectionReceipt, Mapping[str, Any] | None] | None,
    ) -> tuple[list[dict[str, Any]], int]:
        if mode == "diff":
            if base is None or base[1] is None:
                raise ValueError("diff replay is missing its verified base projection")
            return self._replay_diff_buckets(
                release=release,
                artifact_id=artifact_id,
                working=working,
                input_seals=input_seals,
                parts_root=parts_root,
                lineage=lineage,
                base_receipt=base[0],
                base_manifest=base[1],
            )
        if base is not None:
            raise ValueError("snapshot join unexpectedly received a base projection")
        output_parts: list[dict[str, Any]] = []
        output_rows = 0
        for bucket in range(self.limits.bucket_count):
            papers = self._read_bucket_dataset(working["papers"].get(bucket, []), "papers")
            abstracts = self._read_bucket_dataset(
                working["abstracts"].get(bucket, []), "abstracts"
            )
            paper_ids = self._read_bucket_dataset(
                working["paper-ids"].get(bucket, []), "paper-ids"
            )
            total_rows = len(papers) + len(abstracts) + len(paper_ids)
            total_bytes = sum(
                len(row.payload_json.encode("utf-8"))
                for rows in (papers.values(), abstracts.values(), paper_ids.values())
                for row in rows
            )
            if total_rows > self.limits.max_bucket_rows:
                raise ValueError(
                    f"bucket {bucket} exceeded max_bucket_rows; increase bucket_count"
                )
            if total_bytes > self.limits.max_bucket_payload_bytes:
                raise ValueError(
                    f"bucket {bucket} exceeded max_bucket_payload_bytes; "
                    "increase bucket_count"
                )

            aliases: dict[str, list[_InputRow]] = {}
            sha_owners: dict[str, str] = {}
            for row in paper_ids.values():
                if row.sha_alias is None:
                    raise ValueError("paper-ids row lost its SHA alias")
                owner = sha_owners.get(row.sha_alias)
                if owner is not None and owner != row.corpus_id:
                    raise ValueError(
                        f"SHA alias {row.sha_alias} maps to contradictory corpus IDs"
                    )
                sha_owners[row.sha_alias] = row.corpus_id
                aliases.setdefault(row.corpus_id, []).append(row)
            unanchored = (set(abstracts) | set(aliases)) - set(papers)
            if unanchored:
                sample = min(unanchored, key=_corpus_sort_key)
                raise ValueError(f"bucket {bucket} has no paper anchor for corpus {sample}")

            buffer: list[dict[str, Any]] = []
            buffer_bytes = 0
            part_index = 0
            for corpus_id in sorted(papers, key=_corpus_sort_key):
                row = _projection_row(
                    source=self.source,
                    release=release,
                    artifact_id=artifact_id,
                    input_seals=input_seals,
                    paper=papers[corpus_id],
                    abstract=abstracts.get(corpus_id),
                    aliases=sorted(
                        aliases.get(corpus_id, []),
                        key=lambda item: item.sha_alias or "",
                    ),
                    lineage=lineage,
                    applied_events=(),
                )
                row_bytes = _projection_row_bytes(row)
                if row_bytes > self.limits.max_output_row_bytes:
                    raise ValueError(
                        f"projection row {corpus_id} exceeded max_output_row_bytes"
                    )
                if buffer and (
                    len(buffer) >= self.limits.output_part_rows
                    or buffer_bytes + row_bytes
                    > self.limits.max_output_buffer_bytes
                ):
                    output_parts.append(
                        _write_projection_part(
                            parts_root,
                            bucket=bucket,
                            part_index=part_index,
                            rows=buffer,
                        )
                    )
                    output_rows += len(buffer)
                    part_index += 1
                    buffer = []
                    buffer_bytes = 0
                buffer.append(row)
                buffer_bytes += row_bytes
            if buffer:
                output_parts.append(
                    _write_projection_part(
                        parts_root,
                        bucket=bucket,
                        part_index=part_index,
                        rows=buffer,
                    )
                )
                output_rows += len(buffer)
        return output_parts, output_rows

    def _replay_diff_buckets(
        self,
        *,
        release: str,
        artifact_id: str,
        working: Mapping[str, Mapping[int, list[_WorkingPart]]],
        input_seals: Mapping[str, Mapping[str, Any]],
        parts_root: Path,
        lineage: Mapping[str, Any],
        base_receipt: ProjectionReceipt,
        base_manifest: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        base_parts: dict[int, list[Mapping[str, Any]]] = {}
        for part in base_manifest["parts"]:
            base_parts.setdefault(int(part["bucket"]), []).append(part)
        transition_count = len(lineage["transitions"])
        output_parts: list[dict[str, Any]] = []
        output_rows = 0
        for bucket in range(self.limits.bucket_count):
            states = self._read_base_bucket(
                base_receipt.path,
                base_parts.get(bucket, []),
                bucket,
            )
            events = {
                dataset: self._read_bucket_events(
                    working[dataset].get(bucket, []), dataset
                )
                for dataset in _DATASETS
            }
            event_rows = sum(len(values) for values in events.values())
            event_bytes = sum(
                len(row.payload_json.encode("utf-8"))
                for values in events.values()
                for row in values
            )
            state_bytes = sum(_paper_state_bytes(state) for state in states.values())
            if len(states) + event_rows > self.limits.max_bucket_rows:
                raise ValueError(
                    f"diff bucket {bucket} exceeded max_bucket_rows; "
                    "increase bucket_count"
                )
            if state_bytes + event_bytes > self.limits.max_bucket_payload_bytes:
                raise ValueError(
                    f"diff bucket {bucket} exceeded max_bucket_payload_bytes; "
                    "increase bucket_count"
                )
            scheduled = _schedule_diff_events(events, transition_count)
            for dataset, operation, row in scheduled:
                self._apply_diff_event(states, dataset, operation, row)
            _validate_final_states(states, bucket)

            buffer: list[dict[str, Any]] = []
            buffer_bytes = 0
            part_index = 0
            for corpus_id in sorted(states, key=_corpus_sort_key):
                state = states[corpus_id]
                aliases = state.aliases or {}
                applied_events = state.applied_events or []
                if state.tombstone is not None:
                    row = _tombstone_projection_row(
                        source=self.source,
                        release=release,
                        artifact_id=artifact_id,
                        input_seals=input_seals,
                        lineage=lineage,
                        tombstone=state.tombstone,
                        applied_events=applied_events,
                    )
                else:
                    if state.paper is None:
                        raise ValueError(
                            f"active corpus {corpus_id} has no paper state"
                        )
                    row = _projection_row(
                        source=self.source,
                        release=release,
                        artifact_id=artifact_id,
                        input_seals=input_seals,
                        paper=state.paper,
                        abstract=state.abstract,
                        aliases=sorted(
                            aliases.values(),
                            key=lambda item: item.sha_alias or "",
                        ),
                        lineage=lineage,
                        applied_events=applied_events,
                    )
                row_bytes = _projection_row_bytes(row)
                if row_bytes > self.limits.max_output_row_bytes:
                    raise ValueError(
                        f"projection row {corpus_id} exceeded max_output_row_bytes"
                    )
                if buffer and (
                    len(buffer) >= self.limits.output_part_rows
                    or buffer_bytes + row_bytes
                    > self.limits.max_output_buffer_bytes
                ):
                    output_parts.append(
                        _write_projection_part(
                            parts_root,
                            bucket=bucket,
                            part_index=part_index,
                            rows=buffer,
                        )
                    )
                    output_rows += len(buffer)
                    part_index += 1
                    buffer = []
                    buffer_bytes = 0
                buffer.append(row)
                buffer_bytes += row_bytes
            if buffer:
                output_parts.append(
                    _write_projection_part(
                        parts_root,
                        bucket=bucket,
                        part_index=part_index,
                        rows=buffer,
                    )
                )
                output_rows += len(buffer)
        return output_parts, output_rows

    def _read_base_bucket(
        self,
        root: Path,
        parts: Sequence[Mapping[str, Any]],
        bucket: int,
    ) -> dict[str, _PaperState]:
        states: dict[str, _PaperState] = {}
        payload_bytes = 0
        for part in parts:
            path = root / str(part["path"])
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=self.limits.join_batch_rows
            ):
                if batch.nbytes > self.limits.max_join_batch_bytes:
                    raise ValueError("base projection batch exceeded byte limit")
                for raw in _iter_arrow_rows(batch):
                    state = _state_from_projection_row(raw)
                    if _bucket(state.corpus_id, self.limits.bucket_count) != bucket:
                        raise ValueError("base projection state is in the wrong bucket")
                    if state.corpus_id in states:
                        raise ValueError("base projection contains a duplicate corpus ID")
                    payload_bytes += _paper_state_bytes(state)
                    if len(states) + 1 > self.limits.max_bucket_rows:
                        raise ValueError("base projection bucket exceeded max_bucket_rows")
                    if payload_bytes > self.limits.max_bucket_payload_bytes:
                        raise ValueError(
                            "base projection bucket exceeded max_bucket_payload_bytes"
                        )
                    states[state.corpus_id] = state
        return states

    def _read_bucket_events(
        self,
        parts: Sequence[_WorkingPart],
        dataset: str,
    ) -> list[_InputRow]:
        events: list[_InputRow] = []
        payload_bytes = 0
        seen: dict[tuple[int, str, str], _InputRow] = {}
        prior_order: tuple[int, int, int] | None = None
        for expected_index, part in enumerate(parts):
            expected_name = f"part-{expected_index:06d}.parquet"
            if part.path.name != expected_name:
                raise ValueError(f"{dataset} diff bucket has a noncanonical part path")
            _verify_working_part(part)
            for batch in pq.ParquetFile(part.path).iter_batches(
                batch_size=self.limits.join_batch_rows
            ):
                if batch.nbytes > self.limits.max_join_batch_bytes:
                    raise ValueError(f"{dataset} diff batch exceeded byte limit")
                for raw in _iter_arrow_rows(batch):
                    row = _decode_work_row(dataset, raw)
                    order = row.application_order
                    if not isinstance(order, Mapping) or order.get("mode") != "diff":
                        raise ValueError(f"{dataset} diff row lost application order")
                    diff_index = _nonnegative_integer(
                        order.get("diff_index"), "diff_index"
                    )
                    order_key = (
                        diff_index,
                        0 if row.operation == "upsert" else 1,
                        _nonnegative_integer(
                            order.get("operation_index"),
                            "operation_index",
                        ),
                    )
                    if prior_order is not None and order_key < prior_order:
                        raise ValueError(f"{dataset} diff event order moved backward")
                    prior_order = order_key
                    key = row.sha_alias if dataset == "paper-ids" else row.corpus_id
                    if key is None:
                        raise ValueError(f"{dataset} diff row has no key")
                    identity = (diff_index, row.operation, key)
                    previous = seen.get(identity)
                    if previous is not None:
                        if not _same_input_record(previous, row):
                            raise ValueError(
                                f"{dataset} has contradictory {row.operation} "
                                f"events for {key} in transition {diff_index}"
                            )
                        continue
                    seen[identity] = row
                    events.append(row)
                    payload_bytes += len(row.payload_json.encode("utf-8"))
                    if len(events) > self.limits.max_bucket_rows:
                        raise ValueError(f"{dataset} diff bucket exceeded row limit")
                    if payload_bytes > self.limits.max_bucket_payload_bytes:
                        raise ValueError(f"{dataset} diff bucket exceeded byte limit")
        return events

    def _apply_diff_event(
        self,
        states: dict[str, _PaperState],
        dataset: str,
        operation: str,
        row: _InputRow,
    ) -> None:
        state = states.get(row.corpus_id)
        if operation == "upsert" and state is None:
            state = _PaperState(corpus_id=row.corpus_id)
            states[row.corpus_id] = state
        if state is None:
            raise ValueError(
                f"{dataset} delete references missing corpus {row.corpus_id}"
            )
        if state.applied_events is None or state.aliases is None:
            raise ValueError("paper state was not initialized")
        state.applied_events.append((dataset, row))
        if dataset == "papers":
            if operation == "upsert":
                state.paper = row
                state.tombstone = None
                return
            if state.paper is None or state.tombstone is not None:
                raise ValueError(
                    f"paper delete references inactive corpus {row.corpus_id}"
                )
            state.paper = None
            state.abstract = None
            state.aliases.clear()
            state.tombstone = row
            return
        if state.paper is None or state.tombstone is not None:
            raise ValueError(
                f"{dataset} event references inactive corpus {row.corpus_id}"
            )
        if dataset == "abstracts":
            if operation == "upsert":
                state.abstract = row
            elif state.abstract is None:
                raise ValueError(
                    f"abstract delete references missing corpus {row.corpus_id}"
                )
            else:
                state.abstract = None
            return
        if row.sha_alias is None:
            raise ValueError("paper-ids event has no SHA alias")
        if operation == "upsert":
            state.aliases[row.sha_alias] = row
        elif row.sha_alias not in state.aliases:
            raise ValueError(
                f"paper-ids delete references missing SHA {row.sha_alias}"
            )
        else:
            del state.aliases[row.sha_alias]

    def _validate_global_alias_ownership(
        self,
        *,
        projection_root: Path,
        parts: Sequence[Mapping[str, Any]],
        work_root: Path,
    ) -> None:
        """Externalize a second hash pass so SHA ownership is globally unique."""

        work_root.mkdir()
        buffers: dict[int, list[dict[str, str]]] = {}
        buffered_rows = 0
        buffered_bytes = 0
        indexes: dict[int, int] = {}
        alias_parts: dict[int, list[_WorkingPart]] = {}

        def flush() -> None:
            nonlocal buffers, buffered_rows, buffered_bytes
            for alias_bucket in sorted(buffers):
                rows = buffers[alias_bucket]
                bucket_root = work_root / f"bucket-{alias_bucket:06d}"
                bucket_root.mkdir(exist_ok=True)
                index = indexes.get(alias_bucket, 0)
                path = bucket_root / f"part-{index:06d}.parquet"
                table = pa.Table.from_pylist(rows, schema=_ALIAS_SCHEMA)
                pq.write_table(
                    table,
                    path,
                    compression="zstd",
                    write_statistics=True,
                )
                alias_parts.setdefault(alias_bucket, []).append(
                    _WorkingPart(
                        path=path,
                        row_count=len(rows),
                        byte_count=path.stat().st_size,
                        sha256=_file_sha256(path),
                    )
                )
                indexes[alias_bucket] = index + 1
            buffers = {}
            buffered_rows = 0
            buffered_bytes = 0

        for part in parts:
            path = projection_root / str(part["path"])
            for batch in pq.ParquetFile(path).iter_batches(
                columns=("corpus_id", "tombstone", "sha_aliases"),
                batch_size=self.limits.scan_batch_rows,
            ):
                if batch.nbytes > self.limits.max_scan_batch_bytes:
                    raise ValueError("alias validation batch exceeded byte limit")
                for row in _iter_arrow_rows(batch):
                    corpus_id = _normalize_corpus_id(
                        row.get("corpus_id"), "alias validation"
                    )
                    aliases = row.get("sha_aliases")
                    if not isinstance(aliases, list):
                        raise ValueError("projection SHA aliases must be an array")
                    if row.get("tombstone") is True and aliases:
                        raise ValueError("projection tombstone retained SHA aliases")
                    for raw_sha in aliases:
                        if not isinstance(raw_sha, str):
                            raise ValueError("projection SHA alias must be text")
                        sha = raw_sha.casefold()
                        if raw_sha != sha or _SHA1.fullmatch(sha) is None:
                            raise ValueError("projection SHA alias is invalid")
                        row_bytes = len(sha) + len(corpus_id)
                        if buffers and (
                            buffered_rows >= self.limits.partition_buffer_rows
                            or buffered_bytes + row_bytes
                            > self.limits.partition_buffer_bytes
                        ):
                            flush()
                        alias_bucket = _bucket(sha, self.limits.bucket_count)
                        buffers.setdefault(alias_bucket, []).append(
                            {"sha_alias": sha, "corpus_id": corpus_id}
                        )
                        buffered_rows += 1
                        buffered_bytes += row_bytes
        if buffers:
            flush()

        for alias_bucket, bucket_parts in alias_parts.items():
            owners: dict[str, str] = {}
            payload_bytes = 0
            row_count = 0
            for expected_index, part in enumerate(bucket_parts):
                if part.path.name != f"part-{expected_index:06d}.parquet":
                    raise ValueError("alias validation part order is invalid")
                if part.path.is_symlink() or not part.path.is_file():
                    raise ValueError("alias validation part is missing")
                if (
                    part.path.stat().st_size != part.byte_count
                    or _file_sha256(part.path) != part.sha256
                ):
                    raise ValueError("alias validation part checksum drift")
                metadata = pq.read_metadata(part.path)
                if (
                    metadata.schema.to_arrow_schema() != _ALIAS_SCHEMA
                    or metadata.num_rows != part.row_count
                ):
                    raise ValueError("alias validation part metadata drift")
                for batch in pq.ParquetFile(part.path).iter_batches(
                    batch_size=self.limits.join_batch_rows
                ):
                    if batch.nbytes > self.limits.max_join_batch_bytes:
                        raise ValueError("alias ownership batch exceeded byte limit")
                    for row in _iter_arrow_rows(batch):
                        sha = str(row["sha_alias"])
                        corpus_id = str(row["corpus_id"])
                        row_count += 1
                        payload_bytes += len(sha) + len(corpus_id)
                        if row_count > self.limits.max_bucket_rows:
                            raise ValueError(
                                f"alias bucket {alias_bucket} exceeded row limit; "
                                "increase bucket_count"
                            )
                        if payload_bytes > self.limits.max_bucket_payload_bytes:
                            raise ValueError(
                                f"alias bucket {alias_bucket} exceeded byte limit; "
                                "increase bucket_count"
                            )
                        owner = owners.get(sha)
                        if owner is not None and owner != corpus_id:
                            raise ValueError(
                                f"SHA alias {sha} maps to contradictory corpus IDs"
                            )
                        owners[sha] = corpus_id

    def _read_bucket_dataset(
        self,
        parts: Sequence[_WorkingPart],
        dataset: str,
    ) -> dict[str, _InputRow]:
        rows: dict[str, _InputRow] = {}
        row_count = 0
        payload_bytes = 0
        for expected_index, part in enumerate(parts):
            expected_name = f"part-{expected_index:06d}.parquet"
            if part.path.name != expected_name:
                raise ValueError(f"{dataset} working bucket has a noncanonical part path")
            _verify_working_part(part)
            parquet = pq.ParquetFile(part.path)
            for batch in parquet.iter_batches(batch_size=self.limits.join_batch_rows):
                if batch.nbytes > self.limits.max_join_batch_bytes:
                    raise ValueError(
                        f"{dataset} join batch exceeded max_join_batch_bytes"
                    )
                for raw in _iter_arrow_rows(batch):
                    row = _decode_work_row(dataset, raw)
                    row_count += 1
                    payload_bytes += len(row.payload_json.encode("utf-8"))
                    if row_count > self.limits.max_bucket_rows:
                        raise ValueError(
                            f"{dataset} bucket exceeded max_bucket_rows"
                        )
                    if payload_bytes > self.limits.max_bucket_payload_bytes:
                        raise ValueError(
                            f"{dataset} bucket exceeded max_bucket_payload_bytes"
                        )
                    key = row.sha_alias if dataset == "paper-ids" else row.corpus_id
                    if key is None:
                        raise ValueError(f"{dataset} working row has no join key")
                    existing = rows.get(key)
                    if existing is None:
                        rows[key] = row
                    elif not _same_input_record(existing, row):
                        raise ValueError(
                            f"{dataset} contains contradictory duplicate {key}"
                        )
        return rows

    def _verify_projection(
        self,
        path: Path,
        *,
        release: str,
        artifact_id: str,
        input_seals: Mapping[str, Mapping[str, Any]],
        expected_lineage: Mapping[str, Any] | None = None,
        algorithm: str | None = None,
    ) -> dict[str, Any]:
        algorithm = _ALGORITHM if algorithm is None else algorithm
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"projection path is invalid: {path}")
        if path != self._final_path(release, artifact_id):
            raise ValueError("projection path does not match its identity")
        manifest = _read_json(path / "manifest.json")
        expected_manifest_fields = {
            "algorithm",
            "artifact_id",
            "bucket_count",
            "created_at",
            "format",
            "inputs",
            "lineage",
            "layout",
            "manifest_sha256",
            "part_count",
            "parts",
            "projection_mode",
            "release",
            "row_count",
            "schema",
            "source",
        }
        if set(manifest) != expected_manifest_fields:
            raise ValueError("projection manifest fields are invalid")
        digest_value = dict(manifest)
        digest = digest_value.pop("manifest_sha256", None)
        if digest != _json_sha256(digest_value):
            raise ValueError("projection manifest checksum mismatch")
        mode = _input_mode(input_seals)
        lineage = _validate_projection_lineage(
            manifest.get("lineage"),
            mode=mode,
            release=release,
            input_seals=input_seals,
        )
        if expected_lineage is not None and lineage != dict(expected_lineage):
            raise ValueError("projection lineage does not match requested replay")
        if self._artifact_id(
            release, input_seals, mode, lineage, algorithm=algorithm
        ) != artifact_id:
            raise ValueError("projection artifact ID does not bind its lineage")
        expected_identity = {
            "format": _FORMAT,
            "algorithm": algorithm,
            "artifact_id": artifact_id,
            "source": self.source,
            "release": release,
            "projection_mode": mode,
            "lineage": lineage,
            "layout": self._layout(),
            "schema": str(PROJECTION_SCHEMA),
            "bucket_count": self.limits.bucket_count,
        }
        if any(manifest.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("projection manifest identity mismatch")
        _required_text(manifest.get("created_at"), "created_at")
        inputs = manifest.get("inputs")
        if not isinstance(inputs, Mapping) or set(inputs) != set(_DATASETS):
            raise ValueError("projection input manifest is invalid")
        for dataset in _DATASETS:
            item = inputs[dataset]
            expected_fields = {
                "application_mode",
                "dataset",
                "release",
                "release_sha256",
                "sealed_row_count",
                "shard_count",
                "validated_content_sha256",
                "validated_row_count",
            }
            if not isinstance(item, Mapping) or set(item) != expected_fields:
                raise ValueError(f"projection {dataset} input fields are invalid")
            sealed_rows = _nonnegative_integer(
                item.get("sealed_row_count"), f"{dataset} sealed_row_count"
            )
            validated_rows = _nonnegative_integer(
                item.get("validated_row_count"), f"{dataset} validated_row_count"
            )
            if (
                item.get("dataset") != dataset
                or item.get("release") != release
                or item.get("application_mode") != mode
                or item.get("release_sha256")
                != input_seals[dataset]["release_sha256"]
                or item.get("shard_count") != input_seals[dataset]["shard_count"]
                or sealed_rows != input_seals[dataset]["row_count"]
                or validated_rows != sealed_rows
                or _digest(item.get("validated_content_sha256"), "input content")
                != item.get("validated_content_sha256")
            ):
                raise ValueError(f"projection {dataset} input identity mismatch")

        parts = manifest.get("parts")
        part_count = _nonnegative_integer(manifest.get("part_count"), "part_count")
        row_count = _nonnegative_integer(manifest.get("row_count"), "row_count")
        if not isinstance(parts, list) or len(parts) != part_count:
            raise ValueError("projection part count mismatch")
        parts_root = path / "parts"
        _require_plain_directory(parts_root, "projection parts directory")
        expected_directories: set[str] = set()
        expected_files: dict[str, set[str]] = {}
        seen_positions: set[tuple[int, int]] = set()
        rows_seen = 0
        prior_by_bucket: dict[int, str] = {}
        indexes_by_bucket: dict[int, list[int]] = {}
        for part in parts:
            if not isinstance(part, Mapping) or set(part) != {
                "bucket",
                "byte_count",
                "max_corpus_id",
                "min_corpus_id",
                "name",
                "path",
                "row_count",
                "sha256",
            }:
                raise ValueError("projection part entry is invalid")
            bucket = _nonnegative_integer(part.get("bucket"), "part bucket")
            if bucket >= self.limits.bucket_count:
                raise ValueError("projection part bucket is out of range")
            name = _required_text(part.get("name"), "part name")
            match = re.fullmatch(r"part-(\d{6})\.parquet", name)
            if match is None:
                raise ValueError("projection part name is noncanonical")
            index = int(match.group(1))
            position = (bucket, index)
            if position in seen_positions:
                raise ValueError("projection contains a duplicate part position")
            seen_positions.add(position)
            indexes_by_bucket.setdefault(bucket, []).append(index)
            relative = f"parts/bucket-{bucket:06d}/{name}"
            if part.get("path") != relative:
                raise ValueError("projection part path is noncanonical")
            part_path = path / relative
            if part_path.is_symlink() or not part_path.is_file():
                raise ValueError(f"projection part is missing: {part_path}")
            byte_count = _nonnegative_integer(part.get("byte_count"), "part byte_count")
            expected_rows = _nonnegative_integer(part.get("row_count"), "part row_count")
            if part_path.stat().st_size != byte_count:
                raise ValueError(f"projection part byte count mismatch: {part_path}")
            if _file_sha256(part_path) != _digest(part.get("sha256"), "part sha256"):
                raise ValueError(f"projection part checksum mismatch: {part_path}")
            metadata = pq.read_metadata(part_path)
            if metadata.schema.to_arrow_schema() != PROJECTION_SCHEMA:
                raise ValueError(f"projection part schema mismatch: {part_path}")
            if metadata.num_rows != expected_rows or expected_rows < 1:
                raise ValueError(f"projection part row count mismatch: {part_path}")
            observed = 0
            first: str | None = None
            last = prior_by_bucket.get(bucket)
            for batch in pq.ParquetFile(part_path).iter_batches(
                columns=(
                    "source_record_id",
                    "source",
                    "release",
                    "operation",
                    "tombstone",
                    "corpus_id",
                ),
                batch_size=self.limits.join_batch_rows,
            ):
                if batch.nbytes > self.limits.max_join_batch_bytes:
                    raise ValueError("projection verification batch exceeded byte limit")
                for row in _iter_arrow_rows(batch):
                    corpus_id = _normalize_corpus_id(row["corpus_id"], "projection")
                    if first is None:
                        first = corpus_id
                    if last is not None and _corpus_sort_key(corpus_id) <= _corpus_sort_key(last):
                        raise ValueError("projection corpus IDs are not strictly ordered")
                    if _bucket(corpus_id, self.limits.bucket_count) != bucket:
                        raise ValueError("projection row is stored in the wrong bucket")
                    operation = row["operation"]
                    tombstone = row["tombstone"]
                    if (
                        row["source_record_id"]
                        != f"semantic-scholar:corpus:{corpus_id}"
                        or row["source"] != self.source
                        or row["release"] != release
                        or (operation, tombstone)
                        not in {("upsert", False), ("delete", True)}
                    ):
                        raise ValueError("projection row identity is invalid")
                    last = corpus_id
                    observed += 1
            if observed != expected_rows:
                raise ValueError(f"projection part scan count mismatch: {part_path}")
            if first != part.get("min_corpus_id") or last != part.get("max_corpus_id"):
                raise ValueError(f"projection part corpus bounds mismatch: {part_path}")
            if last is None:
                raise ValueError(f"projection part unexpectedly contained no rows: {part_path}")
            prior_by_bucket[bucket] = last
            rows_seen += observed
            directory = f"bucket-{bucket:06d}"
            expected_directories.add(directory)
            expected_files.setdefault(directory, set()).add(name)
        for bucket, indexes in indexes_by_bucket.items():
            if indexes != list(range(len(indexes))):
                raise ValueError(
                    f"projection bucket {bucket} part indexes are gapped or unordered"
                )
        if rows_seen != row_count:
            raise ValueError("projection row count mismatch")
        actual_directories = {entry.name for entry in parts_root.iterdir()}
        if actual_directories != expected_directories:
            raise ValueError("projection parts directory has unexpected entries")
        for directory, names in expected_files.items():
            bucket_path = parts_root / directory
            _require_plain_directory(bucket_path, "projection bucket directory")
            if {entry.name for entry in bucket_path.iterdir()} != names:
                raise ValueError("projection bucket directory has unexpected entries")
        if {entry.name for entry in path.iterdir()} != {
            "manifest.json",
            "SEAL.json",
            "parts",
        }:
            raise ValueError("projection directory has unexpected entries")

        seal = _read_json(path / "SEAL.json")
        expected_seal_fields = {
            "artifact_id",
            "data_sha256",
            "format",
            "manifest_sha256",
            "part_count",
            "row_count",
            "seal_sha256",
            "sealed_at",
        }
        if set(seal) != expected_seal_fields:
            raise ValueError("projection seal fields are invalid")
        seal_value = dict(seal)
        seal_digest = seal_value.pop("seal_sha256", None)
        if seal_digest != _json_sha256(seal_value):
            raise ValueError("projection seal checksum mismatch")
        if (
            seal.get("format") != _FORMAT
            or seal.get("artifact_id") != artifact_id
            or seal.get("manifest_sha256") != manifest["manifest_sha256"]
            or seal.get("data_sha256") != _json_sha256({"parts": parts})
            or seal.get("row_count") != row_count
            or seal.get("part_count") != part_count
            or seal.get("sealed_at") != manifest["created_at"]
        ):
            raise ValueError("projection seal does not match its manifest")
        return manifest


def _decode_input_row(
    dataset: str,
    values: Mapping[str, Any],
    *,
    max_json_bytes: int,
    mode: str,
    application_order: Mapping[str, Any],
    lake_locator: str,
) -> _InputRow:
    payload_json = values.get("payload_json")
    if not isinstance(payload_json, str):
        raise ValueError(f"{dataset} payload_json must be text")
    if len(payload_json.encode("utf-8")) > max_json_bytes:
        raise ValueError(f"{dataset} payload_json exceeded max_payload_json_bytes")
    payload = _load_canonical_json_object(payload_json, dataset)
    content_sha256 = _digest(values.get("content_sha256"), "content_sha256")
    if hashlib.sha256(payload_json.encode()).hexdigest() != content_sha256:
        raise ValueError(f"{dataset} payload content checksum mismatch")
    operation = _required_text(values.get("operation"), "operation")
    order = _validate_application_order(
        application_order,
        expected_mode=mode,
        row_operation=operation,
    )
    corpus_id = _normalize_corpus_id(payload.get("corpusid"), dataset)
    sha_alias: str | None = None
    if dataset == "paper-ids":
        raw_sha = payload.get("sha")
        if not isinstance(raw_sha, str) or _SHA1.fullmatch(raw_sha.strip().casefold()) is None:
            raise ValueError("paper-ids row has no valid SHA-1 alias")
        sha_alias = raw_sha.strip().casefold()
        expected_id = f"semantic-scholar:paper-id:{sha_alias}"
    else:
        expected_id = f"semantic-scholar:corpus:{corpus_id}"
    source_record_id = _required_text(values.get("source_record_id"), "source_record_id")
    if source_record_id != expected_id:
        raise ValueError(f"{dataset} source_record_id does not match its payload")
    ingested_at = values.get("ingested_at")
    if not isinstance(ingested_at, datetime) or ingested_at.tzinfo is None:
        raise ValueError(f"{dataset} ingested_at must be timezone-aware")
    return _InputRow(
        corpus_id=corpus_id,
        source_record_id=source_record_id,
        operation=operation,
        payload_json=payload_json,
        payload=payload,
        content_sha256=content_sha256,
        ingested_at=ingested_at.astimezone(UTC).isoformat(),
        sha_alias=sha_alias,
        application_order=order,
        lake_locator=_required_text(lake_locator, "lake locator"),
    )


def _decode_work_row(dataset: str, values: Mapping[str, Any]) -> _InputRow:
    if set(values) != set(_WORK_SCHEMA.names):
        raise ValueError(f"{dataset} working row fields are invalid")
    payload_json = values.get("payload_json")
    if not isinstance(payload_json, str):
        raise ValueError(f"{dataset} working payload must be text")
    payload = _load_canonical_json_object(payload_json, dataset)
    order_json = values.get("application_order_json")
    if not isinstance(order_json, str):
        raise ValueError(f"{dataset} working application order must be text")
    order_value = _load_canonical_json_object(order_json, dataset)
    operation = _required_text(values.get("operation"), "operation")
    order = _validate_application_order(
        order_value,
        expected_mode=None,
        row_operation=operation,
    )
    row = _InputRow(
        corpus_id=_normalize_corpus_id(values.get("corpus_id"), dataset),
        source_record_id=_required_text(values.get("source_record_id"), "source_record_id"),
        operation=operation,
        payload_json=payload_json,
        payload=payload,
        content_sha256=_digest(values.get("content_sha256"), "content_sha256"),
        ingested_at=_required_text(values.get("ingested_at"), "ingested_at"),
        sha_alias=values.get("sha_alias"),
        application_order=order,
        lake_locator=_required_text(values.get("lake_locator"), "lake locator"),
    )
    expected = _decode_work_identity(dataset, row)
    if row.source_record_id != expected:
        raise ValueError(f"{dataset} working row identity mismatch")
    if hashlib.sha256(payload_json.encode()).hexdigest() != row.content_sha256:
        raise ValueError(f"{dataset} working row content checksum mismatch")
    return row


def _decode_work_identity(dataset: str, row: _InputRow) -> str:
    payload_corpus = _normalize_corpus_id(row.payload.get("corpusid"), dataset)
    if payload_corpus != row.corpus_id:
        raise ValueError(f"{dataset} working corpus ID mismatch")
    if dataset != "paper-ids":
        if row.sha_alias is not None:
            raise ValueError(f"{dataset} working row has a SHA alias")
        return f"semantic-scholar:corpus:{row.corpus_id}"
    raw_sha = row.payload.get("sha")
    if not isinstance(raw_sha, str):
        raise ValueError("paper-ids working row has no SHA alias")
    sha = raw_sha.strip().casefold()
    if _SHA1.fullmatch(sha) is None or row.sha_alias != sha:
        raise ValueError("paper-ids working SHA alias mismatch")
    return f"semantic-scholar:paper-id:{sha}"


def _projection_row(
    *,
    source: str,
    release: str,
    artifact_id: str,
    input_seals: Mapping[str, Mapping[str, Any]],
    paper: _InputRow,
    abstract: _InputRow | None,
    aliases: Sequence[_InputRow],
    lineage: Mapping[str, Any],
    applied_events: Sequence[tuple[str, _InputRow]],
) -> dict[str, Any]:
    paper_payload = paper.payload
    title = _optional_string(paper_payload.get("title"), "paper title")
    paper_abstract = _optional_string(paper_payload.get("abstract"), "paper abstract")
    abstract_text = None
    if abstract is not None:
        abstract_text = _optional_string(
            abstract.payload.get("abstract"), "abstract text"
        )
    if (
        paper_abstract is not None
        and abstract_text is not None
        and paper_abstract != abstract_text
    ):
        raise ValueError(
            f"paper and abstract datasets contradict corpus {paper.corpus_id}"
        )
    text = abstract_text if abstract_text is not None else paper_abstract
    external_ids = paper_payload.get("externalids")
    if external_ids is None:
        external_ids = {}
    if not isinstance(external_ids, Mapping):
        raise ValueError(f"paper {paper.corpus_id} externalids must be an object")
    authors = paper_payload.get("authors")
    if authors is None:
        authors = []
    if not isinstance(authors, list):
        raise ValueError(f"paper {paper.corpus_id} authors must be an array")
    venue = _optional_string(paper_payload.get("venue"), "paper venue")
    year = _optional_year(paper_payload.get("year"), paper.corpus_id)
    publication_date = _optional_publication_date(
        paper_payload.get("publicationdate"), paper.corpus_id
    )
    raw_payload = {
        "paper": dict(paper_payload),
        "abstract": None if abstract is None else dict(abstract.payload),
        "paper_ids": [dict(alias.payload) for alias in aliases],
        "tombstone": None,
        "applied_events": [
            {
                "dataset": dataset,
                "operation": event.operation,
                "payload": dict(event.payload),
            }
            for dataset, event in applied_events
        ],
    }
    urls = _extract_urls(raw_payload)
    urls.extend(_external_identifier_urls(external_ids))
    urls = _unique_urls(urls)
    direct_url = paper_payload.get("url")
    if direct_url is not None:
        if not isinstance(direct_url, str):
            raise ValueError(f"paper {paper.corpus_id} url must be text")
        _validate_web_url(direct_url, f"paper {paper.corpus_id} url")
    canonical_url = direct_url or (urls[0]["url"] if urls else None)
    evidence_inputs = [
        _evidence_item("papers", paper, "$.paper"),
    ]
    if abstract is not None:
        evidence_inputs.append(_evidence_item("abstracts", abstract, "$.abstract"))
    evidence_inputs.extend(
        _evidence_item("paper-ids", alias, f"$.paper_ids[{index}]")
        for index, alias in enumerate(aliases)
    )
    evidence = {
        "source": source,
        "release": release,
        "projection_artifact_id": artifact_id,
        "lineage": dict(lineage),
        "input_release_sha256": {
            dataset: input_seals[dataset]["release_sha256"]
            for dataset in _DATASETS
        },
        "inputs": evidence_inputs,
        "applied_events": [
            _evidence_item(
                dataset,
                event,
                f"$.applied_events[{index}].payload",
            )
            for index, (dataset, event) in enumerate(applied_events)
        ],
    }
    return {
        "source_record_id": paper.source_record_id,
        "source": source,
        "release": release,
        "operation": "upsert",
        "tombstone": False,
        "corpus_id": paper.corpus_id,
        "title": title,
        "abstract": text,
        "text": text,
        "external_ids_json": _canonical_json(dict(external_ids)),
        "authors_json": _canonical_json(authors),
        "venue": venue,
        "year": year,
        "publication_date": publication_date,
        "canonical_url": canonical_url,
        "urls_json": _canonical_json(urls),
        "sha_aliases": [alias.sha_alias for alias in aliases],
        "raw_payload_json": _canonical_json(raw_payload),
        "evidence_json": _canonical_json(evidence),
    }


def _tombstone_projection_row(
    *,
    source: str,
    release: str,
    artifact_id: str,
    input_seals: Mapping[str, Mapping[str, Any]],
    lineage: Mapping[str, Any],
    tombstone: _InputRow,
    applied_events: Sequence[tuple[str, _InputRow]],
) -> dict[str, Any]:
    if tombstone.operation != "delete":
        raise ValueError("paper tombstone must originate from a delete operation")
    raw_payload = {
        "paper": None,
        "abstract": None,
        "paper_ids": [],
        "tombstone": dict(tombstone.payload),
        "applied_events": [
            {
                "dataset": dataset,
                "operation": event.operation,
                "payload": dict(event.payload),
            }
            for dataset, event in applied_events
        ],
    }
    evidence = {
        "source": source,
        "release": release,
        "projection_artifact_id": artifact_id,
        "lineage": dict(lineage),
        "input_release_sha256": {
            dataset: input_seals[dataset]["release_sha256"]
            for dataset in _DATASETS
        },
        "inputs": [_evidence_item("papers", tombstone, "$.tombstone")],
        "applied_events": [
            _evidence_item(
                dataset,
                event,
                f"$.applied_events[{index}].payload",
            )
            for index, (dataset, event) in enumerate(applied_events)
        ],
    }
    return {
        "source_record_id": f"semantic-scholar:corpus:{tombstone.corpus_id}",
        "source": source,
        "release": release,
        "operation": "delete",
        "tombstone": True,
        "corpus_id": tombstone.corpus_id,
        "title": None,
        "abstract": None,
        "text": None,
        "external_ids_json": "{}",
        "authors_json": "[]",
        "venue": None,
        "year": None,
        "publication_date": None,
        "canonical_url": None,
        "urls_json": "[]",
        "sha_aliases": [],
        "raw_payload_json": _canonical_json(raw_payload),
        "evidence_json": _canonical_json(evidence),
    }
def _evidence_item(dataset: str, row: _InputRow, locator: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "source_record_id": row.source_record_id,
        "content_sha256": row.content_sha256,
        "ingested_at": row.ingested_at,
        "operation": row.operation,
        "application_order": dict(row.application_order or {}),
        "lake_locator": row.lake_locator,
        "raw_payload_locator": locator,
    }


def _state_from_projection_row(value: Mapping[str, Any]) -> _PaperState:
    if set(value) != set(PROJECTION_SCHEMA.names):
        raise ValueError("base projection row fields are invalid")
    corpus_id = _normalize_corpus_id(value.get("corpus_id"), "base projection")
    if value.get("source_record_id") != f"semantic-scholar:corpus:{corpus_id}":
        raise ValueError("base projection source record identity is invalid")
    raw_json = value.get("raw_payload_json")
    evidence_json = value.get("evidence_json")
    if not isinstance(raw_json, str) or not isinstance(evidence_json, str):
        raise ValueError("base projection raw payload and evidence must be text")
    raw = _load_canonical_json_object(raw_json, "base projection raw")
    evidence = _load_canonical_json_object(
        evidence_json,
        "base projection evidence",
    )
    if set(raw) != {
        "abstract",
        "applied_events",
        "paper",
        "paper_ids",
        "tombstone",
    }:
        raise ValueError("base projection raw state fields are invalid")
    evidence_inputs = evidence.get("inputs")
    if not isinstance(evidence_inputs, list):
        raise ValueError("base projection evidence inputs must be an array")
    by_locator: dict[str, Mapping[str, Any]] = {}
    for item in evidence_inputs:
        if not isinstance(item, Mapping):
            raise ValueError("base projection evidence input must be an object")
        locator = _required_text(item.get("raw_payload_locator"), "raw payload locator")
        if locator in by_locator:
            raise ValueError("base projection has duplicate evidence locators")
        by_locator[locator] = item
    raw_events = raw.get("applied_events")
    evidence_events = evidence.get("applied_events")
    if not isinstance(raw_events, list) or not isinstance(evidence_events, list):
        raise ValueError("base projection applied-event audit is invalid")
    if len(raw_events) != len(evidence_events):
        raise ValueError("base projection applied-event audit count mismatch")

    tombstone_flag = value.get("tombstone")
    operation = value.get("operation")
    state = _PaperState(corpus_id=corpus_id)
    if tombstone_flag is True:
        if operation != "delete":
            raise ValueError("base tombstone operation is invalid")
        if raw.get("paper") is not None or raw.get("abstract") is not None:
            raise ValueError("base tombstone retained active paper state")
        if raw.get("paper_ids") != [] or not isinstance(raw.get("tombstone"), Mapping):
            raise ValueError("base tombstone child state is invalid")
        if set(by_locator) != {"$.tombstone"}:
            raise ValueError("base tombstone evidence is incomplete")
        state.tombstone = _input_from_evidence(
            "papers",
            raw["tombstone"],
            by_locator["$.tombstone"],
            corpus_id,
        )
        if state.tombstone.operation != "delete":
            raise ValueError("base tombstone evidence is not a delete")
        return state
    if tombstone_flag is not False or operation != "upsert":
        raise ValueError("base active projection operation is invalid")
    paper_payload = raw.get("paper")
    if not isinstance(paper_payload, Mapping) or raw.get("tombstone") is not None:
        raise ValueError("base active paper payload is missing")
    abstract_payload = raw.get("abstract")
    if abstract_payload is not None and not isinstance(abstract_payload, Mapping):
        raise ValueError("base abstract payload must be an object")
    paper_ids = raw.get("paper_ids")
    if not isinstance(paper_ids, list) or any(
        not isinstance(item, Mapping) for item in paper_ids
    ):
        raise ValueError("base paper-id payloads must be objects")
    expected_locators = {"$.paper"}
    if abstract_payload is not None:
        expected_locators.add("$.abstract")
    expected_locators.update(f"$.paper_ids[{index}]" for index in range(len(paper_ids)))
    if set(by_locator) != expected_locators:
        raise ValueError("base active evidence does not match raw component state")
    state.paper = _input_from_evidence(
        "papers",
        paper_payload,
        by_locator["$.paper"],
        corpus_id,
    )
    if state.paper.operation != "upsert":
        raise ValueError("base active paper evidence is not an upsert")
    if abstract_payload is not None:
        state.abstract = _input_from_evidence(
            "abstracts",
            abstract_payload,
            by_locator["$.abstract"],
            corpus_id,
        )
        if state.abstract.operation != "upsert":
            raise ValueError("base active abstract evidence is not an upsert")
    if state.aliases is None:
        raise ValueError("base paper state was not initialized")
    for index, payload in enumerate(paper_ids):
        alias = _input_from_evidence(
            "paper-ids",
            payload,
            by_locator[f"$.paper_ids[{index}]"],
            corpus_id,
        )
        if alias.operation != "upsert" or alias.sha_alias is None:
            raise ValueError("base active SHA evidence is not an upsert")
        if alias.sha_alias in state.aliases:
            raise ValueError("base active projection has a duplicate SHA alias")
        state.aliases[alias.sha_alias] = alias
    return state


def _input_from_evidence(
    dataset: str,
    payload: Mapping[str, Any],
    evidence: Mapping[str, Any],
    corpus_id: str,
) -> _InputRow:
    expected_fields = {
        "application_order",
        "content_sha256",
        "dataset",
        "ingested_at",
        "lake_locator",
        "operation",
        "raw_payload_locator",
        "source_record_id",
    }
    if set(evidence) != expected_fields or evidence.get("dataset") != dataset:
        raise ValueError("base component evidence fields are invalid")
    payload_json = _canonical_json(dict(payload))
    content_sha256 = _digest(evidence.get("content_sha256"), "content_sha256")
    if hashlib.sha256(payload_json.encode()).hexdigest() != content_sha256:
        raise ValueError("base component evidence content checksum mismatch")
    operation = _required_text(evidence.get("operation"), "operation")
    application_order = evidence.get("application_order")
    if not isinstance(application_order, Mapping):
        raise ValueError("base component application order is invalid")
    order = _validate_application_order(
        application_order,
        expected_mode=None,
        row_operation=operation,
    )
    payload_corpus = _normalize_corpus_id(payload.get("corpusid"), dataset)
    if payload_corpus != corpus_id:
        raise ValueError("base component corpus ID mismatch")
    sha_alias = None
    if dataset == "paper-ids":
        raw_sha = payload.get("sha")
        if not isinstance(raw_sha, str):
            raise ValueError("base paper-id evidence has no SHA")
        sha_alias = raw_sha.strip().casefold()
        if _SHA1.fullmatch(sha_alias) is None:
            raise ValueError("base paper-id evidence SHA is invalid")
    row = _InputRow(
        corpus_id=corpus_id,
        source_record_id=_required_text(
            evidence.get("source_record_id"), "source_record_id"
        ),
        operation=operation,
        payload_json=payload_json,
        payload=dict(payload),
        content_sha256=content_sha256,
        ingested_at=_required_iso_timestamp(evidence.get("ingested_at")),
        sha_alias=sha_alias,
        application_order=order,
        lake_locator=_required_text(evidence.get("lake_locator"), "lake locator"),
    )
    if row.source_record_id != _decode_work_identity(dataset, row):
        raise ValueError("base component source record identity mismatch")
    return row


def _schedule_diff_events(
    events: Mapping[str, Sequence[_InputRow]],
    transition_count: int,
) -> list[tuple[str, str, _InputRow]]:
    scheduled: list[tuple[str, str, _InputRow]] = []
    phases = (
        ("papers", "upsert"),
        ("abstracts", "upsert"),
        ("paper-ids", "upsert"),
        ("abstracts", "delete"),
        ("paper-ids", "delete"),
        ("papers", "delete"),
    )
    for diff_index in range(transition_count):
        for dataset, operation in phases:
            for row in events[dataset]:
                order = row.application_order or {}
                if (
                    order.get("diff_index") == diff_index
                    and row.operation == operation
                ):
                    scheduled.append((dataset, operation, row))
    if len(scheduled) != sum(len(rows) for rows in events.values()):
        raise ValueError("diff events do not fit the sealed transition chain")
    return scheduled


def _validate_final_states(states: Mapping[str, _PaperState], bucket: int) -> None:
    sha_owners: dict[str, str] = {}
    for corpus_id, state in states.items():
        aliases = state.aliases or {}
        if state.tombstone is not None:
            if state.paper is not None or state.abstract is not None or aliases:
                raise ValueError(f"tombstone {corpus_id} retained active child state")
            continue
        if state.paper is None:
            raise ValueError(f"bucket {bucket} has no paper anchor for corpus {corpus_id}")
        for sha, alias in aliases.items():
            if alias.corpus_id != corpus_id or alias.sha_alias != sha:
                raise ValueError("paper state contains a mismatched SHA alias")
            owner = sha_owners.get(sha)
            if owner is not None and owner != corpus_id:
                raise ValueError(f"SHA alias {sha} maps to contradictory corpus IDs")
            sha_owners[sha] = corpus_id


def _paper_state_bytes(state: _PaperState) -> int:
    rows = [state.paper, state.abstract, state.tombstone]
    rows.extend((state.aliases or {}).values())
    return sum(
        len(row.payload_json.encode("utf-8"))
        for row in rows
        if row is not None
    )


def _input_mode(seals: Mapping[str, Mapping[str, Any]]) -> str:
    if set(seals) != set(_DATASETS):
        raise ValueError("Semantic Scholar input must contain all required datasets")
    modes = {seal.get("application_mode") for seal in seals.values()}
    if len(modes) != 1 or next(iter(modes)) not in {"snapshot", "diff"}:
        raise ValueError("Semantic Scholar datasets have mixed or invalid release modes")
    return str(next(iter(modes)))


def _shared_diff_transitions(
    seals: Mapping[str, Mapping[str, Any]],
    release: str,
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] | None = None
    for dataset in _DATASETS:
        transitions: dict[int, tuple[str, str]] = {}
        for shard in seals[dataset]["shards"]:
            order = shard.get("application_order")
            if not isinstance(order, Mapping) or order.get("mode") != "diff":
                raise ValueError(f"{dataset} release is missing diff application order")
            index = _nonnegative_integer(order.get("diff_index"), "diff_index")
            transition = (
                _required_text(order.get("from_release"), "from_release"),
                _required_text(order.get("to_release"), "to_release"),
            )
            prior = transitions.get(index)
            if prior is not None and prior != transition:
                raise ValueError(f"{dataset} has contradictory diff transitions")
            transitions[index] = transition
        indexes = sorted(transitions)
        if indexes != list(range(len(transitions))):
            raise ValueError(f"{dataset} diff transition indexes are gapped")
        values = [
            {
                "diff_index": index,
                "from_release": transitions[index][0],
                "to_release": transitions[index][1],
            }
            for index in sorted(transitions)
        ]
        if not values or values[-1]["to_release"] != release:
            raise ValueError(f"{dataset} diff chain does not end at target release")
        if expected is None:
            expected = values
        elif values != expected:
            raise ValueError("Semantic Scholar dataset diff transition chains disagree")
    if expected is None:
        raise ValueError("Semantic Scholar diff transition chain is empty")
    return expected


def _snapshot_lineage(release: str) -> dict[str, Any]:
    return {
        "base_release": None,
        "base_artifact_id": None,
        "target_release": release,
        "transitions": [],
    }


def _diff_lineage(
    release: str,
    base: ProjectionReceipt,
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "base_release": base.release,
        "base_artifact_id": base.artifact_id,
        "target_release": release,
        "transitions": [dict(value) for value in transitions],
    }


def _validate_projection_lineage(
    value: Any,
    *,
    mode: str,
    release: str,
    input_seals: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "base_artifact_id",
        "base_release",
        "target_release",
        "transitions",
    }:
        raise ValueError("projection lineage fields are invalid")
    if mode == "snapshot":
        expected = _snapshot_lineage(release)
        if dict(value) != expected:
            raise ValueError("snapshot projection lineage is invalid")
        return expected
    transitions = _shared_diff_transitions(input_seals, release)
    base_release = _required_text(value.get("base_release"), "base_release")
    base_artifact_id = _digest(value.get("base_artifact_id"), "base_artifact_id")
    expected = {
        "base_release": base_release,
        "base_artifact_id": base_artifact_id,
        "target_release": release,
        "transitions": transitions,
    }
    if base_release != transitions[0]["from_release"] or dict(value) != expected:
        raise ValueError("diff projection lineage is invalid")
    return expected


def _validate_application_order(
    value: Mapping[str, Any],
    *,
    expected_mode: str | None,
    row_operation: str,
) -> dict[str, Any]:
    mode = _required_text(value.get("mode"), "application order mode")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError("row application order mode does not match release")
    if mode == "snapshot":
        if set(value) != {"mode", "manifest_index"}:
            raise ValueError("snapshot application order fields are invalid")
        result = {
            "mode": "snapshot",
            "manifest_index": _nonnegative_integer(
                value.get("manifest_index"), "manifest_index"
            ),
        }
        if row_operation != "upsert":
            raise ValueError("snapshot row must be an upsert")
        return result
    if mode != "diff" or set(value) != {
        "diff_index",
        "from_release",
        "mode",
        "operation",
        "operation_index",
        "to_release",
    }:
        raise ValueError("diff application order fields are invalid")
    operation = _required_text(value.get("operation"), "diff operation")
    if operation not in {"upsert", "delete"} or operation != row_operation:
        raise ValueError("diff row operation does not match application order")
    start = _required_text(value.get("from_release"), "from_release")
    end = _required_text(value.get("to_release"), "to_release")
    if start == end:
        raise ValueError("diff transition must change releases")
    return {
        "mode": "diff",
        "diff_index": _nonnegative_integer(value.get("diff_index"), "diff_index"),
        "operation": operation,
        "operation_index": _nonnegative_integer(
            value.get("operation_index"), "operation_index"
        ),
        "from_release": start,
        "to_release": end,
    }


def _required_iso_timestamp(value: Any) -> str:
    text = _required_text(value, "ingested_at")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError("ingested_at must be an ISO timestamp") from None
    if parsed.tzinfo is None:
        raise ValueError("ingested_at must be timezone-aware")
    return parsed.astimezone(UTC).isoformat()


def _required_release_date(value: Any, label: str) -> str:
    text = _required_text(value, label)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{label} must be an ISO calendar date") from None
    if parsed.isoformat() != text:
        raise ValueError(f"{label} must use YYYY-MM-DD format")
    return text


def _extract_urls(value: Any, locator: str = "$") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            found.extend(_extract_urls(value[key], f"{locator}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_extract_urls(child, f"{locator}[{index}]"))
    elif isinstance(value, str):
        candidate = value.strip()
        if candidate == value and candidate.casefold().startswith(("http://", "https://")):
            _validate_web_url(candidate, f"URL at {locator}")
            found.append({"locator": locator, "url": candidate})
    return _unique_urls(found)


def _external_identifier_urls(external_ids: Mapping[str, Any]) -> list[dict[str, str]]:
    """Expose canonical URLs for exact external identifiers supported by S2."""

    found: list[dict[str, str]] = []
    for key, raw_value in external_ids.items():
        if not isinstance(key, str) or not isinstance(raw_value, str):
            continue
        identifier = raw_value.strip()
        if not identifier or identifier != raw_value:
            continue
        normalized_key = key.casefold()
        if normalized_key == "arxiv" and re.fullmatch(
            r"(?:(?:[a-z-]+/)?\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?",
            identifier,
            re.IGNORECASE,
        ):
            url = f"https://arxiv.org/abs/{quote(identifier, safe='./')}"
        elif normalized_key == "doi" and re.fullmatch(
            r"10\.\d{4,9}/[^\s]+", identifier, re.IGNORECASE
        ):
            url = f"https://doi.org/{quote(identifier, safe='/()') }"
        else:
            continue
        found.append({"locator": f"$.paper.externalids.{key}", "url": url})
    return found


def _unique_urls(found: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for item in sorted(found, key=lambda item: (item["url"], item["locator"])):
        identity = (item["url"], item["locator"])
        if identity not in seen:
            seen.add(identity)
            unique.append({"locator": item["locator"], "url": item["url"]})
    return unique


def _validate_web_url(value: str, label: str) -> None:
    parts = urlsplit(value)
    if (
        parts.scheme.casefold() not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{label} is not a public web locator")
    try:
        _port = parts.port
    except ValueError:
        raise ValueError(f"{label} has an invalid port") from None


def _optional_year(value: Any, corpus_id: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"paper {corpus_id} year must be an integer")
    if not -(2**31) <= value < 2**31:
        raise ValueError(f"paper {corpus_id} year is out of range")
    return value


def _optional_publication_date(value: Any, corpus_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"paper {corpus_id} publicationdate must be text")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"paper {corpus_id} publicationdate must be an ISO date"
        ) from None
    if parsed.isoformat() != value:
        raise ValueError(f"paper {corpus_id} publicationdate is noncanonical")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    return value


def _load_canonical_json_object(value: str, dataset: str) -> Mapping[str, Any]:
    try:
        result = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (RecursionError, TypeError, ValueError):
        raise ValueError(f"{dataset} payload_json is malformed JSON") from None
    if not isinstance(result, Mapping):
        raise ValueError(f"{dataset} payload_json must contain an object")
    if _canonical_json(dict(result)) != value:
        raise ValueError(f"{dataset} payload_json is not canonical")
    return result


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


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
        raise ValueError("payload contains a non-JSON value") from None


def _normalize_corpus_id(value: Any, dataset: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{dataset} row has no valid corpusid")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{dataset} row has no valid corpusid")
        return str(value)
    if isinstance(value, str) and value.isascii() and value.isdigit():
        normalized = str(int(value))
        if normalized != "0":
            return normalized
    raise ValueError(f"{dataset} row has no valid corpusid")


def _same_input_record(left: _InputRow, right: _InputRow) -> bool:
    return (
        left.corpus_id,
        left.source_record_id,
        left.operation,
        left.payload_json,
        left.content_sha256,
        left.sha_alias,
    ) == (
        right.corpus_id,
        right.source_record_id,
        right.operation,
        right.payload_json,
        right.content_sha256,
        right.sha_alias,
    )


def _input_row_bytes(row: _InputRow) -> int:
    return sum(
        len(value.encode("utf-8"))
        for value in (
            row.corpus_id,
            row.source_record_id,
            row.operation,
            row.payload_json,
            row.content_sha256,
            row.ingested_at,
            row.sha_alias or "",
            _canonical_json(dict(row.application_order or {})),
            row.lake_locator,
        )
    )


def _projection_row_bytes(row: Mapping[str, Any]) -> int:
    return len(_canonical_json(dict(row)).encode("utf-8"))


def _input_digest_value(row: _InputRow) -> bytes:
    return _canonical_json(
        {
            "source_record_id": row.source_record_id,
            "operation": row.operation,
            "payload_json": row.payload_json,
            "content_sha256": row.content_sha256,
            "ingested_at": row.ingested_at,
            "application_order": dict(row.application_order or {}),
            "lake_locator": row.lake_locator,
        }
    ).encode()


def _update_framed_digest(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _bucket(corpus_id: str, bucket_count: int) -> int:
    digest = hashlib.sha256(corpus_id.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % bucket_count


def _corpus_sort_key(value: str) -> tuple[int, str]:
    return (len(value), value)


def _iter_arrow_rows(batch: pa.RecordBatch) -> Iterator[dict[str, Any]]:
    for index in range(batch.num_rows):
        yield {
            name: batch.column(position)[index].as_py()
            for position, name in enumerate(batch.schema.names)
        }


def _verify_working_part(part: _WorkingPart) -> None:
    if part.path.is_symlink() or not part.path.is_file():
        raise ValueError(f"working partition is missing: {part.path}")
    if part.path.stat().st_size != part.byte_count:
        raise ValueError(f"working partition byte count mismatch: {part.path}")
    if _file_sha256(part.path) != part.sha256:
        raise ValueError(f"working partition checksum mismatch: {part.path}")
    metadata = pq.read_metadata(part.path)
    if metadata.num_rows != part.row_count:
        raise ValueError(f"working partition row count mismatch: {part.path}")
    if metadata.schema.to_arrow_schema() != _WORK_SCHEMA:
        raise ValueError(f"working partition schema mismatch: {part.path}")


def _write_projection_part(
    parts_root: Path,
    *,
    bucket: int,
    part_index: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    bucket_root = parts_root / f"bucket-{bucket:06d}"
    bucket_root.mkdir(exist_ok=True)
    name = f"part-{part_index:06d}.parquet"
    path = bucket_root / name
    table = pa.Table.from_pylist(rows, schema=PROJECTION_SCHEMA)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return {
        "bucket": bucket,
        "name": name,
        "path": f"parts/{bucket_root.name}/{name}",
        "row_count": len(rows),
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
        "min_corpus_id": rows[0]["corpus_id"],
        "max_corpus_id": rows[-1]["corpus_id"],
    }


def _receipt(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    already_materialized: bool,
) -> ProjectionReceipt:
    return ProjectionReceipt(
        source=str(manifest["source"]),
        release=str(manifest["release"]),
        artifact_id=str(manifest["artifact_id"]),
        row_count=int(manifest["row_count"]),
        part_count=int(manifest["part_count"]),
        bucket_count=int(manifest["bucket_count"]),
        path=path,
        already_materialized=already_materialized,
    )


def _path_component(value: str) -> str:
    raw = _required_text(value, "path component")
    readable = _COMPONENT.sub("-", raw).strip(".-_")[:48] or "item"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{readable}-{digest}"


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    digest = value.casefold()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(value)).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required projection metadata is missing: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, RecursionError, TypeError, ValueError):
        raise ValueError(f"projection metadata is malformed: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"projection metadata must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _require_plain_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is invalid: {path}")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    for directory, _subdirectories, files in os.walk(path, topdown=False):
        current = Path(directory)
        for filename in files:
            _fsync_file(current / filename)
        _fsync_directory(current)


__all__ = [
    "MaterializationLimits",
    "PROJECTION_SCHEMA",
    "ProjectionPlan",
    "ProjectionReceipt",
    "SemanticScholarProjectionMaterializer",
]
