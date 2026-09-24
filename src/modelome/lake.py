from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_FORMAT = "modelome-parquet-lake-v3"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_UNSET = object()
_RECORD_SCHEMA = pa.schema(
    [
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("operation", pa.string(), nullable=False),
        pa.field("payload_json", pa.large_string(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


@dataclass(frozen=True, slots=True)
class LakeRecord:
    source_record_id: str
    payload: Mapping[str, Any]
    operation: str = "upsert"


@dataclass(frozen=True, slots=True)
class ShardApplicationOrder:
    """Logical position of a shard in an immutable release replay."""

    mode: str
    manifest_index: int | None = None
    diff_index: int | None = None
    operation: str | None = None
    operation_index: int | None = None
    from_release: str | None = None
    to_release: str | None = None

    def __post_init__(self) -> None:
        mode = _required(self.mode, "application-order mode").casefold()
        object.__setattr__(self, "mode", mode)
        if mode == "single":
            if any(
                value is not None
                for value in (
                    self.manifest_index,
                    self.diff_index,
                    self.operation,
                    self.operation_index,
                    self.from_release,
                    self.to_release,
                )
            ):
                raise ValueError("single-shard order cannot contain position fields")
            return
        if mode == "snapshot":
            _nonnegative_integer(self.manifest_index, "manifest_index")
            if any(
                value is not None
                for value in (
                    self.diff_index,
                    self.operation,
                    self.operation_index,
                    self.from_release,
                    self.to_release,
                )
            ):
                raise ValueError("snapshot order can contain only manifest_index")
            return
        if mode != "diff":
            raise ValueError("application-order mode must be single, snapshot, or diff")
        _nonnegative_integer(self.diff_index, "diff_index")
        _nonnegative_integer(self.operation_index, "operation_index")
        operation = _required(self.operation, "diff operation").casefold()
        if operation not in {"upsert", "delete"}:
            raise ValueError("diff operation must be upsert or delete")
        object.__setattr__(self, "operation", operation)
        start = _required(self.from_release, "from_release")
        end = _required(self.to_release, "to_release")
        if start == end:
            raise ValueError("diff transition must change releases")
        object.__setattr__(self, "from_release", start)
        object.__setattr__(self, "to_release", end)
        if self.manifest_index is not None:
            raise ValueError("diff order cannot contain manifest_index")

    @classmethod
    def single(cls) -> ShardApplicationOrder:
        return cls(mode="single")

    @classmethod
    def snapshot(cls, manifest_index: int) -> ShardApplicationOrder:
        return cls(mode="snapshot", manifest_index=manifest_index)

    @classmethod
    def diff(
        cls,
        *,
        diff_index: int,
        operation: str,
        operation_index: int,
        from_release: str,
        to_release: str,
    ) -> ShardApplicationOrder:
        return cls(
            mode="diff",
            diff_index=diff_index,
            operation=operation,
            operation_index=operation_index,
            from_release=from_release,
            to_release=to_release,
        )

    def as_dict(self) -> dict[str, Any]:
        if self.mode == "single":
            return {"mode": "single"}
        if self.mode == "snapshot":
            return {
                "mode": "snapshot",
                "manifest_index": self.manifest_index,
            }
        return {
            "mode": "diff",
            "diff_index": self.diff_index,
            "operation": self.operation,
            "operation_index": self.operation_index,
            "from_release": self.from_release,
            "to_release": self.to_release,
        }


@dataclass(frozen=True, slots=True)
class ShardReceipt:
    source: str
    dataset: str
    release: str
    shard: str
    control_sha256: str
    upstream_sha256: str
    upstream_url: str | None
    upstream_bytes: int | None
    row_count: int
    part_count: int
    application_order: ShardApplicationOrder
    path: Path
    already_committed: bool = False


@dataclass(frozen=True, slots=True)
class ReleaseReceipt:
    source: str
    dataset: str
    release: str
    shard_count: int
    row_count: int
    application_mode: str
    path: Path
    already_sealed: bool = False


class ParquetLandingZone:
    """Streaming, immutable Parquet storage for complete upstream bulk shards.

    A shard becomes visible only after every bounded row batch and its manifest are
    durable. A release becomes enumerable only after ``seal_release`` verifies the
    exact expected shard set. No operation loads a complete corpus into Python.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.shards_root = self.root / "shards"
        self.staging_root = self.root / ".staging"

    def initialize(self) -> None:
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(f"Parquet landing-zone path is not a directory: {self.root}")
        self.shards_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        for path in (self.root, self.shards_root, self.staging_root):
            if path.is_symlink():
                raise ValueError(f"Parquet landing-zone path must not be a symlink: {path}")

    def lookup_committed_shard(
        self,
        *,
        source: str,
        dataset: str,
        release: str,
        shard: str,
        control_sha256: str,
        upstream_url: str | None,
        application_order: ShardApplicationOrder | Mapping[str, Any] | None = None,
    ) -> ShardReceipt | None:
        """Return one fully verified control match without trusting path names."""

        self.initialize()
        source = _required(source, "source")
        dataset = _required(dataset, "dataset")
        release = _required(release, "release")
        shard = _required(shard, "shard")
        control_digest = _sha256(control_sha256, "control_sha256")
        stable_url = _optional_text(upstream_url, "upstream_url")
        order = _application_order(
            ShardApplicationOrder.single()
            if application_order is None
            else application_order
        )
        matches: list[tuple[Path, dict[str, Any]]] = []
        for path, manifest in self._verified_shard_bucket(
            source=source,
            dataset=dataset,
            release=release,
            shard=shard,
        ):
            if manifest["control_sha256"] != control_digest:
                continue
            if manifest["application_order"] != order.as_dict():
                raise ValueError(
                    "committed control has a different application order"
                )
            if manifest.get("upstream_url") != stable_url:
                raise ValueError("committed control has a different upstream URL")
            matches.append((path, manifest))
        if len(matches) > 1:
            raise ValueError("committed shard control identity is ambiguous")
        if not matches:
            return None
        path, manifest = matches[0]
        return _shard_receipt(path, manifest, already_committed=True)

    def commit_shard(
        self,
        *,
        source: str,
        dataset: str,
        release: str,
        shard: str,
        control_sha256: str,
        upstream_sha256: str,
        records: Iterable[LakeRecord],
        upstream_url: str | None = None,
        upstream_bytes: int | None = None,
        application_order: ShardApplicationOrder | Mapping[str, Any] | None = None,
        expected_rows: int | None = None,
        batch_rows: int = 10_000,
    ) -> ShardReceipt:
        """Stream one verified upstream object into immutable Parquet fragments."""

        self.initialize()
        stable_url = _optional_text(upstream_url, "upstream_url")
        byte_count = _optional_nonnegative_integer(upstream_bytes, "upstream_bytes")
        identity = _identity(
            source,
            dataset,
            release,
            shard,
            control_sha256,
            upstream_sha256,
        )
        order = _application_order(
            ShardApplicationOrder.single()
            if application_order is None
            else application_order
        )
        if expected_rows is not None and expected_rows < 0:
            raise ValueError("expected_rows must be nonnegative")
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        existing = self.lookup_committed_shard(
            source=identity["source"],
            dataset=identity["dataset"],
            release=identity["release"],
            shard=identity["shard"],
            control_sha256=identity["control_sha256"],
            upstream_url=stable_url,
            application_order=order,
        )
        if existing is not None:
            if existing.upstream_sha256 != identity["upstream_sha256"]:
                raise ValueError(
                    "stable shard control resolved to a different upstream digest"
                )
            if byte_count is not None and existing.upstream_bytes != byte_count:
                raise ValueError(
                    "stable shard control resolved to a different upstream byte count"
                )
            if expected_rows is not None and existing.row_count != expected_rows:
                raise ValueError(
                    "stable shard control has a different committed row count"
                )
            return existing
        final_dir = self._shard_path(identity)
        if final_dir.exists():
            manifest = self._verify_shard(
                final_dir,
                identity,
                application_order=order,
                upstream_url=stable_url,
                upstream_bytes=byte_count,
            )
            return _shard_receipt(final_dir, manifest, already_committed=True)

        bucket_dir = final_dir.parent
        bucket_dir.mkdir(parents=True, exist_ok=True)
        if bucket_dir.is_symlink():
            raise ValueError(f"landing-zone shard bucket must not be a symlink: {bucket_dir}")
        stage = Path(tempfile.mkdtemp(prefix="shard-", dir=self.staging_root))
        parts_dir = stage / "parts"
        parts_dir.mkdir()
        row_count = 0
        part_rows: list[dict[str, Any]] = []
        parts: list[dict[str, Any]] = []
        ingested_at = datetime.now(UTC)

        try:
            for record in records:
                part_rows.append(_record_row(record, ingested_at))
                row_count += 1
                if len(part_rows) >= batch_rows:
                    parts.append(_write_part(parts_dir, len(parts), part_rows))
                    part_rows = []
            if part_rows:
                parts.append(_write_part(parts_dir, len(parts), part_rows))
            if expected_rows is not None and row_count != expected_rows:
                raise ValueError(
                    f"{source}/{dataset}/{release}/{shard}: expected "
                    f"{expected_rows} rows, received {row_count}"
                )

            manifest = {
                "format": _FORMAT,
                **identity,
                "application_order": order.as_dict(),
                "upstream_url": stable_url,
                "upstream_bytes": byte_count,
                "row_count": row_count,
                "part_count": len(parts),
                "parts": parts,
                "schema": str(_RECORD_SCHEMA),
                "committed_at": ingested_at.isoformat(),
            }
            manifest["manifest_sha256"] = _json_sha256(manifest)
            _write_json_exclusive(stage / "manifest.json", manifest)
            _fsync_tree(stage)
            try:
                os.replace(stage, final_dir)
            except OSError:
                if not final_dir.exists():
                    raise
                existing = self._verify_shard(
                    final_dir,
                    identity,
                    application_order=order,
                    upstream_url=stable_url,
                    upstream_bytes=byte_count,
                )
                return _shard_receipt(final_dir, existing, already_committed=True)
            _fsync_directory(bucket_dir)
            _fsync_directory(bucket_dir.parent)
            return _shard_receipt(final_dir, manifest, already_committed=False)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def seal_release(
        self,
        *,
        source: str,
        dataset: str,
        release: str,
        expected_shards: Mapping[str, str | ShardReceipt],
    ) -> ReleaseReceipt:
        """Atomically declare an exact, checksum-verified set of shards complete."""

        self.initialize()
        source = _required(source, "source")
        dataset = _required(dataset, "dataset")
        release = _required(release, "release")
        if not expected_shards:
            raise ValueError("expected_shards must not be empty")

        release_dir = self._release_path(source, dataset, release)
        seal_path = release_dir / "RELEASE.json"
        if seal_path.exists():
            existing = self._verify_release(seal_path, source, dataset, release)
            existing_by_shard = {
                str(item["shard"]): item for item in existing["shards"]
            }
            requested_by_shard = {
                _required(shard, "shard"): selector
                for shard, selector in expected_shards.items()
            }
            if set(existing_by_shard) != set(requested_by_shard):
                raise ValueError(
                    f"sealed release {source}/{dataset}/{release} has a different shard set"
                )
            for shard, selector in requested_by_shard.items():
                item = existing_by_shard[shard]
                if isinstance(selector, ShardReceipt):
                    matches = (
                        selector.source == source
                        and selector.dataset == dataset
                        and selector.release == release
                        and selector.shard == shard
                        and selector.control_sha256 == item["control_sha256"]
                        and selector.upstream_sha256 == item["upstream_sha256"]
                    )
                else:
                    matches = (
                        _sha256(selector, "upstream_sha256")
                        == item["upstream_sha256"]
                    )
                if not matches:
                    raise ValueError(
                        f"sealed release {source}/{dataset}/{release} "
                        "has a different shard set"
                    )
                shard_path = self.root / str(item["path"])
                identity = _identity(
                    source,
                    dataset,
                    release,
                    shard,
                    item["control_sha256"],
                    item["upstream_sha256"],
                )
                manifest = self._verify_shard(shard_path, identity)
                sealed_fields = (
                    "application_order",
                    "upstream_url",
                    "upstream_bytes",
                    "row_count",
                )
                if any(
                    manifest.get(field) != item.get(field)
                    for field in sealed_fields
                ):
                    raise ValueError(
                        f"release seal does not match shard manifest: {shard_path}"
                    )
            return _release_receipt(seal_path, existing, already_sealed=True)

        manifests = []
        seen: set[str] = set()
        for raw_shard, selector in expected_shards.items():
            shard = _required(raw_shard, "shard")
            if shard in seen:
                raise ValueError(f"duplicate shard identity: {shard}")
            seen.add(shard)
            path, manifest = self._resolve_expected_shard(
                source=source,
                dataset=dataset,
                release=release,
                shard=shard,
                selector=selector,
            )
            manifests.append(
                {
                    "shard": manifest["shard"],
                    "control_sha256": manifest["control_sha256"],
                    "upstream_sha256": manifest["upstream_sha256"],
                    "upstream_url": manifest["upstream_url"],
                    "upstream_bytes": manifest["upstream_bytes"],
                    "row_count": manifest["row_count"],
                    "application_order": manifest["application_order"],
                    "path": str(path.relative_to(self.root)),
                }
            )

        manifests = _ordered_shards(manifests, release=release)
        application_mode = str(manifests[0]["application_order"]["mode"])

        release_dir.mkdir(parents=True, exist_ok=True)
        seal = {
            "format": _FORMAT,
            "source": source,
            "dataset": dataset,
            "release": release,
            "application_mode": application_mode,
            "shard_count": len(manifests),
            "row_count": sum(int(item["row_count"]) for item in manifests),
            "shards": manifests,
        }
        seal["release_sha256"] = _json_sha256(seal)

        if seal_path.exists():
            existing = _read_json(seal_path)
            if existing != seal:
                raise ValueError(
                    f"sealed release {source}/{dataset}/{release} has a different shard set"
                )
            return _release_receipt(seal_path, seal, already_sealed=True)

        temporary = release_dir / f".RELEASE-{os.getpid()}-{os.urandom(6).hex()}.tmp"
        try:
            _write_json_exclusive(temporary, seal)
            try:
                os.link(temporary, seal_path)
            except FileExistsError:
                existing = self._verify_release(
                    seal_path, source, dataset, release
                )
                if existing != seal:
                    raise ValueError(
                        f"sealed release {source}/{dataset}/{release} "
                        "has a different shard set"
                    ) from None
                return _release_receipt(seal_path, seal, already_sealed=True)
            _fsync_directory(release_dir)
        finally:
            if temporary.exists():
                temporary.unlink()
        return _release_receipt(seal_path, seal, already_sealed=False)

    def list_releases(
        self,
        *,
        source: str | None = None,
        dataset: str | None = None,
        verify_shards: bool = True,
    ) -> tuple[ReleaseReceipt, ...]:
        """List sealed releases and optionally verify every referenced shard.

        Counts describe physical snapshot or diff rows, not deduplicated model
        entities. Unsealed shard directories are intentionally absent.
        """

        source_filter = _optional_text(source, "source")
        dataset_filter = _optional_text(dataset, "dataset")
        if not isinstance(verify_shards, bool):
            raise TypeError("verify_shards must be boolean")
        self.initialize()
        receipts: list[ReleaseReceipt] = []
        for source_dir in _plain_directories(self.shards_root):
            for dataset_dir in _plain_directories(source_dir):
                for release_dir in _plain_directories(dataset_dir):
                    seal_path = release_dir / "RELEASE.json"
                    if not seal_path.exists():
                        continue
                    if seal_path.is_symlink() or not seal_path.is_file():
                        raise ValueError(f"release seal path is invalid: {seal_path}")
                    untrusted = _read_json(seal_path)
                    release_source = _required(untrusted.get("source"), "source")
                    release_dataset = _required(untrusted.get("dataset"), "dataset")
                    release_name = _required(untrusted.get("release"), "release")
                    expected_path = (
                        self._release_path(
                            release_source,
                            release_dataset,
                            release_name,
                        )
                        / "RELEASE.json"
                    )
                    if seal_path != expected_path:
                        raise ValueError(
                            f"release seal is stored at a noncanonical path: {seal_path}"
                        )
                    if source_filter is not None and release_source != source_filter:
                        continue
                    if dataset_filter is not None and release_dataset != dataset_filter:
                        continue
                    seal = self._verify_release(
                        seal_path,
                        release_source,
                        release_dataset,
                        release_name,
                    )
                    if verify_shards:
                        self._verify_sealed_release_shards(seal)
                    receipts.append(
                        _release_receipt(seal_path, seal, already_sealed=True)
                    )
        receipts.sort(
            key=lambda item: (
                item.source,
                item.dataset,
                item.release,
                str(item.path),
            )
        )
        return tuple(receipts)

    def _resolve_expected_shard(
        self,
        *,
        source: str,
        dataset: str,
        release: str,
        shard: str,
        selector: str | ShardReceipt,
    ) -> tuple[Path, dict[str, Any]]:
        if isinstance(selector, ShardReceipt):
            if (
                selector.source,
                selector.dataset,
                selector.release,
                selector.shard,
            ) != (source, dataset, release, shard):
                raise ValueError("expected shard receipt has a different identity")
            identity = _identity(
                source,
                dataset,
                release,
                shard,
                selector.control_sha256,
                selector.upstream_sha256,
            )
            path = self._shard_path(identity)
            if selector.path != path:
                raise ValueError("expected shard receipt has a noncanonical path")
            return path, self._verify_shard(
                path,
                identity,
                application_order=selector.application_order,
                upstream_url=selector.upstream_url,
                upstream_bytes=selector.upstream_bytes,
            )
        upstream_digest = _sha256(selector, "upstream_sha256")
        matches = [
            (path, manifest)
            for path, manifest in self._verified_shard_bucket(
                source=source,
                dataset=dataset,
                release=release,
                shard=shard,
            )
            if manifest["upstream_sha256"] == upstream_digest
        ]
        if not matches:
            raise ValueError(f"required landing-zone shard is missing: {shard}")
        if len(matches) > 1:
            raise ValueError(
                f"required landing-zone shard selector is ambiguous: {shard}"
            )
        return matches[0]

    def iter_release_batches(
        self,
        *,
        source: str,
        dataset: str,
        release: str,
        columns: Sequence[str] | None = None,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Scan a sealed release lazily as Arrow record batches."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        seal_path = self._release_path(source, dataset, release) / "RELEASE.json"
        seal = self._verify_release(seal_path, source, dataset, release)
        verified_shards = self._verify_sealed_release_shards(seal)
        for _shard, shard_dir, manifest in verified_shards:
            for part in manifest["parts"]:
                part_path = shard_dir / "parts" / part["name"]
                yield from pq.ParquetFile(part_path).iter_batches(
                    batch_size=batch_size,
                    columns=columns,
                )

    def _verify_sealed_release_shards(
        self,
        seal: Mapping[str, Any],
    ) -> list[tuple[Mapping[str, Any], Path, dict[str, Any]]]:
        source = _required(seal.get("source"), "source")
        dataset = _required(seal.get("dataset"), "dataset")
        release = _required(seal.get("release"), "release")
        verified = []
        for shard in seal["shards"]:
            shard_dir = self.root / str(shard["path"])
            identity = _identity(
                source,
                dataset,
                release,
                str(shard["shard"]),
                str(shard["control_sha256"]),
                str(shard["upstream_sha256"]),
            )
            manifest = self._verify_shard(shard_dir, identity)
            sealed_fields = (
                "application_order",
                "upstream_url",
                "upstream_bytes",
                "row_count",
            )
            if any(manifest.get(field) != shard.get(field) for field in sealed_fields):
                raise ValueError(
                    f"release seal does not match shard manifest: {shard_dir}"
                )
            verified.append((shard, shard_dir, manifest))
        return verified

    def _verify_release(
        self,
        path: Path,
        source: str,
        dataset: str,
        release: str,
    ) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release is not sealed: {source}/{dataset}/{release}")
        seal = _read_json(path)
        if set(seal) != {
            "application_mode",
            "dataset",
            "format",
            "release",
            "release_sha256",
            "row_count",
            "shard_count",
            "shards",
            "source",
        }:
            raise ValueError(f"release seal fields are invalid: {path}")
        expected = {
            "format": _FORMAT,
            "source": _required(source, "source"),
            "dataset": _required(dataset, "dataset"),
            "release": _required(release, "release"),
        }
        if any(seal.get(key) != value for key, value in expected.items()):
            raise ValueError(f"invalid release seal: {path}")
        digest_payload = dict(seal)
        digest = digest_payload.pop("release_sha256", None)
        if digest != _json_sha256(digest_payload):
            raise ValueError(f"release seal checksum mismatch: {path}")
        shards = seal.get("shards")
        shard_count = _nonnegative_integer(seal.get("shard_count"), "shard_count")
        _nonnegative_integer(seal.get("row_count"), "row_count")
        if not isinstance(shards, list) or len(shards) != shard_count:
            raise ValueError(f"release seal shard count mismatch: {path}")
        if not shards:
            raise ValueError(f"release seal has no shards: {path}")
        seen_shards: set[str] = set()
        for item in shards:
            if not isinstance(item, Mapping):
                raise ValueError(f"invalid release shard entry: {path}")
            if set(item) != {
                "application_order",
                "control_sha256",
                "path",
                "row_count",
                "shard",
                "upstream_bytes",
                "upstream_sha256",
                "upstream_url",
            }:
                raise ValueError(f"release shard entry fields are invalid: {path}")
            identity = _identity(
                source,
                dataset,
                release,
                item.get("shard"),
                item.get("control_sha256"),
                item.get("upstream_sha256"),
            )
            if identity["shard"] in seen_shards:
                raise ValueError(f"release seal contains a duplicate shard: {path}")
            seen_shards.add(identity["shard"])
            _optional_text(item.get("upstream_url"), "upstream_url")
            _optional_nonnegative_integer(
                item.get("upstream_bytes"), "upstream_bytes"
            )
            _nonnegative_integer(item.get("row_count"), "row_count")
            _application_order(item.get("application_order"))
            stored_path = self.root / _required(item.get("path"), "release shard path")
            if stored_path != self._shard_path(identity):
                raise ValueError(f"release seal contains a noncanonical shard path: {path}")
        ordered = _ordered_shards(shards, release=release)
        if ordered != shards:
            raise ValueError(f"release seal shards are not in application order: {path}")
        if seal.get("application_mode") != ordered[0]["application_order"]["mode"]:
            raise ValueError(f"release seal application mode mismatch: {path}")
        if sum(int(item["row_count"]) for item in shards) != seal.get("row_count"):
            raise ValueError(f"release seal row count mismatch: {path}")
        return seal

    def _verify_shard(
        self,
        path: Path,
        identity: Mapping[str, str],
        *,
        application_order: ShardApplicationOrder | None = None,
        upstream_url: str | None | object = _UNSET,
        upstream_bytes: int | None | object = _UNSET,
    ) -> dict[str, Any]:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"invalid landing-zone shard path: {path}")
        manifest_path = path / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"landing-zone shard manifest is missing: {path}")
        manifest = _read_json(manifest_path)
        expected_manifest_fields = {
            "application_order",
            "committed_at",
            "control_sha256",
            "dataset",
            "format",
            "manifest_sha256",
            "part_count",
            "parts",
            "release",
            "row_count",
            "schema",
            "shard",
            "source",
            "upstream_bytes",
            "upstream_sha256",
            "upstream_url",
        }
        if set(manifest) != expected_manifest_fields:
            raise ValueError(f"landing-zone shard manifest fields are invalid: {path}")
        digest_payload = dict(manifest)
        manifest_digest = digest_payload.pop("manifest_sha256", None)
        if manifest_digest != _json_sha256(digest_payload):
            raise ValueError(f"landing-zone shard manifest checksum mismatch: {path}")
        if manifest.get("format") != _FORMAT or any(
            manifest.get(key) != value for key, value in identity.items()
        ):
            raise ValueError(f"landing-zone shard identity mismatch: {path}")
        if self._shard_path(identity) != path:
            raise ValueError(f"landing-zone shard path does not match its manifest: {path}")
        stored_order = _application_order(manifest.get("application_order"))
        if application_order is not None and stored_order != application_order:
            raise ValueError(f"landing-zone shard application order mismatch: {path}")
        stored_url = _optional_text(manifest.get("upstream_url"), "upstream_url")
        if upstream_url is not _UNSET and stored_url != upstream_url:
            raise ValueError(f"landing-zone shard upstream URL mismatch: {path}")
        stored_bytes = _optional_nonnegative_integer(
            manifest.get("upstream_bytes"), "upstream_bytes"
        )
        if upstream_bytes is not _UNSET and stored_bytes != upstream_bytes:
            raise ValueError(f"landing-zone shard upstream byte count mismatch: {path}")
        if manifest.get("schema") != str(_RECORD_SCHEMA):
            raise ValueError(f"landing-zone shard schema declaration mismatch: {path}")
        _required(manifest.get("committed_at"), "committed_at")
        parts = manifest.get("parts")
        part_count = _nonnegative_integer(manifest.get("part_count"), "part_count")
        expected_row_count = _nonnegative_integer(
            manifest.get("row_count"), "row_count"
        )
        if not isinstance(parts, list) or len(parts) != part_count:
            raise ValueError(f"landing-zone shard part count mismatch: {path}")
        parts_dir = path / "parts"
        if parts_dir.is_symlink() or not parts_dir.is_dir():
            raise ValueError(f"landing-zone parts directory is invalid: {parts_dir}")
        row_count = 0
        expected_names: set[str] = set()
        for index, item in enumerate(parts):
            if not isinstance(item, Mapping):
                raise ValueError(f"invalid landing-zone part manifest: {path}")
            if set(item) != {"name", "row_count", "byte_count", "sha256"}:
                raise ValueError(f"invalid landing-zone part manifest: {path}")
            expected_name = f"part-{index:06d}.parquet"
            if item.get("name") != expected_name:
                raise ValueError(f"landing-zone part order is invalid: {path}")
            expected_names.add(expected_name)
            part_path = parts_dir / expected_name
            if part_path.is_symlink() or not part_path.is_file():
                raise ValueError(f"landing-zone part is missing: {part_path}")
            if part_path.stat().st_size != _nonnegative_integer(
                item.get("byte_count"), "part byte_count"
            ):
                raise ValueError(f"landing-zone part byte count mismatch: {part_path}")
            part_digest = _sha256(item.get("sha256"), "part sha256")
            if _file_sha256(part_path) != part_digest:
                raise ValueError(f"landing-zone part checksum mismatch: {part_path}")
            metadata = pq.read_metadata(part_path)
            part_rows = _nonnegative_integer(item.get("row_count"), "part row_count")
            if metadata.num_rows != part_rows:
                raise ValueError(f"landing-zone part row count mismatch: {part_path}")
            if metadata.schema.to_arrow_schema() != _RECORD_SCHEMA:
                raise ValueError(f"landing-zone part schema mismatch: {part_path}")
            row_count += metadata.num_rows
        actual_names = {item.name for item in parts_dir.iterdir()}
        if actual_names != expected_names:
            raise ValueError(f"landing-zone parts directory has unexpected entries: {path}")
        if row_count != expected_row_count:
            raise ValueError(f"landing-zone shard row count mismatch: {path}")
        return manifest

    def _verified_shard_bucket(
        self,
        *,
        source: str,
        dataset: str,
        release: str,
        shard: str,
    ) -> list[tuple[Path, dict[str, Any]]]:
        bucket = self._shard_bucket_path(source, dataset, release, shard)
        if not bucket.exists():
            return []
        if bucket.is_symlink() or not bucket.is_dir():
            raise ValueError(f"invalid landing-zone shard bucket: {bucket}")
        verified = []
        expected_bucket_identity = {
            "source": source,
            "dataset": dataset,
            "release": release,
            "shard": shard,
        }
        for path in sorted(bucket.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"invalid entry in landing-zone shard bucket: {path}")
            manifest_path = path / "manifest.json"
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError(f"landing-zone shard manifest is missing: {path}")
            manifest = _read_json(manifest_path)
            identity = _manifest_identity(manifest)
            if any(
                identity[key] != value
                for key, value in expected_bucket_identity.items()
            ):
                raise ValueError(
                    f"landing-zone shard bucket contains a mismatched manifest: {path}"
                )
            verified.append((path, self._verify_shard(path, identity)))
        return verified

    def _release_path(self, source: str, dataset: str, release: str) -> Path:
        return (
            self.shards_root
            / _path_component(source)
            / _path_component(dataset)
            / _path_component(release)
        )

    def _shard_path(self, identity: Mapping[str, str]) -> Path:
        return self._shard_bucket_path(
            identity["source"],
            identity["dataset"],
            identity["release"],
            identity["shard"],
        ) / identity["control_sha256"]

    def _shard_bucket_path(
        self,
        source: str,
        dataset: str,
        release: str,
        shard: str,
    ) -> Path:
        return self._release_path(source, dataset, release) / _path_component(shard)


def _identity(
    source: str,
    dataset: str,
    release: str,
    shard: str,
    control_sha256: str,
    upstream_sha256: str,
) -> dict[str, str]:
    return {
        "source": _required(source, "source"),
        "dataset": _required(dataset, "dataset"),
        "release": _required(release, "release"),
        "shard": _required(shard, "shard"),
        "control_sha256": _sha256(control_sha256, "control_sha256"),
        "upstream_sha256": _sha256(upstream_sha256, "upstream_sha256"),
    }


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, str]:
    return _identity(
        manifest.get("source"),
        manifest.get("dataset"),
        manifest.get("release"),
        manifest.get("shard"),
        manifest.get("control_sha256"),
        manifest.get("upstream_sha256"),
    )


def _plain_directories(parent: Path) -> tuple[Path, ...]:
    directories = []
    for path in parent.iterdir():
        if path.is_symlink():
            raise ValueError(f"landing-zone path must not be a symlink: {path}")
        if not path.is_dir():
            raise ValueError(f"landing-zone path is not a directory: {path}")
        directories.append(path)
    return tuple(sorted(directories, key=lambda path: path.name))


def _record_row(record: LakeRecord, ingested_at: datetime) -> dict[str, Any]:
    if not isinstance(record, LakeRecord):
        raise TypeError("records must contain LakeRecord values")
    source_record_id = _required(record.source_record_id, "source_record_id")
    operation = _required(record.operation, "operation").casefold()
    if operation not in {"upsert", "delete"}:
        raise ValueError("record operation must be 'upsert' or 'delete'")
    if not isinstance(record.payload, Mapping):
        raise TypeError("record payload must be a mapping")
    payload_json = json.dumps(
        dict(record.payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "source_record_id": source_record_id,
        "operation": operation,
        "payload_json": payload_json,
        "content_sha256": hashlib.sha256(payload_json.encode()).hexdigest(),
        "ingested_at": ingested_at,
    }


def _write_part(parts_dir: Path, index: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    name = f"part-{index:06d}.parquet"
    path = parts_dir / name
    table = pa.Table.from_pylist(rows, schema=_RECORD_SCHEMA)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    _fsync_file(path)
    return {
        "name": name,
        "row_count": len(rows),
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _shard_receipt(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    already_committed: bool,
) -> ShardReceipt:
    return ShardReceipt(
        source=str(manifest["source"]),
        dataset=str(manifest["dataset"]),
        release=str(manifest["release"]),
        shard=str(manifest["shard"]),
        control_sha256=str(manifest["control_sha256"]),
        upstream_sha256=str(manifest["upstream_sha256"]),
        upstream_url=manifest.get("upstream_url"),
        upstream_bytes=manifest.get("upstream_bytes"),
        row_count=int(manifest["row_count"]),
        part_count=int(manifest["part_count"]),
        application_order=_application_order(manifest["application_order"]),
        path=path,
        already_committed=already_committed,
    )


def _release_receipt(
    path: Path,
    seal: Mapping[str, Any],
    *,
    already_sealed: bool,
) -> ReleaseReceipt:
    return ReleaseReceipt(
        source=str(seal["source"]),
        dataset=str(seal["dataset"]),
        release=str(seal["release"]),
        shard_count=int(seal["shard_count"]),
        row_count=int(seal["row_count"]),
        application_mode=str(seal["application_mode"]),
        path=path,
        already_sealed=already_sealed,
    )


def _application_order(
    value: ShardApplicationOrder | Mapping[str, Any],
) -> ShardApplicationOrder:
    if isinstance(value, ShardApplicationOrder):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("application_order must be a JSON object")
    mode_value = value.get("mode")
    if not isinstance(mode_value, str) or not mode_value.strip():
        raise ValueError("application_order.mode must not be empty")
    mode = mode_value.strip().casefold()
    if mode == "single":
        expected_keys = {"mode"}
        result = ShardApplicationOrder.single()
    elif mode == "snapshot":
        expected_keys = {"mode", "manifest_index"}
        result = ShardApplicationOrder.snapshot(
            _nonnegative_integer(value.get("manifest_index"), "manifest_index")
        )
    elif mode == "diff":
        expected_keys = {
            "mode",
            "diff_index",
            "operation",
            "operation_index",
            "from_release",
            "to_release",
        }
        result = ShardApplicationOrder.diff(
            diff_index=_nonnegative_integer(value.get("diff_index"), "diff_index"),
            operation=_required(value.get("operation"), "diff operation"),
            operation_index=_nonnegative_integer(
                value.get("operation_index"), "operation_index"
            ),
            from_release=_required(value.get("from_release"), "from_release"),
            to_release=_required(value.get("to_release"), "to_release"),
        )
    else:
        raise ValueError("application-order mode must be single, snapshot, or diff")
    if set(value) != expected_keys:
        raise ValueError(
            f"{mode} application_order must contain exactly "
            f"{', '.join(sorted(expected_keys))}"
        )
    if dict(value) != result.as_dict():
        raise ValueError("application_order must use its canonical representation")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _ordered_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    release: str,
) -> list[Mapping[str, Any]]:
    """Validate and return an exact deterministic shard replay order."""

    if not shards:
        raise ValueError("a release must contain at least one shard")
    entries = [
        (item, _application_order(item.get("application_order")))
        for item in shards
    ]
    modes = {order.mode for _, order in entries}
    if len(modes) != 1:
        raise ValueError("release has ambiguous mixed application-order modes")
    mode = next(iter(modes))

    if mode == "single":
        if len(entries) != 1:
            raise ValueError("single application order requires exactly one shard")
        return [entries[0][0]]

    if mode == "snapshot":
        indexes = [order.manifest_index for _, order in entries]
        if len(set(indexes)) != len(indexes):
            raise ValueError("snapshot application order has duplicate manifest_index")
        expected = list(range(len(indexes)))
        if sorted(indexes) != expected:
            raise ValueError(
                "snapshot application order has a gapped or nonzero manifest_index"
            )
        return [
            item
            for item, _ in sorted(
                entries,
                key=lambda entry: entry[1].manifest_index,
            )
        ]

    positions = [
        (order.diff_index, order.operation, order.operation_index)
        for _, order in entries
    ]
    if len(set(positions)) != len(positions):
        raise ValueError("diff application order has a duplicate shard position")
    diff_indexes = sorted({order.diff_index for _, order in entries})
    if diff_indexes != list(range(len(diff_indexes))):
        raise ValueError("diff application order has a gapped or nonzero diff_index")

    transitions: list[tuple[str, str]] = []
    for diff_index in diff_indexes:
        transition_orders = [
            order for _, order in entries if order.diff_index == diff_index
        ]
        transition = {
            (order.from_release, order.to_release) for order in transition_orders
        }
        if len(transition) != 1:
            raise ValueError(
                f"diff application order has an ambiguous transition at {diff_index}"
            )
        transitions.append(next(iter(transition)))
        for operation in ("upsert", "delete"):
            operation_indexes = [
                order.operation_index
                for order in transition_orders
                if order.operation == operation
            ]
            if not operation_indexes:
                continue
            if sorted(operation_indexes) != list(range(len(operation_indexes))):
                raise ValueError(
                    "diff application order has a gapped or nonzero "
                    f"{operation} operation_index at transition {diff_index}"
                )

    for index, (start, _end) in enumerate(transitions[1:], start=1):
        previous_end = transitions[index - 1][1]
        if start != previous_end:
            raise ValueError(
                "diff application order has a non-contiguous transition chain"
            )
    if transitions[-1][1] != _required(release, "release"):
        raise ValueError("diff application order does not end at the sealed release")

    phase = {"upsert": 0, "delete": 1}
    return [
        item
        for item, _ in sorted(
            entries,
            key=lambda entry: (
                entry[1].diff_index,
                phase[entry[1].operation],
                entry[1].operation_index,
            ),
        )
    ]


def _path_component(value: str) -> str:
    raw = _required(value, "path component")
    readable = _COMPONENT.sub("-", raw).strip(".-_")[:48] or "item"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{readable}-{digest}"


def canonical_control_sha256(descriptor: Mapping[str, Any]) -> str:
    """Hash a JSON control descriptor with deterministic key ordering."""

    if not isinstance(descriptor, Mapping):
        raise TypeError("control descriptor must be a mapping")
    try:
        return _json_sha256(descriptor)
    except (TypeError, ValueError):
        raise ValueError("control descriptor must contain canonical JSON values") from None


def _required(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _optional_text(value: Any, label: str) -> str | None:
    return None if value is None else _required(value, label)


def _sha256(value: Any, label: str) -> str:
    digest = _required(value, label).casefold()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return digest


def _optional_nonnegative_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, label)


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


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
    "canonical_control_sha256",
    "LakeRecord",
    "ParquetLandingZone",
    "ReleaseReceipt",
    "ShardApplicationOrder",
    "ShardReceipt",
]
