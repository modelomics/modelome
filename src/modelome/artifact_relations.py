from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.normalize import canonicalize_url

_FORMAT = "modelome-artifact-relation-projection-v1"
_ALGORITHM = "current-evidence-external-memory-join-v1"
_STORE_FORMATS = frozenset(
    {"modelome-parquet-snapshot-v1", "modelome-parquet-snapshot-v2"}
)
_COMMIT_ID = re.compile(r"^\d{20}-[0-9a-f]{16}-[0-9a-f]{12}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NON_ARTIFACT_URL_RELATIONS = frozenset({"license"})

RELATION_SCHEMA = pa.schema(
    [
        pa.field("relation_id", pa.string(), nullable=False),
        pa.field("subject_artifact_id", pa.string(), nullable=False),
        pa.field("predicate", pa.string(), nullable=False),
        pa.field("target_artifact_id", pa.string(), nullable=False),
        pa.field("symmetric", pa.bool_(), nullable=False),
        pa.field("evidence_type", pa.string(), nullable=False),
        pa.field("confidence", pa.float64(), nullable=False),
        pa.field("match_namespace", pa.string()),
        pa.field("match_value", pa.large_string()),
        pa.field("locator", pa.large_string()),
        pa.field("subject_artifact_revision_id", pa.string(), nullable=False),
        pa.field("subject_kind", pa.string(), nullable=False),
        pa.field("subject_source", pa.string(), nullable=False),
        pa.field("subject_source_record_id", pa.string(), nullable=False),
        pa.field("subject_canonical_url", pa.large_string(), nullable=False),
        pa.field("target_artifact_revision_id", pa.string(), nullable=False),
        pa.field("target_kind", pa.string(), nullable=False),
        pa.field("target_source", pa.string(), nullable=False),
        pa.field("target_source_record_id", pa.string(), nullable=False),
        pa.field("target_canonical_url", pa.large_string(), nullable=False),
        pa.field("evidence_json", pa.large_string(), nullable=False),
        pa.field("source_commit", pa.string(), nullable=False),
    ]
)

_WORK_SCHEMA = pa.schema(
    [
        pa.field("key", pa.large_string(), nullable=False),
        pa.field("payload_json", pa.large_string(), nullable=False),
    ]
)

_INPUT_COLUMNS: dict[str, tuple[str, ...]] = {
    "artifacts": (
        "id",
        "source",
        "source_record_id",
        "kind",
        "canonical_url",
        "canonical_url_normalized",
        "current_revision_id",
        "active",
    ),
    "evidence_provenance": (
        "id",
        "artifact_revision_id",
        "subject_type",
        "subject_id",
        "predicate",
        "value_json",
        "locator",
        "extractor",
        "confidence",
        "created_at",
    ),
    "url_discoveries": (
        "id",
        "url_id",
        "artifact_revision_id",
        "relation",
        "locator",
        "discovered_at",
        "depth",
    ),
    "url_frontier": ("id", "url"),
    "artifact_model_links": (
        "id",
        "artifact_id",
        "artifact_revision_id",
        "model_id",
        "local_id",
        "resolution_status",
        "confidence",
        "locator",
        "extractor",
        "created_at",
    ),
    "artifact_release_links": (
        "id",
        "artifact_id",
        "artifact_revision_id",
        "release_id",
        "local_id",
        "resolution_status",
        "confidence",
        "locator",
        "extractor",
        "created_at",
    ),
}


@dataclass(frozen=True, slots=True)
class ArtifactRelationLimits:
    """Memory and output bounds for relation materialization."""

    # The public registry can now contain nearly a million current artifacts.
    # More deterministic buckets preserve the existing per-bucket memory ceiling
    # instead of relaxing it as the historical corpus grows.
    bucket_count: int = 1_024
    scan_batch_rows: int = 4_096
    max_scan_batch_bytes: int = 128 * 1024 * 1024
    partition_buffer_rows: int = 4_096
    partition_buffer_bytes: int = 64 * 1024 * 1024
    max_work_row_bytes: int = 8 * 1024 * 1024
    join_batch_rows: int = 4_096
    max_join_batch_bytes: int = 128 * 1024 * 1024
    max_bucket_rows: int = 1_000_000
    max_bucket_bytes: int = 512 * 1024 * 1024
    output_part_rows: int = 25_000
    max_output_buffer_bytes: int = 128 * 1024 * 1024
    max_output_row_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.bucket_count > 65_536:
            raise ValueError("bucket_count must not exceed 65536")
        if self.max_work_row_bytes > self.partition_buffer_bytes:
            raise ValueError("max_work_row_bytes must not exceed partition_buffer_bytes")
        if self.max_output_row_bytes > self.max_output_buffer_bytes:
            raise ValueError("max_output_row_bytes must not exceed max_output_buffer_bytes")


@dataclass(frozen=True, slots=True)
class ArtifactRelationReceipt:
    source_commit: str
    source_generation: int
    source_state_digest: str
    artifact_id: str
    row_count: int
    part_count: int
    bucket_count: int
    relation_counts: tuple[tuple[str, int], ...]
    path: Path
    already_materialized: bool = False


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    root: Path
    commit: str
    generation: int
    state_digest: str
    path: Path
    table_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _WorkPart:
    path: Path
    row_count: int
    byte_count: int
    sha256: str


class _PartitionSink:
    def __init__(
        self,
        root: Path,
        limits: ArtifactRelationLimits,
    ) -> None:
        self.root = root
        self.limits = limits
        self.root.mkdir(parents=True)
        self.parts: dict[int, list[_WorkPart]] = {}
        self._indexes: dict[int, int] = {}
        self._buffers: dict[int, list[dict[str, str]]] = {}
        self._buffer_rows = 0
        self._buffer_bytes = 0

    def add(self, key: str, payload: Mapping[str, Any]) -> None:
        key = _required_text(key, "partition key")
        payload_json = _canonical_json(dict(payload))
        row_bytes = len(key.encode()) + len(payload_json.encode())
        if row_bytes > self.limits.max_work_row_bytes:
            raise ValueError("working row exceeded max_work_row_bytes")
        if self._buffer_rows and (
            self._buffer_rows >= self.limits.partition_buffer_rows
            or self._buffer_bytes + row_bytes > self.limits.partition_buffer_bytes
        ):
            self.flush()
        bucket = _bucket(key, self.limits.bucket_count)
        self._buffers.setdefault(bucket, []).append({"key": key, "payload_json": payload_json})
        self._buffer_rows += 1
        self._buffer_bytes += row_bytes

    def flush(self) -> None:
        for bucket in sorted(self._buffers):
            rows = self._buffers[bucket]
            if not rows:
                continue
            bucket_root = self.root / f"bucket-{bucket:06d}"
            bucket_root.mkdir(exist_ok=True)
            index = self._indexes.get(bucket, 0)
            path = bucket_root / f"part-{index:06d}.parquet"
            pq.write_table(
                pa.Table.from_pylist(rows, schema=_WORK_SCHEMA),
                path,
                compression="zstd",
                write_statistics=True,
            )
            self.parts.setdefault(bucket, []).append(
                _WorkPart(
                    path=path,
                    row_count=len(rows),
                    byte_count=path.stat().st_size,
                    sha256=_file_sha256(path),
                )
            )
            self._indexes[bucket] = index + 1
        self._buffers.clear()
        self._buffer_rows = 0
        self._buffer_bytes = 0


class ArtifactRelationMaterializer:
    """Build immutable artifact relation claims from current registry evidence.

    The materializer never joins by a model name. Directed relations come from an
    explicit or locally classified URL mention. Symmetric connections come from
    exact artifact identifiers or registry model/release identities, whose own
    resolution is identifier-based. Current revisions and active artifacts are
    the only inputs, so corrected-away evidence cannot survive a rebuild.
    """

    def __init__(
        self,
        store_root: str | Path,
        *,
        output_root: str | Path | None = None,
        limits: ArtifactRelationLimits | None = None,
    ) -> None:
        self.store_root = Path(store_root).expanduser().absolute()
        self.output_root = (
            self.store_root / "projections" / "artifact-relations"
            if output_root is None
            else Path(output_root).expanduser().absolute()
        )
        self.limits = limits or ArtifactRelationLimits()

    def materialize(self) -> ArtifactRelationReceipt:
        self._initialize_output()
        source = self._current_source_snapshot()
        artifact_id = self._artifact_id(source)
        final_dir = self._final_path(source.commit, artifact_id)
        if final_dir.exists() or final_dir.is_symlink():
            manifest = self._verify_projection(final_dir, source, artifact_id)
            return _receipt(final_dir, manifest, already_materialized=True)

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        _require_plain_directory(final_dir.parent, "projection commit directory")
        stage = Path(
            tempfile.mkdtemp(
                prefix="artifact-relations-",
                dir=self.output_root / ".staging",
            )
        )
        try:
            work_root = stage / "work"
            parts_root = stage / "parts"
            work_root.mkdir()
            parts_root.mkdir()
            sinks, scanned, selected = self._partition_inputs(source, work_root)
            relation_sink = _PartitionSink(work_root / "relations", self.limits)
            self._join_current_revisions(sinks)
            self._join_url_ids(sinks)
            self._join_urls(sinks, relation_sink)
            self._join_identity_groups(sinks["identifier_claims"], relation_sink)
            self._join_model_groups(sinks["model_claims"], relation_sink)
            self._join_release_groups(sinks["release_claims"], relation_sink)
            relation_sink.flush()
            parts, row_count, counts = self._write_output_parts(
                source,
                artifact_id,
                relation_sink,
                parts_root,
            )
            shutil.rmtree(work_root)
            created_at = datetime.now(UTC).isoformat()
            manifest: dict[str, Any] = {
                "format": _FORMAT,
                "algorithm": _ALGORITHM,
                "artifact_id": artifact_id,
                "source": {
                    "commit": source.commit,
                    "generation": source.generation,
                    "state_digest": source.state_digest,
                    "table_counts": dict(source.table_counts),
                    "scanned_counts": scanned,
                    "selected_counts": selected,
                },
                "semantics": self._semantics(),
                "layout": self._layout(),
                "schema": str(RELATION_SCHEMA),
                "row_count": row_count,
                "part_count": len(parts),
                "relation_counts": counts,
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
                existing = self._verify_projection(final_dir, source, artifact_id)
                return _receipt(final_dir, existing, already_materialized=True)
            _fsync_directory(final_dir.parent)
            verified = self._verify_projection(final_dir, source, artifact_id)
            return _receipt(final_dir, verified, already_materialized=False)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def current_receipt(self) -> ArtifactRelationReceipt | None:
        """Return the verified projection for the current store commit, if sealed.

        This is a read-only lookup for consumers such as the deferred entry exporter.
        It never creates the projection root, stages work, or rebuilds an outdated
        result: callers must explicitly run :meth:`materialize` when no current
        projection exists.
        """

        source = self._current_source_snapshot()
        artifact_id = self._artifact_id(source)
        path = self._final_path(source.commit, artifact_id)
        if not path.exists() and not path.is_symlink():
            return None
        manifest = self._verify_projection(path, source, artifact_id)
        return _receipt(path, manifest, already_materialized=True)

    def iter_batches(
        self,
        receipt: ArtifactRelationReceipt,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        if not isinstance(receipt, ArtifactRelationReceipt):
            raise TypeError("receipt must be an ArtifactRelationReceipt")
        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if columns is not None:
            unknown = set(columns) - set(RELATION_SCHEMA.names)
            if unknown:
                raise ValueError(f"unknown relation projection columns: {sorted(unknown)}")
        source = self._source_snapshot(receipt.source_commit)
        manifest = self._verify_projection(receipt.path, source, receipt.artifact_id)
        if (
            receipt.source_generation != source.generation
            or receipt.source_state_digest != source.state_digest
            or receipt.row_count != manifest["row_count"]
            or receipt.part_count != manifest["part_count"]
            or receipt.bucket_count != manifest["layout"]["bucket_count"]
            or dict(receipt.relation_counts) != manifest["relation_counts"]
        ):
            raise ValueError("relation projection receipt does not match its artifact")
        for part in manifest["parts"]:
            yield from pq.ParquetFile(receipt.path / part["path"]).iter_batches(
                columns=columns,
                batch_size=batch_size,
            )

    def _partition_inputs(
        self,
        source: _SourceSnapshot,
        work_root: Path,
    ) -> tuple[dict[str, _PartitionSink], dict[str, int], dict[str, int]]:
        sinks = {
            name: _PartitionSink(work_root / name, self.limits)
            for name in (
                "revision_artifacts",
                "revision_evidence",
                "revision_discoveries",
                "revision_model_links",
                "revision_release_links",
                "url_frontier",
                "url_targets",
                "identifier_claims",
                "model_claims",
                "release_claims",
                "enriched_discoveries",
                "url_requests",
            )
        }
        scanned: dict[str, int] = {}
        selected: dict[str, int] = {name: 0 for name in _INPUT_COLUMNS}

        for table_name, columns in _INPUT_COLUMNS.items():
            count = 0
            for row in self._iter_source_rows(source, table_name, columns):
                count += 1
                if table_name == "artifacts":
                    if not bool(row["active"]) or not row["current_revision_id"]:
                        continue
                    artifact = _artifact_payload(row)
                    artifact["source_commit"] = source.commit
                    sinks["revision_artifacts"].add(artifact["artifact_revision_id"], artifact)
                    target_url = _web_url(artifact["canonical_url_normalized"])
                    if target_url is not None:
                        target = {**artifact, "match": {"type": "canonical_url"}}
                        sinks["url_targets"].add(target_url, target)
                elif table_name == "evidence_provenance":
                    if not (
                        (
                            row["subject_type"] == "artifact"
                            and row["predicate"] == "external_identifier"
                        )
                        or row["subject_type"] == "url"
                    ):
                        continue
                    sinks["revision_evidence"].add(row["artifact_revision_id"], row)
                elif table_name == "url_discoveries":
                    sinks["revision_discoveries"].add(row["artifact_revision_id"], row)
                elif table_name == "url_frontier":
                    sinks["url_frontier"].add(row["id"], row)
                elif table_name == "artifact_model_links":
                    sinks["revision_model_links"].add(row["artifact_revision_id"], row)
                else:
                    sinks["revision_release_links"].add(row["artifact_revision_id"], row)
                selected[table_name] += 1
            scanned[table_name] = count
            if count != source.table_counts[table_name]:
                raise ValueError(f"source table row count drifted while scanning {table_name}")
        for sink in sinks.values():
            sink.flush()
        return sinks, scanned, selected

    def _join_current_revisions(
        self,
        sinks: Mapping[str, _PartitionSink],
    ) -> None:
        artifacts_sink = sinks["revision_artifacts"]
        evidence_sink = sinks["revision_evidence"]
        discovery_sink = sinks["revision_discoveries"]
        model_sink = sinks["revision_model_links"]
        release_sink = sinks["revision_release_links"]
        for bucket in range(self.limits.bucket_count):
            artifacts = _read_work_bucket(artifacts_sink.parts.get(bucket, ()), self.limits)
            evidence = _read_work_bucket(evidence_sink.parts.get(bucket, ()), self.limits)
            discoveries = _read_work_bucket(discovery_sink.parts.get(bucket, ()), self.limits)
            model_links = _read_work_bucket(model_sink.parts.get(bucket, ()), self.limits)
            release_links = _read_work_bucket(release_sink.parts.get(bucket, ()), self.limits)
            _check_join_bound(
                (artifacts, evidence, discoveries, model_links, release_links),
                self.limits,
                f"revision bucket {bucket}",
            )
            by_revision: dict[str, Mapping[str, Any]] = {}
            for key, artifact in artifacts:
                if key in by_revision:
                    raise ValueError(f"duplicate current artifact revision {key}")
                by_revision[key] = artifact

            url_evidence: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
            for revision_id, claim in evidence:
                artifact = by_revision.get(revision_id)
                if artifact is None:
                    continue
                if claim["subject_type"] == "artifact":
                    if claim["subject_id"] != artifact["artifact_id"]:
                        raise ValueError("artifact identifier evidence has a mismatched subject")
                    value = _json_object(claim["value_json"], "identifier evidence")
                    namespace = _required_text(value.get("namespace"), "identifier namespace")
                    identifier = _required_text(value.get("value"), "identifier value")
                    group_key = _canonical_json({"namespace": namespace, "value": identifier})
                    sinks["identifier_claims"].add(
                        group_key,
                        {
                            "artifact": artifact,
                            "namespace": namespace,
                            "value": identifier,
                            "claim": _evidence_summary(claim),
                        },
                    )
                    if namespace.casefold() == "url":
                        alias = _web_url(identifier)
                        if alias is not None:
                            sinks["url_targets"].add(
                                alias,
                                {
                                    **artifact,
                                    "match": {
                                        "type": "identifier_alias",
                                        "evidence": _evidence_summary(claim),
                                    },
                                },
                            )
                    continue
                value = _json_object(claim["value_json"], "URL evidence")
                if value.get("artifact_id") != artifact["artifact_id"]:
                    raise ValueError("URL evidence has a mismatched origin artifact")
                evidence_url = _web_url(value.get("url"))
                if evidence_url is None:
                    raise ValueError("URL evidence contains a non-web URL")
                evidence_key = (
                    revision_id,
                    str(claim["subject_id"]),
                    str(claim["predicate"]),
                    str(claim["locator"] or ""),
                )
                url_evidence.setdefault(evidence_key, []).append(
                    {**_evidence_summary(claim), "value": {**value, "url": evidence_url}}
                )

            self._stage_model_or_release_claims(
                model_links,
                by_revision,
                sinks["model_claims"],
                identity_field="model_id",
            )
            self._stage_model_or_release_claims(
                release_links,
                by_revision,
                sinks["release_claims"],
                identity_field="release_id",
            )
            for revision_id, discovery in discoveries:
                origin = by_revision.get(revision_id)
                if origin is None:
                    continue
                # A shared license locator is retained as source metadata, but
                # does not assert that two artifacts are related to one another.
                # Keeping it out of the artifact-only join also prevents a common
                # corpus license from becoming a single unbounded URL work group.
                if str(discovery["relation"]).casefold() in _NON_ARTIFACT_URL_RELATIONS:
                    continue
                key = (
                    revision_id,
                    str(discovery["url_id"]),
                    str(discovery["relation"]),
                    str(discovery["locator"] or ""),
                )
                claims = sorted(url_evidence.get(key, ()), key=lambda row: str(row["id"]))
                if not claims:
                    raise ValueError("current URL discovery has no provenance evidence")
                representative = claims[0]
                sinks["enriched_discoveries"].add(
                    discovery["url_id"],
                    {
                        "origin": origin,
                        "discovery": discovery,
                        "claim": representative,
                        "claim_count": len(claims),
                    },
                )
        for name in (
            "url_targets",
            "identifier_claims",
            "model_claims",
            "release_claims",
            "enriched_discoveries",
        ):
            sinks[name].flush()

    def _stage_model_or_release_claims(
        self,
        rows: Sequence[tuple[str, Mapping[str, Any]]],
        by_revision: Mapping[str, Mapping[str, Any]],
        sink: _PartitionSink,
        *,
        identity_field: str,
    ) -> None:
        for revision_id, link in rows:
            artifact = by_revision.get(revision_id)
            if artifact is None:
                continue
            if link["artifact_id"] != artifact["artifact_id"]:
                raise ValueError(f"{identity_field} link has a mismatched artifact")
            identity = _required_text(link[identity_field], identity_field)
            sink.add(
                identity,
                {
                    "artifact": artifact,
                    "identity": identity,
                    "claim": {
                        "id": link["id"],
                        "artifact_revision_id": revision_id,
                        "local_id": link["local_id"],
                        "resolution_status": link["resolution_status"],
                        "locator": link["locator"],
                        "extractor": link["extractor"],
                        "confidence": float(link["confidence"]),
                        "created_at": link["created_at"],
                    },
                },
            )

    def _join_url_ids(
        self,
        sinks: Mapping[str, _PartitionSink],
    ) -> None:
        for bucket in range(self.limits.bucket_count):
            frontier = _read_work_bucket(sinks["url_frontier"].parts.get(bucket, ()), self.limits)
            discoveries = _read_work_bucket(
                sinks["enriched_discoveries"].parts.get(bucket, ()), self.limits
            )
            _check_join_bound((frontier, discoveries), self.limits, f"URL-ID bucket {bucket}")
            urls: dict[str, str] = {}
            for key, row in frontier:
                url = _web_url(row["url"])
                if url is None:
                    raise ValueError("URL frontier contains a non-web URL")
                previous = urls.setdefault(key, url)
                if previous != url:
                    raise ValueError(f"URL frontier identity {key} is contradictory")
            for url_id, discovery in discoveries:
                url = urls.get(url_id)
                if url is None:
                    raise ValueError(f"URL discovery references missing frontier item {url_id}")
                if discovery["claim"]["value"]["url"] != url:
                    raise ValueError("URL discovery evidence disagrees with the frontier URL")
                sinks["url_requests"].add(url, {**discovery, "url": url})
        sinks["url_requests"].flush()

    def _join_urls(
        self,
        sinks: Mapping[str, _PartitionSink],
        relation_sink: _PartitionSink,
    ) -> None:
        for bucket in range(self.limits.bucket_count):
            targets = _read_work_bucket(sinks["url_targets"].parts.get(bucket, ()), self.limits)
            requests = _read_work_bucket(sinks["url_requests"].parts.get(bucket, ()), self.limits)
            _check_join_bound((targets, requests), self.limits, f"URL bucket {bucket}")
            targets_by_url: dict[str, dict[str, Mapping[str, Any]]] = {}
            for url, target in targets:
                by_artifact = targets_by_url.setdefault(url, {})
                artifact_id = str(target["artifact_id"])
                previous = by_artifact.get(artifact_id)
                if previous is None or _target_match_rank(target) < _target_match_rank(previous):
                    by_artifact[artifact_id] = target
            for url, request in requests:
                origin = request["origin"]
                discovery = request["discovery"]
                for target in sorted(
                    targets_by_url.get(url, {}).values(),
                    key=lambda row: str(row["artifact_id"]),
                ):
                    if origin["artifact_id"] == target["artifact_id"]:
                        continue
                    evidence = {
                        "url": url,
                        "url_discovery": discovery,
                        "origin_claim": request["claim"],
                        "origin_claim_count": request["claim_count"],
                        "target_match": target["match"],
                    }
                    row = _relation_row(
                        source_commit=self._source_commit_from_artifact(origin),
                        subject=origin,
                        predicate=_required_text(discovery["relation"], "URL relation"),
                        target=target,
                        symmetric=False,
                        evidence_type="url_mention",
                        confidence=float(request["claim"]["confidence"]),
                        match_namespace="url",
                        match_value=url,
                        locator=discovery["locator"],
                        evidence=evidence,
                        evidence_identity=str(discovery["id"]),
                    )
                    relation_sink.add(row["relation_id"], row)

    def _join_identity_groups(
        self,
        sink: _PartitionSink,
        relation_sink: _PartitionSink,
    ) -> None:
        self._join_groups(
            sink,
            relation_sink,
            predicate="same_artifact",
            evidence_type="shared_artifact_identifier",
            namespace=lambda row: str(row["namespace"]),
            value=lambda row: str(row["value"]),
        )

    def _join_model_groups(
        self,
        sink: _PartitionSink,
        relation_sink: _PartitionSink,
    ) -> None:
        self._join_groups(
            sink,
            relation_sink,
            predicate="documents_same_model",
            evidence_type="shared_model_identity",
            namespace=lambda _row: "registry:model",
            value=lambda row: str(row["identity"]),
        )

    def _join_release_groups(
        self,
        sink: _PartitionSink,
        relation_sink: _PartitionSink,
    ) -> None:
        self._join_groups(
            sink,
            relation_sink,
            predicate="documents_same_release",
            evidence_type="shared_release_identity",
            namespace=lambda _row: "registry:release",
            value=lambda row: str(row["identity"]),
        )

    def _join_groups(
        self,
        sink: _PartitionSink,
        relation_sink: _PartitionSink,
        *,
        predicate: str,
        evidence_type: str,
        namespace: Any,
        value: Any,
    ) -> None:
        for bucket in range(self.limits.bucket_count):
            rows = _read_work_bucket(sink.parts.get(bucket, ()), self.limits)
            groups: dict[str, list[Mapping[str, Any]]] = {}
            for key, row in rows:
                groups.setdefault(key, []).append(row)
            for group_key in sorted(groups):
                by_artifact: dict[str, list[Mapping[str, Any]]] = {}
                for claim in groups[group_key]:
                    artifact_id = str(claim["artifact"]["artifact_id"])
                    by_artifact.setdefault(artifact_id, []).append(claim)
                if len(by_artifact) < 2:
                    continue
                artifact_ids = sorted(by_artifact)
                anchor_id = artifact_ids[0]
                anchor_claims = sorted(
                    by_artifact[anchor_id], key=lambda row: _canonical_json(dict(row))
                )
                anchor = anchor_claims[0]
                for target_id in artifact_ids[1:]:
                    target_claims = sorted(
                        by_artifact[target_id], key=lambda row: _canonical_json(dict(row))
                    )
                    target_claim = target_claims[0]
                    match_namespace = namespace(anchor)
                    match_value = value(anchor)
                    evidence = {
                        "group_key": group_key,
                        "subject_claim": anchor["claim"],
                        "subject_claim_count": len(anchor_claims),
                        "target_claim": target_claim["claim"],
                        "target_claim_count": len(target_claims),
                    }
                    confidence = min(
                        float(anchor["claim"]["confidence"]),
                        float(target_claim["claim"]["confidence"]),
                    )
                    row = _relation_row(
                        source_commit=self._source_commit_from_artifact(anchor["artifact"]),
                        subject=anchor["artifact"],
                        predicate=predicate,
                        target=target_claim["artifact"],
                        symmetric=True,
                        evidence_type=evidence_type,
                        confidence=confidence,
                        match_namespace=match_namespace,
                        match_value=match_value,
                        locator=None,
                        evidence=evidence,
                        evidence_identity=group_key,
                    )
                    relation_sink.add(row["relation_id"], row)

    @staticmethod
    def _source_commit_from_artifact(artifact: Mapping[str, Any]) -> str:
        value = artifact.get("source_commit")
        return _required_text(value, "source_commit")

    def _write_output_parts(
        self,
        source: _SourceSnapshot,
        artifact_id: str,
        sink: _PartitionSink,
        parts_root: Path,
    ) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
        parts: list[dict[str, Any]] = []
        row_count = 0
        counts: dict[str, int] = {}
        for bucket in range(self.limits.bucket_count):
            rows = _read_work_bucket(sink.parts.get(bucket, ()), self.limits)
            unique: dict[str, Mapping[str, Any]] = {}
            for relation_id, row in rows:
                prior = unique.get(relation_id)
                if prior is not None and prior != row:
                    raise ValueError(f"relation identity {relation_id} is contradictory")
                unique[relation_id] = row
            ordered = [unique[key] for key in sorted(unique)]
            buffer: list[Mapping[str, Any]] = []
            buffer_bytes = 0
            part_index = 0
            for row in ordered:
                if row["source_commit"] != source.commit:
                    raise ValueError("relation row lost its source commit")
                expected_id = _relation_id_from_row(row)
                if row["relation_id"] != expected_id:
                    raise ValueError("relation row identity is not reproducible")
                row_bytes = len(_canonical_json(dict(row)).encode())
                if row_bytes > self.limits.max_output_row_bytes:
                    raise ValueError("relation row exceeded max_output_row_bytes")
                if buffer and (
                    len(buffer) >= self.limits.output_part_rows
                    or buffer_bytes + row_bytes > self.limits.max_output_buffer_bytes
                ):
                    parts.append(_write_output_part(parts_root, bucket, part_index, buffer))
                    row_count += len(buffer)
                    part_index += 1
                    buffer = []
                    buffer_bytes = 0
                buffer.append(row)
                buffer_bytes += row_bytes
                kind = str(row["evidence_type"])
                counts[kind] = counts.get(kind, 0) + 1
            if buffer:
                parts.append(_write_output_part(parts_root, bucket, part_index, buffer))
                row_count += len(buffer)
        return parts, row_count, dict(sorted(counts.items()))

    def _initialize_output(self) -> None:
        if self.store_root.is_symlink() or not self.store_root.is_dir():
            raise ValueError(f"registry store root is not a plain directory: {self.store_root}")
        if self.output_root.exists() and not self.output_root.is_dir():
            raise ValueError(f"relation projection root is not a directory: {self.output_root}")
        if self.output_root.is_symlink():
            raise ValueError("relation projection root must not be a symlink")
        self.output_root.mkdir(parents=True, exist_ok=True)
        staging = self.output_root / ".staging"
        staging.mkdir(exist_ok=True)
        _require_plain_directory(self.output_root, "relation projection root")
        _require_plain_directory(staging, "relation projection staging directory")

    def _current_source_snapshot(self) -> _SourceSnapshot:
        head_path = self.store_root / "HEAD.json"
        if head_path.is_symlink() or not head_path.is_file():
            raise ValueError(f"registry store HEAD is missing or invalid: {head_path}")
        head = _read_json(head_path)
        store_format = head.get("format")
        if store_format not in _STORE_FORMATS:
            raise ValueError("unsupported registry store format")
        commit = _required_text(head.get("commit"), "HEAD commit")
        source = self._source_snapshot(commit, store_format)
        if source.generation != _nonnegative_int(
            head.get("generation"), "HEAD generation"
        ) or source.state_digest != _digest(head.get("state_digest"), "HEAD state_digest"):
            raise ValueError("registry HEAD does not match its immutable commit")
        return source

    def _source_snapshot(self, commit: str, store_format: str | None = None) -> _SourceSnapshot:
        commit = _required_text(commit, "source commit")
        if _COMMIT_ID.fullmatch(commit) is None:
            raise ValueError("source commit identifier is invalid")
        commit_path = self.store_root / "commits" / commit
        manifest_path = commit_path / "manifest.json"
        tables_path = commit_path / "tables"
        _require_plain_directory(commit_path, "source commit directory")
        _require_plain_directory(tables_path, "source tables directory")
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("source commit manifest is missing")
        manifest = _read_json(manifest_path)
        manifest_format = manifest.get("format")
        if manifest_format not in _STORE_FORMATS or manifest.get("commit") != commit:
            raise ValueError("source commit manifest identity mismatch")
        if store_format is not None and manifest_format != store_format:
            raise ValueError("source commit manifest identity mismatch")
        generation = _nonnegative_int(manifest.get("generation"), "source generation")
        state_digest = _digest(manifest.get("state_digest"), "source state_digest")
        raw_counts = manifest.get("table_counts")
        if not isinstance(raw_counts, Mapping):
            raise ValueError("source table_counts is invalid")
        counts: dict[str, int] = {}
        for table_name in _INPUT_COLUMNS:
            count = _nonnegative_int(raw_counts.get(table_name), f"{table_name} row count")
            table_path = tables_path / f"{table_name}.parquet"
            if table_path.is_symlink() or not table_path.is_file():
                raise ValueError(f"source table is missing: {table_path}")
            if pq.read_metadata(table_path).num_rows != count:
                raise ValueError(f"source table metadata count mismatch: {table_name}")
            counts[table_name] = count
        return _SourceSnapshot(
            root=self.store_root,
            commit=commit,
            generation=generation,
            state_digest=state_digest,
            path=commit_path,
            table_counts=counts,
        )

    def _iter_source_rows(
        self,
        source: _SourceSnapshot,
        table_name: str,
        columns: Sequence[str],
    ) -> Iterator[dict[str, Any]]:
        path = source.path / "tables" / f"{table_name}.parquet"
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows == 0:
            return
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"source table {table_name} is missing columns {sorted(missing)}")
        for batch in parquet.iter_batches(
            columns=columns,
            batch_size=self.limits.scan_batch_rows,
        ):
            if batch.nbytes > self.limits.max_scan_batch_bytes:
                raise ValueError(f"source table {table_name} scan batch exceeded byte limit")
            yield from _iter_arrow_rows(batch)

    def _artifact_id(self, source: _SourceSnapshot) -> str:
        return _json_sha256(
            {
                "format": _FORMAT,
                "algorithm": _ALGORITHM,
                "source_commit": source.commit,
                "source_generation": source.generation,
                "source_state_digest": source.state_digest,
                "semantics": self._semantics(),
                "layout": self._layout(),
                "schema": str(RELATION_SCHEMA),
            }
        )

    def _layout(self) -> dict[str, int]:
        return {
            "bucket_count": self.limits.bucket_count,
            "output_part_rows": self.limits.output_part_rows,
            "max_output_buffer_bytes": self.limits.max_output_buffer_bytes,
        }

    @staticmethod
    def _semantics() -> dict[str, str]:
        return {
            "directed_edges": "one-claim-per-current-url-discovery-and-target",
            "symmetric_groups": "lexicographic-star-preserving-connectivity",
            "identity_resolution": "exact-identifiers-only-never-names",
        }

    def _final_path(self, commit: str, artifact_id: str) -> Path:
        return self.output_root / commit / artifact_id

    def _verify_projection(
        self,
        path: Path,
        source: _SourceSnapshot,
        artifact_id: str,
    ) -> Mapping[str, Any]:
        expected_path = self._final_path(source.commit, artifact_id)
        if path.absolute() != expected_path:
            raise ValueError("relation projection path is noncanonical")
        _require_plain_directory(path, "relation projection directory")
        manifest = _read_json(path / "manifest.json")
        manifest_digest = manifest.get("manifest_sha256")
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256", None)
        if manifest_digest != _json_sha256(unsigned):
            raise ValueError("relation projection manifest checksum mismatch")
        expected = {
            "format": _FORMAT,
            "algorithm": _ALGORITHM,
            "artifact_id": artifact_id,
            "semantics": self._semantics(),
            "layout": self._layout(),
            "schema": str(RELATION_SCHEMA),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("relation projection manifest identity mismatch")
        source_value = manifest.get("source")
        if not isinstance(source_value, Mapping) or any(
            source_value.get(key) != value
            for key, value in {
                "commit": source.commit,
                "generation": source.generation,
                "state_digest": source.state_digest,
                "table_counts": dict(source.table_counts),
            }.items()
        ):
            raise ValueError("relation projection source identity mismatch")
        row_count = _nonnegative_int(manifest.get("row_count"), "projection row_count")
        part_count = _nonnegative_int(manifest.get("part_count"), "projection part_count")
        relation_counts = manifest.get("relation_counts")
        if not isinstance(relation_counts, Mapping):
            raise ValueError("relation projection counts are invalid")
        if any(
            not isinstance(key, str) or not key or _nonnegative_int(value, "relation count") < 1
            for key, value in relation_counts.items()
        ):
            raise ValueError("relation projection counts are invalid")
        if sum(int(value) for value in relation_counts.values()) != row_count:
            raise ValueError("relation projection counts do not sum to row_count")
        parts = manifest.get("parts")
        if not isinstance(parts, list) or len(parts) != part_count:
            raise ValueError("relation projection part count mismatch")
        seen_positions: set[tuple[int, int]] = set()
        expected_entries: dict[str, set[str]] = {}
        observed_rows = 0
        prior_by_bucket: dict[int, str] = {}
        for part in parts:
            if not isinstance(part, Mapping):
                raise ValueError("relation projection part is invalid")
            bucket = _nonnegative_int(part.get("bucket"), "part bucket")
            if bucket >= self.limits.bucket_count:
                raise ValueError("relation projection part bucket is out of range")
            name = _required_text(part.get("name"), "part name")
            match = re.fullmatch(r"part-(\d{6})\.parquet", name)
            if match is None:
                raise ValueError("relation projection part name is noncanonical")
            position = (bucket, int(match.group(1)))
            if position in seen_positions:
                raise ValueError("relation projection part position is duplicated")
            seen_positions.add(position)
            relative = f"parts/bucket-{bucket:06d}/{name}"
            if part.get("path") != relative:
                raise ValueError("relation projection part path is noncanonical")
            part_path = path / relative
            if part_path.is_symlink() or not part_path.is_file():
                raise ValueError(f"relation projection part is missing: {part_path}")
            byte_count = _nonnegative_int(part.get("byte_count"), "part byte_count")
            expected_rows = _nonnegative_int(part.get("row_count"), "part row_count")
            if expected_rows < 1 or part_path.stat().st_size != byte_count:
                raise ValueError("relation projection part size is invalid")
            if _file_sha256(part_path) != _digest(part.get("sha256"), "part sha256"):
                raise ValueError("relation projection part checksum mismatch")
            metadata = pq.read_metadata(part_path)
            if metadata.schema.to_arrow_schema() != RELATION_SCHEMA:
                raise ValueError("relation projection part schema mismatch")
            if metadata.num_rows != expected_rows:
                raise ValueError("relation projection part row count mismatch")
            observed = 0
            first: str | None = None
            last = prior_by_bucket.get(bucket)
            for batch in pq.ParquetFile(part_path).iter_batches(
                columns=("relation_id", "source_commit"),
                batch_size=self.limits.join_batch_rows,
            ):
                if batch.nbytes > self.limits.max_join_batch_bytes:
                    raise ValueError("projection verification batch exceeded byte limit")
                for row in _iter_arrow_rows(batch):
                    relation_id = _required_text(row["relation_id"], "relation_id")
                    if first is None:
                        first = relation_id
                    if last is not None and relation_id <= last:
                        raise ValueError("relation projection IDs are not strictly ordered")
                    if _bucket(relation_id, self.limits.bucket_count) != bucket:
                        raise ValueError("relation projection row is in the wrong bucket")
                    if row["source_commit"] != source.commit:
                        raise ValueError("relation projection row source commit mismatch")
                    last = relation_id
                    observed += 1
            if observed != expected_rows:
                raise ValueError("relation projection part scan count mismatch")
            if first != part.get("min_relation_id") or last != part.get("max_relation_id"):
                raise ValueError("relation projection part bounds mismatch")
            prior_by_bucket[bucket] = str(last)
            observed_rows += observed
            expected_entries.setdefault(f"bucket-{bucket:06d}", set()).add(name)
        if observed_rows != row_count:
            raise ValueError("relation projection row count mismatch")
        parts_root = path / "parts"
        _require_plain_directory(parts_root, "relation projection parts directory")
        if {entry.name for entry in parts_root.iterdir()} != set(expected_entries):
            raise ValueError("relation projection parts contain unexpected directories")
        for directory, names in expected_entries.items():
            bucket_path = parts_root / directory
            _require_plain_directory(bucket_path, "relation projection bucket")
            if {entry.name for entry in bucket_path.iterdir()} != names:
                raise ValueError("relation projection bucket contains unexpected files")
        if {entry.name for entry in path.iterdir()} != {"manifest.json", "SEAL.json", "parts"}:
            raise ValueError("relation projection directory contains unexpected entries")
        seal = _read_json(path / "SEAL.json")
        unsigned_seal = dict(seal)
        seal_digest = unsigned_seal.pop("seal_sha256", None)
        if seal_digest != _json_sha256(unsigned_seal):
            raise ValueError("relation projection seal checksum mismatch")
        if any(
            seal.get(key) != value
            for key, value in {
                "format": _FORMAT,
                "artifact_id": artifact_id,
                "manifest_sha256": manifest_digest,
                "data_sha256": _json_sha256({"parts": parts}),
                "row_count": row_count,
                "part_count": part_count,
                "sealed_at": manifest.get("created_at"),
            }.items()
        ):
            raise ValueError("relation projection seal does not match its manifest")
        return manifest


def _artifact_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    canonical = _required_text(row["canonical_url_normalized"], "normalized URL")
    return {
        "artifact_id": _required_text(row["id"], "artifact id"),
        "artifact_revision_id": _required_text(
            row["current_revision_id"], "current artifact revision"
        ),
        "kind": _required_text(row["kind"], "artifact kind"),
        "source": _required_text(row["source"], "artifact source"),
        "source_record_id": _required_text(row["source_record_id"], "source record id"),
        "canonical_url": _required_text(row["canonical_url"], "canonical URL"),
        "canonical_url_normalized": canonical,
        # Filled by the materializer before a relation row is emitted.
        "source_commit": "",
    }


def _evidence_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "artifact_revision_id": row["artifact_revision_id"],
        "subject_type": row["subject_type"],
        "subject_id": row["subject_id"],
        "predicate": row["predicate"],
        "locator": row["locator"],
        "extractor": row["extractor"],
        "confidence": float(row["confidence"]),
        "created_at": row["created_at"],
    }


def _relation_row(
    *,
    source_commit: str,
    subject: Mapping[str, Any],
    predicate: str,
    target: Mapping[str, Any],
    symmetric: bool,
    evidence_type: str,
    confidence: float,
    match_namespace: str | None,
    match_value: str | None,
    locator: str | None,
    evidence: Mapping[str, Any],
    evidence_identity: str,
) -> dict[str, Any]:
    if subject["artifact_id"] == target["artifact_id"]:
        raise ValueError("artifact relation cannot be a self edge")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("artifact relation confidence must be between zero and one")
    identity = {
        "subject_artifact_id": subject["artifact_id"],
        "predicate": predicate,
        "target_artifact_id": target["artifact_id"],
        "symmetric": symmetric,
        "evidence_type": evidence_type,
        "evidence_identity": evidence_identity,
    }
    relation_id = _json_sha256(identity)
    return {
        "relation_id": relation_id,
        "subject_artifact_id": subject["artifact_id"],
        "predicate": predicate,
        "target_artifact_id": target["artifact_id"],
        "symmetric": symmetric,
        "evidence_type": evidence_type,
        "confidence": confidence,
        "match_namespace": match_namespace,
        "match_value": match_value,
        "locator": locator,
        "subject_artifact_revision_id": subject["artifact_revision_id"],
        "subject_kind": subject["kind"],
        "subject_source": subject["source"],
        "subject_source_record_id": subject["source_record_id"],
        "subject_canonical_url": subject["canonical_url"],
        "target_artifact_revision_id": target["artifact_revision_id"],
        "target_kind": target["kind"],
        "target_source": target["source"],
        "target_source_record_id": target["source_record_id"],
        "target_canonical_url": target["canonical_url"],
        "evidence_json": _canonical_json(dict(evidence)),
        "source_commit": source_commit,
    }


def _relation_id_from_row(row: Mapping[str, Any]) -> str:
    evidence = _json_object(row["evidence_json"], "relation evidence")
    if row["evidence_type"] == "url_mention":
        evidence_identity = str(evidence["url_discovery"]["id"])
    else:
        evidence_identity = str(evidence["group_key"])
    return _json_sha256(
        {
            "subject_artifact_id": row["subject_artifact_id"],
            "predicate": row["predicate"],
            "target_artifact_id": row["target_artifact_id"],
            "symmetric": row["symmetric"],
            "evidence_type": row["evidence_type"],
            "evidence_identity": evidence_identity,
        }
    )


def _target_match_rank(row: Mapping[str, Any]) -> tuple[int, str]:
    match = row.get("match")
    kind = match.get("type") if isinstance(match, Mapping) else None
    return (0 if kind == "canonical_url" else 1, _canonical_json(dict(row)))


def _write_output_part(
    parts_root: Path,
    bucket: int,
    part_index: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bucket_root = parts_root / f"bucket-{bucket:06d}"
    bucket_root.mkdir(exist_ok=True)
    name = f"part-{part_index:06d}.parquet"
    path = bucket_root / name
    pq.write_table(
        pa.Table.from_pylist(list(rows), schema=RELATION_SCHEMA),
        path,
        compression="zstd",
        write_statistics=True,
    )
    return {
        "bucket": bucket,
        "name": name,
        "path": f"parts/{bucket_root.name}/{name}",
        "row_count": len(rows),
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
        "min_relation_id": rows[0]["relation_id"],
        "max_relation_id": rows[-1]["relation_id"],
    }


def _read_work_bucket(
    parts: Sequence[_WorkPart],
    limits: ArtifactRelationLimits,
) -> list[tuple[str, Mapping[str, Any]]]:
    result: list[tuple[str, Mapping[str, Any]]] = []
    payload_bytes = 0
    for expected_index, part in enumerate(parts):
        if part.path.name != f"part-{expected_index:06d}.parquet":
            raise ValueError("working partition part sequence is noncanonical")
        if part.path.is_symlink() or not part.path.is_file():
            raise ValueError("working partition part is missing")
        if part.path.stat().st_size != part.byte_count or _file_sha256(part.path) != part.sha256:
            raise ValueError("working partition part checksum mismatch")
        metadata = pq.read_metadata(part.path)
        if metadata.num_rows != part.row_count or metadata.schema.to_arrow_schema() != _WORK_SCHEMA:
            raise ValueError("working partition metadata mismatch")
        for batch in pq.ParquetFile(part.path).iter_batches(batch_size=limits.join_batch_rows):
            if batch.nbytes > limits.max_join_batch_bytes:
                raise ValueError("working partition join batch exceeded byte limit")
            for raw in _iter_arrow_rows(batch):
                key = _required_text(raw["key"], "working partition key")
                payload_json = raw["payload_json"]
                payload = _json_object(payload_json, "working partition payload")
                payload_bytes += len(key.encode()) + len(payload_json.encode())
                result.append((key, payload))
                if len(result) > limits.max_bucket_rows:
                    raise ValueError("working partition exceeded max_bucket_rows")
                if payload_bytes > limits.max_bucket_bytes:
                    raise ValueError("working partition exceeded max_bucket_bytes")
    return result


def _check_join_bound(
    groups: Sequence[Sequence[tuple[str, Mapping[str, Any]]]],
    limits: ArtifactRelationLimits,
    label: str,
) -> None:
    rows = sum(len(group) for group in groups)
    if rows > limits.max_bucket_rows:
        raise ValueError(f"{label} exceeded max_bucket_rows; increase bucket_count")
    payload_bytes = sum(
        len(key.encode()) + len(_canonical_json(dict(payload)).encode())
        for group in groups
        for key, payload in group
    )
    if payload_bytes > limits.max_bucket_bytes:
        raise ValueError(f"{label} exceeded max_bucket_bytes; increase bucket_count")


def _receipt(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    already_materialized: bool,
) -> ArtifactRelationReceipt:
    source = manifest["source"]
    return ArtifactRelationReceipt(
        source_commit=str(source["commit"]),
        source_generation=int(source["generation"]),
        source_state_digest=str(source["state_digest"]),
        artifact_id=str(manifest["artifact_id"]),
        row_count=int(manifest["row_count"]),
        part_count=int(manifest["part_count"]),
        bucket_count=int(manifest["layout"]["bucket_count"]),
        relation_counts=tuple(sorted(manifest["relation_counts"].items())),
        path=path,
        already_materialized=already_materialized,
    )


def _web_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    result = canonicalize_url(value)
    parts = urlsplit(result)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    return result


def _bucket(key: str, count: int) -> int:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def _iter_arrow_rows(batch: pa.RecordBatch) -> Iterator[dict[str, Any]]:
    for index in range(batch.num_rows):
        yield {
            name: batch.column(position)[index].as_py()
            for position, name in enumerate(batch.schema.names)
        }


def _json_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be JSON text")
    try:
        result = json.loads(value)
    except (RecursionError, TypeError, ValueError):
        raise ValueError(f"{label} is malformed") from None
    if not isinstance(result, Mapping):
        raise ValueError(f"{label} must contain an object")
    if _canonical_json(dict(result)) != value:
        raise ValueError(f"{label} must be canonical JSON")
    return result


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
        raise ValueError("value is not canonicalizable JSON") from None


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(value)).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value.casefold()) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value.casefold()


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON metadata file is missing or invalid: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"JSON metadata file is malformed: {path}") from None
    if not isinstance(value, dict):
        raise ValueError(f"JSON metadata must contain an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _require_plain_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a plain directory: {path}")


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
