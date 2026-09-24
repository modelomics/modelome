from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelRelationHint,
    ModelStatus,
    ReleaseHint,
    SourceIssue,
    SourcePage,
    SourceRecord,
    SyncStats,
)
from modelome.normalize import (
    canonicalize_url,
    content_hash,
    extract_url_mentions,
    identifier_from_url,
    infer_url_relation,
    normalize_name,
)

_STATUS_RANK = {
    ModelStatus.STUB.value: 0,
    ModelStatus.CANDIDATE.value: 1,
    ModelStatus.DOCUMENTED.value: 2,
    ModelStatus.RELEASED.value: 3,
}
_RUN_LEASE_SECONDS = 6 * 60 * 60
_STORE_FORMAT = "modelome-parquet-snapshot-v2"
_LEGACY_STORE_FORMATS = frozenset(
    {"modelome-parquet-snapshot-v1", "ulr-parquet-snapshot-v1"}
)
_SUPPORTED_STORE_FORMATS = _LEGACY_STORE_FORMATS | {_STORE_FORMAT}
_COMMIT_ID_PATTERN = re.compile(r"^\d{20}-[0-9a-f]{16}-[0-9a-f]{12}$")
_TABLES = (
    "source_checkpoints",
    "sync_runs",
    "artifacts",
    "artifact_revisions",
    "artifact_identifiers",
    "models",
    "model_aliases",
    "model_external_identifiers",
    "model_identifier_claims",
    "artifact_model_links",
    "model_relation_claims",
    "url_frontier",
    "url_discoveries",
    "evidence_provenance",
    "dead_letters",
    "model_releases",
    "release_external_identifiers",
    "release_identifier_claims",
    "artifact_release_links",
)

# Transaction rollback needs isolated rows only for tables whose existing rows
# can be updated in place. The other tables are append-only (or are replaced as
# whole lists while pruning a derived extraction), so copying their lists is
# sufficient and avoids duplicating the historical evidence corpus for every
# page checkpoint.
_MUTABLE_ROW_TABLES = frozenset(
    {
        "source_checkpoints",
        "sync_runs",
        "artifacts",
        "models",
        "model_external_identifiers",
        "url_frontier",
        "dead_letters",
        "model_releases",
        "release_external_identifiers",
    }
)


class Database:
    """Atomic, append-only Parquet persistence for registry evidence.

    Every successful write creates a new immutable commit directory containing
    one Parquet file per logical table. ``HEAD.json`` is the only mutable data
    pointer and is replaced atomically after the commit is durable. All joins,
    filtering, identity resolution, and projections are performed in Python.

    Names never resolve identity. A model or release is reused only when an
    exact namespaced identifier has a bound claim in current evidence.
    """

    def __init__(self, path: str | Path):
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        if str(path) == ":memory:":
            self._temporary = tempfile.TemporaryDirectory(prefix="modelome-parquet-")
            self.root = Path(self._temporary.name).resolve()
        else:
            self.root = Path(path).expanduser().resolve()
        self.path = str(self.root)
        self._thread_lock = threading.RLock()
        self._tables: dict[str, list[dict[str, Any]]] = _empty_tables()
        self._head_commit: str | None = None
        self._head_generation = 0
        self._table_digests: dict[str, str] = {}
        self._table_schemas: dict[str, str] = {}
        self._transaction_base_tables: dict[str, list[dict[str, Any]]] | None = None
        self._ingest_batch_active = False
        self._initialized = False
        self._by_id: dict[str, dict[Any, dict[str, Any]]] = {}
        self._artifact_by_source_record: dict[tuple[str, str], dict[str, Any]] = {}
        self._artifact_identifier_keys: set[tuple[str, str, str]] = set()
        self._model_alias_keys: set[tuple[str, str]] = set()
        self._model_identifier_claims_by_key: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = {}
        self._model_external_identifiers_by_key: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = {}
        self._derived_extraction_keys: set[tuple[str, str]] = set()
        self._source_by_name: dict[str, dict[str, Any]] = {}
        self._frontier_by_url: dict[str, dict[str, Any]] = {}
        self._url_discovery_url_ids_by_relation: dict[str, set[str]] = {}
        self._url_discoveries_by_url_id: dict[str, list[dict[str, Any]]] = {}
        self._weight_artifact_urls: set[str] = set()
        self._reindex()

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def initialize(self) -> None:
        with self._thread_lock:
            if self.root.exists() and not self.root.is_dir():
                raise ValueError(
                    f"Parquet store path is not a directory: {self.root}. "
                    "Choose a new store directory; single-file stores are not supported."
                )
            self.root.mkdir(parents=True, exist_ok=True)
            self._validate_store_layout()
            with self._file_lock():
                head_path = self.root / "HEAD.json"
                if head_path.is_symlink():
                    raise ValueError(f"Parquet store HEAD must not be a symlink: {head_path}")
                if not head_path.exists():
                    existing_commits = tuple((self.root / "commits").iterdir())
                    if existing_commits:
                        raise ValueError(
                            "Parquet store HEAD is missing but immutable commits exist; "
                            "refusing to publish an empty replacement. Restore HEAD from "
                            "backup after inspecting the retained commits."
                        )
                    self._tables = _empty_tables()
                    self._head_commit = None
                    self._head_generation = 0
                    self._table_digests = {}
                    self._table_schemas = {}
                    self._commit("initialize")
                else:
                    self._load(force=True)
            self._initialized = True

    def table_rows(self, table: str) -> list[dict[str, Any]]:
        """Return decoded copies of rows in the current logical table."""

        if table not in _TABLES:
            raise KeyError(f"unknown logical table: {table}")
        self._read_current()
        return deepcopy(self._tables[table])

    def observed_declared_url_discoveries(
        self, relations: set[str]
    ) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
        """Yield indexed observed URLs with structured discoveries for relations."""

        self._read_current()
        for relation in relations:
            for url_id in self._url_discovery_url_ids_by_relation.get(relation, ()):
                frontier = self._by_id["url_frontier"].get(url_id)
                if frontier is None or frontier.get("status") != "observed":
                    continue
                for discovery in self._url_discoveries_by_url_id.get(url_id, ()):
                    if str(discovery.get("relation", "")).casefold() == relation:
                        yield frontier, discovery

    def declared_url_discoveries_for_ids(
        self, url_ids: set[str]
    ) -> Iterable[dict[str, Any]]:
        """Yield indexed URL discoveries for a bounded set of frontier IDs."""

        self._read_current()
        for url_id in url_ids:
            frontier = self._by_id["url_frontier"].get(url_id)
            if frontier is None:
                continue
            yield from self._url_discoveries_by_url_id.get(url_id, ())

    def has_weight_artifact_url(self, url: str) -> bool:
        self._read_current()
        return canonicalize_url(url) in self._weight_artifact_urls

    def get_source_state(self, name: str) -> dict[str, Any]:
        self._read_current()
        row = self._source_by_name.get(name)
        return {} if row is None else _load_object(row["state_json"])

    def list_control_records(
        self,
        source: str,
        *,
        record_type: str | None = None,
        limit: int | None = None,
    ) -> list[SourceRecord]:
        """Return active current catalog controls for a bulk data plane.

        Control records are ordinary immutable source evidence. This method
        deliberately reconstructs only records with no model or release claims;
        bulk downloaders cannot accidentally reinterpret registry assertions as
        transfer instructions.
        """

        source = _required(source, "source")
        requested_type = _optional_text(record_type)
        if limit is not None and (isinstance(limit, bool) or limit < 1):
            raise ValueError("control-record limit must be positive")
        self._read_current()
        result: list[SourceRecord] = []
        artifacts = sorted(
            (
                artifact
                for artifact in self._tables["artifacts"]
                if artifact["source"] == source
                and artifact["kind"] == ArtifactKind.CATALOG_RECORD.value
                and artifact["active"] == 1
            ),
            key=lambda artifact: artifact["source_record_id"],
        )
        for artifact in artifacts:
            revision_id = artifact.get("current_revision_id")
            revision = self._by_id["artifact_revisions"].get(revision_id)
            if revision is None:
                raise ValueError(
                    f"active control artifact has no current revision: {artifact['id']}"
                )
            payload = _load_object(revision["record_json"])
            record = _control_record_from_payload(payload)
            raw_type = _optional_text(record.raw.get("record_type"))
            if requested_type is not None and raw_type != requested_type:
                continue
            result.append(record)
            if limit is not None and len(result) >= limit:
                break
        return result

    def start_run(self, source: str) -> int:
        source = _required(source, "source")
        now = _now()
        stale_before = _as_datetime(now) - timedelta(seconds=_RUN_LEASE_SECONDS)
        with self._write_transaction("start_run"):
            self._ensure_source(source, now)
            running = next(
                (
                    row
                    for row in self._tables["sync_runs"]
                    if row["source"] == source and row["status"] == "running"
                ),
                None,
            )
            if running is not None:
                heartbeat = _as_datetime(running.get("heartbeat_at") or running["started_at"])
                if heartbeat >= stale_before:
                    raise RuntimeError(
                        f"{source}: another sync run is active (run {running['id']})"
                    )
                running.update(
                    status="abandoned",
                    finished_at=now,
                    heartbeat_at=now,
                    error=running.get("error") or "sync run lease expired",
                )
            run_id = 1 + max((int(row["id"]) for row in self._tables["sync_runs"]), default=0)
            self._tables["sync_runs"].append(
                {
                    "id": run_id,
                    "source": source,
                    "status": "running",
                    "started_at": now,
                    "finished_at": None,
                    "heartbeat_at": now,
                    "stats_json": "{}",
                    "error": None,
                }
            )
            self._reindex()
            return run_id

    def finish_run(
        self,
        run_id: int,
        status: str,
        stats: SyncStats | Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        status = _required(status, "status")
        stats_value: Mapping[str, Any] = (
            stats.as_dict() if isinstance(stats, SyncStats) else stats or {}
        )
        stats_json = _dump_json(dict(stats_value))
        with self._write_transaction("finish_run"):
            run = self._by_id["sync_runs"].get(run_id)
            if run is None:
                raise KeyError(f"unknown sync run: {run_id}")
            now = _now()
            run.update(
                status=status,
                finished_at=now,
                heartbeat_at=now,
                stats_json=stats_json,
                error=error,
            )
            return _run_dict(run)

    def ingest_page(
        self,
        source: str,
        records: Iterable[SourceRecord] | SourcePage,
        next_state: Mapping[str, Any] | None = None,
        run_id: int | None = None,
        extractor: str | Any = "source",
        *,
        complete: bool | None = None,
        upstream_count: int | None = None,
        model_hints: Mapping[str, Iterable[ModelHint]]
        | Callable[[SourceRecord], Iterable[ModelHint]]
        | None = None,
        quarantine_errors: bool = True,
        enqueue_links: bool = True,
        link_depth: int = 0,
        authoritative_snapshot: bool | None = None,
        allow_large_snapshot_shrink: bool = False,
        artifact_source: str | None = None,
    ) -> dict[str, int]:
        """Persist a page and checkpoint in one atomic Parquet commit.

        ``source`` owns run/checkpoint state. ``artifact_source`` can retain the
        upstream identity namespace when a separate traversal, such as a historical
        backfill, reads records from that same source.
        """

        source = _required(source, "source")
        artifact_source = _required(artifact_source or source, "artifact_source")
        if link_depth < 0:
            raise ValueError("link_depth must be non-negative")
        retry_state: Mapping[str, Any] | None = None
        advance_on_source_issues = False
        if isinstance(records, SourcePage):
            page_records: Sequence[SourceRecord] = records.records
            page_issues: Sequence[SourceIssue] = records.issues
            if next_state is None:
                next_state = records.next_state
            if complete is None:
                complete = records.complete
            if upstream_count is None:
                upstream_count = records.upstream_count
            if authoritative_snapshot is None:
                authoritative_snapshot = records.authoritative_snapshot
            retry_state = records.retry_state
            advance_on_source_issues = records.advance_on_source_issues
        else:
            page_records = tuple(records)
            page_issues = ()
        next_state = next_state or {}
        complete = bool(complete) if complete is not None else False
        authoritative_snapshot = bool(authoritative_snapshot)
        if upstream_count is not None and upstream_count < 0:
            raise ValueError("upstream_count must be non-negative")
        if authoritative_snapshot and not complete:
            raise ValueError("an authoritative snapshot must be complete")
        if authoritative_snapshot and run_id is None:
            raise ValueError("an authoritative snapshot requires a sync run")
        extractor_name, extractor_callback = _extractor_parts(extractor)
        source_extractor = "source-declared" if extractor_callback is not None else extractor_name
        derived_extractor = extractor_name if extractor_callback is not None else None
        _dump_json(dict(next_state))
        if retry_state is not None:
            _dump_json(dict(retry_state))

        result = {
            "records_seen": len(page_records) + len(page_issues),
            "new_artifacts": 0,
            "new_revisions": 0,
            "models_touched": 0,
            "links_discovered": 0,
            "errors": 0,
            "record_errors": 0,
            "checkpoint_advanced": 1,
        }
        touched_models: set[str] = set()
        now = _now()
        transaction = (
            nullcontext()
            if self._ingest_batch_active
            else self._write_transaction("ingest_page")
        )
        with transaction:
            self._ensure_source(source, now)
            checkpoint = self._source_by_name[source]
            previous_upstream_count = checkpoint["upstream_count"]
            if run_id is not None:
                run = self._by_id["sync_runs"].get(run_id)
                if run is None:
                    raise KeyError(f"unknown sync run: {run_id}")
                if run["source"] != source:
                    raise ValueError(
                        f"sync run {run_id} belongs to {run['source']!r}, not {source!r}"
                    )
                if run["status"] != "running":
                    raise ValueError(f"sync run {run_id} is not running")
            if (
                authoritative_snapshot
                and upstream_count == 0
                and previous_upstream_count is not None
                and previous_upstream_count > 0
            ):
                raise ValueError(
                    f"{source}: refusing an empty authoritative snapshot after "
                    f"a prior upstream count of {previous_upstream_count}"
                )
            if (
                authoritative_snapshot
                and not allow_large_snapshot_shrink
                and upstream_count is not None
                and previous_upstream_count is not None
                and previous_upstream_count > 0
                and upstream_count * 4 < previous_upstream_count * 3
            ):
                raise ValueError(
                    f"{source}: refusing authoritative snapshot count drop from "
                    f"{previous_upstream_count} to {upstream_count}; pass "
                    "allow_large_snapshot_shrink=True only after operator review"
                )

            for index, issue in enumerate(page_issues):
                issue_record_id = str(issue.source_record_id).strip() or _stable_id(
                    "source-issue", source, str(index), _safe_json(issue.summary)
                )
                self._upsert_dead_letter(
                    source,
                    issue_record_id,
                    _required(issue.stage, "source issue stage"),
                    _required(issue.error, "source issue error"),
                    issue.summary,
                    run_id,
                    now,
                )
                if authoritative_snapshot and run_id is not None:
                    artifact = self._artifact_by_source_record.get(
                        (artifact_source, issue_record_id)
                    )
                    if artifact is not None:
                        artifact.update(
                            active=1,
                            tombstoned_at=None,
                            last_seen_run_id=run_id,
                            updated_at=now,
                        )
                result["errors"] += 1

            for index, record in enumerate(page_records):
                try:
                    extras: Iterable[ModelHint] = ()
                    if extractor_callback is not None and not record.deleted:
                        extras = extractor_callback(record)
                    if model_hints is not None and not record.deleted:
                        supplied = (
                            model_hints(record)
                            if callable(model_hints)
                            else model_hints.get(record.source_record_id, ())
                        )
                        extras = (*extras, *supplied)
                    extra_tuple = tuple(extras)
                    _validate_record(record, extra_tuple)
                except (ValueError, TypeError, OverflowError) as exc:
                    if not quarantine_errors:
                        raise
                    self._upsert_dead_letter(
                        source,
                        _dead_letter_record_id(record, index),
                        "record_ingest",
                        str(exc) or type(exc).__name__,
                        _record_summary(record),
                        run_id,
                        now,
                    )
                    if authoritative_snapshot and run_id is not None:
                        failed_id = str(getattr(record, "source_record_id", "")).strip()
                        artifact = self._artifact_by_source_record.get(
                            (artifact_source, failed_id)
                        )
                        if artifact is not None:
                            artifact.update(
                                active=1,
                                tombstoned_at=None,
                                last_seen_run_id=run_id,
                                updated_at=now,
                            )
                    result["errors"] += 1
                    result["record_errors"] += 1
                    continue
                record_result = self._ingest_record(
                    artifact_source,
                    record,
                    extra_tuple,
                    run_id,
                    source_extractor,
                    derived_extractor,
                    now,
                    enqueue_links,
                    link_depth,
                )
                result["new_artifacts"] += record_result["new_artifacts"]
                result["new_revisions"] += record_result["new_revisions"]
                result["links_discovered"] += record_result["links_discovered"]
                touched_models.update(record_result["model_ids"])

            if authoritative_snapshot and result["errors"] == 0:
                for artifact in self._tables["artifacts"]:
                    if (
                        artifact["source"] == artifact_source
                        and artifact["active"] == 1
                        and artifact.get("last_seen_run_id") != run_id
                    ):
                        artifact.update(
                            active=0,
                            tombstoned_at=artifact.get("tombstoned_at") or now,
                            updated_at=now,
                        )

            checkpoint["pages_ingested"] += 1
            checkpoint["records_seen"] += len(page_records) + len(page_issues)
            checkpoint["last_run_id"] = run_id or checkpoint.get("last_run_id")
            checkpoint["updated_at"] = now
            if result["record_errors"] or (
                page_issues and not advance_on_source_issues
            ):
                result["checkpoint_advanced"] = 0
                if retry_state is not None:
                    checkpoint["state_json"] = _dump_json(dict(retry_state))
            else:
                checkpoint["state_json"] = _dump_json(dict(next_state))
                checkpoint["complete"] = bool(complete)
                if upstream_count is not None:
                    checkpoint["upstream_count"] = upstream_count
            if run_id is not None:
                self._by_id["sync_runs"][run_id]["heartbeat_at"] = now
            if not self._ingest_batch_active:
                self._reindex()

        result["models_touched"] = len(touched_models)
        return result

    @contextmanager
    def ingest_batch(self) -> Iterator[None]:
        """Atomically persist several already-fetched source pages.

        The caller must still invoke :meth:`ingest_page` once for every source
        page.  That retains exact page-level checkpoint and quarantine behavior,
        while avoiding a full snapshot clone, hash, and Parquet write for every
        individual provider page.
        """

        if self._ingest_batch_active:
            raise RuntimeError("ingest batches cannot be nested")
        with self._write_transaction("ingest_pages"):
            self._ingest_batch_active = True
            try:
                yield
            finally:
                self._ingest_batch_active = False

    def search_models(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        self._read_current()
        limit = min(limit, 500)
        normalized = normalize_name(query)
        raw_normalized = query.strip().casefold()
        aliases_by_model: dict[str, list[dict[str, Any]]] = {}
        for alias in self._tables["model_aliases"]:
            aliases_by_model.setdefault(alias["model_id"], []).append(alias)
        identifiers_by_model: dict[str, list[dict[str, Any]]] = {}
        for identifier in self._tables["model_external_identifiers"]:
            identifiers_by_model.setdefault(identifier["model_id"], []).append(identifier)
        links_by_model: dict[str, list[dict[str, Any]]] = {}
        for link in self._tables["artifact_model_links"]:
            links_by_model.setdefault(link["model_id"], []).append(link)

        ranked: list[tuple[int, float, str, str, dict[str, Any]]] = []
        for model in self._tables["models"]:
            aliases = aliases_by_model.get(model["id"], [])
            identifiers = identifiers_by_model.get(model["id"], [])
            names = [model["normalized_name"], *(row["normalized_alias"] for row in aliases)]
            identifier_values = [
                f"{row['namespace']}:{row['value']}".casefold() for row in identifiers
            ]
            if normalized == "":
                rank = 3
            elif normalized in names:
                rank = 0
            elif any(name.startswith(normalized) for name in names):
                rank = 1
            elif any(raw_normalized in value for value in identifier_values):
                rank = 0
            elif any(normalized in name for name in names):
                rank = 2
            else:
                continue
            model_links = links_by_model.get(model["id"], [])
            artifact_ids = {link["artifact_id"] for link in model_links}
            current_ids = {
                link["artifact_id"]
                for link in model_links
                if self._by_id["artifacts"][link["artifact_id"]]["current_revision_id"]
                == link["artifact_revision_id"]
            }
            active_ids = {
                artifact_id
                for artifact_id in current_ids
                if self._by_id["artifacts"][artifact_id]["active"] == 1
            }
            result = {
                "id": model["id"],
                "name": model["canonical_name"],
                "canonical_name": model["canonical_name"],
                "status": model["status"],
                "confidence": model["confidence"],
                "aliases": sorted(
                    (row["alias"] for row in aliases),
                    key=lambda value: (normalize_name(value), value),
                ),
                "identifiers": sorted(
                    ({"namespace": row["namespace"], "value": row["value"]} for row in identifiers),
                    key=lambda value: (value["namespace"], value["value"]),
                ),
                "artifact_count": len(artifact_ids),
                "current_artifact_count": len(current_ids),
                "active_current_artifact_count": len(active_ids),
            }
            ranked.append(
                (
                    rank,
                    -float(model["confidence"]),
                    model["canonical_name"],
                    model["id"],
                    result,
                )
            )
        ranked.sort(key=lambda value: value[:4])
        return [value[4] for value in ranked[:limit]]

    def model_detail(self, model_id: str) -> dict[str, Any] | None:
        self._read_current()
        model = self._by_id["models"].get(model_id)
        if model is None:
            return None
        aliases = sorted(
            (row["alias"] for row in self._tables["model_aliases"] if row["model_id"] == model_id),
            key=lambda value: (normalize_name(value), value),
        )
        identifiers = sorted(
            (
                {"namespace": row["namespace"], "value": row["value"]}
                for row in self._tables["model_external_identifiers"]
                if row["model_id"] == model_id
            ),
            key=lambda value: (value["namespace"], value["value"]),
        )
        identifier_claims = []
        for claim in self._tables["model_identifier_claims"]:
            if claim["model_id"] != model_id:
                continue
            artifact = self._artifact_for_revision(claim["artifact_revision_id"])
            identifier_claims.append(
                {
                    **deepcopy(claim),
                    "artifact_id": artifact["id"],
                    "source": artifact["source"],
                    "source_record_id": artifact["source_record_id"],
                    "canonical_url": artifact["canonical_url"],
                    "title": artifact["title"],
                }
            )
        identifier_claims.sort(key=lambda row: (row["created_at"], row["id"]))

        artifacts = []
        for link in self._tables["artifact_model_links"]:
            if link["model_id"] != model_id:
                continue
            artifact = self._by_id["artifacts"][link["artifact_id"]]
            artifacts.append(
                {
                    "link_id": link["id"],
                    "artifact_revision_id": link["artifact_revision_id"],
                    "local_id": link["local_id"],
                    "status": link["status"],
                    "resolution_status": link["resolution_status"],
                    "confidence": link["confidence"],
                    "locator": link["locator"],
                    "extractor": link["extractor"],
                    "artifact_id": artifact["id"],
                    "source": artifact["source"],
                    "source_record_id": artifact["source_record_id"],
                    "kind": artifact["kind"],
                    "canonical_url": artifact["canonical_url"],
                    "title": artifact["title"],
                    "published_at": artifact["published_at"],
                    "modified_at": artifact["modified_at"],
                    "active": artifact["active"],
                    "tombstoned_at": artifact["tombstoned_at"],
                    "last_seen_run_id": artifact["last_seen_run_id"],
                    "is_current": int(
                        link["artifact_revision_id"] == artifact["current_revision_id"]
                    ),
                }
            )
        artifacts.sort(
            key=lambda row: (
                row["published_at"] or "",
                row["source"],
                row["source_record_id"],
                row["link_id"],
            )
        )

        relations = []
        for claim in self._tables["model_relation_claims"]:
            if (
                claim.get("subject_model_id") != model_id
                and claim.get("target_model_id") != model_id
            ):
                continue
            artifact = self._artifact_for_revision(claim["artifact_revision_id"])
            item = deepcopy(claim)
            item["target_identifiers"] = json.loads(item.pop("target_identifiers_json"))
            item.update(
                artifact_id=artifact["id"],
                source=artifact["source"],
                source_record_id=artifact["source_record_id"],
                canonical_url=artifact["canonical_url"],
                title=artifact["title"],
                direction=("outgoing" if claim.get("subject_model_id") == model_id else "incoming"),
            )
            relations.append(item)
        relations.sort(key=lambda row: (row["created_at"], row["id"]))

        evidence = []
        for row in self._tables["evidence_provenance"]:
            if row["subject_type"] != "model" or row["subject_id"] != model_id:
                continue
            artifact = self._artifact_for_revision(row["artifact_revision_id"])
            evidence.append(
                {
                    **deepcopy(row),
                    "value": json.loads(row["value_json"]),
                    "artifact_id": artifact["id"],
                    "source": artifact["source"],
                    "source_record_id": artifact["source_record_id"],
                    "canonical_url": artifact["canonical_url"],
                    "title": artifact["title"],
                }
            )
        evidence.sort(key=lambda row: (row["created_at"], row["id"]))
        return {
            "id": model["id"],
            "name": model["canonical_name"],
            "canonical_name": model["canonical_name"],
            "normalized_name": model["normalized_name"],
            "status": model["status"],
            "confidence": model["confidence"],
            "created_at": model["created_at"],
            "updated_at": model["updated_at"],
            "aliases": aliases,
            "identifiers": identifiers,
            "identifier_claims": identifier_claims,
            "artifacts": artifacts,
            "connected_artifacts": self._connected_artifacts(model_id),
            "releases": self._model_releases(model_id),
            "relations": relations,
            "evidence": evidence,
        }

    def stats(self) -> dict[str, int]:
        self._read_current()
        mapping = {
            "sources": "source_checkpoints",
            "sync_runs": "sync_runs",
            "artifacts": "artifacts",
            "artifact_revisions": "artifact_revisions",
            "models": "models",
            "model_releases": "model_releases",
            "artifact_model_links": "artifact_model_links",
            "artifact_release_links": "artifact_release_links",
            "relation_claims": "model_relation_claims",
            "evidence": "evidence_provenance",
            "frontier_urls": "url_frontier",
            "dead_letters": "dead_letters",
        }
        result = {key: len(self._tables[table]) for key, table in mapping.items()}
        result["resolved_relation_claims"] = sum(
            row["resolution_status"] == "resolved" for row in self._tables["model_relation_claims"]
        )
        result["unresolved_relation_claims"] = sum(
            row["resolution_status"] != "resolved" for row in self._tables["model_relation_claims"]
        )
        result["identifier_conflicts"] = sum(
            row["resolution_status"] == "conflict"
            for row in self._tables["model_identifier_claims"]
        )
        result["release_identifier_conflicts"] = sum(
            row["resolution_status"] == "conflict"
            for row in self._tables["release_identifier_claims"]
        )
        result["pending_urls"] = sum(
            row["status"] == "pending" for row in self._tables["url_frontier"]
        )
        result["active_artifacts"] = sum(row["active"] == 1 for row in self._tables["artifacts"])
        return result

    def coverage_metrics(self) -> dict[str, Any]:
        self._read_current()
        sources = []
        for checkpoint in sorted(self._tables["source_checkpoints"], key=lambda row: row["source"]):
            artifacts = [
                row for row in self._tables["artifacts"] if row["source"] == checkpoint["source"]
            ]
            artifact_ids = {row["id"] for row in artifacts}
            sources.append(
                {
                    "source": checkpoint["source"],
                    "upstream_count": checkpoint["upstream_count"],
                    "complete": bool(checkpoint["complete"]),
                    "artifacts": len(artifacts),
                    "active_artifacts": sum(row["active"] == 1 for row in artifacts),
                    "current_revisions": len(
                        {
                            row["current_revision_id"]
                            for row in artifacts
                            if row["current_revision_id"]
                        }
                    ),
                    "model_links": sum(
                        row["artifact_id"] in artifact_ids
                        for row in self._tables["artifact_model_links"]
                    ),
                    "release_links": sum(
                        row["artifact_id"] in artifact_ids
                        for row in self._tables["artifact_release_links"]
                    ),
                }
            )
        status_counts = {
            status: 0
            for status in (
                ModelStatus.CANDIDATE.value,
                ModelStatus.DOCUMENTED.value,
                ModelStatus.RELEASED.value,
                ModelStatus.STUB.value,
            )
        }
        for model in self._tables["models"]:
            status_counts[model["status"]] = status_counts.get(model["status"], 0) + 1
        relation_counts = {
            "resolved": sum(
                row["resolution_status"] == "resolved"
                for row in self._tables["model_relation_claims"]
            ),
            "unresolved": sum(
                row["resolution_status"] != "resolved"
                for row in self._tables["model_relation_claims"]
            ),
        }
        frontier_counts = {
            status: sum(row["status"] == status for row in self._tables["url_frontier"])
            for status in ("pending", "observed", "failed")
        }
        exact_models = len({row["model_id"] for row in self._tables["model_external_identifiers"]})
        return {
            "sources": sources,
            "global": {
                "models_with_exact_identifiers": exact_models,
                "model_statuses": status_counts,
                "relation_claims": relation_counts,
                "frontier": frontier_counts,
            },
        }

    def source_status(self) -> list[dict[str, Any]]:
        self._read_current()
        result = []
        for checkpoint in sorted(self._tables["source_checkpoints"], key=lambda row: row["source"]):
            runs = [
                row for row in self._tables["sync_runs"] if row["source"] == checkpoint["source"]
            ]
            last_run = max(runs, key=lambda row: int(row["id"]), default=None)
            result.append(
                {
                    "source": checkpoint["source"],
                    "state": _load_object(checkpoint["state_json"]),
                    "complete": bool(checkpoint["complete"]),
                    "upstream_count": checkpoint["upstream_count"],
                    "pages_ingested": checkpoint["pages_ingested"],
                    "records_seen": checkpoint["records_seen"],
                    "updated_at": checkpoint["updated_at"],
                    "last_run_id": checkpoint["last_run_id"],
                    "last_run": None
                    if last_run is None
                    else {
                        "status": last_run["status"],
                        "started_at": last_run["started_at"],
                        "finished_at": last_run["finished_at"],
                        "stats": _load_object(last_run["stats_json"]),
                        "error": last_run["error"],
                    },
                }
            )
        return result

    def list_frontier(self, status: str = "pending", limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        self._read_current()
        rows = [row for row in self._tables["url_frontier"] if row["status"] == status]
        rows.sort(key=lambda row: (row["first_seen_at"], row["url"]))
        return deepcopy(rows[: min(limit, 1000)])

    def claim_frontier(
        self,
        limit: int = 100,
        *,
        lease_seconds: int = 3600,
        refresh_after_seconds: int = 24 * 60 * 60,
    ) -> list[dict[str, Any]]:
        """Atomically lease pending URLs and recover stale claims/fetches."""

        if limit < 1:
            return []
        if lease_seconds < 1:
            raise ValueError("frontier lease_seconds must be positive")
        if refresh_after_seconds < 1:
            raise ValueError("frontier refresh_after_seconds must be positive")
        now = _now()
        now_value = _as_datetime(now)
        stale_before = now_value - timedelta(seconds=lease_seconds)
        refresh_before = now_value - timedelta(seconds=refresh_after_seconds)
        claimed: list[dict[str, Any]] = []
        with self._write_transaction("claim_frontier"):
            for row in self._tables["url_frontier"]:
                fetched_at = row.get("last_fetched_at")
                if (
                    row["status"] in {"done", "failed"}
                    and fetched_at is not None
                    and _as_datetime(fetched_at) <= refresh_before
                ):
                    row.update(status="pending", attempts=0, claimed_at=None, last_error=None)
                if (
                    row["status"] == "claimed"
                    and row.get("claimed_at") is not None
                    and _as_datetime(row["claimed_at"]) < stale_before
                ):
                    row.update(status="pending", claimed_at=None)
            pending = [row for row in self._tables["url_frontier"] if row["status"] == "pending"]
            pending.sort(
                key=lambda row: (
                    row.get("last_fetched_at") is not None,
                    row["first_seen_at"],
                    row["url"],
                )
            )
            for row in pending[: min(limit, 1000)]:
                row.update(
                    status="claimed",
                    attempts=int(row["attempts"]) + 1,
                    claimed_at=now,
                )
                claimed.append(deepcopy(row))
        return claimed

    def update_frontier(
        self,
        url: str,
        status: str,
        *,
        error: str | None = None,
        increment_attempts: bool = False,
    ) -> None:
        canonical_url = _web_url(url)
        if canonical_url is None:
            raise ValueError(f"not an HTTP(S) URL: {url!r}")
        status = _required(status, "status")
        with self._write_transaction("update_frontier"):
            row = self._frontier_by_url.get(canonical_url)
            if row is None:
                raise KeyError(f"unknown frontier URL: {canonical_url}")
            now = _now()
            row.update(
                status=status,
                last_error=error,
                attempts=int(row["attempts"]) + int(increment_attempts),
                claimed_at=now if status == "claimed" else None,
            )
            if status in {"done", "failed"}:
                row["last_fetched_at"] = now

    def add_dead_letter(
        self,
        source: str,
        source_record_id: str,
        stage: str,
        error: str | Exception,
        summary: Mapping[str, Any] | str | None = None,
        *,
        run_id: int | None = None,
    ) -> str:
        source = _required(source, "source")
        source_record_id = _required(source_record_id, "source_record_id")
        stage = _required(stage, "stage")
        error_text = _required(str(error), "error")
        now = _now()
        with self._write_transaction("add_dead_letter"):
            self._ensure_source(source, now)
            if run_id is not None:
                run = self._by_id["sync_runs"].get(run_id)
                if run is None:
                    raise KeyError(f"unknown sync run: {run_id}")
                if run["source"] != source:
                    raise ValueError(f"sync run {run_id} belongs to {run['source']!r}")
            return self._upsert_dead_letter(
                source,
                source_record_id,
                stage,
                error_text,
                summary or {},
                run_id,
                now,
            )

    def list_dead_letters(
        self, source: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        self._read_current()
        rows = [
            row for row in self._tables["dead_letters"] if source is None or row["source"] == source
        ]
        rows.sort(key=lambda row: (row["last_seen_at"], row["id"]), reverse=True)
        result = []
        for row in rows[: min(limit, 1000)]:
            item = deepcopy(row)
            item["summary"] = json.loads(item.pop("summary_json"))
            result.append(item)
        return result

    def _upsert_dead_letter(
        self,
        source: str,
        source_record_id: str,
        stage: str,
        error: str,
        summary: Any,
        run_id: int | None,
        now: str,
    ) -> str:
        summary_json = _safe_json(summary)
        dead_letter_id = _stable_id("dead-letter", source, source_record_id, stage, error)
        existing = next(
            (
                row
                for row in self._tables["dead_letters"]
                if row["source"] == source
                and row["source_record_id"] == source_record_id
                and row["stage"] == stage
                and row["error"] == error
            ),
            None,
        )
        if existing is None:
            self._tables["dead_letters"].append(
                {
                    "id": dead_letter_id,
                    "source": source,
                    "source_record_id": source_record_id,
                    "stage": stage,
                    "error": error,
                    "summary_json": summary_json,
                    "run_id": run_id,
                    "attempts": 1,
                    "first_seen_at": now,
                    "last_seen_at": now,
                }
            )
        else:
            existing.update(
                summary_json=summary_json,
                run_id=run_id or existing.get("run_id"),
                attempts=int(existing["attempts"]) + 1,
                last_seen_at=now,
            )
        self._reindex()
        return dead_letter_id

    def _ensure_source(self, source: str, now: str) -> None:
        if source in self._source_by_name:
            return
        row = {
            "source": source,
            "state_json": "{}",
            "complete": False,
            "upstream_count": None,
            "pages_ingested": 0,
            "records_seen": 0,
            "last_run_id": None,
            "updated_at": now,
        }
        self._tables["source_checkpoints"].append(row)
        self._source_by_name[source] = row

    def _ingest_record(
        self,
        source: str,
        record: SourceRecord,
        extra_hints: Sequence[ModelHint],
        run_id: int | None,
        source_extractor: str,
        derived_extractor: str | None,
        now: str,
        enqueue_links: bool,
        link_depth: int,
    ) -> dict[str, Any]:
        source_record_id = _required(record.source_record_id, "source_record_id")
        canonical_url = _required(record.canonical_url, "canonical_url")
        canonical_url_normalized = canonicalize_url(canonical_url)
        artifact_id = _stable_id("artifact", source, source_record_id)
        artifact = self._artifact_by_source_record.get((source, source_record_id))
        new_artifact = artifact is None
        active = int(not record.deleted)
        tombstoned_at = now if record.deleted else None
        if artifact is None:
            artifact = {
                "id": artifact_id,
                "source": source,
                "source_record_id": source_record_id,
                "kind": record.kind.value,
                "canonical_url": canonical_url,
                "canonical_url_normalized": canonical_url_normalized,
                "title": record.title,
                "published_at": record.published_at,
                "modified_at": record.modified_at,
                "current_revision_id": None,
                "created_at": now,
                "updated_at": now,
                "active": active,
                "tombstoned_at": tombstoned_at,
                "last_seen_run_id": run_id,
            }
            self._tables["artifacts"].append(artifact)
            self._artifact_by_source_record[(source, source_record_id)] = artifact
            self._by_id["artifacts"][artifact_id] = artifact
        current_revision_id = artifact["current_revision_id"]
        record_json = _dump_json(_record_payload(record))
        revision_hash = content_hash(record_json)
        revision_id = _stable_id("artifact-revision", artifact_id, revision_hash)
        revision = self._by_id["artifact_revisions"].get(revision_id)
        new_revision = revision is None
        if revision is None:
            revision = {
                "id": revision_id,
                "artifact_id": artifact_id,
                "revision_hash": revision_hash,
                "raw_json": _dump_json(dict(record.raw)),
                "text": record.text,
                "record_json": record_json,
                "source_modified_at": record.modified_at,
                "run_id": run_id,
                "ingested_at": now,
            }
            self._tables["artifact_revisions"].append(revision)
            self._by_id["artifact_revisions"][revision_id] = revision
        artifact.update(
            active=active,
            tombstoned_at=tombstoned_at,
            last_seen_run_id=run_id or artifact.get("last_seen_run_id"),
        )

        for namespace, value in _identifier_keys(record.identifiers):
            identifier_key = (artifact_id, namespace, value)
            if identifier_key not in self._artifact_identifier_keys:
                self._tables["artifact_identifiers"].append(
                    {
                        "artifact_id": artifact_id,
                        "namespace": namespace,
                        "value": value,
                        "first_seen_revision_id": revision_id,
                    }
                )
                self._artifact_identifier_keys.add(identifier_key)
            self._insert_evidence(
                revision_id,
                "artifact",
                artifact_id,
                "external_identifier",
                {"namespace": namespace, "value": value},
                None,
                source_extractor,
                1.0,
                now,
            )

        links_discovered = self._persist_links(
            artifact_id,
            revision_id,
            record,
            source_extractor,
            now,
            enqueue_links,
            link_depth,
        )
        hints = _combine_hints(record.models, extra_hints)
        declared_local_ids = {hint.local_id for hint in record.models}
        if derived_extractor is None:
            hint_groups = ((hints, source_extractor),)
            derived_hints: tuple[ModelHint, ...] = ()
        else:
            self._prune_derived_extraction(revision_id, derived_extractor)
            source_hints = tuple(
                hint for hint in hints if hint.local_id in declared_local_ids
            )
            derived_hints = tuple(
                hint for hint in hints if hint.local_id not in declared_local_ids
            )
            hint_groups = (
                (source_hints, source_extractor),
                (derived_hints, derived_extractor),
            )
        local_models: dict[str, str] = {}
        model_resolutions: dict[str, str] = {}
        touched: set[str] = set()
        for group, provenance in hint_groups:
            for hint in group:
                model_id, resolution = self._upsert_model_hint(
                    artifact_id, revision_id, hint, provenance, now
                )
                local_models[hint.local_id] = model_id
                model_resolutions[hint.local_id] = resolution
                touched.add(model_id)
        for release in _combine_release_hints(record.releases):
            self._upsert_release_hint(
                artifact_id,
                revision_id,
                release,
                local_models,
                source_extractor,
                now,
            )
        for relation in record.model_relations:
            touched.update(
                self._persist_relation(
                    artifact_id,
                    revision_id,
                    relation,
                    local_models,
                    model_resolutions,
                    source_extractor,
                    now,
                )
            )
        if derived_extractor is not None and derived_hints:
            self._derived_extraction_keys.add((revision_id, derived_extractor))
        if current_revision_id != revision_id:
            artifact.update(
                kind=record.kind.value,
                canonical_url=canonical_url,
                canonical_url_normalized=canonical_url_normalized,
                title=record.title,
                published_at=record.published_at,
                modified_at=record.modified_at,
                current_revision_id=revision_id,
                updated_at=now,
            )
        # The enclosing page transaction reindexes once after all records have
        # been applied. Every lookup needed by this record is updated eagerly
        # above; rebuilding every table index here makes large source pages
        # quadratic in their record count.
        return {
            "new_artifacts": int(new_artifact),
            "new_revisions": int(new_revision),
            "links_discovered": links_discovered,
            "model_ids": touched,
        }

    def _prune_derived_extraction(self, revision_id: str, extractor: str) -> None:
        if (revision_id, extractor) not in self._derived_extraction_keys:
            # Newly observed immutable revisions cannot have stale derived rows.
            # This fast path prevents one full-table scan and reindex per record
            # on large historical backfills.
            return
        for table in (
            "artifact_model_links",
            "model_identifier_claims",
            "release_identifier_claims",
            "artifact_release_links",
            "model_relation_claims",
            "evidence_provenance",
        ):
            self._tables[table] = [
                row
                for row in self._tables[table]
                if not (
                    row["artifact_revision_id"] == revision_id and row["extractor"] == extractor
                )
            ]
        self._reindex()

    def _upsert_model_hint(
        self,
        artifact_id: str,
        revision_id: str,
        hint: ModelHint,
        extractor: str,
        now: str,
    ) -> tuple[str, str]:
        local_id = _required(hint.local_id, "model local_id")
        name = _required(hint.name, "model name")
        confidence = _confidence(hint.confidence)
        identifiers = _identifier_keys(hint.identifiers)
        matching_models: set[str] = set()
        projected_models: set[str] = set()
        for namespace, value in identifiers:
            for claim in self._model_identifier_claims_by_key.get((namespace, value), ()):
                if claim["resolution_status"] != "bound":
                    continue
                artifact = self._artifact_for_revision(claim["artifact_revision_id"])
                if (
                    claim["artifact_revision_id"] == artifact["current_revision_id"]
                    or claim["artifact_revision_id"] == revision_id
                ):
                    matching_models.add(claim["model_id"])
            projected_models.update(
                row["model_id"]
                for row in self._model_external_identifiers_by_key.get((namespace, value), ())
            )
        if len(matching_models) == 1:
            model_id = next(iter(matching_models))
            resolution = "exact_identifier"
        elif len(matching_models) > 1:
            model_id = _stable_id("model-claim", artifact_id, local_id)
            resolution = "identifier_conflict"
        elif identifiers:
            namespace, value = identifiers[0]
            base_model_id = _stable_id("model-identifier", namespace, value)
            if projected_models or base_model_id in self._by_id["models"]:
                model_id = _stable_id(
                    "model-identifier-rebind",
                    artifact_id,
                    local_id,
                    _dump_json(identifiers),
                )
            else:
                model_id = base_model_id
            resolution = "exact_identifier_new"
        else:
            model_id = _stable_id("model-claim", artifact_id, local_id)
            resolution = "source_scoped"

        status = hint.status.value
        model = self._by_id["models"].get(model_id)
        if model is None:
            model = {
                "id": model_id,
                "canonical_name": name,
                "normalized_name": normalize_name(name),
                "status": status,
                "confidence": confidence,
                "created_at": now,
                "updated_at": now,
            }
            self._tables["models"].append(model)
            self._by_id["models"][model_id] = model
        else:
            next_status = model["status"]
            if _STATUS_RANK[status] > _STATUS_RANK.get(model["status"], -1):
                next_status = status
            next_confidence = max(float(model["confidence"]), confidence)
            if next_status != model["status"] or next_confidence != model["confidence"]:
                model.update(
                    status=next_status,
                    confidence=next_confidence,
                    updated_at=now,
                )

        for alias_value in dict.fromkeys((name, *hint.aliases)):
            alias = _required(alias_value, "model alias")
            alias_key = (model_id, alias)
            if alias_key not in self._model_alias_keys:
                self._tables["model_aliases"].append(
                    {
                        "model_id": model_id,
                        "alias": alias,
                        "normalized_alias": normalize_name(alias),
                        "first_seen_revision_id": revision_id,
                    }
                )
                self._model_alias_keys.add(alias_key)
            self._insert_evidence(
                revision_id,
                "model",
                model_id,
                "alias",
                alias,
                hint.locator,
                extractor,
                confidence,
                now,
            )

        identifier_status = "conflict" if resolution == "identifier_conflict" else "bound"
        for namespace, value in identifiers:
            if identifier_status == "bound":
                projections = self._model_external_identifiers_by_key.get(
                    (namespace, value), ()
                )
                projection = projections[0] if projections else None
                if projection is None:
                    projection = {
                        "model_id": model_id,
                        "namespace": namespace,
                        "value": value,
                        "first_seen_revision_id": revision_id,
                    }
                    self._tables["model_external_identifiers"].append(projection)
                    self._model_external_identifiers_by_key.setdefault(
                        (namespace, value), []
                    ).append(projection)
                elif projection["model_id"] != model_id:
                    projection.update(model_id=model_id, first_seen_revision_id=revision_id)
            claim_id = _stable_id(
                "model-identifier-claim",
                revision_id,
                model_id,
                namespace,
                value,
                hint.locator or "",
                extractor,
            )
            if claim_id not in self._by_id["model_identifier_claims"]:
                claim = {
                    "id": claim_id,
                    "artifact_revision_id": revision_id,
                    "model_id": model_id,
                    "namespace": namespace,
                    "value": value,
                    "resolution_status": identifier_status,
                    "locator": hint.locator,
                    "extractor": extractor,
                    "confidence": confidence,
                    "created_at": now,
                }
                self._tables["model_identifier_claims"].append(claim)
                self._by_id["model_identifier_claims"][claim_id] = claim
                self._model_identifier_claims_by_key.setdefault(
                    (namespace, value), []
                ).append(claim)
            self._insert_evidence(
                revision_id,
                "model",
                model_id,
                "external_identifier",
                {
                    "namespace": namespace,
                    "value": value,
                    "resolution_status": identifier_status,
                },
                hint.locator,
                extractor,
                confidence,
                now,
            )

        link_id = _stable_id("artifact-model-link", revision_id, model_id, local_id, extractor)
        if link_id not in self._by_id["artifact_model_links"]:
            link = {
                "id": link_id,
                "artifact_id": artifact_id,
                "artifact_revision_id": revision_id,
                "model_id": model_id,
                "local_id": local_id,
                "status": status,
                "resolution_status": resolution,
                "confidence": confidence,
                "locator": hint.locator,
                "extractor": extractor,
                "created_at": now,
            }
            self._tables["artifact_model_links"].append(link)
            self._by_id["artifact_model_links"][link_id] = link
        self._insert_evidence(
            revision_id,
            "model",
            model_id,
            "documented_by",
            {
                "artifact_id": artifact_id,
                "local_id": local_id,
                "name": name,
                "status": status,
                "resolution_status": resolution,
            },
            hint.locator,
            extractor,
            confidence,
            now,
        )
        return model_id, resolution

    def _upsert_release_hint(
        self,
        artifact_id: str,
        artifact_revision_id: str,
        hint: ReleaseHint,
        local_models: Mapping[str, str],
        extractor: str,
        now: str,
    ) -> tuple[str, str]:
        local_id = _required(hint.local_id, "release local_id")
        model_local_id = _required(hint.model_local_id, "release model_local_id")
        model_id = local_models.get(model_local_id)
        if model_id is None:
            raise ValueError(
                f"release {local_id!r} references unknown model local_id {model_local_id!r}"
            )
        version = _optional_text(hint.version)
        revision = _optional_text(hint.revision)
        identifiers = _identifier_keys(hint.identifiers)
        if version is None and revision is None and not identifiers:
            raise ValueError(
                f"release {local_id!r} requires a version, revision, or external identifier"
            )
        confidence = _confidence(hint.confidence)
        metadata_json = _dump_json(dict(hint.metadata))
        matching_releases: set[str] = set()
        projected_releases: set[str] = set()
        for namespace, value in identifiers:
            for claim in self._tables["release_identifier_claims"]:
                if (
                    claim["namespace"] == namespace
                    and claim["value"] == value
                    and claim["resolution_status"] == "bound"
                ):
                    artifact = self._artifact_for_revision(claim["artifact_revision_id"])
                    if (
                        claim["artifact_revision_id"] == artifact["current_revision_id"]
                        or claim["artifact_revision_id"] == artifact_revision_id
                    ):
                        matching_releases.add(claim["release_id"])
            projected_releases.update(
                row["release_id"]
                for row in self._tables["release_external_identifiers"]
                if row["namespace"] == namespace and row["value"] == value
            )
        matched_release_id = None
        if len(matching_releases) == 1:
            candidate_id = next(iter(matching_releases))
            candidate = self._by_id["model_releases"].get(candidate_id)
            if candidate is not None and candidate["model_id"] == model_id:
                matched_release_id = candidate_id
        if matched_release_id is not None:
            release_id = matched_release_id
            resolution = "exact_identifier"
        elif matching_releases:
            release_id = _stable_id("release-claim", artifact_id, local_id)
            resolution = "identifier_conflict"
        elif identifiers:
            namespace, value = identifiers[0]
            base_release_id = _stable_id("release-identifier", namespace, value)
            if projected_releases or base_release_id in self._by_id["model_releases"]:
                release_id = _stable_id(
                    "release-identifier-rebind",
                    artifact_id,
                    local_id,
                    _dump_json(identifiers),
                )
            else:
                release_id = base_release_id
            resolution = "exact_identifier_new"
        else:
            release_id = _stable_id("release-claim", artifact_id, local_id)
            resolution = "source_scoped"

        release = self._by_id["model_releases"].get(release_id)
        if release is None:
            release = {
                "id": release_id,
                "model_id": model_id,
                "version": version,
                "revision": revision,
                "released_at": hint.released_at,
                "metadata_json": metadata_json,
                "confidence": confidence,
                "created_at": now,
                "updated_at": now,
            }
            self._tables["model_releases"].append(release)
            self._by_id["model_releases"][release_id] = release
        else:
            if release["model_id"] != model_id:
                raise ValueError(
                    f"release identity {release_id!r} is already tied to another model"
                )
            next_metadata = release["metadata_json"]
            if _load_object(next_metadata) == {} and metadata_json != "{}":
                next_metadata = metadata_json
            next_values = {
                "version": release["version"] or version,
                "revision": release["revision"] or revision,
                "released_at": release["released_at"] or hint.released_at,
                "metadata_json": next_metadata,
                "confidence": max(float(release["confidence"]), confidence),
            }
            if any(release[key] != value for key, value in next_values.items()):
                release.update(**next_values, updated_at=now)
        model = self._by_id["models"][model_id]
        if _STATUS_RANK.get(model["status"], -1) < _STATUS_RANK[ModelStatus.RELEASED.value]:
            model.update(
                status=ModelStatus.RELEASED.value,
                confidence=max(float(model["confidence"]), confidence),
                updated_at=now,
            )

        identifier_status = "conflict" if resolution == "identifier_conflict" else "bound"
        for namespace, value in identifiers:
            if identifier_status == "bound":
                projection = next(
                    (
                        row
                        for row in self._tables["release_external_identifiers"]
                        if row["namespace"] == namespace and row["value"] == value
                    ),
                    None,
                )
                if projection is None:
                    self._tables["release_external_identifiers"].append(
                        {
                            "release_id": release_id,
                            "namespace": namespace,
                            "value": value,
                            "first_seen_revision_id": artifact_revision_id,
                        }
                    )
                elif projection["release_id"] != release_id:
                    projection.update(
                        release_id=release_id,
                        first_seen_revision_id=artifact_revision_id,
                    )
            claim_id = _stable_id(
                "release-identifier-claim",
                artifact_revision_id,
                release_id,
                namespace,
                value,
                hint.locator or "",
                extractor,
            )
            if claim_id not in self._by_id["release_identifier_claims"]:
                claim = {
                    "id": claim_id,
                    "artifact_revision_id": artifact_revision_id,
                    "release_id": release_id,
                    "namespace": namespace,
                    "value": value,
                    "resolution_status": identifier_status,
                    "locator": hint.locator,
                    "extractor": extractor,
                    "confidence": confidence,
                    "created_at": now,
                }
                self._tables["release_identifier_claims"].append(claim)
                self._by_id["release_identifier_claims"][claim_id] = claim
            self._insert_evidence(
                artifact_revision_id,
                "release",
                release_id,
                "external_identifier",
                {
                    "namespace": namespace,
                    "value": value,
                    "resolution_status": identifier_status,
                },
                hint.locator,
                extractor,
                confidence,
                now,
            )
        link_id = _stable_id(
            "artifact-release-link",
            artifact_revision_id,
            release_id,
            local_id,
            extractor,
        )
        if link_id not in self._by_id["artifact_release_links"]:
            link = {
                "id": link_id,
                "artifact_id": artifact_id,
                "artifact_revision_id": artifact_revision_id,
                "model_id": model_id,
                "release_id": release_id,
                "local_id": local_id,
                "model_local_id": model_local_id,
                "version": version,
                "revision": revision,
                "released_at": hint.released_at,
                "metadata_json": metadata_json,
                "resolution_status": resolution,
                "confidence": confidence,
                "locator": hint.locator,
                "extractor": extractor,
                "created_at": now,
            }
            self._tables["artifact_release_links"].append(link)
            self._by_id["artifact_release_links"][link_id] = link
        self._insert_evidence(
            artifact_revision_id,
            "release",
            release_id,
            "release_of",
            {
                "model_id": model_id,
                "artifact_id": artifact_id,
                "local_id": local_id,
                "model_local_id": model_local_id,
                "version": version,
                "revision": revision,
                "released_at": hint.released_at,
                "metadata": dict(hint.metadata),
                "resolution_status": resolution,
            },
            hint.locator,
            extractor,
            confidence,
            now,
        )
        return release_id, resolution

    def _persist_relation(
        self,
        artifact_id: str,
        revision_id: str,
        relation: ModelRelationHint,
        local_models: Mapping[str, str],
        model_resolutions: Mapping[str, str],
        extractor: str,
        now: str,
    ) -> set[str]:
        subject_local_id = _required(relation.subject_local_id, "relation subject_local_id")
        predicate = _required(relation.predicate, "relation predicate")
        confidence = _confidence(relation.confidence)
        subject_model_id = local_models.get(subject_local_id)
        touched = {subject_model_id} if subject_model_id is not None else set()
        target = relation.target
        target_local_id = _required(target.local_id, "relation target local_id")
        target_model_id = local_models.get(target_local_id)
        target_resolution = model_resolutions.get(target_local_id)
        if target_model_id is None and target.identifiers:
            candidate_id, target_resolution = self._upsert_model_hint(
                artifact_id, revision_id, target, extractor, now
            )
            touched.add(candidate_id)
            if target_resolution != "identifier_conflict":
                target_model_id = candidate_id
        elif target_model_id is not None:
            touched.add(target_model_id)
        if subject_model_id is None:
            resolution_status = "unresolved_subject"
        elif target_model_id is None and target_resolution == "identifier_conflict":
            resolution_status = "conflicting_target_identifiers"
        elif target_model_id is None:
            resolution_status = "unresolved_target"
        else:
            resolution_status = "resolved"
        target_payload = _hint_payload(target)
        claim_id = _stable_id(
            "model-relation-claim",
            revision_id,
            subject_local_id,
            predicate,
            _dump_json(target_payload),
            relation.locator or "",
            extractor,
        )
        if claim_id not in self._by_id["model_relation_claims"]:
            claim = {
                "id": claim_id,
                "artifact_revision_id": revision_id,
                "subject_local_id": subject_local_id,
                "subject_model_id": subject_model_id,
                "predicate": predicate,
                "target_local_id": target_local_id,
                "target_name": target.name,
                "target_identifiers_json": _dump_json(target_payload["identifiers"]),
                "target_model_id": target_model_id,
                "resolution_status": resolution_status,
                "confidence": confidence,
                "locator": relation.locator,
                "extractor": extractor,
                "created_at": now,
            }
            self._tables["model_relation_claims"].append(claim)
            self._by_id["model_relation_claims"][claim_id] = claim
        self._insert_evidence(
            revision_id,
            "model_relation_claim",
            claim_id,
            predicate,
            {
                "subject_local_id": subject_local_id,
                "subject_model_id": subject_model_id,
                "target": target_payload,
                "target_model_id": target_model_id,
                "resolution_status": resolution_status,
            },
            relation.locator,
            extractor,
            confidence,
            now,
        )
        return touched

    def _persist_links(
        self,
        artifact_id: str,
        revision_id: str,
        record: SourceRecord,
        extractor: str,
        now: str,
        enqueue_links: bool,
        link_depth: int,
    ) -> int:
        discoveries: list[tuple[str, str, str | None, bool, bool]] = [
            (link.url, link.relation, link.locator, True, link.crawl)
            for link in record.links
        ]
        discoveries.extend(
            (url, infer_url_relation(record.text, locator), locator, False, True)
            for url, locator in extract_url_mentions(record.text)
        )
        new_urls = 0
        seen: set[tuple[str, str, str | None, bool]] = set()
        record_url = canonicalize_url(record.canonical_url)
        for raw_url, raw_relation, locator, is_explicit, crawl in discoveries:
            url = _web_url(raw_url)
            if url is None or url == record_url:
                continue
            relation = raw_relation.strip() or "references"
            key = (url, relation, locator, crawl)
            if key in seen:
                continue
            seen.add(key)
            should_enqueue = enqueue_links and crawl and (
                is_explicit or _is_embedded_repository_candidate(url, relation)
            )
            url_id = _stable_id("url", url)
            frontier = self._frontier_by_url.get(url)
            if frontier is None:
                frontier = {
                    "id": url_id,
                    "url": url,
                    "status": "pending" if should_enqueue else "observed",
                    "attempts": 0,
                    "first_seen_at": now,
                    "last_seen_at": now,
                    "claimed_at": None,
                    "last_error": None,
                    "depth": link_depth,
                    "last_fetched_at": None,
                }
                self._tables["url_frontier"].append(frontier)
                self._frontier_by_url[url] = frontier
                self._by_id["url_frontier"][url_id] = frontier
                new_urls += 1
            else:
                frontier["depth"] = min(int(frontier["depth"]), link_depth)
                if should_enqueue and frontier["status"] == "observed":
                    frontier["status"] = "pending"
            discovery_id = _stable_id("url-discovery", revision_id, url_id, relation, locator or "")
            if discovery_id not in self._by_id["url_discoveries"]:
                discovery = {
                    "id": discovery_id,
                    "url_id": url_id,
                    "artifact_revision_id": revision_id,
                    "relation": relation,
                    "locator": locator,
                    "discovered_at": now,
                    "depth": link_depth,
                }
                self._tables["url_discoveries"].append(discovery)
                self._by_id["url_discoveries"][discovery_id] = discovery
                if new_urls == 0:
                    frontier["last_seen_at"] = now
            self._insert_evidence(
                revision_id,
                "url",
                url_id,
                relation,
                {"artifact_id": artifact_id, "url": url},
                locator,
                extractor,
                1.0,
                now,
            )
        return new_urls

    def _insert_evidence(
        self,
        revision_id: str,
        subject_type: str,
        subject_id: str,
        predicate: str,
        value: Any,
        locator: str | None,
        extractor: str,
        confidence: float,
        now: str,
    ) -> None:
        value_json = _dump_json(value)
        evidence_id = _stable_id(
            "evidence",
            revision_id,
            subject_type,
            subject_id,
            predicate,
            value_json,
            locator or "",
            extractor,
        )
        if evidence_id in self._by_id["evidence_provenance"]:
            return
        row = {
            "id": evidence_id,
            "artifact_revision_id": revision_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "predicate": predicate,
            "value_json": value_json,
            "locator": locator,
            "extractor": extractor,
            "confidence": _confidence(confidence),
            "created_at": now,
        }
        self._tables["evidence_provenance"].append(row)
        self._by_id["evidence_provenance"][evidence_id] = row

    def _model_releases(self, model_id: str) -> list[dict[str, Any]]:
        rows = [row for row in self._tables["model_releases"] if row["model_id"] == model_id]
        rows.sort(
            key=lambda row: (
                row["released_at"] or "",
                row["version"] or "",
                row["revision"] or "",
                row["id"],
            )
        )
        releases = []
        for row in rows:
            release_id = row["id"]
            identifiers = sorted(
                (
                    {"namespace": item["namespace"], "value": item["value"]}
                    for item in self._tables["release_external_identifiers"]
                    if item["release_id"] == release_id
                ),
                key=lambda value: (value["namespace"], value["value"]),
            )
            identifier_claims = []
            for claim in self._tables["release_identifier_claims"]:
                if claim["release_id"] != release_id:
                    continue
                artifact = self._artifact_for_revision(claim["artifact_revision_id"])
                identifier_claims.append(
                    {
                        **deepcopy(claim),
                        "artifact_id": artifact["id"],
                        "source": artifact["source"],
                        "source_record_id": artifact["source_record_id"],
                        "canonical_url": artifact["canonical_url"],
                        "title": artifact["title"],
                    }
                )
            identifier_claims.sort(key=lambda value: (value["created_at"], value["id"]))
            evidence = []
            for link in self._tables["artifact_release_links"]:
                if link["release_id"] != release_id:
                    continue
                artifact = self._by_id["artifacts"][link["artifact_id"]]
                claim = {
                    **deepcopy(link),
                    "source": artifact["source"],
                    "source_record_id": artifact["source_record_id"],
                    "kind": artifact["kind"],
                    "canonical_url": artifact["canonical_url"],
                    "title": artifact["title"],
                }
                claim["metadata"] = json.loads(claim.pop("metadata_json"))
                evidence.append(claim)
            evidence.sort(key=lambda value: (value["created_at"], value["id"]))
            releases.append(
                {
                    "id": release_id,
                    "model_id": row["model_id"],
                    "version": row["version"],
                    "revision": row["revision"],
                    "released_at": row["released_at"],
                    "metadata": json.loads(row["metadata_json"]),
                    "confidence": row["confidence"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "identifiers": identifiers,
                    "identifier_claims": identifier_claims,
                    "evidence": evidence,
                }
            )
        return releases

    def _connected_artifacts(self, model_id: str) -> list[dict[str, Any]]:
        direct_ids = {
            link["artifact_id"]
            for link in self._tables["artifact_model_links"]
            if link["model_id"] == model_id
            and link["artifact_revision_id"]
            == self._by_id["artifacts"][link["artifact_id"]]["current_revision_id"]
        }
        connected: dict[str, dict[str, Any]] = {}
        seen: set[tuple[str, ...]] = set()

        def entry(artifact: dict[str, Any]) -> dict[str, Any]:
            return connected.setdefault(
                artifact["id"],
                {
                    "artifact_id": artifact["id"],
                    "source": artifact["source"],
                    "source_record_id": artifact["source_record_id"],
                    "kind": artifact["kind"],
                    "canonical_url": artifact["canonical_url"],
                    "title": artifact["title"],
                    "published_at": artifact["published_at"],
                    "modified_at": artifact["modified_at"],
                    "active": bool(artifact["active"]),
                    "tombstoned_at": artifact["tombstoned_at"],
                    "last_seen_run_id": artifact["last_seen_run_id"],
                    "connections": [],
                },
            )

        current_identifiers: dict[str, list[tuple[str, str]]] = {}
        current_url_aliases: dict[str, set[str]] = {}
        for artifact in self._tables["artifacts"]:
            revision_id = artifact["current_revision_id"]
            for evidence in self._tables["evidence_provenance"]:
                if (
                    evidence["artifact_revision_id"] == revision_id
                    and evidence["subject_type"] == "artifact"
                    and evidence["subject_id"] == artifact["id"]
                    and evidence["predicate"] == "external_identifier"
                ):
                    value = json.loads(evidence["value_json"])
                    pair = (value["namespace"], value["value"])
                    current_identifiers.setdefault(artifact["id"], []).append(pair)
                    if value["namespace"] == "url":
                        current_url_aliases.setdefault(artifact["id"], set()).add(value["value"])

        for origin_id in direct_ids:
            origin = self._by_id["artifacts"][origin_id]
            for namespace, value in current_identifiers.get(origin_id, []):
                for target in self._tables["artifacts"]:
                    if target["id"] in direct_ids:
                        continue
                    if (namespace, value) not in current_identifiers.get(target["id"], []):
                        continue
                    key = ("shared_identifier", target["id"], namespace, value, origin_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    entry(target)["connections"].append(
                        {
                            "connection_type": "shared_identifier",
                            "namespace": namespace,
                            "value": value,
                            "evidence_source": {
                                "artifact_id": origin_id,
                                "artifact_revision_id": origin["current_revision_id"],
                                "source": origin["source"],
                                "source_record_id": origin["source_record_id"],
                            },
                            "matched_evidence": {
                                "artifact_id": target["id"],
                                "artifact_revision_id": target["current_revision_id"],
                                "source": target["source"],
                                "source_record_id": target["source_record_id"],
                            },
                        }
                    )

        discoveries_by_revision: dict[str, list[dict[str, Any]]] = {}
        for discovery in self._tables["url_discoveries"]:
            discoveries_by_revision.setdefault(discovery["artifact_revision_id"], []).append(
                discovery
            )
        for origin_id in direct_ids:
            origin = self._by_id["artifacts"][origin_id]
            for discovery in discoveries_by_revision.get(origin["current_revision_id"], []):
                frontier = self._by_id["url_frontier"][discovery["url_id"]]
                for target in self._tables["artifacts"]:
                    if target["id"] in direct_ids:
                        continue
                    if not self._artifact_matches_url(target, frontier["url"], current_url_aliases):
                        continue
                    self._append_url_connection(
                        connected,
                        seen,
                        entry,
                        target,
                        origin,
                        discovery,
                        frontier["url"],
                        "outgoing",
                    )
        for connected_artifact in self._tables["artifacts"]:
            if connected_artifact["id"] in direct_ids:
                continue
            for discovery in discoveries_by_revision.get(
                connected_artifact["current_revision_id"], []
            ):
                frontier = self._by_id["url_frontier"][discovery["url_id"]]
                for direct_id in direct_ids:
                    direct = self._by_id["artifacts"][direct_id]
                    if not self._artifact_matches_url(direct, frontier["url"], current_url_aliases):
                        continue
                    self._append_url_connection(
                        connected,
                        seen,
                        entry,
                        connected_artifact,
                        connected_artifact,
                        discovery,
                        frontier["url"],
                        "incoming",
                    )
        for artifact in connected.values():
            artifact["connections"].sort(
                key=lambda item: (
                    item["connection_type"],
                    item.get("namespace", ""),
                    item.get("value", item.get("url", "")),
                    item["evidence_source"]["source"],
                    item["evidence_source"]["source_record_id"],
                )
            )
        return sorted(
            connected.values(),
            key=lambda artifact: (
                artifact["source"],
                artifact["source_record_id"],
                artifact["artifact_id"],
            ),
        )

    @staticmethod
    def _artifact_matches_url(
        artifact: Mapping[str, Any],
        url: str,
        aliases: Mapping[str, set[str]],
    ) -> bool:
        return artifact["canonical_url_normalized"] == url or url in aliases.get(
            str(artifact["id"]), set()
        )

    @staticmethod
    def _append_url_connection(
        connected: dict[str, dict[str, Any]],
        seen: set[tuple[str, ...]],
        entry: Callable[[dict[str, Any]], dict[str, Any]],
        target: dict[str, Any],
        evidence_artifact: dict[str, Any],
        discovery: dict[str, Any],
        url: str,
        direction: str,
    ) -> None:
        key = (
            "discovered_url",
            target["id"],
            url,
            evidence_artifact["id"],
            direction,
            discovery["relation"],
            discovery["locator"] or "",
        )
        if key in seen:
            return
        seen.add(key)
        entry(target)["connections"].append(
            {
                "connection_type": "discovered_url",
                "direction": direction,
                "url": url,
                "relation": discovery["relation"],
                "locator": discovery["locator"],
                "depth": discovery["depth"],
                "evidence_source": {
                    "artifact_id": evidence_artifact["id"],
                    "artifact_revision_id": evidence_artifact["current_revision_id"],
                    "url_discovery_id": discovery["id"],
                    "source": evidence_artifact["source"],
                    "source_record_id": evidence_artifact["source_record_id"],
                },
            }
        )

    def _artifact_for_revision(self, revision_id: str) -> dict[str, Any]:
        revision = self._by_id["artifact_revisions"].get(revision_id)
        if revision is None:
            raise KeyError(f"unknown artifact revision: {revision_id}")
        return self._by_id["artifacts"][revision["artifact_id"]]

    def _read_current(self) -> None:
        if not self._initialized:
            self.initialize()
        with self._thread_lock:
            self._load(force=False)

    @contextmanager
    def _write_transaction(self, operation: str) -> Iterator[None]:
        if not self._initialized:
            self.initialize()
        with self._thread_lock, self._file_lock():
            # We hold the cross-process lock now.  ``_load(force=False)`` still
            # performs the complete verified reload if another writer advanced
            # HEAD, but preserves this process's already-indexed state after its
            # own preceding commit.  For high-volume sources that avoids reading
            # and reindexing the whole registry once per page.
            self._load(force=False)
            prior_tables = self._tables
            self._tables = {
                table_name: (
                    [dict(row) for row in rows]
                    if table_name in _MUTABLE_ROW_TABLES
                    else list(rows)
                )
                for table_name, rows in prior_tables.items()
            }
            self._reindex()
            self._transaction_base_tables = prior_tables
            try:
                yield
                self._reindex()
                self._commit(operation)
            except Exception:
                self._tables = prior_tables
                self._reindex()
                raise
            finally:
                self._transaction_base_tables = None

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        lock_path = self.root / ".write.lock"
        if lock_path.is_symlink():
            raise ValueError(f"Parquet store lock must not be a symlink: {lock_path}")
        lock_path.touch(exist_ok=True)
        if not lock_path.is_file():
            raise ValueError(f"Parquet store lock is not a file: {lock_path}")
        with lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _load(self, *, force: bool) -> None:
        head_path = self.root / "HEAD.json"
        if head_path.is_symlink():
            raise ValueError(f"Parquet store HEAD must not be a symlink: {head_path}")
        if not head_path.exists():
            raise ValueError(
                "Parquet store HEAD is missing; refusing to replace the visible "
                "registry state. Restore HEAD from backup after inspecting the "
                "retained commits."
            )
        head = _read_json_object(head_path)
        store_format = head.get("format")
        if store_format not in _SUPPORTED_STORE_FORMATS:
            raise ValueError(f"unsupported Parquet store format in {head_path}")
        commit_id = _required(str(head.get("commit", "")), "HEAD commit")
        if _COMMIT_ID_PATTERN.fullmatch(commit_id) is None:
            raise ValueError(f"invalid Parquet commit identifier in {head_path}")
        if not force and commit_id == self._head_commit:
            return
        commit_dir = self.root / "commits" / commit_id
        if commit_dir.is_symlink() or not commit_dir.is_dir():
            raise ValueError(f"invalid Parquet commit directory: {commit_dir}")
        tables_dir = commit_dir / "tables"
        manifest_path = commit_dir / "manifest.json"
        if tables_dir.is_symlink() or manifest_path.is_symlink():
            raise ValueError(f"Parquet commit contains a symlink: {commit_dir}")
        manifest = _read_json_object(manifest_path)
        if manifest.get("format") != store_format or manifest.get("commit") != commit_id:
            raise ValueError(f"invalid Parquet commit manifest: {commit_dir}")
        if (
            manifest.get("state_digest") != head.get("state_digest")
            or int(manifest.get("generation", -1)) != int(head.get("generation", -2))
        ):
            raise ValueError(f"HEAD metadata does not match Parquet commit {commit_id}")
        parquet_digests = _manifest_digest_map(manifest, "parquet_sha256")
        manifest_table_digests = _manifest_digest_map(manifest, "table_digests")
        if (parquet_digests is None) != (manifest_table_digests is None):
            raise ValueError(
                f"Parquet commit has incomplete table integrity metadata: {commit_dir}"
            )
        if store_format == _STORE_FORMAT:
            if parquet_digests is None or manifest_table_digests is None:
                raise ValueError(f"Parquet v2 commit lacks table integrity metadata: {commit_dir}")
            if _state_digest_from_table_digests(manifest_table_digests) != manifest.get(
                "state_digest"
            ):
                raise ValueError(f"state digest mismatch for Parquet commit {commit_id}")
        tables: dict[str, list[dict[str, Any]]] = {}
        expected_counts = manifest.get("table_counts", {})
        if not isinstance(expected_counts, dict):
            raise ValueError(f"invalid table_counts in {commit_dir / 'manifest.json'}")
        for table_name in _TABLES:
            table_path = tables_dir / f"{table_name}.parquet"
            if table_path.is_symlink():
                raise ValueError(f"Parquet table must not be a symlink: {table_path}")
            if parquet_digests is not None and _file_sha256(table_path) != parquet_digests[
                table_name
            ]:
                raise ValueError(f"Parquet table checksum mismatch: {table_path}")
            arrow_table = pq.read_table(table_path)
            if arrow_table.column_names == ["__empty__"]:
                rows: list[dict[str, Any]] = []
            else:
                rows = arrow_table.to_pylist()
            expected = expected_counts.get(table_name)
            if expected != len(rows):
                raise ValueError(
                    f"row count mismatch for {table_name}: expected {expected}, got {len(rows)}"
                )
            tables[table_name] = rows
        if parquet_digests is None:
            # Older commits predate physical-file checksums. Continue to verify
            # their canonical logical digest exactly, then publish checksums on
            # the next successful write.
            state_digest, table_digests = _state_and_table_digests(tables)
            if state_digest != manifest.get("state_digest"):
                raise ValueError(f"state digest mismatch for Parquet commit {commit_id}")
        else:
            table_digests = manifest_table_digests
        self._tables = tables
        self._head_commit = commit_id
        self._head_generation = int(head.get("generation", 0))
        self._table_digests = table_digests
        raw_schemas = manifest.get("schemas")
        self._table_schemas = (
            {name: str(raw_schemas[name]) for name in _TABLES}
            if isinstance(raw_schemas, Mapping)
            and all(isinstance(raw_schemas.get(name), str) for name in _TABLES)
            else {}
        )
        self._reindex()

    def _commit(self, operation: str) -> None:
        table_digests = self._current_table_digests()
        state_digest = _state_digest_from_table_digests(table_digests)
        generation = self._head_generation + 1
        # The nonce prevents a pre-HEAD crash from reserving the deterministic
        # generation/digest name and blocking a clean retry. Such an orphan is
        # harmless because readers follow only HEAD.
        commit_id = f"{generation:020d}-{state_digest[:16]}-{uuid.uuid4().hex[:12]}"
        commits_dir = self.root / "commits"
        final_dir = commits_dir / commit_id
        if final_dir.exists():
            raise FileExistsError(f"Parquet commit already exists: {commit_id}")
        staging_dir = self.root / ".staging" / f"{commit_id}-{uuid.uuid4().hex}"
        tables_dir = staging_dir / "tables"
        tables_dir.mkdir(parents=True)
        try:
            schemas: dict[str, str] = {}
            parquet_digests: dict[str, str] = {}
            parent_tables_dir = (
                self.root / "commits" / self._head_commit / "tables"
                if self._head_commit is not None
                else None
            )
            for table_name in _TABLES:
                table_path = tables_dir / f"{table_name}.parquet"
                parent_path = (
                    parent_tables_dir / f"{table_name}.parquet"
                    if parent_tables_dir is not None
                    else None
                )
                if (
                    parent_path is not None
                    and self._table_digests.get(table_name) == table_digests[table_name]
                    and table_name in self._table_schemas
                    and parent_path.is_file()
                    and not parent_path.is_symlink()
                ):
                    # Immutable parent files can be shared exactly. This avoids
                    # rewriting every logical table just to update a run heartbeat
                    # or one source cursor, while each commit directory remains a
                    # complete, independently readable snapshot.
                    os.link(parent_path, table_path)
                    schemas[table_name] = self._table_schemas[table_name]
                    parquet_digests[table_name] = _file_sha256(table_path)
                    continue
                rows = self._tables[table_name]
                if rows:
                    arrow_table = pa.Table.from_pylist(rows)
                else:
                    arrow_table = pa.table({"__empty__": pa.array([], type=pa.bool_())})
                pq.write_table(arrow_table, table_path, compression="zstd")
                _fsync_file(table_path)
                schemas[table_name] = str(arrow_table.schema)
                parquet_digests[table_name] = _file_sha256(table_path)
            manifest = {
                "format": _STORE_FORMAT,
                "commit": commit_id,
                "generation": generation,
                "parent": self._head_commit,
                "created_at": _now(),
                "operation": operation,
                "state_digest": state_digest,
                "table_digests": table_digests,
                "parquet_sha256": parquet_digests,
                "table_counts": {
                    table_name: len(self._tables[table_name]) for table_name in _TABLES
                },
                "schemas": schemas,
            }
            manifest_path = staging_dir / "manifest.json"
            _write_json(manifest_path, manifest)
            _fsync_tree(staging_dir)
            os.replace(staging_dir, final_dir)
            _fsync_directory(commits_dir)
            head = {
                "format": _STORE_FORMAT,
                "commit": commit_id,
                "generation": generation,
                "state_digest": state_digest,
            }
            head_temp = self.root / f".HEAD-{uuid.uuid4().hex}.tmp"
            _write_json(head_temp, head)
            os.replace(head_temp, self.root / "HEAD.json")
            _fsync_directory(self.root)
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise
        self._head_commit = commit_id
        self._head_generation = generation
        self._table_digests = table_digests
        self._table_schemas = schemas

    def _current_table_digests(self) -> dict[str, str]:
        """Hash only tables whose logical rows changed in this transaction."""

        baseline = self._transaction_base_tables
        result = {}
        for table_name in _TABLES:
            if (
                baseline is not None
                and table_name in self._table_digests
                and self._tables[table_name] == baseline[table_name]
            ):
                result[table_name] = self._table_digests[table_name]
            else:
                result[table_name] = _table_digest(self._tables[table_name])
        return result

    def _reindex(self) -> None:
        self._by_id = {}
        for table_name, rows in self._tables.items():
            self._by_id[table_name] = {row["id"]: row for row in rows if "id" in row}
        self._artifact_by_source_record = {
            (row["source"], row["source_record_id"]): row for row in self._tables["artifacts"]
        }
        self._artifact_identifier_keys = {
            (row["artifact_id"], row["namespace"], row["value"])
            for row in self._tables["artifact_identifiers"]
        }
        self._model_alias_keys = {
            (row["model_id"], row["alias"]) for row in self._tables["model_aliases"]
        }
        model_claims: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self._tables["model_identifier_claims"]:
            model_claims[(row["namespace"], row["value"])].append(row)
        self._model_identifier_claims_by_key = dict(model_claims)
        model_identifiers: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
            list
        )
        for row in self._tables["model_external_identifiers"]:
            model_identifiers[(row["namespace"], row["value"])].append(row)
        self._model_external_identifiers_by_key = dict(model_identifiers)
        self._derived_extraction_keys = {
            (str(row["artifact_revision_id"]), str(row["extractor"]))
            for table_name in (
                "artifact_model_links",
                "model_identifier_claims",
                "release_identifier_claims",
                "artifact_release_links",
                "model_relation_claims",
                "evidence_provenance",
            )
            for row in self._tables[table_name]
            if row.get("artifact_revision_id") and row.get("extractor")
        }
        self._source_by_name = {row["source"]: row for row in self._tables["source_checkpoints"]}
        self._frontier_by_url = {row["url"]: row for row in self._tables["url_frontier"]}
        discovery_url_ids_by_relation: defaultdict[str, set[str]] = defaultdict(set)
        discoveries_by_url_id: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._tables["url_discoveries"]:
            relation = str(row.get("relation", "")).casefold()
            url_id = str(row["url_id"])
            discovery_url_ids_by_relation[relation].add(url_id)
            discoveries_by_url_id[url_id].append(row)
        self._url_discovery_url_ids_by_relation = dict(discovery_url_ids_by_relation)
        self._url_discoveries_by_url_id = dict(discoveries_by_url_id)
        self._weight_artifact_urls = {
            canonicalize_url(str(row["canonical_url"]))
            for row in self._tables["artifacts"]
            if row.get("kind") == ArtifactKind.WEIGHTS.value and row.get("canonical_url")
        }

    def _validate_store_layout(self) -> None:
        for name in ("commits", ".staging"):
            path = self.root / name
            if path.is_symlink():
                raise ValueError(f"Parquet store directory must not be a symlink: {path}")
            path.mkdir(exist_ok=True)
            if not path.is_dir() or not path.resolve().is_relative_to(self.root):
                raise ValueError(f"invalid Parquet store directory: {path}")
        lock_path = self.root / ".write.lock"
        if lock_path.is_symlink():
            raise ValueError(f"Parquet store lock must not be a symlink: {lock_path}")
        lock_path.touch(exist_ok=True)
        if not lock_path.is_file():
            raise ValueError(f"Parquet store lock is not a file: {lock_path}")


def _empty_tables() -> dict[str, list[dict[str, Any]]]:
    return {table_name: [] for table_name in _TABLES}


def _validate_record(record: SourceRecord, extras: Sequence[ModelHint]) -> None:
    _required(record.source_record_id, "source_record_id")
    _required(record.canonical_url, "canonical_url")
    _required(record.title, "title")
    _dump_json(_record_payload(record))
    _dump_json(dict(record.raw))
    if not isinstance(record.deleted, bool):
        raise TypeError("deleted must be boolean")
    if record.deleted and (record.models or record.model_relations or record.releases or extras):
        raise ValueError("deleted records cannot assert models, relations, or releases")
    _identifier_keys(record.identifiers)
    hints = _combine_hints(record.models, extras)
    local_ids: set[str] = set()
    for hint in hints:
        local_id = _required(hint.local_id, "model local_id")
        _required(hint.name, "model name")
        _confidence(hint.confidence)
        _identifier_keys(hint.identifiers)
        for alias in hint.aliases:
            _required(alias, "model alias")
        local_ids.add(local_id)
    for link in record.links:
        if not isinstance(link.model_local_ids, tuple):
            raise TypeError("link model_local_ids must be a tuple")
        scoped_model_ids = tuple(
            _required(local_id, "link model_local_id")
            for local_id in link.model_local_ids
        )
        if len(set(scoped_model_ids)) != len(scoped_model_ids):
            raise ValueError("link model_local_ids must not contain duplicates")
        missing_models = set(scoped_model_ids) - local_ids
        if missing_models:
            raise ValueError(
                "link scopes resources to unknown model local IDs: "
                + ", ".join(sorted(missing_models))
            )
    for release in _combine_release_hints(record.releases):
        local_id = _required(release.local_id, "release local_id")
        model_local_id = _required(release.model_local_id, "release model_local_id")
        if model_local_id not in local_ids:
            raise ValueError(
                f"release {local_id!r} references unknown model local_id {model_local_id!r}"
            )
        identifiers = _identifier_keys(release.identifiers)
        version = _optional_text(release.version)
        revision = _optional_text(release.revision)
        if version is None and revision is None and not identifiers:
            raise ValueError(
                f"release {local_id!r} requires a version, revision, or external identifier"
            )
        _confidence(release.confidence)
        _dump_json(dict(release.metadata))
    for relation in record.model_relations:
        _required(relation.subject_local_id, "relation subject_local_id")
        _required(relation.predicate, "relation predicate")
        _confidence(relation.confidence)
        _required(relation.target.local_id, "relation target local_id")
        _required(relation.target.name, "model name")
        _identifier_keys(relation.target.identifiers)
        _confidence(relation.target.confidence)


def _record_payload(record: SourceRecord) -> dict[str, Any]:
    return {
        "source_record_id": record.source_record_id,
        "kind": record.kind.value,
        "canonical_url": record.canonical_url,
        "title": record.title,
        "raw": dict(record.raw),
        "text": record.text,
        "published_at": record.published_at,
        "modified_at": record.modified_at,
        "deleted": record.deleted,
        "identifiers": [
            {"namespace": namespace, "value": value}
            for namespace, value in _identifier_keys(record.identifiers)
        ],
        "links": sorted(
            (_link_payload(link) for link in record.links),
            key=lambda value: (
                value["url"],
                value["relation"],
                value["locator"] or "",
                tuple(value.get("model_local_ids", ())),
            ),
        ),
        "models": sorted((_hint_payload(model) for model in record.models), key=_dump_json),
        "model_relations": sorted(
            (_relation_payload(relation) for relation in record.model_relations),
            key=_dump_json,
        ),
        "releases": sorted(
            (_release_payload(release) for release in record.releases), key=_dump_json
        ),
    }


def _link_payload(link: Link) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "url": link.url,
        "relation": link.relation,
        "locator": link.locator,
        "crawl": link.crawl,
    }
    if link.model_local_ids:
        payload["model_local_ids"] = sorted(link.model_local_ids)
    return payload


def _stored_link_model_local_ids(value: Mapping[str, Any], label: str) -> tuple[str, ...]:
    raw = value.get("model_local_ids", [])
    if not isinstance(raw, list):
        raise ValueError(f"{label} model_local_ids must be an array")
    result = tuple(_required(item, f"{label} model_local_id") for item in raw)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} model_local_ids must not contain duplicates")
    return result


def _control_record_from_payload(value: Any) -> SourceRecord:
    if not isinstance(value, Mapping):
        raise ValueError("stored control record payload must be an object")
    kind = _required(value.get("kind"), "control kind")
    if kind != ArtifactKind.CATALOG_RECORD.value:
        raise ValueError("stored control record is not a catalog record")
    for field in ("models", "model_relations", "releases"):
        claims = value.get(field)
        if not isinstance(claims, list) or claims:
            raise ValueError(f"stored control record must have an empty {field} array")
    raw = value.get("raw")
    if not isinstance(raw, Mapping):
        raise ValueError("stored control record raw value must be an object")
    identifiers_value = value.get("identifiers")
    links_value = value.get("links")
    if not isinstance(identifiers_value, list) or not isinstance(links_value, list):
        raise ValueError("stored control record identifiers and links must be arrays")
    identifiers: list[Identifier] = []
    for item in identifiers_value:
        if not isinstance(item, Mapping):
            raise ValueError("stored control identifier must be an object")
        identifiers.append(
            Identifier(
                _required(item.get("namespace"), "control identifier namespace"),
                _required(item.get("value"), "control identifier value"),
            )
        )
    links: list[Link] = []
    for item in links_value:
        if not isinstance(item, Mapping):
            raise ValueError("stored control link must be an object")
        crawl = item.get("crawl")
        if not isinstance(crawl, bool):
            raise ValueError("stored control link crawl value must be boolean")
        model_local_ids = _stored_link_model_local_ids(item, "control link")
        if model_local_ids:
            raise ValueError("stored control link must not scope resources to models")
        links.append(
            Link(
                url=_required(item.get("url"), "control link URL"),
                relation=_required(item.get("relation"), "control link relation"),
                locator=_optional_text(item.get("locator")),
                crawl=crawl,
                model_local_ids=model_local_ids,
            )
        )
    deleted = value.get("deleted")
    if deleted is not False:
        raise ValueError("active control record cannot be marked deleted")
    return SourceRecord(
        source_record_id=_required(value.get("source_record_id"), "control source_record_id"),
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=_required(value.get("canonical_url"), "control canonical_url"),
        title=_required(value.get("title"), "control title"),
        raw=dict(raw),
        text=_optional_text(value.get("text")) or "",
        published_at=_optional_text(value.get("published_at")),
        modified_at=_optional_text(value.get("modified_at")),
        identifiers=tuple(identifiers),
        links=tuple(links),
        deleted=False,
    )


def _hint_payload(hint: ModelHint) -> dict[str, Any]:
    return {
        "local_id": hint.local_id,
        "name": hint.name,
        "identifiers": [
            {"namespace": namespace, "value": value}
            for namespace, value in _identifier_keys(hint.identifiers)
        ],
        "aliases": list(hint.aliases),
        "status": hint.status.value,
        "confidence": hint.confidence,
        "locator": hint.locator,
    }


def _release_payload(hint: ReleaseHint) -> dict[str, Any]:
    return {
        "local_id": hint.local_id,
        "model_local_id": hint.model_local_id,
        "version": hint.version,
        "revision": hint.revision,
        "identifiers": [
            {"namespace": namespace, "value": value}
            for namespace, value in _identifier_keys(hint.identifiers)
        ],
        "released_at": hint.released_at,
        "metadata": dict(hint.metadata),
        "confidence": hint.confidence,
        "locator": hint.locator,
    }


def _relation_payload(hint: ModelRelationHint) -> dict[str, Any]:
    return {
        "subject_local_id": hint.subject_local_id,
        "predicate": hint.predicate,
        "target": _hint_payload(hint.target),
        "confidence": hint.confidence,
        "locator": hint.locator,
    }


def _combine_hints(
    declared: Sequence[ModelHint], extras: Sequence[ModelHint]
) -> tuple[ModelHint, ...]:
    combined: dict[str, ModelHint] = {}
    for hint in (*declared, *extras):
        previous = combined.get(hint.local_id)
        if previous is not None and previous != hint:
            raise ValueError(f"conflicting model hints use local_id {hint.local_id!r}")
        combined[hint.local_id] = hint
    return tuple(combined.values())


def _combine_release_hints(declared: Sequence[ReleaseHint]) -> tuple[ReleaseHint, ...]:
    combined: dict[str, ReleaseHint] = {}
    for hint in declared:
        previous = combined.get(hint.local_id)
        if previous is not None and previous != hint:
            raise ValueError(f"conflicting release hints use local_id {hint.local_id!r}")
        combined[hint.local_id] = hint
    return tuple(combined.values())


def _extractor_parts(
    extractor: str | Any,
) -> tuple[str, Callable[[SourceRecord], Iterable[ModelHint]] | None]:
    if isinstance(extractor, str):
        return _required(extractor, "extractor"), None
    name = _required(getattr(extractor, "name", ""), "extractor name")
    extract = getattr(extractor, "extract", None)
    if not callable(extract):
        raise TypeError("extractor must be a name or expose extract(record)")
    return name, extract


def _identifier_keys(identifiers: Iterable[Identifier]) -> tuple[tuple[str, str], ...]:
    result: set[tuple[str, str]] = set()
    for identifier in identifiers:
        namespace = _required(identifier.namespace, "identifier namespace").casefold()
        value = _required(identifier.value, "identifier value")
        result.add((namespace, value))
    return tuple(sorted(result))


def _stable_id(kind: str, *parts: str) -> str:
    digest = hashlib.sha256()
    for value in (kind, *parts):
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    prefix = kind.replace("_", "-")
    return f"{prefix}_{digest.hexdigest()}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _as_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _required(value: str, label: str) -> str:
    result = value.strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    result = value.strip()
    return result or None


def _confidence(value: float) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return result


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_json(value: Any) -> str:
    try:
        return _dump_json(value)
    except (TypeError, ValueError, OverflowError):
        return _dump_json({"repr": repr(value)[:8000]})


def _record_summary(record: SourceRecord) -> dict[str, Any]:
    summary = {
        "source_record_id": getattr(record, "source_record_id", None),
        "kind": getattr(getattr(record, "kind", None), "value", None),
        "canonical_url": getattr(record, "canonical_url", None),
        "title": getattr(record, "title", None),
        "raw": getattr(record, "raw", None),
    }
    try:
        _dump_json(summary)
        return summary
    except (TypeError, ValueError, OverflowError):
        summary["raw"] = repr(summary["raw"])[:8000]
        return summary


def _dead_letter_record_id(record: SourceRecord, index: int) -> str:
    value = str(getattr(record, "source_record_id", "")).strip()
    if value:
        return value
    return _stable_id("page-record", str(index), _safe_json(_record_summary(record)))


def _load_object(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("expected a JSON object")
    return loaded


def _web_url(value: str) -> str | None:
    canonical = canonicalize_url(value)
    parts = urlsplit(canonical)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    return canonical


def _is_embedded_repository_candidate(url: str, relation: str) -> bool:
    if relation in {"implementation", "official_implementation"}:
        return True
    identifier = identifier_from_url(url)
    return identifier is not None and identifier.namespace == "github:repository"


def _run_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "source": row["source"],
        "status": row["status"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "stats": _load_object(str(row["stats_json"])),
        "error": row["error"],
    }


def _state_digest(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    return _state_and_table_digests(tables)[0]


def _state_and_table_digests(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, dict[str, str]]:
    """Return the legacy-compatible whole-state digest and per-table digests.

    The whole-state encoding intentionally remains unchanged so existing commit
    manifests continue to verify. Per-table hashes are an internal optimization:
    they identify immutable Parquet files that may safely be hard-linked into a
    new complete snapshot.
    """

    digest = hashlib.sha256()
    table_digests: dict[str, str] = {}
    for table_name in _TABLES:
        # ``json.dumps(list(rows), separators=(",", ":"), sort_keys=True)`` is
        # exactly an opening bracket, each independently encoded row separated
        # by commas, and a closing bracket. Hash it incrementally instead of
        # materializing multi-gigabyte evidence tables a second time merely for
        # integrity verification. The resulting bytes (and legacy digest) are
        # unchanged.
        table_digest = hashlib.sha256()
        payload_length = 0
        for payload in _json_array_chunks(tables[table_name]):
            table_digest.update(payload)
            payload_length += len(payload)
        table_digests[table_name] = table_digest.hexdigest()
        digest.update(len(table_name).to_bytes(4, "big"))
        digest.update(table_name.encode("utf-8"))
        digest.update(payload_length.to_bytes(8, "big"))
        for payload in _json_array_chunks(tables[table_name]):
            digest.update(payload)
    return digest.hexdigest(), table_digests


def _table_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for payload in _json_array_chunks(rows):
        digest.update(payload)
    return digest.hexdigest()


def _state_digest_from_table_digests(table_digests: Mapping[str, str]) -> str:
    """Return the v2 registry digest from every logical table digest."""

    digest = hashlib.sha256()
    for table_name in _TABLES:
        table_digest = table_digests.get(table_name)
        if table_digest is None or re.fullmatch(r"[0-9a-f]{64}", table_digest) is None:
            raise ValueError(f"invalid table digest for {table_name}")
        digest.update(len(table_name).to_bytes(4, "big"))
        digest.update(table_name.encode("utf-8"))
        digest.update(table_digest.encode("ascii"))
    return digest.hexdigest()


def _json_array_chunks(rows: Sequence[Mapping[str, Any]]) -> Iterator[bytes]:
    """Yield the exact compact JSON encoding of a list without copying it."""

    yield b"["
    for index, row in enumerate(rows):
        if index:
            yield b","
        yield _dump_json(row).encode("utf-8")
    yield b"]"


def _manifest_digest_map(
    manifest: Mapping[str, Any], key: str
) -> dict[str, str] | None:
    """Return an exact per-table SHA-256 map, or ``None`` for legacy commits."""

    raw = manifest.get(key)
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != set(_TABLES):
        raise ValueError(f"invalid {key} in Parquet commit manifest")
    result = {table_name: str(raw[table_name]) for table_name in _TABLES}
    if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in result.values()):
        raise ValueError(f"invalid {key} in Parquet commit manifest")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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
