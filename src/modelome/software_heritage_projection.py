from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.lake import ParquetLandingZone, ReleaseReceipt, ShardReceipt
from modelome.models import ArtifactKind, Identifier, Link, SourcePage, SourceRecord
from modelome.storage import Database

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCP_GITHUB_RE = re.compile(
    r"^git@github\.com:(?P<owner>[^/]+)/(?P<repository>[^/]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SoftwareHeritageGitHubCandidate:
    repository_name: str
    repository_url: str
    origin_url: str
    origin_url_sha256: str
    origin_source_record_id: str
    landing_record_sha256: str
    release: str
    object_key: str
    object_url: str
    object_sha256: str
    source_row_ordinal: int
    locator: str

    def as_source_record(self) -> SourceRecord:
        identity = self.repository_name.casefold()
        return SourceRecord(
            source_record_id=f"software-heritage:github-repository:{identity}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self.object_url,
            title=f"GitHub repository archived by Software Heritage: {self.repository_name}",
            identifiers=(
                Identifier("github:repository", self.repository_name),
                Identifier(
                    "software-heritage:origin-url-sha256",
                    self.origin_url_sha256,
                ),
            ),
            links=(
                Link(
                    self.repository_url,
                    relation="archived_repository_origin",
                    locator=self.locator,
                    crawl=True,
                ),
            ),
            raw={
                "record_type": "software_heritage_github_repository_candidate",
                "repository_name": self.repository_name,
                "repository_url": self.repository_url,
                "origin_url": self.origin_url,
                "origin_url_sha256": self.origin_url_sha256,
                "origin_source_record_id": self.origin_source_record_id,
                "landing_record_sha256": self.landing_record_sha256,
                "release": self.release,
                "object_key": self.object_key,
                "object_url": self.object_url,
                "object_sha256": self.object_sha256,
                "source_row_ordinal": self.source_row_ordinal,
                "locator": self.locator,
                "discovery_basis": "software_heritage_archived_origin",
                "model_admission_performed": False,
                "repository_content_present": False,
                "historical_github_census": False,
            },
        )


@dataclass(frozen=True, slots=True)
class SoftwareHeritageProjectionPage:
    release: str
    start_row: int
    next_row: int
    total_rows: int
    rows_examined: int
    candidates: tuple[SoftwareHeritageGitHubCandidate, ...]
    complete: bool

    @property
    def records(self) -> tuple[SourceRecord, ...]:
        return tuple(candidate.as_source_record() for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class SoftwareHeritageProjectionOutcome:
    status: str
    run_id: int | None
    release: str
    start_row: int
    next_row: int
    total_rows: int
    rows_examined: int
    repositories: int
    links_discovered: int
    complete: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SoftwareHeritageShardProjectionOutcome:
    status: str
    run_id: int | None
    release: str
    shard: str
    manifest_index: int
    start_row: int
    next_row: int
    total_rows: int
    rows_examined: int
    repositories: int
    links_discovered: int
    complete: bool
    error: str | None = None


class SoftwareHeritageGitHubProjector:
    """Project exact GitHub origins while retaining every origin in Parquet.

    This stage recognizes GitHub transport URLs structurally. It does not search
    repository names or decide whether a repository contains a neural model.
    Exact HTTPS repository targets enter the ordinary frontier, where existing
    GitHub metadata/README extraction performs model admission.
    """

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        source: str = "software-heritage",
        dataset: str = "origins",
    ) -> None:
        if not isinstance(landing_zone, ParquetLandingZone):
            raise TypeError("landing_zone must be a ParquetLandingZone")
        self.landing_zone = landing_zone
        self.source = _required_text(source, "source")
        self.dataset = _required_text(dataset, "dataset")

    def page(
        self,
        release: str,
        *,
        start_row: int = 0,
        max_origin_rows: int = 250_000,
        max_repositories: int = 25_000,
        arrow_batch_size: int = 16_384,
    ) -> SoftwareHeritageProjectionPage:
        release = _release(release)
        start_row = _nonnegative_integer(start_row, "start_row")
        max_origin_rows = _positive_integer(max_origin_rows, "max_origin_rows")
        max_repositories = _positive_integer(max_repositories, "max_repositories")
        arrow_batch_size = _positive_integer(arrow_batch_size, "arrow_batch_size")
        receipt = self._release(release)
        if start_row > receipt.row_count:
            raise ValueError("start_row is beyond the sealed release row count")
        if start_row == receipt.row_count:
            return SoftwareHeritageProjectionPage(
                release=release,
                start_row=start_row,
                next_row=start_row,
                total_rows=receipt.row_count,
                rows_examined=0,
                candidates=(),
                complete=True,
            )

        candidates: dict[str, SoftwareHeritageGitHubCandidate] = {}
        row_index = 0
        examined = 0
        stopped = False
        for batch in self.landing_zone.iter_release_batches(
            source=self.source,
            dataset=self.dataset,
            release=release,
            columns=("source_record_id", "payload_json", "content_sha256"),
            batch_size=arrow_batch_size,
        ):
            for row in batch.to_pylist():
                if row_index < start_row:
                    row_index += 1
                    continue
                if examined >= max_origin_rows:
                    stopped = True
                    break
                candidate = _candidate(row, expected_release=release)
                if candidate is not None:
                    identity = candidate.repository_name.casefold()
                    if identity not in candidates and len(candidates) >= max_repositories:
                        stopped = True
                        break
                    candidates.setdefault(identity, candidate)
                examined += 1
                row_index += 1
            if stopped:
                break

        next_row = start_row + examined
        if next_row > receipt.row_count:
            raise ValueError("projection read beyond the sealed release row count")
        return SoftwareHeritageProjectionPage(
            release=release,
            start_row=start_row,
            next_row=next_row,
            total_rows=receipt.row_count,
            rows_examined=examined,
            candidates=tuple(candidates.values()),
            complete=next_row == receipt.row_count,
        )

    def page_shard(
        self,
        source_receipt: ShardReceipt,
        *,
        start_row: int = 0,
        max_origin_rows: int = 250_000,
        max_repositories: int = 25_000,
        arrow_batch_size: int = 16_384,
    ) -> SoftwareHeritageProjectionPage:
        """Project a committed shard immediately, before its release is sealed."""

        receipt = self._verified_shard(source_receipt)
        release = _release(receipt.release)
        start_row = _nonnegative_integer(start_row, "start_row")
        max_origin_rows = _positive_integer(max_origin_rows, "max_origin_rows")
        max_repositories = _positive_integer(max_repositories, "max_repositories")
        arrow_batch_size = _positive_integer(arrow_batch_size, "arrow_batch_size")
        if start_row > receipt.row_count:
            raise ValueError("start_row is beyond the committed shard row count")
        if start_row == receipt.row_count:
            return SoftwareHeritageProjectionPage(
                release=release,
                start_row=start_row,
                next_row=start_row,
                total_rows=receipt.row_count,
                rows_examined=0,
                candidates=(),
                complete=True,
            )
        return self._page_batches(
            release=release,
            total_rows=receipt.row_count,
            batches=self._iter_shard_batches(receipt, batch_size=arrow_batch_size),
            start_row=start_row,
            max_origin_rows=max_origin_rows,
            max_repositories=max_repositories,
        )

    def _page_batches(
        self,
        *,
        release: str,
        total_rows: int,
        batches: Iterator[pa.RecordBatch],
        start_row: int,
        max_origin_rows: int,
        max_repositories: int,
    ) -> SoftwareHeritageProjectionPage:
        candidates: dict[str, SoftwareHeritageGitHubCandidate] = {}
        row_index = 0
        examined = 0
        stopped = False
        for batch in batches:
            for row in batch.to_pylist():
                if row_index < start_row:
                    row_index += 1
                    continue
                if examined >= max_origin_rows:
                    stopped = True
                    break
                candidate = _candidate(row, expected_release=release)
                if candidate is not None:
                    identity = candidate.repository_name.casefold()
                    if identity not in candidates and len(candidates) >= max_repositories:
                        stopped = True
                        break
                    candidates.setdefault(identity, candidate)
                examined += 1
                row_index += 1
            if stopped:
                break
        next_row = start_row + examined
        if next_row > total_rows:
            raise ValueError("projection read beyond the input row count")
        return SoftwareHeritageProjectionPage(
            release=release,
            start_row=start_row,
            next_row=next_row,
            total_rows=total_rows,
            rows_examined=examined,
            candidates=tuple(candidates.values()),
            complete=next_row == total_rows,
        )

    def _verified_shard(self, receipt: ShardReceipt) -> ShardReceipt:
        if not isinstance(receipt, ShardReceipt):
            raise TypeError("source_receipt must be a ShardReceipt")
        if receipt.source != self.source or receipt.dataset != self.dataset:
            raise ValueError("source receipt does not identify this projector's dataset")
        verified = self.landing_zone.lookup_committed_shard(
            source=receipt.source,
            dataset=receipt.dataset,
            release=receipt.release,
            shard=receipt.shard,
            control_sha256=receipt.control_sha256,
            upstream_url=receipt.upstream_url,
            application_order=receipt.application_order,
        )
        if verified is None:
            raise ValueError("source shard is not committed")
        if _shard_identity(receipt) != _shard_identity(verified):
            raise ValueError("source shard receipt does not match committed evidence")
        return verified

    def _iter_shard_batches(
        self,
        receipt: ShardReceipt,
        *,
        batch_size: int,
    ) -> Iterator[pa.RecordBatch]:
        manifest_path = receipt.path / "manifest.json"
        try:
            raw = manifest_path.read_bytes()
        except OSError as error:
            raise ValueError("committed shard manifest is unreadable") from error
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("committed shard manifest exceeds byte bound")
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("committed shard manifest is invalid") from error
        if not isinstance(manifest, Mapping) or not isinstance(manifest.get("parts"), list):
            raise ValueError("committed shard manifest has no parts")
        for part in manifest["parts"]:
            if not isinstance(part, Mapping):
                raise ValueError("committed shard part entry is invalid")
            name = _required_exact_text(part.get("name"), "part name")
            if Path(name).name != name:
                raise ValueError("committed shard part name is unsafe")
            yield from pq.ParquetFile(receipt.path / "parts" / name).iter_batches(
                columns=("source_record_id", "payload_json", "content_sha256"),
                batch_size=batch_size,
            )

    def _release(self, release: str) -> ReleaseReceipt:
        matches = tuple(
            item
            for item in self.landing_zone.list_releases(
                source=self.source,
                dataset=self.dataset,
                verify_shards=False,
            )
            if item.release == release
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected one sealed {self.source}/{self.dataset}/{release} release, "
                f"found {len(matches)}"
            )
        return matches[0]


def run_software_heritage_github_projection(
    database: Database,
    landing_zone: ParquetLandingZone,
    *,
    release: str,
    source: str = "software-heritage-github-repositories",
    max_origin_rows: int = 250_000,
    max_repositories: int = 25_000,
    arrow_batch_size: int = 16_384,
) -> SoftwareHeritageProjectionOutcome:
    """Publish one restart-safe origin page and enqueue exact GitHub URLs."""

    if not isinstance(database, Database):
        raise TypeError("database must be a Database")
    release = _release(release)
    source = _required_text(source, "source")
    state = database.get_source_state(source)
    previous_release = _optional_release(state.get("release"))
    previous_complete = state.get("complete") is True
    if (
        previous_release is not None
        and release <= previous_release
        and (previous_complete or release < previous_release)
    ):
        total = _nonnegative_integer(state.get("total_rows", 0), "total_rows")
        same = release == previous_release
        return SoftwareHeritageProjectionOutcome(
            status="complete",
            run_id=None,
            release=release,
            start_row=total if same else 0,
            next_row=total if same else 0,
            total_rows=total if same else 0,
            rows_examined=0,
            repositories=0,
            links_discovered=0,
            complete=True,
        )
    if previous_release is not None and release > previous_release and not previous_complete:
        raise ValueError(
            f"projection release {previous_release} is incomplete; resume it before {release}"
        )
    start_row = (
        _nonnegative_integer(state.get("next_row", 0), "next_row")
        if previous_release == release
        else 0
    )
    projector = SoftwareHeritageGitHubProjector(landing_zone)
    run_id = database.start_run(source)
    try:
        page = projector.page(
            release,
            start_row=start_row,
            max_origin_rows=max_origin_rows,
            max_repositories=max_repositories,
            arrow_batch_size=arrow_batch_size,
        )
        next_state = {
            "release": release,
            "next_row": page.next_row,
            "total_rows": page.total_rows,
            "complete": page.complete,
        }
        result = database.ingest_page(
            source,
            SourcePage(
                records=page.records,
                next_state=next_state,
                complete=page.complete,
                upstream_count=page.total_rows,
                retry_state=state,
            ),
            run_id=run_id,
            extractor="software-heritage-origin-github-v1",
            enqueue_links=True,
            link_depth=0,
        )
        if result.get("errors"):
            raise ValueError("one or more Software Heritage candidates were quarantined")
        status = "complete" if page.complete else "partial"
        database.finish_run(
            run_id,
            status,
            {
                "release": release,
                "start_row": page.start_row,
                "next_row": page.next_row,
                "total_rows": page.total_rows,
                "rows_examined": page.rows_examined,
                "repositories": len(page.candidates),
                "links_discovered": result["links_discovered"],
                "complete": page.complete,
            },
        )
        return SoftwareHeritageProjectionOutcome(
            status=status,
            run_id=run_id,
            release=release,
            start_row=page.start_row,
            next_row=page.next_row,
            total_rows=page.total_rows,
            rows_examined=page.rows_examined,
            repositories=len(page.candidates),
            links_discovered=result["links_discovered"],
            complete=page.complete,
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        database.finish_run(
            run_id,
            "failed",
            {"release": release, "start_row": start_row, "complete": False},
            error=message,
        )
        return SoftwareHeritageProjectionOutcome(
            status="failed",
            run_id=run_id,
            release=release,
            start_row=start_row,
            next_row=start_row,
            total_rows=0,
            rows_examined=0,
            repositories=0,
            links_discovered=0,
            complete=False,
            error=message,
        )


def run_software_heritage_github_shard_projection(
    database: Database,
    landing_zone: ParquetLandingZone,
    *,
    source_receipt: ShardReceipt,
    source: str = "software-heritage-github-shards",
    max_origin_rows: int = 250_000,
    max_repositories: int = 25_000,
    arrow_batch_size: int = 16_384,
) -> SoftwareHeritageShardProjectionOutcome:
    """Project each committed ORC shard without waiting for all 128 shards."""

    if not isinstance(database, Database):
        raise TypeError("database must be a Database")
    if not isinstance(source_receipt, ShardReceipt):
        raise TypeError("source_receipt must be a ShardReceipt")
    source = _required_text(source, "source")
    if source_receipt.application_order.mode != "snapshot":
        raise ValueError("Software Heritage shard must have snapshot application order")
    manifest_index = source_receipt.application_order.manifest_index
    if manifest_index is None:
        raise ValueError("Software Heritage shard lacks manifest_index")
    release = _release(source_receipt.release)
    shard = _required_exact_text(source_receipt.shard, "shard")
    state = database.get_source_state(source)
    previous_release = _optional_release(state.get("release"))
    previous_index = (
        _nonnegative_integer(state.get("manifest_index"), "manifest_index")
        if previous_release is not None
        else None
    )
    previous_complete = state.get("complete") is True
    current_position = (release, manifest_index)
    previous_position = (
        (previous_release, previous_index)
        if previous_release is not None and previous_index is not None
        else None
    )
    if (
        previous_position is not None
        and current_position <= previous_position
        and (previous_complete or current_position < previous_position)
    ):
        if current_position == previous_position:
            expected = (
                state.get("shard"),
                state.get("control_sha256"),
                state.get("upstream_sha256"),
            )
            observed = (
                source_receipt.shard,
                source_receipt.control_sha256,
                source_receipt.upstream_sha256,
            )
            if expected != observed:
                raise ValueError("completed shard position resolved to different evidence")
        total = _nonnegative_integer(state.get("total_rows", 0), "total_rows")
        same = current_position == previous_position
        return SoftwareHeritageShardProjectionOutcome(
            status="complete",
            run_id=None,
            release=release,
            shard=shard,
            manifest_index=manifest_index,
            start_row=total if same else 0,
            next_row=total if same else 0,
            total_rows=total if same else 0,
            rows_examined=0,
            repositories=0,
            links_discovered=0,
            complete=True,
        )
    if previous_position is not None and current_position > previous_position:
        if not previous_complete:
            raise ValueError(
                f"projection shard {state.get('shard')} is incomplete; resume it first"
            )
        if release == previous_release and manifest_index != previous_index + 1:
            raise ValueError("projection cannot skip a Software Heritage manifest index")
        if release > previous_release and manifest_index != 0:
            raise ValueError("a new Software Heritage release must start at manifest index 0")
    if previous_position is None and manifest_index != 0:
        raise ValueError("initial Software Heritage projection must start at manifest index 0")

    same = current_position == previous_position
    if same:
        expected = (
            state.get("shard"),
            state.get("control_sha256"),
            state.get("upstream_sha256"),
        )
        observed = (
            source_receipt.shard,
            source_receipt.control_sha256,
            source_receipt.upstream_sha256,
        )
        if expected != observed:
            raise ValueError("incomplete shard position resolved to different evidence")
    start_row = _nonnegative_integer(state.get("next_row", 0), "next_row") if same else 0
    projector = SoftwareHeritageGitHubProjector(landing_zone)
    run_id = database.start_run(source)
    try:
        page = projector.page_shard(
            source_receipt,
            start_row=start_row,
            max_origin_rows=max_origin_rows,
            max_repositories=max_repositories,
            arrow_batch_size=arrow_batch_size,
        )
        next_state = {
            "release": release,
            "manifest_index": manifest_index,
            "shard": shard,
            "control_sha256": source_receipt.control_sha256,
            "upstream_sha256": source_receipt.upstream_sha256,
            "next_row": page.next_row,
            "total_rows": page.total_rows,
            "complete": page.complete,
        }
        result = database.ingest_page(
            source,
            SourcePage(
                records=page.records,
                next_state=next_state,
                complete=page.complete,
                upstream_count=page.total_rows,
                retry_state=state,
            ),
            run_id=run_id,
            extractor="software-heritage-origin-github-shard-v1",
            enqueue_links=True,
            link_depth=0,
        )
        if result.get("errors"):
            raise ValueError("one or more Software Heritage candidates were quarantined")
        status = "complete" if page.complete else "partial"
        database.finish_run(
            run_id,
            status,
            {
                "release": release,
                "shard": shard,
                "manifest_index": manifest_index,
                "start_row": page.start_row,
                "next_row": page.next_row,
                "total_rows": page.total_rows,
                "rows_examined": page.rows_examined,
                "repositories": len(page.candidates),
                "links_discovered": result["links_discovered"],
                "complete": page.complete,
            },
        )
        return SoftwareHeritageShardProjectionOutcome(
            status=status,
            run_id=run_id,
            release=release,
            shard=shard,
            manifest_index=manifest_index,
            start_row=page.start_row,
            next_row=page.next_row,
            total_rows=page.total_rows,
            rows_examined=page.rows_examined,
            repositories=len(page.candidates),
            links_discovered=result["links_discovered"],
            complete=page.complete,
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        database.finish_run(
            run_id,
            "failed",
            {
                "release": release,
                "shard": shard,
                "manifest_index": manifest_index,
                "start_row": start_row,
                "complete": False,
            },
            error=message,
        )
        return SoftwareHeritageShardProjectionOutcome(
            status="failed",
            run_id=run_id,
            release=release,
            shard=shard,
            manifest_index=manifest_index,
            start_row=start_row,
            next_row=start_row,
            total_rows=0,
            rows_examined=0,
            repositories=0,
            links_discovered=0,
            complete=False,
            error=message,
        )


def _candidate(
    row: Mapping[str, Any],
    *,
    expected_release: str,
) -> SoftwareHeritageGitHubCandidate | None:
    source_record_id = _required_text(row.get("source_record_id"), "source_record_id")
    landing_digest = _sha256(row.get("content_sha256"), "content_sha256")
    payload_json = _required_exact_text(row.get("payload_json"), "payload_json")
    if hashlib.sha256(payload_json.encode()).hexdigest() != landing_digest:
        raise ValueError("origin row content_sha256 does not match payload_json")
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as error:
        raise ValueError("origin row payload_json is invalid") from error
    if not isinstance(payload, Mapping) or payload.get("record_type") != "software_heritage_origin":
        raise ValueError("origin row payload has an unexpected record type")
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("origin row is missing evidence")
    encoded = _required_exact_text(payload.get("origin_url_base64"), "origin_url_base64")
    try:
        origin_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("origin_url_base64 is invalid") from error
    origin_digest = _sha256(payload.get("origin_url_sha256"), "origin_url_sha256")
    if hashlib.sha256(origin_bytes).hexdigest() != origin_digest:
        raise ValueError("origin URL digest does not match its exact bytes")
    if source_record_id != f"software-heritage:origin-url-sha256:{origin_digest}":
        raise ValueError("origin row source_record_id is inconsistent")
    origin_text = payload.get("origin_url")
    if origin_text is None:
        return None
    origin_text = _required_exact_text(origin_text, "origin_url")
    if origin_text.encode("utf-8") != origin_bytes:
        raise ValueError("origin_url text does not match its exact bytes")
    release = _release(evidence.get("release"))
    if release != expected_release:
        raise ValueError("origin release disagrees with its sealed release")
    object_key = _required_exact_text(evidence.get("object_key"), "object_key")
    object_url = _https_url(evidence.get("object_url"), "object_url")
    object_sha256 = _sha256(evidence.get("object_sha256"), "object_sha256")
    ordinal = _nonnegative_integer(evidence.get("source_row_ordinal"), "source_row_ordinal")
    locator = _required_exact_text(evidence.get("locator"), "locator")
    if object_sha256 not in locator or f":row:{ordinal}:" not in locator:
        raise ValueError("origin evidence locator is inconsistent")
    repository = _github_repository(origin_text)
    if repository is None:
        return None
    repository_name, repository_url = repository
    return SoftwareHeritageGitHubCandidate(
        repository_name=repository_name,
        repository_url=repository_url,
        origin_url=origin_text,
        origin_url_sha256=origin_digest,
        origin_source_record_id=source_record_id,
        landing_record_sha256=landing_digest,
        release=release,
        object_key=object_key,
        object_url=object_url,
        object_sha256=object_sha256,
        source_row_ordinal=ordinal,
        locator=locator,
    )


def _shard_identity(receipt: ShardReceipt) -> tuple[Any, ...]:
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
        receipt.application_order.as_dict(),
        receipt.path,
    )


def _github_repository(value: str) -> tuple[str, str] | None:
    scp = _SCP_GITHUB_RE.fullmatch(value)
    if scp is not None:
        owner = scp.group("owner")
        repository = scp.group("repository")
        return _github_name_and_url(owner, repository)

    parts = urlsplit(value)
    scheme = parts.scheme.casefold()
    if (parts.hostname or "").casefold().rstrip(".") != "github.com":
        return None
    if parts.query or parts.fragment:
        return None
    if scheme in {"http", "https", "git"}:
        if parts.username is not None or parts.password is not None:
            return None
        allowed_ports = {None, 80} if scheme == "http" else {None, 443}
        if scheme == "git":
            allowed_ports = {None, 9418}
        if parts.port not in allowed_ports:
            return None
    elif scheme == "ssh":
        if parts.username not in {None, "git"} or parts.password is not None:
            return None
        if parts.port not in {None, 22}:
            return None
    else:
        return None
    decoded = unquote(parts.path)
    if "\\" in decoded or "\x00" in decoded:
        return None
    segments = decoded.strip("/").split("/")
    if len(segments) != 2:
        return None
    return _github_name_and_url(segments[0], segments[1])


def _github_name_and_url(owner: str, repository: str) -> tuple[str, str] | None:
    if repository.casefold().endswith(".git"):
        repository = repository[:-4]
    components = (owner, repository)
    if any(
        not component
        or component in {".", ".."}
        or not component.isascii()
        or any(ord(character) < 33 or character in "/\\?#" for character in component)
        for component in components
    ):
        return None
    name = f"{owner}/{repository}"
    url = f"https://github.com/{quote(owner, safe='')}/{quote(repository, safe='')}"
    return name, url


def _https_url(value: Any, label: str) -> str:
    text = _required_exact_text(value, label)
    parts = urlsplit(text)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.port not in {None, 443}
    ):
        raise ValueError(f"{label} must be a credential-free HTTPS URL")
    return text


def _release(value: Any) -> str:
    text = _required_exact_text(value, "release")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError("release must be an ISO calendar date") from error
    if parsed.isoformat() != text:
        raise ValueError("release must be canonical")
    return text


def _optional_release(value: Any) -> str | None:
    return None if value is None else _release(value)


def _sha256(value: Any, label: str) -> str:
    text = _required_exact_text(value, label).casefold()
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _required_exact_text(value: Any, label: str) -> str:
    text = _required_text(value, label)
    if text != value:
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return text


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


__all__ = [
    "SoftwareHeritageGitHubCandidate",
    "SoftwareHeritageGitHubProjector",
    "SoftwareHeritageProjectionOutcome",
    "SoftwareHeritageProjectionPage",
    "SoftwareHeritageShardProjectionOutcome",
    "run_software_heritage_github_projection",
    "run_software_heritage_github_shard_projection",
]
