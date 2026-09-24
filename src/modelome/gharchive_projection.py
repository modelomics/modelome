from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from modelome.lake import ParquetLandingZone, ReleaseReceipt
from modelome.models import ArtifactKind, Identifier, Link, SourcePage, SourceRecord
from modelome.sources.github_release_assets import project_github_release_assets
from modelome.storage import Database

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class GhArchiveRepositoryCandidate:
    repository_id: str
    repository_name: str
    repository_url: str
    repository_api_url: str
    event_id: str
    event_type: str
    event_created_at: str
    event_source_record_id: str
    event_json_sha256: str
    landing_record_sha256: str
    archive_sha256: str
    archive_url: str
    archive_hour: str
    locator: str
    release_asset_records: tuple[SourceRecord, ...] = ()

    def as_source_record(self) -> SourceRecord:
        """Return evidence whose explicit GitHub link enters the URL frontier."""

        return SourceRecord(
            source_record_id=f"github-repository-id:{self.repository_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self.archive_url,
            title=f"GitHub repository activity: {self.repository_name}",
            published_at=self.event_created_at,
            identifiers=(
                Identifier("github:repository-id", self.repository_id),
                Identifier("github:repository", self.repository_name),
            ),
            links=(
                Link(
                    self.repository_url,
                    relation="repository_activity",
                    locator=self.locator,
                    crawl=True,
                ),
            ),
            raw={
                "record_type": "gharchive_repository_candidate",
                "repository_id": self.repository_id,
                "repository_name": self.repository_name,
                "repository_url": self.repository_url,
                "repository_api_url": self.repository_api_url,
                "event_id": self.event_id,
                "event_type": self.event_type,
                "event_created_at": self.event_created_at,
                "event_source_record_id": self.event_source_record_id,
                "event_json_sha256": self.event_json_sha256,
                "landing_record_sha256": self.landing_record_sha256,
                "archive_sha256": self.archive_sha256,
                "archive_url": self.archive_url,
                "archive_hour": self.archive_hour,
                "locator": self.locator,
                "discovery_basis": "public_repository_activity",
                "historical_repository_census": False,
            },
        )


@dataclass(frozen=True, slots=True)
class GhArchiveProjectionPage:
    release: str
    start_row: int
    next_row: int
    total_rows: int
    rows_examined: int
    candidates: tuple[GhArchiveRepositoryCandidate, ...]
    complete: bool
    release_asset_records: tuple[SourceRecord, ...] = ()

    @property
    def records(self) -> tuple[SourceRecord, ...]:
        return (
            *(candidate.as_source_record() for candidate in self.candidates),
            *self.release_asset_records,
        )


@dataclass(frozen=True, slots=True)
class GhArchiveProjectionOutcome:
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


class GhArchiveRepositoryProjector:
    """Project event rows to bounded, exact, idempotent repository candidates.

    Candidate identity uses GitHub's stable numeric repository ID. The familiar
    ``owner/name`` identifier and actionable ``https://github.com/owner/name`` URL
    are retained alongside the event locator and digests. Repeated events cannot
    create duplicate registry artifacts because all observations of one numeric
    repository ID use the same source record ID.
    """

    def __init__(
        self,
        landing_zone: ParquetLandingZone,
        *,
        source: str = "gharchive",
        dataset: str = "events",
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
        max_event_rows: int = 100_000,
        max_repositories: int = 20_000,
        max_release_asset_candidates: int = 2_000,
        arrow_batch_size: int = 8_192,
    ) -> GhArchiveProjectionPage:
        release = _required_text(release, "release")
        start_row = _nonnegative_integer(start_row, "start_row")
        max_event_rows = _positive_integer(max_event_rows, "max_event_rows")
        max_repositories = _positive_integer(max_repositories, "max_repositories")
        max_release_asset_candidates = _positive_integer(
            max_release_asset_candidates,
            "max_release_asset_candidates",
        )
        if max_release_asset_candidates < 100:
            raise ValueError("max_release_asset_candidates must be at least 100")
        arrow_batch_size = _positive_integer(arrow_batch_size, "arrow_batch_size")
        receipt = self._release(release)
        if start_row > receipt.row_count:
            raise ValueError("start_row is beyond the sealed release row count")
        if start_row == receipt.row_count:
            return GhArchiveProjectionPage(
                release=release,
                start_row=start_row,
                next_row=start_row,
                total_rows=receipt.row_count,
                rows_examined=0,
                candidates=(),
                complete=True,
            )

        candidates: dict[str, GhArchiveRepositoryCandidate] = {}
        asset_records: dict[str, SourceRecord] = {}
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
                if examined >= max_event_rows:
                    stopped = True
                    break
                candidate = _candidate(row, expected_release=release)
                if (
                    candidate.repository_id not in candidates
                    and len(candidates) >= max_repositories
                ):
                    stopped = True
                    break
                new_assets = tuple(
                    record
                    for record in candidate.release_asset_records
                    if record.source_record_id not in asset_records
                )
                if len(asset_records) + len(new_assets) > max_release_asset_candidates:
                    stopped = True
                    break
                candidates.setdefault(candidate.repository_id, candidate)
                asset_records.update(
                    (record.source_record_id, record) for record in new_assets
                )
                examined += 1
                row_index += 1
            if stopped:
                break

        next_row = start_row + examined
        if next_row > receipt.row_count:
            raise ValueError("projection read beyond the sealed release row count")
        return GhArchiveProjectionPage(
            release=release,
            start_row=start_row,
            next_row=next_row,
            total_rows=receipt.row_count,
            rows_examined=examined,
            candidates=tuple(candidates.values()),
            complete=next_row == receipt.row_count,
            release_asset_records=tuple(asset_records.values()),
        )

    def _release(self, release: str) -> ReleaseReceipt:
        matches = tuple(
            item
            for item in self.landing_zone.list_releases(
                source=self.source,
                dataset=self.dataset,
                # ``iter_release_batches`` verifies the selected release before
                # reading it. Avoid re-hashing every historical hourly shard just
                # to locate one receipt.
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


def run_gharchive_repository_projection(
    database: Database,
    landing_zone: ParquetLandingZone,
    *,
    release: str,
    source: str = "gharchive-repositories",
    max_event_rows: int = 100_000,
    max_repositories: int = 20_000,
    max_release_asset_candidates: int = 2_000,
    arrow_batch_size: int = 8_192,
) -> GhArchiveProjectionOutcome:
    """Publish one bounded projection page and enqueue exact GitHub URLs.

    Calls are restart-safe through the Database source checkpoint. A caller must
    finish the current release before advancing to another; this makes a daily
    runner unable to skip the tail of a busy hour accidentally.
    """

    if not isinstance(database, Database):
        raise TypeError("database must be a Database")
    release = _required_text(release, "release")
    source = _required_text(source, "source")
    state = database.get_source_state(source)
    previous_release = _optional_text(state.get("release"))
    previous_complete = state.get("complete") is True
    requested_hour = _release_hour(release)
    previous_hour = _release_hour(previous_release) if previous_release is not None else None
    if (
        previous_hour is not None
        and requested_hour <= previous_hour
        and (previous_complete or requested_hour < previous_hour)
    ):
        total = _nonnegative_integer(state.get("total_rows", 0), "total_rows")
        return GhArchiveProjectionOutcome(
            status="complete",
            run_id=None,
            release=release,
            start_row=total if requested_hour == previous_hour else 0,
            next_row=total if requested_hour == previous_hour else 0,
            total_rows=total if requested_hour == previous_hour else 0,
            rows_examined=0,
            repositories=0,
            links_discovered=0,
            complete=True,
        )
    if (
        previous_hour is not None
        and requested_hour > previous_hour
        and not previous_complete
    ):
        raise ValueError(
            f"projection release {previous_release} is incomplete; resume it before {release}"
        )
    start_row = (
        _nonnegative_integer(state.get("next_row", 0), "next_row")
        if previous_release == release
        else 0
    )
    projector = GhArchiveRepositoryProjector(landing_zone)
    run_id = database.start_run(source)
    try:
        page = projector.page(
            release,
            start_row=start_row,
            max_event_rows=max_event_rows,
            max_repositories=max_repositories,
            max_release_asset_candidates=max_release_asset_candidates,
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
            extractor="gharchive-event-repository-v1",
            enqueue_links=True,
            link_depth=0,
        )
        if result.get("errors"):
            raise ValueError("one or more projected repository candidates were quarantined")
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
                "release_asset_candidates": len(page.release_asset_records),
                "links_discovered": result["links_discovered"],
                "complete": page.complete,
            },
        )
        return GhArchiveProjectionOutcome(
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
            {
                "release": release,
                "start_row": start_row,
                "complete": False,
            },
            error=message,
        )
        return GhArchiveProjectionOutcome(
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


def _candidate(
    row: Mapping[str, Any],
    *,
    expected_release: str,
) -> GhArchiveRepositoryCandidate:
    source_record_id = _required_text(row.get("source_record_id"), "source_record_id")
    if not source_record_id.startswith("gharchive:event:"):
        raise ValueError("event row has an unexpected source_record_id")
    landing_record_sha256 = _sha256(row.get("content_sha256"), "content_sha256")
    raw_payload = _required_text(row.get("payload_json"), "payload_json")
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as error:
        raise ValueError("event row payload_json is invalid") from error
    if not isinstance(payload, Mapping) or payload.get("record_type") != "gharchive_public_event":
        raise ValueError("event row payload has an unexpected record type")
    repository = payload.get("repository")
    evidence = payload.get("evidence")
    if not isinstance(repository, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("event row is missing repository or evidence")

    repository_id = _positive_decimal(repository.get("id"), "repository id")
    repository_name = _required_text(repository.get("name"), "repository name")
    repository_url = _github_repository_url(
        repository.get("html_url"),
        expected_name=repository_name,
    )
    repository_api_url = _github_api_url(
        repository.get("api_url"),
        expected_name=repository_name,
    )
    identifier = repository.get("identifier")
    stable_identifier = repository.get("stable_identifier")
    if not isinstance(identifier, Mapping) or (
        identifier.get("namespace"), identifier.get("value")
    ) != ("github:repository", repository_name):
        raise ValueError("event row repository identifier is inconsistent")
    if not isinstance(stable_identifier, Mapping) or (
        stable_identifier.get("namespace"), str(stable_identifier.get("value"))
    ) != ("github:repository-id", repository_id):
        raise ValueError("event row stable repository identifier is inconsistent")

    event_id = _required_text(payload.get("event_id"), "event_id")
    if source_record_id != f"gharchive:event:{event_id}":
        raise ValueError("event row ID is inconsistent")
    archive_hour = _required_text(evidence.get("archive_hour"), "archive_hour")
    if archive_hour != expected_release:
        raise ValueError("event row archive hour disagrees with its sealed release")
    event_json_sha256 = _sha256(
        evidence.get("event_json_sha256"),
        "event_json_sha256",
    )
    archive_sha256 = _sha256(evidence.get("archive_sha256"), "archive_sha256")
    archive_url = _https_url(evidence.get("archive_url"), "archive_url")
    locator = _required_text(evidence.get("locator"), "locator")
    if event_json_sha256 not in locator or source_record_id.split(":", 2)[-1] not in locator:
        raise ValueError("event evidence locator is inconsistent")
    release_asset_records: tuple[SourceRecord, ...] = ()
    event = payload.get("event")
    if isinstance(event, Mapping) and event.get("type") == "ReleaseEvent":
        event_source_record = SourceRecord(
            source_record_id=source_record_id,
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=archive_url,
            title=f"GitHub release activity: {repository_name}",
            raw={
                "record_type": "gharchive_public_event",
                "repository": dict(repository),
                "event": dict(event),
            },
        )
        release_asset_records = tuple(
            replace(
                record,
                raw={
                    **record.raw,
                    "candidate_confidence": 0.2,
                    "candidate_scope": "gharchive-event-release-asset",
                },
            )
            for record in project_github_release_assets(event_source_record)
        )
    return GhArchiveRepositoryCandidate(
        repository_id=repository_id,
        repository_name=repository_name,
        repository_url=repository_url,
        repository_api_url=repository_api_url,
        event_id=event_id,
        event_type=_required_text(payload.get("event_type"), "event_type"),
        event_created_at=_required_text(payload.get("created_at"), "created_at"),
        event_source_record_id=source_record_id,
        event_json_sha256=event_json_sha256,
        landing_record_sha256=landing_record_sha256,
        archive_sha256=archive_sha256,
        archive_url=archive_url,
        archive_hour=archive_hour,
        locator=locator,
        release_asset_records=release_asset_records,
    )


def _github_repository_url(value: Any, *, expected_name: str) -> str:
    url = _https_url(value, "repository URL")
    parts = urlsplit(url)
    path = parts.path.strip("/")
    if (
        (parts.hostname or "").casefold() != "github.com"
        or parts.query
        or parts.fragment
        or path.casefold() != expected_name.casefold()
    ):
        raise ValueError("repository URL disagrees with repository name")
    return url


def _github_api_url(value: Any, *, expected_name: str) -> str:
    url = _https_url(value, "repository API URL")
    parts = urlsplit(url)
    path = parts.path.strip("/")
    if (
        (parts.hostname or "").casefold() != "api.github.com"
        or parts.query
        or parts.fragment
        or path.casefold() != f"repos/{expected_name}".casefold()
    ):
        raise ValueError("repository API URL disagrees with repository name")
    return url


def _https_url(value: Any, label: str) -> str:
    url = _required_text(value, label)
    parts = urlsplit(url)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.port not in {None, 443}
    ):
        raise ValueError(f"{label} must be a credential-free HTTPS URL")
    return url


def _sha256(value: Any, label: str) -> str:
    text = _required_text(value, label).casefold()
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _positive_decimal(value: Any, label: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a positive integer")
    text = str(value)
    if not text.isascii() or not text.isdecimal() or text.startswith("0"):
        raise ValueError(f"{label} must be a positive decimal integer")
    return text


def _release_hour(value: str) -> datetime:
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-(\d{1,2})", value)
    if match is None:
        raise ValueError("release must be a canonical GH Archive hour key")
    hour = int(match.group(2))
    if hour > 23 or match.group(2) != str(hour):
        raise ValueError("release must be a canonical GH Archive hour key")
    try:
        day = datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("release must be a canonical GH Archive hour key") from error
    return day.replace(hour=hour)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    return result or None


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
    "GhArchiveProjectionOutcome",
    "GhArchiveProjectionPage",
    "GhArchiveRepositoryCandidate",
    "GhArchiveRepositoryProjector",
    "run_gharchive_repository_projection",
]
