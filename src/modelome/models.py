from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ArtifactKind(StrEnum):
    PAPER = "paper"
    CODE_REPOSITORY = "code_repository"
    MODEL_CARD = "model_card"
    WEIGHTS = "weights"
    PROVIDER_PAGE = "provider_page"
    BLOG_POST = "blog_post"
    CATALOG_RECORD = "catalog_record"
    WEB_PAGE = "web_page"
    OTHER = "other"


class ModelStatus(StrEnum):
    DOCUMENTED = "documented"
    RELEASED = "released"
    CANDIDATE = "candidate"
    STUB = "stub"


@dataclass(frozen=True, slots=True)
class Identifier:
    namespace: str
    value: str


@dataclass(frozen=True, slots=True)
class Link:
    url: str
    relation: str = "references"
    locator: str | None = None
    crawl: bool = True
    # Empty means the link describes the source record as a whole.  A non-empty
    # tuple explicitly limits the link to these ModelHint.local_id values in
    # the same source record.  This prevents a catalog-wide resource list from
    # being copied onto every entry subject during a later entry build.
    model_local_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelHint:
    """A source-backed assertion that a model exists.

    ``local_id`` is scoped to one source record. It lets relation hints refer to
    a declaration without making the source choose a registry identifier.
    """

    local_id: str
    name: str
    identifiers: tuple[Identifier, ...] = ()
    aliases: tuple[str, ...] = ()
    status: ModelStatus = ModelStatus.DOCUMENTED
    confidence: float = 1.0
    locator: str | None = None


@dataclass(frozen=True, slots=True)
class ModelRelationHint:
    subject_local_id: str
    predicate: str
    target: ModelHint
    confidence: float = 1.0
    locator: str | None = None


@dataclass(frozen=True, slots=True)
class ReleaseHint:
    """A source-backed trained release or checkpoint assertion.

    ``model_local_id`` must name a ``ModelHint`` in the same source record.
    Versions and revisions are descriptive and never resolve identity; only an
    exact namespaced identifier can reuse a release across source records.
    """

    local_id: str
    model_local_id: str
    version: str | None = None
    revision: str | None = None
    identifiers: tuple[Identifier, ...] = ()
    released_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    locator: str | None = None


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_record_id: str
    kind: ArtifactKind
    canonical_url: str
    title: str
    raw: Mapping[str, Any]
    text: str = ""
    published_at: str | None = None
    modified_at: str | None = None
    identifiers: tuple[Identifier, ...] = ()
    links: tuple[Link, ...] = ()
    models: tuple[ModelHint, ...] = ()
    model_relations: tuple[ModelRelationHint, ...] = ()
    releases: tuple[ReleaseHint, ...] = ()
    deleted: bool = False


@dataclass(frozen=True, slots=True)
class SourceIssue:
    """One upstream item that could not be normalized without blocking its page."""

    source_record_id: str
    stage: str
    error: str
    summary: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourcePage:
    records: tuple[SourceRecord, ...]
    next_state: Mapping[str, Any]
    complete: bool
    upstream_count: int | None = None
    authoritative_snapshot: bool = False
    issues: tuple[SourceIssue, ...] = ()
    retry_state: Mapping[str, Any] | None = None
    # Immutable snapshots can contain rows that are permanently malformed. Such
    # rows remain quarantined, but an adapter can explicitly permit the cursor to
    # advance so a single row does not replay an otherwise valid page forever.
    advance_on_source_issues: bool = False


@dataclass(slots=True)
class SyncStats:
    source: str
    pages: int = 0
    records_seen: int = 0
    new_artifacts: int = 0
    new_revisions: int = 0
    models_touched: int = 0
    links_discovered: int = 0
    errors: list[str] = field(default_factory=list)
    complete: bool = False
    upstream_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pages": self.pages,
            "records_seen": self.records_seen,
            "new_artifacts": self.new_artifacts,
            "new_revisions": self.new_revisions,
            "models_touched": self.models_touched,
            "links_discovered": self.links_discovered,
            "errors": list(self.errors),
            "complete": self.complete,
            "upstream_count": self.upstream_count,
        }
