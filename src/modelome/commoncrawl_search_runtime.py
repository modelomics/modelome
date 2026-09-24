from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from modelome.commoncrawl_projection import (
    CommonCrawlDiscoveryReceipt,
    CommonCrawlWetDiscoveryProjector,
)
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, identifier_from_url, normalize_name
from modelome.storage import Database

_FORMAT = "modelome-commoncrawl-search-ingest-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_RECORD_ID = re.compile(r"^commoncrawl:wet-record:([0-9a-f]{64})$")
_TEXT_LOCATOR = re.compile(r"^text:(\d+)(?::|-)(\d+)$")

_ACTION_COMMON_FIELDS = (
    "document_id",
    "source_record_id",
    "source",
    "dataset",
    "release",
    "source_url",
    "content_sha256",
    "crawl_locator_json",
    "source_input_kind",
    "source_input_sha256",
    "source_shard",
    "source_row_ordinal",
    "projection_artifact_id",
)


@dataclass(frozen=True, slots=True)
class CommonCrawlSearchLimits:
    """Fixed work and memory ceilings for one searchable-ingestion page."""

    max_document_rows: int = 10_000
    max_actionable_documents: int = 2_000
    arrow_batch_rows: int = 8
    max_arrow_batch_bytes: int = 160 * 1024 * 1024
    max_candidate_rows_per_document: int = 1_024
    max_relation_rows_per_document: int = 100_000
    max_github_relations_per_document: int = 4_096

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")


@dataclass(frozen=True, slots=True)
class CommonCrawlSearchPage:
    release: str
    projection_artifact_id: str
    start_row: int
    next_row: int
    total_rows: int
    documents_examined: int
    records: tuple[SourceRecord, ...]
    candidate_count: int
    github_relation_count: int
    complete: bool


@dataclass(frozen=True, slots=True)
class CommonCrawlSearchOutcome:
    status: str
    run_id: int | None
    checkpoint_source: str
    release: str
    projection_artifact_id: str
    start_row: int
    next_row: int
    total_rows: int
    documents_examined: int
    actionable_documents: int
    candidate_count: int
    github_relation_count: int
    links_discovered: int
    complete: bool
    error: str | None = None


class CommonCrawlSearchProjector:
    """Merge sealed discovery tables into bounded actionable source records.

    The discovery projection remains the lossless corpus. This adapter copies only
    documents with at least one neural-model assertion or an exactly parsed GitHub
    repository relation into the searchable registry. All three inputs are merged
    monotonically by ``source_row_ordinal``; no model names or disciplines are used
    as admission filters.
    """

    def __init__(
        self,
        projector: CommonCrawlWetDiscoveryProjector,
        *,
        limits: CommonCrawlSearchLimits | None = None,
    ) -> None:
        if not isinstance(projector, CommonCrawlWetDiscoveryProjector):
            raise TypeError("projector must be a CommonCrawlWetDiscoveryProjector")
        self.projector = projector
        self.limits = limits or CommonCrawlSearchLimits()

    def verify(self, receipt: CommonCrawlDiscoveryReceipt) -> None:
        """Verify the receipt, manifest, seal, and every projected Parquet part."""

        iterator = self.projector.iter_batches(
            receipt,
            "documents",
            columns=("source_row_ordinal",),
            batch_size=1,
        )
        try:
            next(iterator)
        except StopIteration:
            pass
        finally:
            iterator.close()

    def page(
        self,
        receipt: CommonCrawlDiscoveryReceipt,
        *,
        start_row: int = 0,
    ) -> CommonCrawlSearchPage:
        if not isinstance(receipt, CommonCrawlDiscoveryReceipt):
            raise TypeError("receipt must be a CommonCrawlDiscoveryReceipt")
        start_row = _nonnegative_integer(start_row, "start_row")
        if start_row > receipt.document_count:
            raise ValueError("start_row is beyond the discovery document count")

        documents = _OrdinalCursor(
            _rows(
                self.projector,
                receipt,
                "documents",
                batch_rows=self.limits.arrow_batch_rows,
                max_batch_bytes=self.limits.max_arrow_batch_bytes,
            ),
            "documents",
        )
        candidates = _OrdinalCursor(
            _rows(
                self.projector,
                receipt,
                "model_candidates",
                batch_rows=self.limits.arrow_batch_rows,
                max_batch_bytes=self.limits.max_arrow_batch_bytes,
            ),
            "model_candidates",
        )
        relations = _OrdinalCursor(
            _rows(
                self.projector,
                receipt,
                "url_relations",
                batch_rows=self.limits.arrow_batch_rows,
                max_batch_bytes=self.limits.max_arrow_batch_bytes,
            ),
            "url_relations",
        )
        for cursor in (documents, candidates, relations):
            cursor.discard_before(start_row)

        records: list[SourceRecord] = []
        candidate_count = 0
        github_relation_count = 0
        examined = 0
        next_row = start_row
        while (
            next_row < receipt.document_count
            and examined < self.limits.max_document_rows
            and len(records) < self.limits.max_actionable_documents
        ):
            document_rows = documents.take(next_row, max_rows=1)
            if len(document_rows) != 1:
                raise ValueError(f"documents projection is missing source_row_ordinal {next_row}")
            document = document_rows[0]
            _validate_document(document, receipt)

            candidate_rows = candidates.take(
                next_row,
                max_rows=self.limits.max_candidate_rows_per_document,
            )
            for row in candidate_rows:
                _validate_candidate(row, document)

            github_rows = relations.take(
                next_row,
                max_rows=self.limits.max_relation_rows_per_document,
                select=lambda row, document=document: _validate_relation(row, document),
                max_selected=self.limits.max_github_relations_per_document,
            )
            if candidate_rows or github_rows:
                records.append(
                    _source_record(
                        document,
                        candidates=candidate_rows,
                        github_relations=github_rows,
                    )
                )
                candidate_count += len(candidate_rows)
                github_relation_count += len(github_rows)

            examined += 1
            next_row += 1

        complete = next_row == receipt.document_count
        if complete:
            if documents.current is not None:
                raise ValueError("documents projection exceeds its declared row count")
            for cursor in (candidates, relations):
                if cursor.current is not None:
                    raise ValueError(f"{cursor.table} contains an orphan source_row_ordinal")

        return CommonCrawlSearchPage(
            release=receipt.release,
            projection_artifact_id=receipt.artifact_id,
            start_row=start_row,
            next_row=next_row,
            total_rows=receipt.document_count,
            documents_examined=examined,
            records=tuple(records),
            candidate_count=candidate_count,
            github_relation_count=github_relation_count,
            complete=complete,
        )


def run_commoncrawl_search_ingestion(
    database: Database,
    projector: CommonCrawlWetDiscoveryProjector,
    receipt: CommonCrawlDiscoveryReceipt,
    *,
    checkpoint_namespace: str = "commoncrawl-search",
    artifact_source: str = "commoncrawl",
    limits: CommonCrawlSearchLimits | None = None,
) -> CommonCrawlSearchOutcome:
    """Ingest one restart-safe page from a sealed Common Crawl projection.

    Each projection artifact owns an isolated checkpoint. A completed replay is an
    integrity-checked no-op. The Database checkpoint and actionable records are
    committed atomically by ``Database.ingest_page``.
    """

    if not isinstance(database, Database):
        raise TypeError("database must be a Database")
    if not isinstance(receipt, CommonCrawlDiscoveryReceipt):
        raise TypeError("receipt must be a CommonCrawlDiscoveryReceipt")
    checkpoint_source = commoncrawl_search_checkpoint_source(
        receipt,
        namespace=checkpoint_namespace,
    )
    artifact_source = _required_text(artifact_source, "artifact_source")
    search = CommonCrawlSearchProjector(projector, limits=limits)
    state = database.get_source_state(checkpoint_source)
    if state:
        _validate_checkpoint_identity(state, receipt)
        start_row = _nonnegative_integer(state.get("next_row"), "next_row")
        if state.get("complete") is True:
            search.verify(receipt)
            return _outcome(
                status="complete",
                run_id=None,
                checkpoint_source=checkpoint_source,
                receipt=receipt,
                start_row=start_row,
                next_row=start_row,
                documents_examined=0,
                actionable_documents=0,
                candidate_count=0,
                github_relation_count=0,
                links_discovered=0,
                complete=True,
            )
    else:
        start_row = 0

    run_id = database.start_run(checkpoint_source)
    try:
        page = search.page(receipt, start_row=start_row)
        next_state = _checkpoint(receipt, page.next_row, page.complete)
        result = database.ingest_page(
            checkpoint_source,
            SourcePage(
                records=page.records,
                next_state=next_state,
                complete=page.complete,
                upstream_count=page.total_rows,
                retry_state=state,
            ),
            run_id=run_id,
            extractor=_FORMAT,
            enqueue_links=True,
            link_depth=0,
            quarantine_errors=False,
            artifact_source=artifact_source,
        )
        status = "complete" if page.complete else "partial"
        stats = {
            "release": receipt.release,
            "projection_artifact_id": receipt.artifact_id,
            "start_row": page.start_row,
            "next_row": page.next_row,
            "total_rows": page.total_rows,
            "documents_examined": page.documents_examined,
            "actionable_documents": len(page.records),
            "candidate_count": page.candidate_count,
            "github_relation_count": page.github_relation_count,
            "links_discovered": result["links_discovered"],
            "complete": page.complete,
        }
        database.finish_run(run_id, status, stats)
        return _outcome(
            status=status,
            run_id=run_id,
            checkpoint_source=checkpoint_source,
            receipt=receipt,
            start_row=page.start_row,
            next_row=page.next_row,
            documents_examined=page.documents_examined,
            actionable_documents=len(page.records),
            candidate_count=page.candidate_count,
            github_relation_count=page.github_relation_count,
            links_discovered=result["links_discovered"],
            complete=page.complete,
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        database.finish_run(
            run_id,
            "failed",
            {
                "release": receipt.release,
                "projection_artifact_id": receipt.artifact_id,
                "start_row": start_row,
                "complete": False,
            },
            error=message,
        )
        return _outcome(
            status="failed",
            run_id=run_id,
            checkpoint_source=checkpoint_source,
            receipt=receipt,
            start_row=start_row,
            next_row=start_row,
            documents_examined=0,
            actionable_documents=0,
            candidate_count=0,
            github_relation_count=0,
            links_discovered=0,
            complete=False,
            error=message,
        )


def commoncrawl_search_checkpoint_source(
    receipt: CommonCrawlDiscoveryReceipt,
    *,
    namespace: str = "commoncrawl-search",
) -> str:
    """Return the isolated Database checkpoint owner for one projection artifact."""

    if not isinstance(receipt, CommonCrawlDiscoveryReceipt):
        raise TypeError("receipt must be a CommonCrawlDiscoveryReceipt")
    prefix = _required_text(namespace, "checkpoint namespace")
    artifact_id = _sha256(receipt.artifact_id, "projection artifact ID")
    return f"{prefix}:{artifact_id}"


class _OrdinalCursor:
    def __init__(self, rows: Iterator[dict[str, Any]], table: str) -> None:
        self._rows = rows
        self.table = table
        self._last_ordinal: int | None = None
        self.current: dict[str, Any] | None = None
        self._advance()

    def discard_before(self, ordinal: int) -> None:
        while self.current is not None and self._ordinal(self.current) < ordinal:
            self._advance()

    def take(
        self,
        ordinal: int,
        *,
        max_rows: int,
        select: Callable[[Mapping[str, Any]], bool] | None = None,
        max_selected: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.current is not None and self._ordinal(self.current) < ordinal:
            raise ValueError(f"{self.table} contains an orphan source_row_ordinal")
        selected: list[dict[str, Any]] = []
        seen = 0
        while self.current is not None and self._ordinal(self.current) == ordinal:
            row = self.current
            seen += 1
            if seen > max_rows:
                raise ValueError(f"{self.table} exceeded its per-document row ceiling")
            if select is None or select(row):
                selected.append(row)
                if max_selected is not None and len(selected) > max_selected:
                    raise ValueError(f"{self.table} exceeded its selected-row ceiling")
            self._advance()
        return selected

    def _advance(self) -> None:
        try:
            row = next(self._rows)
        except StopIteration:
            self.current = None
            return
        ordinal = self._ordinal(row)
        if self._last_ordinal is not None and ordinal < self._last_ordinal:
            raise ValueError(f"{self.table} is not ordered by source_row_ordinal")
        self._last_ordinal = ordinal
        self.current = row

    def _ordinal(self, row: Mapping[str, Any]) -> int:
        return _nonnegative_integer(
            row.get("source_row_ordinal"),
            f"{self.table} source_row_ordinal",
        )


def _rows(
    projector: CommonCrawlWetDiscoveryProjector,
    receipt: CommonCrawlDiscoveryReceipt,
    table: str,
    *,
    batch_rows: int,
    max_batch_bytes: int,
) -> Iterator[dict[str, Any]]:
    for batch in projector.iter_batches(
        receipt,
        table,
        batch_size=batch_rows,
    ):
        if not isinstance(batch, pa.RecordBatch):
            raise TypeError(f"{table} iterator returned a non-Arrow batch")
        if batch.nbytes > max_batch_bytes:
            raise ValueError(f"{table} Arrow batch exceeded max_arrow_batch_bytes")
        yield from batch.to_pylist()


def _validate_document(
    row: Mapping[str, Any],
    receipt: CommonCrawlDiscoveryReceipt,
) -> None:
    ordinal = _nonnegative_integer(row.get("source_row_ordinal"), "document ordinal")
    if row.get("projection_artifact_id") != receipt.artifact_id:
        raise ValueError("document projection artifact identity is inconsistent")
    expected = {
        "source": receipt.source,
        "dataset": receipt.dataset,
        "release": receipt.release,
        "source_input_kind": receipt.source_input_kind,
        "source_input_sha256": receipt.source_input_sha256,
        "source_shard": receipt.source_shard,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"document {field} conflicts with its projection receipt")

    source_record_id = _required_text(row.get("source_record_id"), "source_record_id")
    record_match = _SOURCE_RECORD_ID.fullmatch(source_record_id)
    if record_match is None:
        raise ValueError("document source_record_id is noncanonical")
    source_url = _required_text(row.get("source_url"), "source_url")
    text = _text(row.get("text"), "document text")
    content_sha256 = _sha256(row.get("content_sha256"), "content_sha256")
    if hashlib.sha256(text.encode()).hexdigest() != content_sha256:
        raise ValueError("document content checksum mismatch")
    if _nonnegative_integer(row.get("content_bytes"), "content_bytes") != len(text.encode()):
        raise ValueError("document content byte count mismatch")
    _sha256(row.get("landing_payload_sha256"), "landing_payload_sha256")
    crawl_locator_json = _canonical_json_text(row.get("crawl_locator_json"), "crawl_locator_json")
    crawl_locator = json.loads(crawl_locator_json)
    if not isinstance(crawl_locator, Mapping):
        raise ValueError("crawl locator must be an object")
    if crawl_locator.get("collection_id") != receipt.release:
        raise ValueError("crawl locator release is inconsistent")
    if row.get("warc_record_id") != crawl_locator.get("warc_record_id"):
        raise ValueError("WARC record ID is inconsistent")
    if row.get("crawl_object_url") != crawl_locator.get("object_url"):
        raise ValueError("crawl object URL is inconsistent")
    if row.get("crawl_path") != crawl_locator.get("path"):
        raise ValueError("crawl object path is inconsistent")
    for field, locator_field in (
        ("crawl_manifest_index", "manifest_index"),
        ("crawl_record_index", "record_index"),
        ("crawl_conversion_index", "conversion_index"),
    ):
        if row.get(field) != crawl_locator.get(locator_field):
            raise ValueError(f"document {field} is inconsistent")
    document_identity = {
        "source": receipt.source,
        "source_record_id": source_record_id,
        "source_url": source_url,
        "content_sha256": content_sha256,
        "crawl_locator": dict(crawl_locator),
    }
    if row.get("document_id") != _json_sha256(document_identity):
        raise ValueError("document ID is inconsistent")
    _required_text(row.get("crawled_at"), "crawled_at")
    _required_text(row.get("content_type"), "content_type")
    _validate_source_identifier(row, source_url)
    if ordinal >= receipt.document_count:
        raise ValueError("document ordinal exceeds the projection receipt")


def _validate_source_identifier(row: Mapping[str, Any], source_url: str) -> None:
    try:
        derived = identifier_from_url(source_url)
    except ValueError:
        derived = None
    namespace = row.get("source_identifier_namespace")
    value = row.get("source_identifier_value")
    repository_url = row.get("source_repository_url")
    if derived is None:
        if namespace is not None or value is not None or repository_url is not None:
            raise ValueError("document source identifier is inconsistent")
        return
    if (namespace, value) != (derived.namespace, derived.value):
        raise ValueError("document source identifier is inconsistent")
    expected_repository = (
        f"https://github.com/{derived.value}" if derived.namespace == "github:repository" else None
    )
    if repository_url != expected_repository:
        raise ValueError("document source repository URL is inconsistent")


def _validate_candidate(
    row: Mapping[str, Any],
    document: Mapping[str, Any],
) -> None:
    _validate_action_common(row, document, "candidate")
    name = _required_text(row.get("name"), "candidate name")
    normalized = _required_text(row.get("normalized_name"), "normalized name")
    if normalize_name(name) != normalized:
        raise ValueError("candidate normalized name is inconsistent")
    if row.get("status") != ModelStatus.CANDIDATE.value:
        raise ValueError("Common Crawl searchable candidates must have candidate status")
    confidence = row.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("candidate confidence is invalid")
    extractor = _required_text(row.get("extractor"), "candidate extractor")
    local_id = _required_text(row.get("extractor_local_id"), "candidate local ID")
    locator = _required_text(row.get("locator"), "candidate locator")
    if row.get("evidence_field") != "text":
        raise ValueError("WET candidate evidence must refer to document text")
    start, end = _locator(locator)
    text = _text(document.get("text"), "document text")
    _validate_span(row, text, start, end, "candidate")
    if normalize_name(text[start:end]) != normalized:
        raise ValueError("candidate locator does not identify its name")
    assertion_identity = {
        "document_id": document["document_id"],
        "extractor": extractor,
        "extractor_local_id": local_id,
        "locator": locator,
        "normalized_name": normalized,
        "content_sha256": document["content_sha256"],
    }
    if row.get("candidate_assertion_id") != _json_sha256(assertion_identity):
        raise ValueError("candidate assertion ID is inconsistent")
    for field in (
        "source_identifier_namespace",
        "source_identifier_value",
        "source_repository_url",
    ):
        if row.get(field) != document.get(field):
            raise ValueError(f"candidate {field} is inconsistent")
    derivation_json = _canonical_json_text(row.get("derivation_json"), "candidate derivation")
    derivation = json.loads(derivation_json)
    if not isinstance(derivation, Mapping):
        raise ValueError("candidate derivation must be an object")
    if (
        derivation.get("assertion_identity") != assertion_identity
        or derivation.get("projection_artifact_id") != document["projection_artifact_id"]
        or derivation.get("source_row_ordinal") != document["source_row_ordinal"]
        or derivation.get("crawl_locator") != json.loads(str(document["crawl_locator_json"]))
    ):
        raise ValueError("candidate derivation is inconsistent")


def _validate_relation(
    row: Mapping[str, Any],
    document: Mapping[str, Any],
) -> bool:
    _validate_action_common(row, document, "URL relation")
    predicate = _required_text(row.get("predicate"), "relation predicate")
    target_url = _required_text(row.get("target_url"), "relation target URL")
    locator = _required_text(row.get("locator"), "relation locator")
    start, end = _locator(locator)
    text = _text(document.get("text"), "document text")
    _validate_span(row, text, start, end, "URL relation")
    if canonicalize_url(text[start:end]) != target_url:
        raise ValueError("relation locator does not identify its target URL")
    assertion_identity = {
        "document_id": document["document_id"],
        "predicate": predicate,
        "target_url": target_url,
        "locator": locator,
        "content_sha256": document["content_sha256"],
    }
    if row.get("relation_assertion_id") != _json_sha256(assertion_identity):
        raise ValueError("relation assertion ID is inconsistent")
    try:
        identifier = identifier_from_url(target_url)
    except ValueError:
        identifier = None
    namespace = row.get("target_identifier_namespace")
    value = row.get("target_identifier_value")
    repository_url = row.get("target_repository_url")
    if identifier is None:
        if namespace is not None or value is not None or repository_url is not None:
            raise ValueError("relation target identifier is inconsistent")
        return False
    if (namespace, value) != (identifier.namespace, identifier.value):
        raise ValueError("relation target identifier is inconsistent")
    if identifier.namespace != "github:repository":
        if repository_url is not None:
            raise ValueError("non-GitHub relation has a repository URL")
        return False
    if repository_url != f"https://github.com/{identifier.value}":
        raise ValueError("GitHub relation repository URL is inconsistent")
    return True


def _validate_action_common(
    row: Mapping[str, Any],
    document: Mapping[str, Any],
    label: str,
) -> None:
    for field in _ACTION_COMMON_FIELDS:
        if row.get(field) != document.get(field):
            raise ValueError(f"{label} {field} is inconsistent with its document")


def _validate_span(
    row: Mapping[str, Any],
    text: str,
    start: int,
    end: int,
    label: str,
) -> None:
    if start < 0 or start >= end or end > len(text):
        raise ValueError(f"{label} locator is outside document text")
    if (
        row.get("evidence_start") != start
        or row.get("evidence_end") != end
        or row.get("evidence_text") != text[start:end]
    ):
        raise ValueError(f"{label} evidence span is inconsistent")
    context_start = _nonnegative_integer(row.get("context_start"), f"{label} context_start")
    context_end = _nonnegative_integer(row.get("context_end"), f"{label} context_end")
    if (
        context_start > start
        or context_end < end
        or context_end > len(text)
        or row.get("context_text") != text[context_start:context_end]
    ):
        raise ValueError(f"{label} evidence context is inconsistent")


def _source_record(
    document: Mapping[str, Any],
    *,
    candidates: list[dict[str, Any]],
    github_relations: list[dict[str, Any]],
) -> SourceRecord:
    identifier = _document_identifier(document)
    links = tuple(
        Link(
            url=_required_text(row.get("target_repository_url"), "GitHub repository URL"),
            relation=_required_text(row.get("predicate"), "relation predicate"),
            locator=_required_text(row.get("locator"), "relation locator"),
            crawl=True,
        )
        for row in github_relations
    )
    hints = tuple(
        ModelHint(
            local_id=_required_text(row.get("candidate_assertion_id"), "candidate assertion ID"),
            name=_required_text(row.get("name"), "candidate name"),
            status=ModelStatus.CANDIDATE,
            confidence=float(row["confidence"]),
            locator=_required_text(row.get("locator"), "candidate locator"),
        )
        for row in candidates
    )
    document_provenance = {key: value for key, value in document.items() if key != "text"}
    return SourceRecord(
        source_record_id=_required_text(document.get("source_record_id"), "source_record_id"),
        kind=(
            ArtifactKind.CODE_REPOSITORY
            if identifier is not None and identifier.namespace == "github:repository"
            else ArtifactKind.WEB_PAGE
        ),
        canonical_url=_required_text(document.get("source_url"), "source_url"),
        title=_required_text(document.get("source_url"), "source_url"),
        raw={
            "record_type": "commoncrawl_actionable_document",
            "format": _FORMAT,
            "document": document_provenance,
            "candidate_assertions": candidates,
            "github_relation_assertions": github_relations,
        },
        # The lossless text remains in the sealed projection. Keeping it out here
        # also prevents generic URL extraction from enqueueing a less precise path
        # in addition to the exact repository roots declared above.
        text="",
        identifiers=(identifier,) if identifier is not None else (),
        links=links,
        models=hints,
    )


def _document_identifier(row: Mapping[str, Any]) -> Identifier | None:
    namespace = row.get("source_identifier_namespace")
    value = row.get("source_identifier_value")
    if namespace is None and value is None:
        return None
    return Identifier(
        _required_text(namespace, "source identifier namespace"),
        _required_text(value, "source identifier value"),
    )


def _checkpoint(
    receipt: CommonCrawlDiscoveryReceipt,
    next_row: int,
    complete: bool,
) -> dict[str, Any]:
    return {
        "projection_artifact_id": receipt.artifact_id,
        "release": receipt.release,
        "source_input_kind": receipt.source_input_kind,
        "source_input_sha256": receipt.source_input_sha256,
        "source_shard": receipt.source_shard,
        "next_row": next_row,
        "total_rows": receipt.document_count,
        "complete": complete,
    }


def _validate_checkpoint_identity(
    state: Mapping[str, Any],
    receipt: CommonCrawlDiscoveryReceipt,
) -> None:
    expected = _checkpoint(
        receipt,
        _nonnegative_integer(state.get("next_row"), "next_row"),
        state.get("complete") is True,
    )
    if dict(state) != expected:
        raise ValueError("search checkpoint does not match its discovery projection")
    if expected["next_row"] > receipt.document_count:
        raise ValueError("search checkpoint is beyond the discovery document count")


def _outcome(
    *,
    status: str,
    run_id: int | None,
    checkpoint_source: str,
    receipt: CommonCrawlDiscoveryReceipt,
    start_row: int,
    next_row: int,
    documents_examined: int,
    actionable_documents: int,
    candidate_count: int,
    github_relation_count: int,
    links_discovered: int,
    complete: bool,
    error: str | None = None,
) -> CommonCrawlSearchOutcome:
    return CommonCrawlSearchOutcome(
        status=status,
        run_id=run_id,
        checkpoint_source=checkpoint_source,
        release=receipt.release,
        projection_artifact_id=receipt.artifact_id,
        start_row=start_row,
        next_row=next_row,
        total_rows=receipt.document_count,
        documents_examined=documents_examined,
        actionable_documents=actionable_documents,
        candidate_count=candidate_count,
        github_relation_count=github_relation_count,
        links_discovered=links_discovered,
        complete=complete,
        error=error,
    )


def _locator(value: str) -> tuple[int, int]:
    match = _TEXT_LOCATOR.fullmatch(value)
    if match is None:
        raise ValueError("locator must identify an exact document-text span")
    return int(match.group(1)), int(match.group(2))


def _canonical_json_text(value: Any, label: str) -> str:
    text = _required_text(value, label)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if _canonical_json(decoded) != text:
        raise ValueError(f"{label} must be canonical JSON")
    return text


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256(value: Any, label: str) -> str:
    text = _required_text(value, label).casefold()
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


__all__ = [
    "CommonCrawlSearchLimits",
    "CommonCrawlSearchOutcome",
    "CommonCrawlSearchPage",
    "CommonCrawlSearchProjector",
    "commoncrawl_search_checkpoint_source",
    "run_commoncrawl_search_ingestion",
]
