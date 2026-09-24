"""One-paper evidence ingestion and entry planning.

This module composes the existing source-record store, conservative candidate
extractor, bounded URL frontier, and offline entry planner into the small unit
of work that a later corpus enumerator can repeat.  It deliberately accepts a
normalized source observation rather than fetching a paper by title or crawling
the web: source adapters own identity resolution, and this worker must never
turn a fuzzy search result into a paper record.

Running the worker persists the paper observation and its direct resource
evidence.  It returns an entry *plan*, not a materialized entry bundle.  Public
entry construction remains an explicit later operation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from modelome.entries import plan_entry_seed
from modelome.extract import IntroductionCueExtractor
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelRelationHint,
    ModelStatus,
    ReleaseHint,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, extract_url_mentions, infer_url_relation
from modelome.storage import Database

PAPER_INGEST_RUN_SOURCE = "paper-ingest"


@dataclass(frozen=True, slots=True)
class PaperIngestPreparation:
    """Validated one-paper work item and its no-side-effect entry plan."""

    record: SourceRecord
    entry_seed: Mapping[str, Any]
    plan: Mapping[str, Any]
    candidate_count: int
    direct_resource_count: int
    extractor_name: str


@dataclass(frozen=True, slots=True)
class PaperIngestOutcome:
    """Receipt for one persisted paper observation; no entry bundle is written."""

    source: str
    source_record_id: str
    canonical_url: str
    candidate_count: int
    direct_resource_count: int
    extractor: str
    stats: Mapping[str, int]
    entry_seed: Mapping[str, Any]
    plan: Mapping[str, Any]
    materialized_entries: int = 0


def prepare_paper_ingestion(seed: Mapping[str, Any]) -> PaperIngestPreparation:
    """Validate one exact paper observation and derive its reviewable plan.

    The input uses the portable entry-seed shape plus optional ``text``,
    ``raw``, ``published_at``, and ``modified_at`` observation fields.  It must
    represent one paper.  Direct links already declared by the source and URLs
    directly found in retained text are both retained.  If the source supplied
    no model/technique declaration, the conservative introduction extractor may
    add candidate subjects; it never runs alongside an explicit declaration.
    """

    if not isinstance(seed, Mapping):
        raise ValueError("paper input must be an object")
    record = _record_from_seed(seed)
    if record.models:
        derived: tuple[ModelHint, ...] = ()
        extractor_name = "source"
    else:
        extractor = IntroductionCueExtractor()
        derived = tuple(extractor.extract(record))
        extractor_name = extractor.name

    entry_seed = _entry_seed_with_direct_resources(seed, record, derived)
    plan = plan_entry_seed(entry_seed)
    return PaperIngestPreparation(
        record=record,
        entry_seed=entry_seed,
        plan=plan,
        candidate_count=len(entry_seed["models"]),
        direct_resource_count=len(entry_seed["links"]),
        extractor_name=extractor_name,
    )


def ingest_paper(database: Database, seed: Mapping[str, Any]) -> PaperIngestOutcome:
    """Persist one prepared paper and return its entry plan without building entries."""

    prepared = prepare_paper_ingestion(seed)
    run_id = database.start_run(PAPER_INGEST_RUN_SOURCE)
    extractor: str | IntroductionCueExtractor = (
        "source" if prepared.extractor_name == "source" else IntroductionCueExtractor()
    )
    try:
        stats = database.ingest_page(
            PAPER_INGEST_RUN_SOURCE,
            (prepared.record,),
            {
                "mode": "one-paper",
                "source": _source(prepared.entry_seed),
                "source_record_id": prepared.record.source_record_id,
            },
            run_id=run_id,
            extractor=extractor,
            artifact_source=_source(prepared.entry_seed),
            complete=False,
        )
        status = "complete" if stats["errors"] == 0 else "partial"
        database.finish_run(run_id, status, {"source": PAPER_INGEST_RUN_SOURCE, **stats})
    except Exception as error:
        database.finish_run(
            run_id,
            "failed",
            {"source": PAPER_INGEST_RUN_SOURCE},
            error=f"{type(error).__name__}: {error}",
        )
        raise
    return PaperIngestOutcome(
        source=_source(prepared.entry_seed),
        source_record_id=prepared.record.source_record_id,
        canonical_url=prepared.record.canonical_url,
        candidate_count=prepared.candidate_count,
        direct_resource_count=prepared.direct_resource_count,
        extractor=prepared.extractor_name,
        stats=stats,
        entry_seed=prepared.entry_seed,
        plan=prepared.plan,
    )


def _record_from_seed(seed: Mapping[str, Any]) -> SourceRecord:
    _source(seed)
    source_record_id = _required_text(seed.get("source_record_id"), "source_record_id")
    kind = _paper_kind(seed.get("kind"))
    canonical_url = _canonical_url(seed.get("canonical_url"), "canonical_url")
    title = _required_text(seed.get("title"), "title")
    raw = seed.get("raw", {})
    if not isinstance(raw, Mapping):
        raise ValueError("raw must be an object")
    text = seed.get("text", "")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    published_at = _optional_text(seed.get("published_at"), "published_at")
    modified_at = _optional_text(seed.get("modified_at"), "modified_at")
    return SourceRecord(
        source_record_id=source_record_id,
        kind=kind,
        canonical_url=canonical_url,
        title=title,
        raw=dict(raw),
        text=text,
        published_at=published_at,
        modified_at=modified_at,
        identifiers=_identifiers(seed.get("identifiers", ()), "identifiers"),
        links=_links(seed.get("links", ())),
        models=_model_hints(seed.get("models", ()), "models"),
        model_relations=_model_relations(seed.get("model_relations", ())),
        releases=_releases(seed.get("releases", ())),
    )


def _entry_seed_with_direct_resources(
    source_seed: Mapping[str, Any],
    record: SourceRecord,
    derived: Sequence[ModelHint],
) -> dict[str, Any]:
    """Return the original portable seed plus its source-local text discoveries."""

    models = source_seed.get("models", ())
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes, bytearray)):
        raise ValueError("models must be a list")
    # Omit input ``raw`` and retained text deliberately. An entry seed is a
    # portable public-facing work item; source text can be rights-restricted and
    # remains only in the evidence store. Unknown keys are likewise not a part
    # of the seed contract and must not leak through a later worker queue.
    return {
        "source": _source(source_seed),
        "source_record_id": record.source_record_id,
        "canonical_url": record.canonical_url,
        "title": record.title,
        "kind": record.kind.value,
        "identifiers": [
            {"namespace": item.namespace, "value": item.value}
            for item in record.identifiers
        ],
        "tags": source_seed.get("tags", ()),
        "links": [
            {
                "url": item.url,
                "relation": item.relation,
                "locator": item.locator,
                "crawl": item.crawl,
            }
            for item in (*record.links, *_text_links(record))
        ],
        "models": [*models, *(_model_hint_dict(item) for item in derived)],
        "model_relations": source_seed.get("model_relations", ()),
        "releases": source_seed.get("releases", ()),
    }


def _text_links(record: SourceRecord) -> tuple[Link, ...]:
    """Keep every direct text URL with a span locator and cautious relation."""

    existing = {
        (canonicalize_url(item.url), item.relation, item.locator)
        for item in record.links
    }
    record_url = canonicalize_url(record.canonical_url)
    discovered = []
    for url, locator in extract_url_mentions(record.text):
        if url == record_url:
            continue
        relation = infer_url_relation(record.text, locator)
        key = (url, relation, locator)
        if key in existing:
            continue
        existing.add(key)
        discovered.append(Link(url, relation=relation, locator=locator, crawl=True))
    return tuple(discovered)


def _paper_kind(value: Any) -> ArtifactKind:
    kind = _required_text(value, "kind")
    if kind != ArtifactKind.PAPER.value:
        raise ValueError("ingest-paper input kind must be 'paper'")
    return ArtifactKind.PAPER


def _links(value: Any) -> tuple[Link, ...]:
    rows = _objects(value, "links")
    result = []
    for index, row in enumerate(rows):
        url = _canonical_url(row.get("url"), f"link {index} url")
        relation = _optional_text(row.get("relation"), f"link {index} relation") or "references"
        locator = _optional_text(row.get("locator"), f"link {index} locator")
        crawl = row.get("crawl", True)
        if not isinstance(crawl, bool):
            raise ValueError(f"link {index} crawl must be true or false")
        result.append(Link(url, relation=relation, locator=locator, crawl=crawl))
    return tuple(result)


def _model_hints(value: Any, field: str) -> tuple[ModelHint, ...]:
    result = []
    for index, row in enumerate(_objects(value, field)):
        result.append(_model_hint(row, f"{field} {index}"))
    return tuple(result)


def _model_hint(value: Mapping[str, Any], field: str) -> ModelHint:
    status = _model_status(value.get("status", ModelStatus.DOCUMENTED.value), field)
    return ModelHint(
        local_id=_required_text(value.get("local_id"), f"{field} local_id"),
        name=_required_text(value.get("name"), f"{field} name"),
        identifiers=_identifiers(value.get("identifiers", ()), f"{field} identifiers"),
        aliases=_text_list(value.get("aliases", ()), f"{field} aliases"),
        status=status,
        confidence=_confidence(value.get("confidence", 1.0), f"{field} confidence"),
        locator=_optional_text(value.get("locator"), f"{field} locator"),
    )


def _model_relations(value: Any) -> tuple[ModelRelationHint, ...]:
    result = []
    for index, row in enumerate(_objects(value, "model_relations")):
        target = row.get("target")
        if not isinstance(target, Mapping):
            raise ValueError(f"model relation {index} target must be an object")
        result.append(
            ModelRelationHint(
                subject_local_id=_required_text(
                    row.get("subject_local_id"),
                    f"model relation {index} subject_local_id",
                ),
                predicate=_required_text(row.get("predicate"), f"model relation {index} predicate"),
                target=_model_hint(target, f"model relation {index} target"),
                confidence=_confidence(
                    row.get("confidence", 1.0),
                    f"model relation {index} confidence",
                ),
                locator=_optional_text(row.get("locator"), f"model relation {index} locator"),
            )
        )
    return tuple(result)


def _releases(value: Any) -> tuple[ReleaseHint, ...]:
    result = []
    for index, row in enumerate(_objects(value, "releases")):
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"release {index} metadata must be an object")
        result.append(
            ReleaseHint(
                local_id=_required_text(row.get("local_id"), f"release {index} local_id"),
                model_local_id=_required_text(
                    row.get("model_local_id"),
                    f"release {index} model_local_id",
                ),
                version=_optional_text(row.get("version"), f"release {index} version"),
                revision=_optional_text(row.get("revision"), f"release {index} revision"),
                identifiers=_identifiers(
                    row.get("identifiers", ()),
                    f"release {index} identifiers",
                ),
                released_at=_optional_text(
                    row.get("released_at"),
                    f"release {index} released_at",
                ),
                metadata=dict(metadata),
                confidence=_confidence(row.get("confidence", 1.0), f"release {index} confidence"),
                locator=_optional_text(row.get("locator"), f"release {index} locator"),
            )
        )
    return tuple(result)


def _identifiers(value: Any, field: str) -> tuple[Identifier, ...]:
    result = []
    for index, row in enumerate(_objects(value, field)):
        result.append(
            Identifier(
                _required_text(row.get("namespace"), f"{field} {index} namespace"),
                _required_text(row.get("value"), f"{field} {index} value"),
            )
        )
    return tuple(result)


def _model_hint_dict(value: ModelHint) -> dict[str, Any]:
    return {
        "local_id": value.local_id,
        "name": value.name,
        "aliases": list(value.aliases),
        "identifiers": [
            {"namespace": item.namespace, "value": item.value}
            for item in value.identifiers
        ],
        "status": value.status.value,
        "confidence": value.confidence,
        "locator": value.locator,
    }


def _objects(value: Any, field: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a list")
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field} must contain objects")
        result.append(item)
    return tuple(result)


def _text_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a list")
    return tuple(_required_text(item, field) for item in value)


def _model_status(value: Any, field: str) -> ModelStatus:
    try:
        return ModelStatus(_required_text(value, f"{field} status"))
    except ValueError as error:
        raise ValueError(f"{field} status is not a supported model status") from error


def _confidence(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number from 0 through 1")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a number from 0 through 1") from error
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be a number from 0 through 1")
    return result


def _canonical_url(value: Any, field: str) -> str:
    candidate = _required_text(value, field)
    try:
        return canonicalize_url(candidate)
    except ValueError as error:
        raise ValueError(f"{field} must be an absolute HTTP(S) URL") from error


def _source(value: Mapping[str, Any]) -> str:
    return _required_text(value.get("source"), "source")


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not (result := value.strip()):
        raise ValueError(f"{field} must be non-empty text")
    return result


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field)


__all__ = [
    "PAPER_INGEST_RUN_SOURCE",
    "PaperIngestOutcome",
    "PaperIngestPreparation",
    "ingest_paper",
    "prepare_paper_ingestion",
]
