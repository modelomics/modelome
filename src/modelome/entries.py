"""Deterministic, identity-first construction of Modelome entries.

This module deliberately does *not* crawl, download papers, or touch the registry
store.  It turns source-declared observations into a small, portable entry corpus
when an operator explicitly runs it.  A future worker can feed it one paper or
catalog record at a time, resolve that record's declared resources, and resume from
the resulting seed stream.

An entry is only merged when two observations share an exact namespaced identifier
or one source explicitly declares the other's exact identifier as a model mirror.
Names, tags, repository hosts, and broad source URLs are useful navigation signals,
but are never merge keys. That makes a complete future run conservative by default:
missing evidence creates a separate entry rather than silently combining techniques.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from modelome.models import SourceRecord
from modelome.normalize import canonicalize_url, content_hash, identifier_from_url, normalize_name

ENTRY_FORMAT = "modelome-entry-corpus-v1"

_PAPER_RELATIONS = frozenset(
    {
        "associated_paper",
        "doi",
        "paper",
        "paper_reference",
        "publication",
        "source_paper",
    }
)
_CODE_RELATIONS = frozenset(
    {
        "code",
        "code_reference",
        "implementation",
        "official_implementation",
        "source_implementation",
    }
)
_WEIGHT_RELATIONS = frozenset(
    {
        "checkpoint",
        "inference_artifact",
        "linked_model_artifact",
        "model_artifact",
        "model_weights",
        "weights",
    }
)
_CARD_RELATIONS = frozenset(
    {"model_card", "model_card_metadata", "model_config", "provider_page"}
)
_NON_FETCHABLE_RELATIONS = frozenset(
    {
        "checkpoint",
        "inference_artifact",
        "linked_model_artifact",
        "model_artifact",
        "model_weights",
        "weights",
    }
)
# Generic weights/model_weights links are collected so provenance can identify
# shared URLs, but only URLs that every owner calls a checkpoint may merge.
_CHECKPOINT_IDENTITY_RELATIONS = frozenset({"checkpoint", "model_weights", "weights"})
_CITATION_RELATIONS = frozenset({"cites", "cited_by", "is-cited-by"})


@dataclass(frozen=True, slots=True)
class EntryIdentifier:
    namespace: str
    value: str

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.value}"


@dataclass(frozen=True, slots=True)
class EntryResolvedArtifact:
    """A concrete registry artifact reached through a sealed relation edge."""

    id: str
    revision_id: str
    kind: str
    source: str
    source_record_id: str


@dataclass(frozen=True, slots=True)
class EntryRelationEvidence:
    """Provenance for a cross-record resource link, never an identity key."""

    id: str
    direction: str
    evidence_type: str
    confidence: float
    match_namespace: str | None
    match_value: str | None


@dataclass(frozen=True, slots=True)
class EntryResource:
    url: str
    relation: str
    locator: str | None
    crawl: bool
    category: str
    source: str
    source_record_id: str
    model_local_id: str
    resolved_artifact: EntryResolvedArtifact | None
    relation_evidence: EntryRelationEvidence | None


@dataclass(frozen=True, slots=True)
class EntryMember:
    source: str
    source_record_id: str
    artifact_url: str
    artifact_identifiers: tuple[EntryIdentifier, ...]
    local_id: str
    name: str
    aliases: tuple[str, ...]
    status: str
    confidence: float
    locator: str | None


@dataclass(frozen=True, slots=True)
class EntryRelease:
    """A source-declared model artifact attached to one entry subject."""

    source: str
    source_record_id: str
    model_local_id: str
    local_id: str
    version: str | None
    revision: str | None
    identifiers: tuple[EntryIdentifier, ...]
    released_at: str | None
    metadata_json: str
    confidence: float
    locator: str | None


@dataclass(frozen=True, slots=True)
class EntryModelRelation:
    """A source-declared model-to-model edge retained on its subject entry.

    ``target_entry_id`` is present only when the target resolves through the
    same source-record local ID or an exact model identifier. The relation
    itself never merges entries and an unresolved target remains useful
    provenance rather than a prompt for a name-based guess.
    """

    source: str
    source_record_id: str
    subject_model_local_id: str
    predicate: str
    target_local_id: str
    target_name: str
    target_aliases: tuple[str, ...]
    target_identifiers: tuple[EntryIdentifier, ...]
    target_status: str
    target_confidence: float
    target_locator: str | None
    confidence: float
    locator: str | None
    target_entry_id: str | None


@dataclass(frozen=True, slots=True)
class EntryCitation:
    """An outgoing citation to an existing entry, with all supporting observations."""

    target_entry_id: str
    evidence: tuple[EntryResource, ...]


@dataclass(frozen=True, slots=True)
class Entry:
    """A portable, resource-complete entry assembled from declared observations."""

    id: str
    canonical_name: str
    aliases: tuple[str, ...]
    identifiers: tuple[EntryIdentifier, ...]
    tags: tuple[str, ...]
    members: tuple[EntryMember, ...]
    resources: tuple[EntryResource, ...]
    releases: tuple[EntryRelease, ...]
    model_relations: tuple[EntryModelRelation, ...]
    citations: tuple[EntryCitation, ...] = ()


@dataclass(frozen=True, slots=True)
class EntryBuildResult:
    entries: tuple[Entry, ...]
    seed_count: int
    candidate_count: int
    resource_count: int

    def manifest(self) -> dict[str, Any]:
        return {
            "format": ENTRY_FORMAT,
            "entry_count": len(self.entries),
            "seed_count": self.seed_count,
            "candidate_count": self.candidate_count,
            "resource_count": self.resource_count,
            "citation_count": sum(len(entry.citations) for entry in self.entries),
            "release_count": sum(len(entry.releases) for entry in self.entries),
            "model_relation_count": sum(
                len(entry.model_relations) for entry in self.entries
            ),
            "entry_sha256": content_hash([_entry_dict(entry) for entry in self.entries]),
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    source: str
    source_record_id: str
    canonical_url: str
    artifact_identifiers: tuple[EntryIdentifier, ...]
    local_id: str
    name: str
    aliases: tuple[str, ...]
    identifiers: tuple[EntryIdentifier, ...]
    status: str
    confidence: float
    locator: str | None
    tags: tuple[str, ...]
    resources: tuple[EntryResource, ...]
    releases: tuple[EntryRelease, ...]
    model_relations: tuple[EntryModelRelation, ...]

    @property
    def fallback_key(self) -> str:
        return f"source-record:{self.source}:{self.source_record_id}:{self.local_id}"

    @property
    def anchors(self) -> tuple[str, ...]:
        return tuple(identifier.key for identifier in self.identifiers) or (self.fallback_key,)


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parents = list(range(count))

    def find(self, value: int) -> int:
        parent = self.parents[value]
        while parent != self.parents[parent]:
            self.parents[parent] = self.parents[self.parents[parent]]
            parent = self.parents[parent]
        self.parents[value] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parents[right_root] = left_root
        else:
            self.parents[left_root] = right_root


def source_record_to_entry_seed(
    record: SourceRecord,
    *,
    source: str,
    tags: Iterable[str] = (),
) -> dict[str, Any]:
    """Convert one source record into the stable JSON seed contract.

    Source adapters can call this immediately after one record is obtained.  It
    does not inspect neighboring papers or require a full corpus download.
    """

    return {
        "source": _required_text(source, "source"),
        "source_record_id": _required_text(record.source_record_id, "source_record_id"),
        "canonical_url": _canonical_url(record.canonical_url, "canonical_url"),
        "title": _required_text(record.title, "title"),
        "kind": str(record.kind),
        "identifiers": [
            {"namespace": item.namespace, "value": item.value} for item in record.identifiers
        ],
        "tags": list(tags),
        "links": [
            {
                "url": item.url,
                "relation": item.relation,
                "locator": item.locator,
                "crawl": item.crawl,
                **(
                    {"model_local_ids": list(item.model_local_ids)}
                    if item.model_local_ids
                    else {}
                ),
            }
            for item in record.links
        ],
        "models": [
            {
                "local_id": item.local_id,
                "name": item.name,
                "aliases": list(item.aliases),
                "identifiers": [
                    {"namespace": identifier.namespace, "value": identifier.value}
                    for identifier in item.identifiers
                ],
                "status": item.status.value,
                "confidence": item.confidence,
                "locator": item.locator,
            }
            for item in record.models
        ],
        "model_relations": [
            {
                "subject_local_id": item.subject_local_id,
                "predicate": item.predicate,
                "target": {
                    "local_id": item.target.local_id,
                    "name": item.target.name,
                    "aliases": list(item.target.aliases),
                    "identifiers": [
                        {"namespace": identifier.namespace, "value": identifier.value}
                        for identifier in item.target.identifiers
                    ],
                    "status": item.target.status.value,
                    "confidence": item.target.confidence,
                    "locator": item.target.locator,
                },
                "confidence": item.confidence,
                "locator": item.locator,
            }
            for item in record.model_relations
        ],
        "releases": [
            {
                "local_id": item.local_id,
                "model_local_id": item.model_local_id,
                "version": item.version,
                "revision": item.revision,
                "identifiers": [
                    {"namespace": identifier.namespace, "value": identifier.value}
                    for identifier in item.identifiers
                ],
                "released_at": item.released_at,
                "metadata": dict(item.metadata),
                "confidence": item.confidence,
                "locator": item.locator,
            }
            for item in record.releases
        ],
    }


def plan_entry_seed(seed: Mapping[str, Any]) -> dict[str, Any]:
    """Return a no-side-effect per-seed plan for review or a queue worker.

    ``attach_resource`` actions preserve declared resource relations. Citations
    use ``attach_citation_if_present`` and require corpus membership resolution. A later
    resolver may act on ``resolve_resource`` actions one-at-a-time; reference-only
    checkpoints are retained as links but never scheduled for byte downloads.
    """

    candidates = _candidates_from_seed(seed)
    actions: list[dict[str, Any]] = []
    for candidate in candidates:
        actions.append(
            {
                "action": "upsert_entry",
                "entry_anchor_identifiers": [
                    asdict(identifier) for identifier in candidate.identifiers
                ],
                "fallback_anchor": candidate.fallback_key,
                "name": candidate.name,
                "tags": list(candidate.tags),
                "evidence": {
                    "source": candidate.source,
                    "source_record_id": candidate.source_record_id,
                    "model_local_id": candidate.local_id,
                },
            }
        )
        for resource in candidate.resources:
            payload = {
                "url": resource.url,
                "relation": resource.relation,
                "category": resource.category,
                "locator": resource.locator,
                "evidence": {
                    "source": resource.source,
                    "source_record_id": resource.source_record_id,
                    "model_local_id": resource.model_local_id,
                },
            }
            if resource.resolved_artifact is not None:
                payload["resolved_artifact"] = asdict(resource.resolved_artifact)
            if resource.relation_evidence is not None:
                payload["relation_evidence"] = asdict(resource.relation_evidence)
            if resource.relation.casefold() in _CITATION_RELATIONS:
                actions.append({"action": "attach_citation_if_present", **payload})
                continue
            actions.append({"action": "attach_resource", **payload})
            if resource.crawl and resource.relation not in _NON_FETCHABLE_RELATIONS:
                actions.append({"action": "resolve_resource", **payload})
        for release in candidate.releases:
            actions.append(
                {
                    "action": "attach_release",
                    "release": _release_dict(release),
                    "evidence": {
                        "source": release.source,
                        "source_record_id": release.source_record_id,
                        "model_local_id": release.model_local_id,
                    },
                }
            )
        for relation in candidate.model_relations:
            actions.append(
                {
                    "action": "attach_model_relation",
                    "predicate": relation.predicate,
                    "target": {
                        "local_id": relation.target_local_id,
                        "name": relation.target_name,
                        "aliases": list(relation.target_aliases),
                        "identifiers": [
                            asdict(identifier) for identifier in relation.target_identifiers
                        ],
                        "status": relation.target_status,
                        "confidence": relation.target_confidence,
                        "locator": relation.target_locator,
                    },
                    "confidence": relation.confidence,
                    "locator": relation.locator,
                    "evidence": {
                        "source": relation.source,
                        "source_record_id": relation.source_record_id,
                        "model_local_id": relation.subject_model_local_id,
                    },
                }
            )
    return {
        "format": ENTRY_FORMAT,
        "seed": {
            "source": _required_text(seed.get("source"), "source"),
            "source_record_id": _required_text(seed.get("source_record_id"), "source_record_id"),
        },
        "actions": actions,
    }


def build_entries(seeds: Iterable[Mapping[str, Any]]) -> EntryBuildResult:
    """Assemble an entry corpus from a finite source-seed stream.

    The function is intentionally offline. It is safe to use in tests, an
    incremental queue worker, or a later full rebuild. Cross-seed unions require
    matching exact identifiers, an explicitly declared mirror target's exact
    identifier, or an unambiguous exact checkpoint URL. Tags and names only
    affect presentation.
    """

    seed_count = 0
    candidates: list[_Candidate] = []
    evidence_only_seeds: list[Mapping[str, Any]] = []
    for seed in seeds:
        seed_count += 1
        seed_candidates = _candidates_from_seed(seed)
        if seed_candidates:
            candidates.extend(seed_candidates)
        else:
            evidence_only_seeds.append(seed)
    candidates = _attach_exact_record_resources(candidates, evidence_only_seeds)
    union = _UnionFind(len(candidates))
    identifier_owner: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        # Fallback anchors are deliberately record-scoped and therefore cannot
        # merge two independently observed names.
        for anchor in (item.key for item in candidate.identifiers):
            owner = identifier_owner.setdefault(anchor, index)
            union.union(index, owner)

    # Some catalogs declare an exact mirror ID in another model namespace.
    # Treat that explicit cross-source relation as identity evidence so an
    # OpenCSG card that names its Hugging Face or ModelScope mirror joins the
    # independently sourced card. Other relation predicates remain descriptive.
    for index, candidate in enumerate(candidates):
        for relation in candidate.model_relations:
            if relation.predicate.casefold() != "mirrors":
                continue
            for identifier in relation.target_identifiers:
                owner = identifier_owner.get(identifier.key)
                if owner is not None and candidates[owner].source != candidate.source:
                    union.union(index, owner)

    # A source-declared direct checkpoint URL can bridge otherwise unrelated
    # catalog identities when independent sources point to the same artifact.
    # Generic weights links may name shared tokenizers or backbone files, so a
    # URL is eligible only if every owner calls it a checkpoint. Require one
    # model candidate per source record for that URL: catalog-wide links are
    # ambiguous.
    checkpoint_owners: dict[str, list[int]] = defaultdict(list)
    explicit_checkpoint_owners: dict[str, set[int]] = defaultdict(set)
    for index, candidate in enumerate(candidates):
        urls = set()
        for resource in candidate.resources:
            if (resource.source, resource.source_record_id) != (
                candidate.source,
                candidate.source_record_id,
            ):
                # Imported evidence links enrich a candidate but do not assert
                # its identity. Only the candidate's own source declaration can
                # act as a checkpoint identity bridge.
                continue
            relation = resource.relation.casefold()
            if relation in _CHECKPOINT_IDENTITY_RELATIONS:
                urls.add(resource.url)
            if relation == "checkpoint":
                explicit_checkpoint_owners[resource.url].add(index)
        for url in urls:
            checkpoint_owners[url].append(index)
    for url, owners in checkpoint_owners.items():
        if len(owners) < 2 or len(explicit_checkpoint_owners[url]) != len(owners):
            continue
        records_by_url: dict[tuple[str, str], int] = {}
        for index in owners:
            candidate = candidates[index]
            record_key = (candidate.source, candidate.source_record_id)
            records_by_url[record_key] = records_by_url.get(record_key, 0) + 1
        if any(count > 1 for count in records_by_url.values()):
            continue
        if len({candidates[index].source for index in owners}) < 2:
            continue
        first = owners[0]
        for owner in owners[1:]:
            union.union(first, owner)

    groups: dict[int, list[_Candidate]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[union.find(index)].append(candidate)
    entries = tuple(_entry_from_candidates(group) for _, group in sorted(groups.items()))
    entry_by_member = {
        (member.source, member.source_record_id, member.local_id): entry.id
        for entry in entries
        for member in entry.members
    }
    entry_ids_by_identifier: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        for identifier in entry.identifiers:
            entry_ids_by_identifier[identifier.key].add(entry.id)
    entries = tuple(
        _resolve_entry_model_relations(
            entry,
            entry_by_member,
            entry_ids_by_identifier,
        )
        for entry in entries
    )
    entries = tuple(
        sorted(
            _resolve_entry_citations(entries),
            key=lambda item: (normalize_name(item.canonical_name), item.id),
        )
    )
    return EntryBuildResult(
        entries=entries,
        seed_count=seed_count,
        candidate_count=len(candidates),
        resource_count=sum(len(entry.resources) for entry in entries),
    )


def _attach_exact_record_resources(
    candidates: list[_Candidate], evidence_seeds: Sequence[Mapping[str, Any]]
) -> list[_Candidate]:
    """Attach resource evidence to candidates with an exact artifact identifier.

    Some source records enrich a paper or catalog record with checkpoint links
    but declare no model themselves. Attach their links only when an exact
    artifact identifier resolves to one candidate. A paper that describes
    multiple models does not identify which model owns a checkpoint link.
    """

    candidates_by_artifact_identifier: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        for identifier in candidate.artifact_identifiers:
            candidates_by_artifact_identifier[identifier.key].append(index)

    resources_by_candidate: dict[int, list[EntryResource]] = defaultdict(list)
    for seed in evidence_seeds:
        source = _required_text(seed.get("source"), "source")
        source_record_id = _required_text(seed.get("source_record_id"), "source_record_id")
        evidence_identifiers = _identifiers(seed.get("identifiers"), "identifiers")
        target_indices = {
            index
            for identifier in evidence_identifiers
            for index in candidates_by_artifact_identifier.get(identifier.key, ())
            if (candidates[index].source, candidates[index].source_record_id)
            != (source, source_record_id)
        }
        if len(target_indices) != 1:
            continue
        raw_links = seed.get("links", ())
        if not isinstance(raw_links, Sequence) or isinstance(
            raw_links, (str, bytes, bytearray)
        ):
            raise ValueError("links must be a list")
        record_kind = _optional_text(seed.get("kind")) or "other"
        for link_index, raw_link in enumerate(raw_links):
            if not isinstance(raw_link, Mapping):
                raise ValueError("links must contain objects")
            relation = _optional_text(raw_link.get("relation")) or "references"
            if relation.casefold() not in _WEIGHT_RELATIONS:
                continue
            url = _canonical_url(raw_link.get("url"), f"link {link_index} url")
            crawl = _crawl(raw_link.get("crawl", True), f"link {link_index} crawl")
            resolved_artifact, relation_evidence = _link_relation_metadata(
                raw_link, link_index, url
            )
            scoped_model_ids = _link_model_local_ids(raw_link, link_index)
            for index in target_indices:
                candidate = candidates[index]
                if scoped_model_ids and candidate.local_id not in scoped_model_ids:
                    continue
                resources_by_candidate[index].append(
                    EntryResource(
                        url=url,
                        relation=relation,
                        locator=_optional_text(raw_link.get("locator")),
                        crawl=crawl,
                        category=_category(relation, record_kind),
                        source=source,
                        source_record_id=source_record_id,
                        model_local_id=candidate.local_id,
                        resolved_artifact=resolved_artifact,
                        relation_evidence=relation_evidence,
                    )
                )
    if not resources_by_candidate:
        return candidates
    return [
        replace(candidate, resources=(*candidate.resources, *resources_by_candidate[index]))
        if index in resources_by_candidate
        else candidate
        for index, candidate in enumerate(candidates)
    ]


def _citation_url_keys(url: str) -> set[tuple[str, str]]:
    keys = {("url", url)}
    identifier = identifier_from_url(url)
    if identifier is not None:
        normalized = _identifiers([asdict(identifier)], "citation identifier")[0]
        keys.add((normalized.namespace, normalized.value))
    # OpenAlex work IDs identify papers even when their landing URL is a DOI.
    if match := re.fullmatch(r"https?://openalex\.org/(W\d+)", url):
        keys.add(("openalex", match[1]))
    return keys


def _resolve_entry_citations(entries: Sequence[Entry]) -> tuple[Entry, ...]:
    """Project declared citations onto the closed set of assembled entries.

    Citation URLs never become identity evidence. Ambiguous paper ownership is
    ignored: a paper mentioning several techniques cannot identify one of them.
    """

    owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    record_owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entry in entries:
        keys = {(item.namespace, item.value) for item in entry.identifiers}
        for member in entry.members:
            record_owners[member.source, member.source_record_id].add(entry.id)
            keys.update(_citation_url_keys(member.artifact_url))
            keys.update((item.namespace, item.value) for item in member.artifact_identifiers)
        for resource in entry.resources:
            if resource.relation.casefold() not in _PAPER_RELATIONS | {"openalex_record"}:
                continue
            if resource.relation_evidence is not None and (
                resource.relation_evidence.direction == "incoming"
            ):
                continue
            keys.update(_citation_url_keys(resource.url))
        for key in keys:
            owners[key].add(entry.id)

    edges: dict[str, dict[str, set[EntryResource]]] = defaultdict(lambda: defaultdict(set))
    for entry in entries:
        for resource in entry.resources:
            predicate = resource.relation.casefold()
            if predicate not in _CITATION_RELATIONS:
                continue
            matched = set().union(
                *(owners.get(key, set()) for key in _citation_url_keys(resource.url))
            )
            if resource.resolved_artifact is not None:
                artifact = resource.resolved_artifact
                matched.update(record_owners.get((artifact.source, artifact.source_record_id), ()))
            if len(matched) != 1:
                continue
            source_id, target_id = entry.id, next(iter(matched))
            reverse = predicate != "cites"
            if resource.relation_evidence is not None:
                direction = resource.relation_evidence.direction
                if direction == "symmetric":
                    continue
                reverse ^= direction == "incoming"
            if reverse:
                source_id, target_id = target_id, source_id
            if source_id != target_id:
                edges[source_id][target_id].add(replace(resource, crawl=False))

    return tuple(
        replace(
            entry,
            resources=tuple(
                resource
                for resource in entry.resources
                if resource.relation.casefold() not in _CITATION_RELATIONS
            ),
            citations=tuple(
                EntryCitation(
                    target_entry_id=target_id,
                    evidence=tuple(
                        sorted(evidence, key=lambda item: json.dumps(asdict(item), sort_keys=True))
                    ),
                )
                for target_id, evidence in sorted(edges.get(entry.id, {}).items())
            ),
        )
        for entry in entries
    )


def _resolve_entry_model_relations(
    entry: Entry,
    entry_by_member: Mapping[tuple[str, str, str], str],
    entry_ids_by_identifier: Mapping[str, set[str]],
) -> Entry:
    """Resolve only unambiguous exact targets after all entries exist."""

    resolved = []
    for relation in entry.model_relations:
        target_entry_id = entry_by_member.get(
            (relation.source, relation.source_record_id, relation.target_local_id)
        )
        if target_entry_id is None and relation.target_identifiers:
            matched = {
                entry_id
                for identifier in relation.target_identifiers
                for entry_id in entry_ids_by_identifier.get(identifier.key, set())
            }
            if len(matched) == 1:
                target_entry_id = next(iter(matched))
        resolved.append(replace(relation, target_entry_id=target_entry_id))
    return replace(
        entry,
        model_relations=tuple(
            sorted(
                set(resolved),
                key=lambda relation: (
                    relation.predicate,
                    relation.target_entry_id or "",
                    relation.target_name.casefold(),
                    relation.target_name,
                    relation.source,
                    relation.source_record_id,
                    relation.subject_model_local_id,
                    relation.locator or "",
                ),
            )
        ),
    )


def read_entry_seeds(path: Path) -> tuple[dict[str, Any], ...]:
    """Read either a JSON list/object or newline-delimited entry seed objects."""

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"entry seed input is empty: {path}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, Mapping):
        parsed = parsed.get("seeds", [parsed])
    if not isinstance(parsed, list):
        raise ValueError("entry seed input must be a JSON object, array, or JSONL stream")
    result = []
    for item in parsed:
        if not isinstance(item, Mapping):
            raise ValueError("entry seed input contains a non-object row")
        result.append(dict(item))
    return tuple(result)


def write_entry_bundle(result: EntryBuildResult, output: Path) -> dict[str, Any]:
    """Write one future-build result atomically, refusing to replace an export."""

    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"entry output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        rows = "".join(
            json.dumps(_entry_dict(entry), ensure_ascii=False, sort_keys=True) + "\n"
            for entry in result.entries
        )
        _atomic_text(staging / "entries.jsonl", rows)
        manifest = {**result.manifest(), "files": {"entries.jsonl": content_hash(rows)}}
        _atomic_text(
            staging / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise
    return {**result.manifest(), "output": str(output)}


def _candidates_from_seed(seed: Mapping[str, Any]) -> tuple[_Candidate, ...]:
    source = _required_text(seed.get("source"), "source")
    source_record_id = _required_text(seed.get("source_record_id"), "source_record_id")
    canonical_url = _canonical_url(seed.get("canonical_url"), "canonical_url")
    _required_text(seed.get("title"), "title")
    record_kind = _optional_text(seed.get("kind")) or "other"
    record_identifiers = _identifiers(seed.get("identifiers"), "identifiers")
    record_tags = _tags(seed.get("tags"), "tags")
    raw_links = seed.get("links", ())
    if not isinstance(raw_links, Sequence) or isinstance(raw_links, (str, bytes, bytearray)):
        raise ValueError("links must be a list")
    models = seed.get("models", ())
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes, bytearray)):
        raise ValueError("models must be a list")
    relations_by_subject = _model_relations(
        seed.get("model_relations", ()),
        source,
        source_record_id,
    )
    releases = _releases(seed.get("releases", ()), source, source_record_id)
    # A corpus-scale paper catalog can contain millions of papers unrelated to
    # a model or technique assertion.  Empty ``models`` therefore means “retain
    # this as source evidence, but do not create an entry.”  A curator that
    # intends a paper-defined technique can add an explicit model/technique row
    # with an artifact-local fallback identifier.
    model_rows: Sequence[Any] = models
    result = []
    for raw_model in model_rows:
        if not isinstance(raw_model, Mapping):
            raise ValueError("models must contain objects")
        local_id = _required_text(raw_model.get("local_id"), "model local_id")
        name = _required_text(raw_model.get("name"), "model name")
        aliases = _text_sequence(raw_model.get("aliases"), "model aliases")
        identifiers = _identifiers(raw_model.get("identifiers"), "model identifiers")
        status = _optional_text(raw_model.get("status")) or "documented"
        confidence = _confidence(raw_model.get("confidence", 1.0), "model confidence")
        locator = _optional_text(raw_model.get("locator"))
        tags = _unique((*record_tags, *_tags(raw_model.get("tags"), "model tags")))
        resources = [
            EntryResource(
                url=canonical_url,
                relation="seed",
                locator=None,
                crawl=False,
                category=_category("seed", record_kind),
                source=source,
                source_record_id=source_record_id,
                model_local_id=local_id,
                resolved_artifact=None,
                relation_evidence=None,
            )
        ]
        for link_index, raw_link in enumerate(raw_links):
            if not isinstance(raw_link, Mapping):
                raise ValueError("links must contain objects")
            scoped_model_ids = _link_model_local_ids(raw_link, link_index)
            if scoped_model_ids and local_id not in scoped_model_ids:
                continue
            url = _canonical_url(raw_link.get("url"), f"link {link_index} url")
            relation = _optional_text(raw_link.get("relation")) or "references"
            locator = _optional_text(raw_link.get("locator"))
            crawl = _crawl(raw_link.get("crawl", True), f"link {link_index} crawl")
            resolved_artifact, relation_evidence = _link_relation_metadata(
                raw_link,
                link_index,
                url,
            )
            resources.append(
                EntryResource(
                    url=url,
                    relation=relation,
                    locator=locator,
                    crawl=crawl,
                    category=_category(relation, record_kind),
                    source=source,
                    source_record_id=source_record_id,
                    model_local_id=local_id,
                    resolved_artifact=resolved_artifact,
                    relation_evidence=relation_evidence,
                )
            )
        result.append(
            _Candidate(
                source=source,
                source_record_id=source_record_id,
                canonical_url=canonical_url,
                artifact_identifiers=record_identifiers,
                local_id=local_id,
                name=name,
                aliases=aliases,
                identifiers=identifiers,
                status=status,
                confidence=confidence,
                locator=locator,
                tags=tags,
                resources=tuple(resources),
                releases=tuple(
                    release for release in releases if release.model_local_id == local_id
                ),
                model_relations=relations_by_subject.get(local_id, ()),
            )
        )
    model_local_ids = {candidate.local_id for candidate in result}
    for link_index, raw_link in enumerate(raw_links):
        if not isinstance(raw_link, Mapping):
            raise ValueError("links must contain objects")
        missing_models = set(_link_model_local_ids(raw_link, link_index)) - model_local_ids
        if missing_models:
            raise ValueError(
                f"link {link_index} scopes resources to absent model local IDs: "
                + ", ".join(sorted(missing_models))
            )
    dangling_releases = sorted(
        {release.model_local_id for release in releases} - model_local_ids
    )
    if dangling_releases:
        raise ValueError(
            "releases refer to model local IDs absent from the seed: "
            + ", ".join(dangling_releases)
        )
    return tuple(result)


def _model_relations(
    value: Any,
    source: str,
    source_record_id: str,
) -> dict[str, tuple[EntryModelRelation, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("model_relations must be a list")
    grouped: dict[str, list[EntryModelRelation]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("model_relations must contain objects")
        subject_local_id = _required_text(
            item.get("subject_local_id"),
            f"model relation {index} subject_local_id",
        )
        predicate = _required_text(item.get("predicate"), f"model relation {index} predicate")
        target = item.get("target")
        if not isinstance(target, Mapping):
            raise ValueError(f"model relation {index} target must be an object")
        relation = EntryModelRelation(
            source=source,
            source_record_id=source_record_id,
            subject_model_local_id=subject_local_id,
            predicate=predicate,
            target_local_id=_required_text(
                target.get("local_id"),
                f"model relation {index} target local_id",
            ),
            target_name=_required_text(
                target.get("name"),
                f"model relation {index} target name",
            ),
            target_aliases=_text_sequence(
                target.get("aliases"),
                f"model relation {index} target aliases",
            ),
            target_identifiers=_identifiers(
                target.get("identifiers"),
                f"model relation {index} target identifiers",
            ),
            target_status=_optional_text(target.get("status")) or "documented",
            target_confidence=_confidence(
                target.get("confidence", 1.0),
                f"model relation {index} target confidence",
            ),
            target_locator=_optional_text(target.get("locator")),
            confidence=_confidence(
                item.get("confidence", 1.0),
                f"model relation {index} confidence",
            ),
            locator=_optional_text(item.get("locator")),
            target_entry_id=None,
        )
        grouped.setdefault(subject_local_id, []).append(relation)
    return {
        local_id: tuple(
            sorted(
                set(relations),
                key=lambda relation: (
                    relation.predicate,
                    relation.target_name.casefold(),
                    relation.target_name,
                    tuple(identifier.key for identifier in relation.target_identifiers),
                    relation.locator or "",
                ),
            )
        )
        for local_id, relations in grouped.items()
    }


def _entry_from_candidates(candidates: Sequence[_Candidate]) -> Entry:
    if not candidates:
        raise ValueError("cannot construct an entry from no candidates")
    ordered = sorted(
        candidates,
        key=lambda item: (
            0 if item.identifiers else 1,
            normalize_name(item.name),
            item.name,
            item.source,
            item.source_record_id,
            item.local_id,
        ),
    )
    canonical = ordered[0]
    identifiers = tuple(
        sorted(
            {item for candidate in candidates for item in candidate.identifiers},
            key=lambda item: item.key,
        )
    )
    fallback = tuple(sorted(candidate.fallback_key for candidate in candidates))
    identity_material = [item.key for item in identifiers] or list(fallback)
    entry_id = f"entry:{content_hash({'anchors': identity_material})}"
    aliases = _unique(
        alias
        for candidate in candidates
        for alias in (candidate.name, *candidate.aliases)
        if normalize_name(alias) != normalize_name(canonical.name)
    )
    tags = _unique(tag for candidate in candidates for tag in candidate.tags)
    members = tuple(
        sorted(
            {
                EntryMember(
                    source=item.source,
                    source_record_id=item.source_record_id,
                    artifact_url=item.canonical_url,
                    artifact_identifiers=item.artifact_identifiers,
                    local_id=item.local_id,
                    name=item.name,
                    aliases=item.aliases,
                    status=item.status,
                    confidence=item.confidence,
                    locator=item.locator,
                )
                for item in candidates
            },
            key=lambda item: (item.source, item.source_record_id, item.local_id),
        )
    )
    resources = tuple(
        sorted(
            {
                resource
                for candidate in candidates
                for resource in candidate.resources
            },
            key=lambda item: (
                item.category,
                item.url,
                item.relation,
                item.source,
                item.source_record_id,
                item.model_local_id,
                item.locator or "",
                item.resolved_artifact.id if item.resolved_artifact is not None else "",
                item.relation_evidence.id if item.relation_evidence is not None else "",
            ),
        )
    )
    releases = tuple(
        sorted(
            {release for candidate in candidates for release in candidate.releases},
            key=lambda item: (
                item.model_local_id,
                item.local_id,
                item.source,
                item.source_record_id,
            ),
        )
    )
    model_relations = tuple(
        sorted(
            {
                relation
                for candidate in candidates
                for relation in candidate.model_relations
            },
            key=lambda relation: (
                relation.predicate,
                relation.target_name.casefold(),
                relation.target_name,
                relation.source,
                relation.source_record_id,
                relation.subject_model_local_id,
                relation.locator or "",
            ),
        )
    )
    return Entry(
        id=entry_id,
        canonical_name=canonical.name,
        aliases=aliases,
        identifiers=identifiers,
        tags=tags,
        members=members,
        resources=resources,
        releases=releases,
        model_relations=model_relations,
    )


def _entry_dict(entry: Entry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "canonical_name": entry.canonical_name,
        "aliases": list(entry.aliases),
        "identifiers": [asdict(item) for item in entry.identifiers],
        "tags": list(entry.tags),
        "members": [asdict(item) for item in entry.members],
        "resources": [asdict(item) for item in entry.resources],
        "releases": [_release_dict(item) for item in entry.releases],
        "citations": [asdict(item) for item in entry.citations],
        "model_relations": [
            {
                "source": relation.source,
                "source_record_id": relation.source_record_id,
                "subject_model_local_id": relation.subject_model_local_id,
                "predicate": relation.predicate,
                "target": {
                    "local_id": relation.target_local_id,
                    "name": relation.target_name,
                    "aliases": list(relation.target_aliases),
                    "identifiers": [
                        asdict(identifier) for identifier in relation.target_identifiers
                    ],
                    "status": relation.target_status,
                    "confidence": relation.target_confidence,
                    "locator": relation.target_locator,
                    "entry_id": relation.target_entry_id,
                },
                "confidence": relation.confidence,
                "locator": relation.locator,
            }
            for relation in entry.model_relations
        ],
    }


def _category(relation: str, kind: str) -> str:
    relation = relation.casefold()
    if relation in _WEIGHT_RELATIONS:
        return "weights"
    if relation in _CODE_RELATIONS:
        return "code"
    if relation in _PAPER_RELATIONS:
        return "paper"
    if relation in _CARD_RELATIONS:
        return "model"
    if relation == "seed" and kind == "paper":
        return "paper"
    return "resource"


def _identifiers(value: Any, field: str) -> tuple[EntryIdentifier, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a list")
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field} must contain objects")
        namespace = _required_text(item.get("namespace"), f"{field} namespace")
        identifier_value = _required_text(item.get("value"), f"{field} value")
        # arXiv's version suffix identifies a revision of the same paper. Keep
        # revision identity in the distinct ``arxiv:version`` namespace, while
        # normalizing common versioned forms of the base paper identifier.
        if namespace == "arxiv":
            identifier_value = re.sub(
                r"v\d+$", "", identifier_value, flags=re.IGNORECASE
            ).casefold()
        elif namespace == "doi":
            try:
                doi_url_identifier = identifier_from_url(identifier_value)
            except ValueError:
                doi_url_identifier = None
            if doi_url_identifier is not None and doi_url_identifier.namespace == "doi":
                identifier_value = doi_url_identifier.value
            else:
                identifier_value = identifier_value.strip()
                if identifier_value.casefold().startswith("doi:"):
                    identifier_value = identifier_value[4:]
                identifier_value = identifier_value.casefold()
        result.append(EntryIdentifier(namespace=namespace, value=identifier_value))
    return tuple(sorted(set(result), key=lambda item: item.key))


def _link_relation_metadata(
    link: Mapping[str, Any],
    index: int,
    url: str,
) -> tuple[EntryResolvedArtifact | None, EntryRelationEvidence | None]:
    raw_artifact = link.get("resolved_artifact")
    raw_evidence = link.get("relation_evidence")
    if raw_artifact is None and raw_evidence is None:
        return None, None
    if not isinstance(raw_artifact, Mapping) or not isinstance(raw_evidence, Mapping):
        raise ValueError(
            f"link {index} must provide both resolved_artifact and relation_evidence"
        )
    resolved_url = _canonical_url(
        raw_artifact.get("canonical_url"),
        f"link {index} resolved artifact canonical_url",
    )
    if resolved_url != url:
        raise ValueError(f"link {index} URL disagrees with resolved artifact canonical_url")
    direction = _required_text(raw_evidence.get("direction"), f"link {index} direction")
    if direction not in {"incoming", "outgoing", "symmetric"}:
        raise ValueError(f"link {index} direction is invalid")
    return (
        EntryResolvedArtifact(
            id=_required_text(raw_artifact.get("id"), f"link {index} resolved artifact id"),
            revision_id=_required_text(
                raw_artifact.get("revision_id"),
                f"link {index} resolved artifact revision_id",
            ),
            kind=_required_text(raw_artifact.get("kind"), f"link {index} resolved artifact kind"),
            source=_required_text(
                raw_artifact.get("source"),
                f"link {index} resolved artifact source",
            ),
            source_record_id=_required_text(
                raw_artifact.get("source_record_id"),
                f"link {index} resolved artifact source_record_id",
            ),
        ),
        EntryRelationEvidence(
            id=_required_text(raw_evidence.get("id"), f"link {index} relation evidence id"),
            direction=direction,
            evidence_type=_required_text(
                raw_evidence.get("evidence_type"),
                f"link {index} relation evidence type",
            ),
            confidence=_confidence(
                raw_evidence.get("confidence"),
                f"link {index} relation confidence",
            ),
            match_namespace=_optional_text(raw_evidence.get("match_namespace")),
            match_value=_optional_text(raw_evidence.get("match_value")),
        ),
    )


def _releases(
    value: Any,
    source: str,
    source_record_id: str,
) -> tuple[EntryRelease, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("releases must be a list")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("releases must contain objects")
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"release {index} metadata must be an object")
        result.append(
            EntryRelease(
                source=source,
                source_record_id=source_record_id,
                model_local_id=_required_text(
                    item.get("model_local_id"), f"release {index} model_local_id"
                ),
                local_id=_required_text(item.get("local_id"), f"release {index} local_id"),
                version=_optional_text(item.get("version")),
                revision=_optional_text(item.get("revision")),
                identifiers=_identifiers(item.get("identifiers"), f"release {index} identifiers"),
                released_at=_optional_text(item.get("released_at")),
                metadata_json=_metadata_json(metadata, f"release {index} metadata"),
                confidence=_confidence(item.get("confidence", 1.0), f"release {index} confidence"),
                locator=_optional_text(item.get("locator")),
            )
        )
    return tuple(result)


def _tags(value: Any, field: str) -> tuple[str, ...]:
    return _text_sequence(value, field)


def _text_sequence(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a list")
    return _unique(_required_text(item, field) for item in value)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values}, key=lambda item: (item.casefold(), item)))


def _canonical_url(value: Any, field: str) -> str:
    raw = _required_text(value, field)
    canonical = canonicalize_url(raw)
    if not canonical.startswith(("http://", "https://")):
        raise ValueError(f"{field} must be an absolute HTTP(S) URL")
    return canonical


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    text = unicodedata.normalize("NFKC", value).strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    if any(ord(character) < 32 for character in text):
        raise ValueError(f"{field} must not contain control characters")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return _required_text(value, "text")


def _crawl(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def _link_model_local_ids(link: Mapping[str, Any], index: int) -> tuple[str, ...]:
    """Return an optional exact subject scope for a source-declared link."""

    return _text_sequence(
        link.get("model_local_ids"), f"link {index} model_local_ids"
    )


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


def _metadata_json(value: Mapping[str, Any], field: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be JSON-compatible") from error


def _release_dict(release: EntryRelease) -> dict[str, Any]:
    return {
        "source": release.source,
        "source_record_id": release.source_record_id,
        "model_local_id": release.model_local_id,
        "local_id": release.local_id,
        "version": release.version,
        "revision": release.revision,
        "identifiers": [asdict(item) for item in release.identifiers],
        "released_at": release.released_at,
        "metadata": json.loads(release.metadata_json),
        "confidence": release.confidence,
        "locator": release.locator,
    }


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


__all__ = [
    "ENTRY_FORMAT",
    "Entry",
    "EntryBuildResult",
    "EntryCitation",
    "EntryIdentifier",
    "EntryMember",
    "EntryRelationEvidence",
    "EntryRelease",
    "EntryModelRelation",
    "EntryResolvedArtifact",
    "EntryResource",
    "build_entries",
    "plan_entry_seed",
    "read_entry_seeds",
    "source_record_to_entry_seed",
    "write_entry_bundle",
]
