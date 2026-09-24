"""Stream current registry observations into the offline entry-seed contract.

This is the bridge between ordinary source ingestion and a later unified-entry
build.  It reads only the current immutable Parquet commit, writes no registry
state, and emits only artifacts whose current source record declares at least one
model/technique hint.  It is intentionally not invoked by normal syncs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from modelome.artifact_relations import ArtifactRelationMaterializer
from modelome.entries import ENTRY_FORMAT
from modelome.normalize import canonicalize_url


@dataclass(frozen=True, slots=True)
class EntrySeedExportReceipt:
    commit: str
    output: str
    manifest: str
    source_count: int
    seed_count: int
    skipped_without_assertion: int
    relation_resource_count: int
    direct_resource_count: int
    targeted_resource_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _CurrentArtifact:
    id: str
    revision_id: str
    source: str
    source_record_id: str
    kind: str
    canonical_url: str
    canonical_url_normalized: str


@dataclass(frozen=True, slots=True)
class _ModelDeclaration:
    local_id: str
    name: str
    aliases: tuple[str, ...]
    identifiers: tuple[tuple[str, str], ...]
    status: str
    confidence: float
    locator: str | None


def export_current_entry_seeds(
    store: Path,
    output: Path,
    *,
    source_tags: Mapping[str, Iterable[str]] | None = None,
    sources: Iterable[str] | None = None,
    relation_root: Path | None = None,
    link_current_resources: bool = False,
) -> EntrySeedExportReceipt:
    """Stream current model/technique declarations to a new JSONL seed file.

    ``source_tags`` is deliberately unconstrained metadata. It can attach a
    broad field or department tag to an entire source without changing identity
    behavior. ``sources`` limits a future build to selected ingestion planes.
    ``relation_root`` is an optional, already-sealed current artifact-relation
    projection. It is read only; this exporter never builds it. Direct URLs found
    in the entry subject's own retained source text are always included. In
    addition, ``link_current_resources`` performs a targeted read-only join
    against other current records' URL/identifier evidence, avoiding a global
    relation projection.
    """

    root = store.expanduser().resolve()
    head_path = root / "HEAD.json"
    if not head_path.is_file():
        raise ValueError(f"registry store is not initialized: {root}")
    head = _read_json(head_path, "registry HEAD")
    commit = _required_text(head.get("commit"), "registry HEAD commit")
    tables = root / "commits" / commit / "tables"
    artifact_path = tables / "artifacts.parquet"
    revision_path = tables / "artifact_revisions.parquet"
    if not artifact_path.is_file() or not revision_path.is_file():
        raise ValueError(f"registry commit is missing entry-seed tables: {commit}")

    output = output.expanduser().resolve()
    manifest_path = output.with_name(f"{output.name}.manifest.json")
    if output.exists() or manifest_path.exists():
        raise ValueError(f"entry seed output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    wanted_sources = (
        None
        if sources is None
        else {str(item).strip() for item in sources if str(item).strip()}
    )
    tags_by_source = {
        str(source).strip(): tuple(str(tag) for tag in tags)
        for source, tags in (source_tags or {}).items()
        if str(source).strip()
    }
    if relation_root is not None and link_current_resources:
        raise ValueError(
            "choose either relation_root or link_current_resources, not both"
        )
    # Resource evidence may originate in a different source from the selected
    # entry subject (for example, a Papers with Code link record pointing at a
    # Hugging Face card). Keep the full current snapshot for exact joins, then
    # apply ``--source`` only to the subjects emitted as seeds.
    current = _current_artifacts(artifact_path, None)
    current_by_id = {artifact.id: artifact for artifact in current.values()}
    all_declarations = _current_model_declarations(tables, current)
    declarations = {
        artifact_id: models
        for artifact_id, models in all_declarations.items()
        if wanted_sources is None
        or current_by_id[artifact_id].source in wanted_sources
    }
    eligible_ids = set(declarations)
    selected_artifact_ids = {
        artifact.id
        for artifact in current.values()
        if wanted_sources is None or artifact.source in wanted_sources
    }
    skipped_without_assertion = len(selected_artifact_ids - eligible_ids)
    relation_links = _relation_links(
        root,
        relation_root,
        eligible_ids,
    )
    direct_links, targeted_links = _current_resource_links(
        tables,
        current,
        eligible_ids,
        include_cross_record=link_current_resources,
    )
    cross_record_links = _merge_link_maps(relation_links, direct_links, targeted_links)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}-", suffix=".jsonl", dir=output.parent
    )
    digest = hashlib.sha256()
    seed_count = 0
    source_names: set[str] = set()
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as stream:
            revision_file = pq.ParquetFile(revision_path)
            for batch in revision_file.iter_batches(
                batch_size=8_192,
                columns=("id", "record_json"),
            ):
                for revision in batch.to_pylist():
                    artifact = current.get(str(revision["id"]))
                    if artifact is None:
                        continue
                    payload = _read_record(str(revision["record_json"]))
                    models = declarations.get(artifact.id)
                    if not models:
                        continue
                    links = _array(payload.get("links", []), "stored artifact links")
                    seed = {
                        "source": artifact.source,
                        "source_record_id": artifact.source_record_id,
                        "canonical_url": payload.get("canonical_url"),
                        "title": payload.get("title"),
                        "kind": payload.get("kind"),
                        "identifiers": payload.get("identifiers", []),
                        "links": _deduplicate_links(
                            [*links, *cross_record_links.get(artifact.id, ())]
                        ),
                        "models": [_model_seed(model) for model in models],
                        "model_relations": _array(
                            payload.get("model_relations", []),
                            "stored model relations",
                        ),
                        "releases": payload.get("releases", []),
                        "tags": list(tags_by_source.get(artifact.source, ())),
                    }
                    line = json.dumps(seed, ensure_ascii=False, sort_keys=True) + "\n"
                    stream.write(line)
                    digest.update(line.encode("utf-8"))
                    seed_count += 1
                    source_names.add(artifact.source)
        os.replace(temporary_name, output)
        manifest = {
            "format": f"{ENTRY_FORMAT}-seed-export-v1",
            "store_commit": commit,
            "entry_seed_file": output.name,
            "sha256": digest.hexdigest(),
            "seed_count": seed_count,
            "skipped_without_assertion": skipped_without_assertion,
            "relation_resource_count": sum(len(rows) for rows in relation_links.values()),
            "direct_resource_count": sum(len(rows) for rows in direct_links.values()),
            "targeted_resource_count": sum(len(rows) for rows in targeted_links.values()),
            "source_count": len(source_names),
            "sources": sorted(source_names),
        }
        _write_json(manifest_path, manifest)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return EntrySeedExportReceipt(
        commit=commit,
        output=str(output),
        manifest=str(manifest_path),
        source_count=len(source_names),
        seed_count=seed_count,
        skipped_without_assertion=skipped_without_assertion,
        relation_resource_count=sum(len(rows) for rows in relation_links.values()),
        direct_resource_count=sum(len(rows) for rows in direct_links.values()),
        targeted_resource_count=sum(len(rows) for rows in targeted_links.values()),
        sha256=digest.hexdigest(),
    )


def source_tags_from_configs(configs: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    """Extract optional, free-form ``entry_tags`` from source catalog blocks."""

    result = {}
    for config in configs:
        source = _required_text(config.get("name"), "source name")
        raw_tags = config.get("entry_tags", ())
        if not isinstance(raw_tags, (list, tuple)):
            raise ValueError(f"{source}: entry_tags must be a list")
        result[source] = tuple(
            _required_text(item, f"{source} entry tag") for item in raw_tags
        )
    return result


def assess_entry_readiness(
    store: Path,
    source_configs: Iterable[Mapping[str, Any]],
    *,
    relation_root: Path | None = None,
) -> dict[str, Any]:
    """Report whether the *current evidence snapshot* can support a full entry run.

    This does not claim global model coverage. It verifies the narrower, actionable
    prerequisites for a future run: which configured source checkpoints exist and
    are complete, which current artifacts have admitted model claims, and whether a
    sealed relation graph is available as an optional alternative to the targeted
    current-record resource join.
    """

    root = store.expanduser().resolve()
    head = _read_json(root / "HEAD.json", "registry HEAD")
    commit = _required_text(head.get("commit"), "registry HEAD commit")
    tables = root / "commits" / commit / "tables"
    current = _current_artifacts(tables / "artifacts.parquet", None)
    declarations = _current_model_declarations(tables, current)
    source_configs_by_name: dict[str, bool] = {}
    for config in source_configs:
        name = _required_text(config.get("name"), "source name")
        enabled = config.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{name}: enabled must be true or false")
        source_configs_by_name[name] = enabled
    checkpoints = {
        _required_text(row.get("source"), "source checkpoint source"): row
        for row in _parquet_rows(
            tables / "source_checkpoints.parquet",
            ("source", "complete", "upstream_count"),
        )
    }
    artifacts_by_source: dict[str, set[str]] = {}
    for artifact in current.values():
        artifacts_by_source.setdefault(artifact.source, set()).add(artifact.id)
    claim_counts_by_source: dict[str, int] = {}
    candidate_counts_by_source: dict[str, int] = {}
    artifact_sources = {artifact.id: artifact.source for artifact in current.values()}
    for artifact_id, models in declarations.items():
        source = artifact_sources[artifact_id]
        candidate_counts_by_source[source] = candidate_counts_by_source.get(source, 0) + 1
        claim_counts_by_source[source] = claim_counts_by_source.get(source, 0) + len(models)

    all_sources = sorted(set(source_configs_by_name) | set(artifacts_by_source) | set(checkpoints))
    source_rows = []
    for source in all_sources:
        checkpoint = checkpoints.get(source)
        artifact_count = len(artifacts_by_source.get(source, set()))
        source_rows.append(
            {
                "source": source,
                "configured": source in source_configs_by_name,
                "enabled": source_configs_by_name.get(source),
                "checkpoint_observed": checkpoint is not None,
                "complete": bool(checkpoint["complete"]) if checkpoint is not None else None,
                "upstream_count": (
                    checkpoint.get("upstream_count") if checkpoint is not None else None
                ),
                "current_artifacts": artifact_count,
                "candidate_artifacts": candidate_counts_by_source.get(source, 0),
                "model_claims": claim_counts_by_source.get(source, 0),
                "artifacts_without_model_claim": artifact_count
                - candidate_counts_by_source.get(source, 0),
            }
        )

    checked_relation_root = (
        relation_root.expanduser().resolve()
        if relation_root is not None
        else root / "projections" / "artifact-relations"
    )
    relation_receipt = ArtifactRelationMaterializer(
        root,
        output_root=checked_relation_root,
    ).current_receipt()
    enabled_sources = sorted(
        source for source, enabled in source_configs_by_name.items() if enabled
    )
    configured_observed = all(source in checkpoints for source in enabled_sources)
    configured_complete = all(
        source in checkpoints and bool(checkpoints[source]["complete"])
        for source in enabled_sources
    )
    sealed_cross_record_ready = relation_receipt is not None
    targeted_cross_record_available = bool(declarations)
    current_evidence_ready = targeted_cross_record_available
    return {
        "format": f"{ENTRY_FORMAT}-readiness-v1",
        "store_commit": commit,
        "configured_source_count": len(source_configs_by_name),
        "enabled_source_count": len(enabled_sources),
        "current_artifact_count": len({artifact.id for artifact in current.values()}),
        "candidate_artifact_count": len(declarations),
        "model_claim_count": sum(len(models) for models in declarations.values()),
        "relation_projection": {
            "root": str(checked_relation_root),
            "current": sealed_cross_record_ready,
            "path": str(relation_receipt.path) if relation_receipt is not None else None,
            "row_count": relation_receipt.row_count if relation_receipt is not None else None,
        },
        "gates": {
            "configured_sources_observed": configured_observed,
            "configured_sources_complete": configured_complete,
            "sealed_cross_record_resources_ready": sealed_cross_record_ready,
            "targeted_cross_record_resources_available": targeted_cross_record_available,
            "current_evidence_entry_ready": current_evidence_ready,
            "configured_full_run_ready": configured_observed
            and configured_complete
            and current_evidence_ready,
        },
        "sources": source_rows,
        "limitations": [
            "This audits current configured-source and evidence prerequisites only.",
            "A complete source checkpoint does not prove coverage of private, deleted, "
            "or undiscoverable models.",
            "Use export-entry-seeds --link-current-resources to include the targeted "
            "exact URL/identifier cross-record evidence path.",
            "Artifacts without an admitted model claim remain evidence, not entries, "
            "until a source or conservative extractor supports a subject.",
        ],
    }


def _current_artifacts(
    path: Path,
    sources: set[str] | None,
) -> dict[str, _CurrentArtifact]:
    result = {}
    artifact_file = pq.ParquetFile(path)
    for batch in artifact_file.iter_batches(
        batch_size=8_192,
        columns=(
            "id",
            "source",
            "source_record_id",
            "kind",
            "canonical_url",
            "canonical_url_normalized",
            "current_revision_id",
            "active",
        ),
    ):
        for artifact in batch.to_pylist():
            if int(artifact.get("active") or 0) != 1:
                continue
            source = str(artifact.get("source") or "").strip()
            if not source or (sources is not None and source not in sources):
                continue
            revision_id = str(artifact.get("current_revision_id") or "").strip()
            source_record_id = str(artifact.get("source_record_id") or "").strip()
            artifact_id = str(artifact.get("id") or "").strip()
            kind = str(artifact.get("kind") or "").strip()
            canonical_url = str(artifact.get("canonical_url") or "").strip()
            canonical_url_normalized = str(artifact.get("canonical_url_normalized") or "").strip()
            if (
                artifact_id
                and revision_id
                and source_record_id
                and kind
                and canonical_url
                and canonical_url_normalized
            ):
                result[revision_id] = _CurrentArtifact(
                    id=artifact_id,
                    revision_id=revision_id,
                    source=source,
                    source_record_id=source_record_id,
                    kind=kind,
                    canonical_url=canonical_url,
                    canonical_url_normalized=canonical_url_normalized,
                )
    return result


def _current_model_declarations(
    tables: Path,
    current: Mapping[str, _CurrentArtifact],
) -> dict[str, tuple[_ModelDeclaration, ...]]:
    """Read declared and extractor-derived model claims for current artifacts.

    Artifact revisions retain the unmodified upstream record. Derived introduction
    and README candidates instead live in ``artifact_model_links``; treating those
    rows as authoritative lets a later entry build see every admitted model claim,
    not only source-native model fields.
    """

    revision_to_artifact = dict(current)
    links_by_artifact: dict[str, list[dict[str, Any]]] = {}
    model_ids: set[str] = set()
    for row in _parquet_rows(
        tables / "artifact_model_links.parquet",
        (
            "artifact_id",
            "artifact_revision_id",
            "model_id",
            "local_id",
            "status",
            "confidence",
            "locator",
        ),
    ):
        artifact = revision_to_artifact.get(str(row["artifact_revision_id"]))
        if artifact is None or str(row["artifact_id"]) != artifact.id:
            continue
        model_id = _required_text(row.get("model_id"), "model link model ID")
        links_by_artifact.setdefault(artifact.id, []).append(row)
        model_ids.add(model_id)
    if not model_ids:
        return {}

    models = {
        _required_text(row.get("id"), "model ID"): row
        for row in _parquet_rows(
            tables / "models.parquet",
            ("id", "canonical_name"),
        )
        if _required_text(row.get("id"), "model ID") in model_ids
    }
    missing_models = model_ids - set(models)
    if missing_models:
        raise ValueError(
            "current model links refer to missing model rows: " + ", ".join(sorted(missing_models))
        )

    aliases_by_model: dict[str, set[str]] = {}
    for row in _parquet_rows(
        tables / "model_aliases.parquet",
        ("model_id", "alias"),
    ):
        model_id = str(row["model_id"])
        if model_id in model_ids:
            aliases_by_model.setdefault(model_id, set()).add(
                _required_text(row.get("alias"), "model alias")
            )
    identifiers_by_claim: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in _parquet_rows(
        tables / "model_identifier_claims.parquet",
        ("artifact_revision_id", "model_id", "namespace", "value", "resolution_status"),
    ):
        revision_id = str(row["artifact_revision_id"])
        model_id = str(row["model_id"])
        if (
            revision_id not in revision_to_artifact
            or model_id not in model_ids
            or row.get("resolution_status") != "bound"
        ):
            continue
        identifiers_by_claim.setdefault((revision_id, model_id), set()).add(
            (
                _required_text(row.get("namespace"), "model identifier namespace"),
                _required_text(row.get("value"), "model identifier value"),
            )
        )

    result: dict[str, list[_ModelDeclaration]] = {}
    for artifact_id, links in links_by_artifact.items():
        declarations = result.setdefault(artifact_id, [])
        for link in links:
            model_id = _required_text(link.get("model_id"), "model link model ID")
            model = models[model_id]
            canonical_name = _required_text(model.get("canonical_name"), "model name")
            aliases = tuple(
                sorted(
                    aliases_by_model.get(model_id, set()) - {canonical_name},
                    key=lambda value: (value.casefold(), value),
                )
            )
            declarations.append(
                _ModelDeclaration(
                    local_id=_required_text(link.get("local_id"), "model local ID"),
                    name=canonical_name,
                    aliases=aliases,
                    identifiers=tuple(
                        sorted(
                            identifiers_by_claim.get(
                                (str(link["artifact_revision_id"]), model_id),
                                set(),
                            )
                        )
                    ),
                    status=_required_text(link.get("status"), "model status"),
                    confidence=_confidence(link.get("confidence"), "model confidence"),
                    locator=_optional_text(link.get("locator")),
                )
            )
    return {
        artifact_id: tuple(
            sorted(
                declarations,
                key=lambda item: (item.local_id, item.name, item.identifiers),
            )
        )
        for artifact_id, declarations in result.items()
    }


def _model_seed(model: _ModelDeclaration) -> dict[str, Any]:
    return {
        "local_id": model.local_id,
        "name": model.name,
        "aliases": list(model.aliases),
        "identifiers": [
            {"namespace": namespace, "value": value}
            for namespace, value in model.identifiers
        ],
        "status": model.status,
        "confidence": model.confidence,
        "locator": model.locator,
    }


def _relation_links(
    store: Path,
    relation_root: Path | None,
    eligible_artifact_ids: set[str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    if relation_root is None:
        return {}
    materializer = ArtifactRelationMaterializer(store, output_root=relation_root)
    receipt = materializer.current_receipt()
    if receipt is None:
        raise ValueError(
            "no sealed artifact-relation projection matches the current registry commit; "
            "run link-artifacts explicitly before exporting relation-enriched entry seeds"
        )
    rows: dict[str, list[dict[str, Any]]] = {}
    columns = (
        "relation_id",
        "subject_artifact_id",
        "predicate",
        "target_artifact_id",
        "symmetric",
        "evidence_type",
        "confidence",
        "match_namespace",
        "match_value",
        "locator",
        "subject_artifact_revision_id",
        "subject_kind",
        "subject_source",
        "subject_source_record_id",
        "subject_canonical_url",
        "target_artifact_revision_id",
        "target_kind",
        "target_source",
        "target_source_record_id",
        "target_canonical_url",
    )
    for batch in materializer.iter_batches(receipt, columns=columns):
        for relation in batch.to_pylist():
            subject_id = _required_text(relation.get("subject_artifact_id"), "relation subject")
            target_id = _required_text(relation.get("target_artifact_id"), "relation target")
            if subject_id in eligible_artifact_ids:
                rows.setdefault(subject_id, []).append(
                    _relation_link(relation, endpoint="target", direction="outgoing")
                )
            if target_id in eligible_artifact_ids:
                direction = "symmetric" if bool(relation["symmetric"]) else "incoming"
                rows.setdefault(target_id, []).append(
                    _relation_link(relation, endpoint="subject", direction=direction)
                )
    return {
        artifact_id: tuple(
            sorted(
                links,
                key=lambda item: (
                    str(item["relation_evidence"]["id"]),
                    str(item["relation_evidence"]["direction"]),
                ),
            )
        )
        for artifact_id, links in rows.items()
    }


def _relation_link(
    relation: Mapping[str, Any],
    *,
    endpoint: str,
    direction: str,
) -> dict[str, Any]:
    if endpoint not in {"subject", "target"}:  # pragma: no cover - local invariant
        raise AssertionError(f"unknown relation endpoint: {endpoint}")
    prefix = f"{endpoint}_artifact"
    relation_id = _required_text(relation.get("relation_id"), "relation id")
    canonical_url = _required_text(relation.get(f"{endpoint}_canonical_url"), "relation URL")
    return {
        "url": canonical_url,
        "relation": _required_text(relation.get("predicate"), "relation predicate"),
        "locator": _optional_text(relation.get("locator")) or f"artifact-relation:{relation_id}",
        "crawl": False,
        "resolved_artifact": {
            "id": _required_text(relation.get(f"{prefix}_id"), "relation artifact id"),
            "revision_id": _required_text(
                relation.get(f"{prefix}_revision_id"),
                "relation artifact revision",
            ),
            "kind": _required_text(relation.get(f"{endpoint}_kind"), "relation artifact kind"),
            "source": _required_text(
                relation.get(f"{endpoint}_source"),
                "relation artifact source",
            ),
            "source_record_id": _required_text(
                relation.get(f"{endpoint}_source_record_id"),
                "relation artifact source record",
            ),
            "canonical_url": canonical_url,
        },
        "relation_evidence": {
            "id": relation_id,
            "direction": direction,
            "evidence_type": _required_text(
                relation.get("evidence_type"),
                "relation evidence type",
            ),
            "confidence": relation.get("confidence"),
            "match_namespace": _optional_text(relation.get("match_namespace")),
            "match_value": _optional_text(relation.get("match_value")),
        },
    }


def _current_resource_links(
    tables: Path,
    current: Mapping[str, _CurrentArtifact],
    eligible_artifact_ids: set[str],
    *,
    include_cross_record: bool,
) -> tuple[
    dict[str, tuple[dict[str, Any], ...]],
    dict[str, tuple[dict[str, Any], ...]],
]:
    """Return direct candidate links plus optional exact cross-record evidence.

    The subject set is limited to artifacts that already have an admitted model
    or technique claim. URLs found in the current text of one of those subjects
    are first-party entry evidence and are returned regardless of
    ``include_cross_record``. When requested, the second return value also adds
    incoming mentions and exact same-artifact bridges from other current records.
    The function scans immutable current tables but never writes a partition or
    relation projection. Neither return value can affect entry identity or
    candidate merging.
    """

    if not eligible_artifact_ids:
        return {}, {}
    by_artifact_id = {artifact.id: artifact for artifact in current.values()}
    candidates = {
        artifact_id: by_artifact_id[artifact_id]
        for artifact_id in eligible_artifact_ids
        if artifact_id in by_artifact_id
    }
    if len(candidates) != len(eligible_artifact_ids):
        missing = sorted(eligible_artifact_ids - set(candidates))
        raise ValueError("entry candidates are missing current artifacts: " + ", ".join(missing))

    rows: dict[str, list[dict[str, Any]]] = {}
    candidates_by_url: dict[str, set[str]] = {}
    if include_cross_record:
        for artifact in candidates.values():
            candidates_by_url.setdefault(
                artifact.canonical_url_normalized,
                set(),
            ).add(artifact.id)

    targets_by_url_id: dict[str, tuple[str, ...]] = {}
    if include_cross_record:
        # This is the cardinality reduction that keeps the operation practical:
        # retain only frontier rows that point at a candidate's canonical URL,
        # not a global URL-to-URL graph.
        for frontier in _parquet_rows(tables / "url_frontier.parquet", ("id", "url")):
            url = canonicalize_url(_required_text(frontier.get("url"), "frontier URL"))
            target_ids = candidates_by_url.get(url)
            if target_ids:
                targets_by_url_id[
                    _required_text(frontier.get("id"), "frontier ID")
                ] = tuple(sorted(target_ids))

    candidate_discoveries: list[tuple[str, str, str, str | None]] = []
    candidate_discovery_url_ids: set[str] = set()
    for discovery in _parquet_rows(
        tables / "url_discoveries.parquet",
        ("id", "url_id", "artifact_revision_id", "relation", "locator"),
    ):
        origin = current.get(str(discovery.get("artifact_revision_id") or ""))
        if origin is None:
            continue
        discovery_id = _required_text(discovery.get("id"), "URL discovery ID")
        url_id = _required_text(discovery.get("url_id"), "URL discovery frontier ID")
        relation = _required_text(discovery.get("relation"), "URL discovery relation")
        locator = _optional_text(discovery.get("locator"))
        if origin.id in candidates:
            candidate_discoveries.append((origin.id, url_id, relation, locator))
            candidate_discovery_url_ids.add(url_id)
        if include_cross_record:
            for target_id in targets_by_url_id.get(url_id, ()):
                if origin.id == target_id:
                    continue
                target = candidates[target_id]
                rows.setdefault(target_id, []).append(
                    _targeted_relation_link(
                        endpoint=origin,
                        predicate=relation,
                        locator=locator,
                        direction="incoming",
                        evidence_type="url_mention",
                        match_namespace="url",
                        match_value=target.canonical_url,
                        evidence_identity=discovery_id,
                    )
                )

    # Source-declared links are already in each record payload. URL discoveries
    # additionally include text extraction, so preserve only discoveries absent
    # from the source-declared link set.
    declared_link_keys, candidate_identifier_keys = _candidate_record_evidence(
        tables / "artifact_revisions.parquet",
        current,
        set(candidates),
    )
    urls_by_id: dict[str, str] = {}
    if candidate_discovery_url_ids:
        for frontier in _parquet_rows(tables / "url_frontier.parquet", ("id", "url")):
            url_id = _required_text(frontier.get("id"), "frontier ID")
            if url_id in candidate_discovery_url_ids:
                urls_by_id[url_id] = canonicalize_url(
                    _required_text(frontier.get("url"), "frontier URL")
                )
    direct_rows: dict[str, list[dict[str, Any]]] = {}
    for candidate_id, url_id, relation, locator in candidate_discoveries:
        url = urls_by_id.get(url_id)
        if url is None:
            raise ValueError(f"URL discovery references missing frontier URL: {url_id}")
        if (candidate_id, url, relation, locator) in declared_link_keys:
            continue
        direct_rows.setdefault(candidate_id, []).append(
            {
                "url": url,
                "relation": relation,
                "locator": locator,
                "crawl": True,
            }
        )

    if not include_cross_record:
        return (
            {
                artifact_id: tuple(_deduplicate_links(links))
                for artifact_id, links in direct_rows.items()
            },
            {},
        )

    # Identical canonical URLs are an exact same-artifact bridge. They are not
    # URL mentions; the records themselves declare the same canonical identity.
    for origin in by_artifact_id.values():
        for target_id in candidates_by_url.get(origin.canonical_url_normalized, ()):
            if origin.id == target_id:
                continue
            rows.setdefault(target_id, []).append(
                _targeted_relation_link(
                    endpoint=origin,
                    predicate="same_artifact",
                    locator=None,
                    direction="symmetric",
                    evidence_type="shared_canonical_url",
                    match_namespace="url",
                    match_value=origin.canonical_url,
                    evidence_identity=f"canonical-url:{origin.id}:{target_id}",
                )
            )

    # ``artifact_identifiers`` is append-only. Read the current record JSON
    # instead, so an identifier removed in a newer source revision cannot form
    # a stale cross-record resource link.
    if candidate_identifier_keys:
        revision_file = pq.ParquetFile(tables / "artifact_revisions.parquet")
        for batch in revision_file.iter_batches(batch_size=8_192, columns=("id", "record_json")):
            for revision in batch.to_pylist():
                origin = current.get(str(revision.get("id") or ""))
                if origin is None:
                    continue
                payload = _read_record(str(revision["record_json"]))
                for namespace, value in _record_identifier_pairs(payload):
                    for target_id in candidate_identifier_keys.get((namespace, value), ()):
                        if origin.id == target_id:
                            continue
                        rows.setdefault(target_id, []).append(
                            _targeted_relation_link(
                                endpoint=origin,
                                predicate="same_artifact",
                                locator=None,
                                direction="symmetric",
                                evidence_type="shared_artifact_identifier",
                                match_namespace=namespace,
                                match_value=value,
                                evidence_identity=(
                                    f"artifact-identifier:{namespace}:{value}:"
                                    f"{origin.id}:{target_id}"
                                ),
                            )
                        )
    return (
        {
            artifact_id: tuple(_deduplicate_links(links))
            for artifact_id, links in direct_rows.items()
        },
        {
            artifact_id: tuple(_deduplicate_links(links))
            for artifact_id, links in rows.items()
        },
    )


def _candidate_record_evidence(
    revision_path: Path,
    current: Mapping[str, _CurrentArtifact],
    candidate_ids: set[str],
) -> tuple[set[tuple[str, str, str, str | None]], dict[tuple[str, str], tuple[str, ...]]]:
    """Read exact current candidate identifiers and source-declared link keys."""

    declared_links: set[tuple[str, str, str, str | None]] = set()
    identifiers: dict[tuple[str, str], set[str]] = {}
    revision_file = pq.ParquetFile(revision_path)
    for batch in revision_file.iter_batches(batch_size=8_192, columns=("id", "record_json")):
        for revision in batch.to_pylist():
            artifact = current.get(str(revision.get("id") or ""))
            if artifact is None or artifact.id not in candidate_ids:
                continue
            payload = _read_record(str(revision["record_json"]))
            for item in _array(payload.get("links", []), "stored artifact links"):
                if not isinstance(item, Mapping):
                    raise ValueError("stored artifact links must contain objects")
                declared_links.add(
                    (
                        artifact.id,
                        canonicalize_url(_required_text(item.get("url"), "stored link URL")),
                        _required_text(item.get("relation"), "stored link relation"),
                        _optional_text(item.get("locator")),
                    )
                )
            for key in _record_identifier_pairs(payload):
                identifiers.setdefault(key, set()).add(artifact.id)
    return declared_links, {
        key: tuple(sorted(artifact_ids)) for key, artifact_ids in identifiers.items()
    }


def _record_identifier_pairs(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    values = _array(payload.get("identifiers", []), "stored artifact identifiers")
    result = set()
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError("stored artifact identifiers must contain objects")
        result.add(
            (
                _required_text(item.get("namespace"), "stored identifier namespace"),
                _required_text(item.get("value"), "stored identifier value"),
            )
        )
    return tuple(sorted(result))


def _targeted_relation_link(
    *,
    endpoint: _CurrentArtifact,
    predicate: str,
    locator: str | None,
    direction: str,
    evidence_type: str,
    match_namespace: str,
    match_value: str,
    evidence_identity: str,
) -> dict[str, Any]:
    evidence_id = "targeted:" + hashlib.sha256(evidence_identity.encode("utf-8")).hexdigest()
    return {
        "url": endpoint.canonical_url,
        "relation": predicate,
        "locator": locator or f"targeted-resource-link:{evidence_id}",
        "crawl": False,
        "resolved_artifact": {
            "id": endpoint.id,
            "revision_id": endpoint.revision_id,
            "kind": endpoint.kind,
            "source": endpoint.source,
            "source_record_id": endpoint.source_record_id,
            "canonical_url": endpoint.canonical_url,
        },
        "relation_evidence": {
            "id": evidence_id,
            "direction": direction,
            "evidence_type": evidence_type,
            "confidence": 1.0,
            "match_namespace": match_namespace,
            "match_value": match_value,
        },
    }


def _merge_link_maps(
    *maps: Mapping[str, tuple[dict[str, Any], ...]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for source in maps:
        for artifact_id, links in source.items():
            merged.setdefault(artifact_id, []).extend(links)
    return {
        artifact_id: tuple(_deduplicate_links(links))
        for artifact_id, links in merged.items()
    }


def _deduplicate_links(links: Iterable[Any]) -> list[Any]:
    unique: dict[str, Any] = {}
    for link in links:
        key = json.dumps(link, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, link)
    return [unique[key] for key in sorted(unique)]


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _read_record(value: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("stored artifact record is not JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("stored artifact record must be an object")
    return payload


def _parquet_rows(path: Path, columns: tuple[str, ...]) -> Iterable[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows == 0:
        return ()
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"entry-seed table is missing columns {sorted(missing)}: {path}")
    return (
        row
        for batch in parquet.iter_batches(batch_size=8_192, columns=columns)
        for row in batch.to_pylist()
    )


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return _required_text(value, "text")


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


__all__ = [
    "EntrySeedExportReceipt",
    "assess_entry_readiness",
    "export_current_entry_seeds",
    "source_tags_from_configs",
]
