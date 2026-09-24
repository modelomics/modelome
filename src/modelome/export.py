from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.storage import Database

_PUBLIC_CONFIG_FIELDS = (
    "adapter",
    "artifact_base_url",
    "artifact_source",
    "data_path",
    "dataset_id",
    "index_url",
    "license",
    "metadata_url",
    "schedule",
    "url",
)

_PARQUET_FIELDS: dict[str, tuple[str, ...]] = {
    "models": (
        "model_id",
        "canonical_name",
        "normalized_name",
        "status",
        "confidence",
        "created_at",
        "updated_at",
    ),
    "model_aliases": ("model_id", "alias", "normalized_alias"),
    "model_identifiers": ("model_id", "namespace", "value"),
    "artifacts": (
        "artifact_id",
        "source",
        "source_record_id",
        "kind",
        "canonical_url",
        "canonical_url_normalized",
        "published_at",
        "modified_at",
        "active",
        "created_at",
        "updated_at",
    ),
    "artifact_identifiers": ("artifact_id", "namespace", "value"),
    "model_artifacts": (
        "model_id",
        "artifact_id",
        "artifact_revision_id",
        "status",
        "resolution_status",
        "confidence",
        "locator",
        "extractor",
        "created_at",
    ),
    "artifact_url_links": (
        "source_artifact_id",
        "target_artifact_id",
        "target_url",
        "relation",
        "locator",
        "discovered_at",
        "depth",
    ),
    "model_releases": (
        "release_id",
        "model_id",
        "version",
        "revision",
        "weight_files_json",
        "released_at",
        "confidence",
        "created_at",
        "updated_at",
    ),
    "release_identifiers": ("release_id", "namespace", "value"),
}


@dataclass(frozen=True, slots=True)
class MetadataExportReceipt:
    output: str
    source_commit: str
    source_generation: int
    source_state_digest: str
    files: Mapping[str, Mapping[str, Any]]


def export_public_metadata(
    store: Database,
    output: str | Path,
    *,
    source_configs: Sequence[Mapping[str, Any]],
) -> MetadataExportReceipt:
    """Write a non-destructive public metadata bundle suitable for Hub upload.

    The export intentionally leaves out raw source payloads, source-record text,
    abstracts, model-card bodies, and evidence value blobs. It retains stable
    identifiers and source-attributed joins that let a downstream user connect a
    model with papers, code, catalog records, and releases.
    """

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"metadata export destination already exists: {destination}; "
            "choose a new empty path"
        )
    if destination == destination.parent:
        raise ValueError("metadata export destination must not be a filesystem root")
    destination.parent.mkdir(parents=True, exist_ok=True)

    head = _read_head(store.root)
    tables = {
        name: store.table_rows(name)
        for name in (
            "artifacts",
            "artifact_identifiers",
            "artifact_model_links",
            "artifact_release_links",
            "model_aliases",
            "model_external_identifiers",
            "model_releases",
            "models",
            "release_external_identifiers",
            "source_checkpoints",
            "url_discoveries",
            "url_frontier",
        )
    }
    projections = _project_public_metadata(tables)
    source_manifest = _source_manifest(tables["source_checkpoints"], source_configs)

    temporary = Path(
        tempfile.mkdtemp(prefix=".modelome-metadata-export-", dir=str(destination.parent))
    )
    try:
        files: dict[str, Mapping[str, Any]] = {}
        for name, rows in projections.items():
            path = temporary / f"{name}.parquet"
            _write_parquet(path, rows, _PARQUET_FIELDS[name])
            files[path.name] = _file_metadata(path, len(rows))

        source_manifest_path = temporary / "source-manifest.json"
        _write_json(source_manifest_path, source_manifest)
        files[source_manifest_path.name] = _file_metadata(
            source_manifest_path, len(source_manifest["sources"])
        )

        card_path = temporary / "README.md"
        card_path.write_text(
            _dataset_card(
                source_commit=str(head["commit"]),
                source_generation=int(head["generation"]),
                source_manifest=source_manifest,
                projected_rows={name: len(rows) for name, rows in projections.items()},
            ),
            encoding="utf-8",
        )
        files[card_path.name] = _file_metadata(card_path, None)

        receipt = MetadataExportReceipt(
            output=str(destination),
            source_commit=str(head["commit"]),
            source_generation=int(head["generation"]),
            source_state_digest=str(head["state_digest"]),
            files=files,
        )
        receipt_path = temporary / "export-receipt.json"
        _write_json(receipt_path, asdict(receipt))

        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return receipt


def _project_public_metadata(
    tables: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    artifacts = {row["id"]: row for row in tables["artifacts"]}
    current_revision_to_artifact = {
        row["current_revision_id"]: row
        for row in artifacts.values()
        if row.get("current_revision_id")
    }
    artifacts_by_url = {
        row["canonical_url_normalized"]: row["id"]
        for row in artifacts.values()
        if row.get("canonical_url_normalized")
    }
    release_weight_files: dict[str, set[str]] = {}
    for link in tables["artifact_release_links"]:
        artifact = artifacts.get(link["artifact_id"])
        if artifact is None or not _is_huggingface_model_artifact(artifact):
            continue
        try:
            metadata = json.loads(link.get("metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, Mapping):
            continue
        filenames = metadata.get("weight_files")
        if not isinstance(filenames, list):
            continue
        release_weight_files.setdefault(link["release_id"], set()).update(
            filename
            for filename in filenames
            if _is_safe_huggingface_filename(filename)
        )
    frontier_by_id = {row["id"]: row for row in tables["url_frontier"]}

    result: dict[str, list[dict[str, Any]]] = {
        "models": [
            {
                "model_id": row["id"],
                "canonical_name": row["canonical_name"],
                "normalized_name": row["normalized_name"],
                "status": row["status"],
                "confidence": row["confidence"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in tables["models"]
        ],
        "model_aliases": [
            {
                "model_id": row["model_id"],
                "alias": row["alias"],
                "normalized_alias": row["normalized_alias"],
            }
            for row in tables["model_aliases"]
        ],
        "model_identifiers": [
            {
                "model_id": row["model_id"],
                "namespace": row["namespace"],
                "value": row["value"],
            }
            for row in tables["model_external_identifiers"]
        ],
        "artifacts": [
            {
                "artifact_id": row["id"],
                "source": row["source"],
                "source_record_id": row["source_record_id"],
                "kind": row["kind"],
                "canonical_url": row["canonical_url"],
                "canonical_url_normalized": row["canonical_url_normalized"],
                "published_at": row["published_at"],
                "modified_at": row["modified_at"],
                "active": row["active"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in artifacts.values()
        ],
        "artifact_identifiers": [
            {
                "artifact_id": row["artifact_id"],
                "namespace": row["namespace"],
                "value": row["value"],
            }
            for row in tables["artifact_identifiers"]
            if row["artifact_id"] in artifacts
        ],
        "model_artifacts": [
            {
                "model_id": row["model_id"],
                "artifact_id": row["artifact_id"],
                "artifact_revision_id": row["artifact_revision_id"],
                "status": row["status"],
                "resolution_status": row["resolution_status"],
                "confidence": row["confidence"],
                "locator": row["locator"],
                "extractor": row["extractor"],
                "created_at": row["created_at"],
            }
            for row in tables["artifact_model_links"]
            if current_revision_to_artifact.get(row["artifact_revision_id"]) is not None
        ],
        "artifact_url_links": [
            {
                "source_artifact_id": artifact["id"],
                "target_artifact_id": artifacts_by_url.get(target["url"]),
                "target_url": target["url"],
                "relation": row["relation"],
                "locator": row["locator"],
                "discovered_at": row["discovered_at"],
                "depth": row["depth"],
            }
            for row in tables["url_discoveries"]
            for artifact in (current_revision_to_artifact.get(row["artifact_revision_id"]),)
            if artifact is not None
            for target in (frontier_by_id.get(row["url_id"]),)
            if target is not None
        ],
        "model_releases": [
            {
                "release_id": row["id"],
                "model_id": row["model_id"],
                "version": row["version"],
                "revision": row["revision"],
                "weight_files_json": (
                    json.dumps(sorted(release_weight_files[row["id"]]), ensure_ascii=False)
                    if release_weight_files.get(row["id"])
                    else None
                ),
                "released_at": row["released_at"],
                "confidence": row["confidence"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in tables["model_releases"]
        ],
        "release_identifiers": [
            {
                "release_id": row["release_id"],
                "namespace": row["namespace"],
                "value": row["value"],
            }
            for row in tables["release_external_identifiers"]
        ],
    }
    for name, rows in result.items():
        fields = _PARQUET_FIELDS[name]
        rows.sort(key=lambda row: tuple(_sort_value(row.get(field)) for field in fields))
    return result


def _is_huggingface_model_artifact(artifact: Mapping[str, Any]) -> bool:
    if artifact.get("source") != "huggingface" or artifact.get("kind") != "model_card":
        return False
    parts = urlsplit(str(artifact.get("canonical_url") or ""))
    return parts.scheme == "https" and parts.hostname == "huggingface.co"


def _is_safe_huggingface_filename(value: Any) -> bool:
    if not isinstance(value, str) or not value or len(value) > 1024:
        return False
    if value.startswith("/") or "\\" in value or any(ord(char) < 32 for char in value):
        return False
    segments = value.split("/")
    return all(segment not in {"", ".", ".."} for segment in segments)


def _source_manifest(
    checkpoints: Sequence[Mapping[str, Any]],
    source_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    configs = {
        str(row["name"]): row
        for row in source_configs
        if isinstance(row.get("name"), str)
    }
    checkpoint_by_source = {str(row["source"]): row for row in checkpoints}
    sources = []
    for source in sorted(set(configs) | set(checkpoint_by_source)):
        checkpoint = checkpoint_by_source.get(source)
        config = configs.get(source, {})
        sources.append(
            {
                "source": source,
                "adapter": config.get("adapter"),
                "is_configured": source in configs,
                "checkpoint_observed": checkpoint is not None,
                "schedule": config.get("schedule"),
                "configured": {
                    field: config[field]
                    for field in _PUBLIC_CONFIG_FIELDS
                    if field in config
                },
                "checkpoint": (
                    {
                        "complete": checkpoint["complete"],
                        "upstream_count": checkpoint["upstream_count"],
                        "pages_ingested": checkpoint["pages_ingested"],
                        "records_seen": checkpoint["records_seen"],
                        "updated_at": checkpoint["updated_at"],
                    }
                    if checkpoint is not None
                    else None
                ),
            }
        )
    return {
        "format": "modelome-public-metadata-v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "sources": sources,
    }


def _dataset_card(
    *,
    source_commit: str,
    source_generation: int,
    source_manifest: Mapping[str, Any],
    projected_rows: Mapping[str, int],
) -> str:
    files = "\n".join(
        f"- {name}.parquet: {count:,} rows."
        for name, count in sorted(projected_rows.items())
    )
    sources = "\n".join(
        f"- {row['source']} ({row.get('adapter') or 'unconfigured adapter'})"
        for row in source_manifest["sources"]
    )
    return f"""---
pretty_name: MODELOME public metadata
tags:
- machine-learning
- model-registry
- provenance
- research
---

# MODELOME public metadata

This is a provenance-preserving, metadata-only export of the MODELOME registry.
It records model identities, source-attributed artifact links, and trained-release
identifiers. The source registry commit is {source_commit} (generation
{source_generation}).

## Included files

{files}

model_artifacts.parquet joins model IDs to their current source artifacts.
artifact_url_links.parquet joins a current artifact to a discovered external URL
and, when present in this export, the matching target artifact ID. Together with
artifacts.parquet and identifiers, those tables support paper/code/model and
model/release reconstruction without claiming that a URL is a binary payload.
model_releases.parquet may include weight_files_json, a JSON array of safe relative
filenames declared by a Hugging Face model record. Other release metadata is omitted.

## Deliberately excluded

This release does not include source-record raw payloads, paper abstracts or full
text, model-card bodies, README bodies, source HTML, or unreviewed evidence values.
Follow the external URLs and identifiers for those materials under their own terms.

## Source attribution and rights

source-manifest.json records configured sources and observed source checkpoints,
including configured sources with no observed checkpoint. The checkpoint's
complete flag and counts are reported only when that checkpoint exists; they do
not claim unrun sources were ingested. A true complete flag means the adapter
finished its checkpointed scan; it does not claim exhaustive historical or global
coverage. Source-level terms remain in force.
In particular, derivative users must preserve attribution and applicable
share-alike obligations for records derived from the Papers with Code archival
datasets. The arXiv snapshot contributes factual metadata only; paper-specific
licenses remain attached to the original records and paper content is not included.

Contributing source checkpoints:

{sources}
"""


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    table = pa.table({field: [row.get(field) for row in rows] for field in fields})
    pq.write_table(table, path, compression="zstd")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_metadata(path: Path, rows: int | None) -> dict[str, Any]:
    return {
        "rows": rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _read_head(root: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads((root / "HEAD.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not read the current registry HEAD") from error
    if not all(key in payload for key in ("commit", "generation", "state_digest")):
        raise ValueError("registry HEAD is missing immutable commit metadata")
    return payload


def _sort_value(value: Any) -> tuple[int, Any]:
    return (value is not None, value if value is not None else "")
