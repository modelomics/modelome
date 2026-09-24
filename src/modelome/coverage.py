from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.normalize import normalize_name
from modelome.storage import Database


def evaluate_manifest(database: Database, path: str | Path) -> dict[str, Any]:
    """Measure exact-name recall for an external, read-only acceptance corpus.

    The manifest is deliberately input to validation rather than input to any
    source adapter. Its examples can test discovery, but cannot cause discovery.
    """

    manifest_path = Path(path)
    payload = manifest_path.read_bytes()
    expectations = _read_expectations(payload, manifest_path)
    active_names = _active_name_index(database)
    observations = [_observe(active_names, item) for item in expectations]

    bucket_totals: dict[str, int] = defaultdict(int)
    bucket_found: dict[str, int] = defaultdict(int)
    for observation in observations:
        bucket = observation["bucket"]
        bucket_totals[bucket] += 1
        bucket_found[bucket] += int(observation["found"])

    found = sum(int(item["found"]) for item in observations)
    total = len(observations)
    return {
        "observed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "total": total,
        "found": found,
        "missing": total - found,
        "recall": found / total if total else 1.0,
        "buckets": {
            bucket: {
                "total": bucket_totals[bucket],
                "found": bucket_found[bucket],
                "missing": bucket_totals[bucket] - bucket_found[bucket],
                "recall": bucket_found[bucket] / bucket_totals[bucket],
            }
            for bucket in sorted(bucket_totals)
        },
        "registry": database.coverage_metrics(),
        "expectations": observations,
    }


def _read_expectations(payload: bytes, path: Path) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: coverage manifest must be UTF-8 CSV") from error
    reader = csv.DictReader(text.splitlines())
    headers = reader.fieldnames or []
    duplicate_headers = sorted(
        {header for header in headers if header is not None and headers.count(header) > 1}
    )
    if duplicate_headers:
        raise ValueError(
            f"{path}: coverage manifest has duplicate column(s): "
            f"{', '.join(duplicate_headers)}"
        )
    missing = {"Model", "Bucket"} - set(headers)
    if missing:
        raise ValueError(
            f"{path}: coverage manifest is missing column(s): {', '.join(sorted(missing))}"
        )
    rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: coverage manifest has no expectations")

    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    bucket_labels: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"{path}:{line_number}: coverage row has unexpected extra columns")
        name = (row.get("Model") or "").strip()
        bucket = (row.get("Bucket") or "").strip()
        if not name or not bucket:
            raise ValueError(f"{path}:{line_number}: Model and Bucket must not be empty")
        normalized_name = normalize_name(name)
        if not normalized_name:
            raise ValueError(
                f"{path}:{line_number}: Model must contain letters or numbers"
            )
        normalized_bucket = bucket.casefold()
        key = (normalized_name, normalized_bucket)
        if key in seen:
            raise ValueError(f"{path}:{line_number}: duplicate expectation for {name!r}")
        seen.add(key)
        bucket_labels.setdefault(normalized_bucket, bucket)
        result.append({"name": name, "bucket": bucket_labels[normalized_bucket]})
    return result


def _active_name_index(database: Database) -> dict[str, dict[str, set[str]]]:
    """Index exact active model names once for a manifest-sized recall pass.

    ``model_detail`` is intentionally comprehensive, but calling it for every
    candidate causes repeated full evidence-table scans on a large registry.
    Coverage asks only whether current active artifact evidence exists, so this
    compact join preserves that meaning in one pass.
    """

    artifacts = {
        row["id"]: row
        for row in database.table_rows("artifacts")
        if row["active"] and row["current_revision_id"]
    }
    sources_by_model: dict[str, set[str]] = defaultdict(set)
    for link in database.table_rows("artifact_model_links"):
        artifact = artifacts.get(link["artifact_id"])
        if artifact is None or link["artifact_revision_id"] != artifact["current_revision_id"]:
            continue
        sources_by_model[link["model_id"]].add(str(artifact["source"]))

    names_by_model: dict[str, set[str]] = defaultdict(set)
    for model in database.table_rows("models"):
        name = normalize_name(str(model["canonical_name"]))
        if name:
            names_by_model[name].add(str(model["id"]))
    for alias in database.table_rows("model_aliases"):
        name = normalize_name(str(alias["alias"]))
        if name:
            names_by_model[name].add(str(alias["model_id"]))
    return {
        name: {
            model_id: sources_by_model[model_id]
            for model_id in model_ids
            if sources_by_model.get(model_id)
        }
        for name, model_ids in names_by_model.items()
    }


def _observe(
    active_names: dict[str, dict[str, set[str]]], expectation: dict[str, str]
) -> dict[str, Any]:
    name = expectation["name"]
    normalized = normalize_name(name)
    matches = active_names.get(normalized, {})
    sources = {source for values in matches.values() for source in values}

    return {
        "name": name,
        "bucket": expectation["bucket"],
        "found": bool(matches),
        "model_ids": sorted(matches),
        "sources": sorted(sources),
    }


__all__ = ["evaluate_manifest"]
