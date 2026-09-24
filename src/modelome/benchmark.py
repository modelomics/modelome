from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from io import StringIO
from typing import Any

from modelome.fetchers import PublicUrlPolicy
from modelome.http import HttpClient, HttpResponse
from modelome.normalize import canonical_json, normalize_name
from modelome.storage import Database

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def evaluate_epoch_benchmark(
    database: Database,
    config: Mapping[str, Any],
    *,
    client: HttpClient | Any | None = None,
    url_policy: PublicUrlPolicy | None = None,
    clock: Clock = _utcnow,
) -> dict[str, Any]:
    """Evaluate registry recall against a configured Epoch CSV without ingesting it.

    ``config`` is the Epoch benchmark table. Its URL and model column are
    read from configuration, so this evaluator contains no model-name seeds. A
    model counts as found only when its exact normalized canonical name or alias
    matches the benchmark expectation and a current, active discovery artifact
    supports that registry model.

    Unique normalized model names form the primary recall denominator. The
    returned ``rows`` collection also reports every valid CSV row, including
    duplicate expectations, so upstream duplication remains visible and
    auditable rather than silently affecting the score.
    """

    benchmark_name, url, model_column = _benchmark_settings(config)
    http = client or HttpClient()
    policy = url_policy or PublicUrlPolicy()
    # Fully resolve destinations for the production HTTP client. Injected
    # clients are commonly deterministic test transports; syntax and literal
    # address safety still apply to them without performing unrelated DNS I/O.
    request_url = policy.validate(url, resolve=client is None or isinstance(http, HttpClient))
    response: HttpResponse = http.get(request_url, headers={"Accept": "text/csv"})
    if response.status != 200:
        raise ValueError(
            f"{benchmark_name}: benchmark retrieval returned HTTP {response.status}; "
            "a fresh complete CSV is required"
        )

    payload = bytes(response.body)
    parsed = _parse_expectations(payload, model_column, benchmark_name)
    observation_index = _build_observation_index(database)
    observed_by_name = {
        normalized_name: _observe(
            normalized_name=normalized_name,
            observation_index=observation_index,
        )
        for normalized_name in parsed["groups"]
    }

    expectations = []
    for normalized_name, group in parsed["groups"].items():
        observation = observed_by_name[normalized_name]
        expectations.append(
            {
                "name": group["name"],
                "normalized_name": normalized_name,
                "row_numbers": list(group["row_numbers"]),
                "names": list(group["names"]),
                **observation,
            }
        )

    rows = []
    for row in parsed["rows"]:
        observation = observed_by_name[row["normalized_name"]]
        rows.append({**row, **observation})

    total = len(expectations)
    found = sum(int(item["found"]) for item in expectations)
    row_total = len(rows)
    row_found = sum(int(item["found"]) for item in rows)
    retrieved_at = _isoformat(clock())
    config_fingerprint = hashlib.sha256(
        canonical_json({"name": benchmark_name, "url": url, "model_column": model_column}).encode()
    ).hexdigest()

    return {
        "benchmark": benchmark_name,
        "kind": "external_csv_recall",
        "model_column": model_column,
        "retrieval": _retrieval_metadata(
            response,
            requested_url=request_url,
            payload=payload,
            retrieved_at=retrieved_at,
            config_sha256=config_fingerprint,
        ),
        "raw_rows": parsed["raw_rows"],
        "valid_rows": row_total,
        "unique_expectations": total,
        "duplicate_count": len(parsed["duplicates"]),
        "invalid_count": len(parsed["invalid"]),
        "integrity_ok": not parsed["invalid"],
        "total": total,
        "found": found,
        "missing": total - found,
        "recall": found / total,
        "row_total": row_total,
        "row_found": row_found,
        "row_missing": row_total - row_found,
        "row_recall": row_found / row_total,
        "duplicates": parsed["duplicates"],
        "invalid": parsed["invalid"],
        "expectations": expectations,
        "rows": rows,
        "registry": database.coverage_metrics(),
    }


def _benchmark_settings(config: Mapping[str, Any]) -> tuple[str, str, str]:
    if not isinstance(config, Mapping):
        raise TypeError("Epoch benchmark config must be a mapping")
    if config.get("_catalog_role") not in {None, "benchmark"}:
        raise ValueError("ingestion source configuration cannot be used as a benchmark")
    name = _required_text(config.get("name"), "benchmark name")
    url = _required_text(config.get("url"), f"{name}: benchmark URL")
    adapter = _required_text(config.get("adapter"), f"{name}: adapter").casefold()
    if adapter != "csv":
        raise ValueError(f"{name}: Epoch benchmark adapter must be 'csv'")
    mapping = config.get("mapping")
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name}: benchmark mapping must be a table")
    model_column = _required_text(
        mapping.get("model_field"), f"{name}: mapping.model_field"
    )
    return name, url, model_column


def _parse_expectations(
    payload: bytes, model_column: str, benchmark_name: str
) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{benchmark_name}: benchmark CSV must be UTF-8") from error

    try:
        reader = csv.DictReader(StringIO(text, newline=""), strict=True)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError(f"{benchmark_name}: benchmark CSV has no header")
        duplicate_headers = sorted(
            {
                header
                for header in headers
                if header is not None and headers.count(header) > 1
            }
        )
        if duplicate_headers:
            raise ValueError(
                f"{benchmark_name}: benchmark CSV has duplicate column(s): "
                + ", ".join(duplicate_headers)
            )
        if model_column not in headers:
            raise ValueError(
                f"{benchmark_name}: benchmark CSV is missing configured model column "
                f"{model_column!r}"
            )
        raw_rows = list(reader)
    except csv.Error as error:
        raise ValueError(f"{benchmark_name}: malformed benchmark CSV: {error}") from error

    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}

    for row_number, row in enumerate(raw_rows, start=2):
        if None in row:
            invalid.append(
                {
                    "row_number": row_number,
                    "reason": "unexpected_extra_columns",
                }
            )
            continue
        raw_name = row.get(model_column)
        if raw_name is None or not (name := raw_name.strip()):
            invalid.append({"row_number": row_number, "reason": "empty_model_name"})
            continue
        normalized_name = normalize_name(name)
        if not normalized_name:
            invalid.append(
                {
                    "row_number": row_number,
                    "name": name,
                    "reason": "model_name_has_no_letters_or_numbers",
                }
            )
            continue

        group = groups.get(normalized_name)
        duplicate_of = None if group is None else group["row_numbers"][0]
        row_result = {
            "row_number": row_number,
            "name": name,
            "normalized_name": normalized_name,
            "duplicate_of_row": duplicate_of,
        }
        rows.append(row_result)
        if group is None:
            groups[normalized_name] = {
                "name": name,
                "names": [name],
                "row_numbers": [row_number],
            }
            continue

        group["row_numbers"].append(row_number)
        if name not in group["names"]:
            group["names"].append(name)
        duplicates.append(
            {
                "row_number": row_number,
                "name": name,
                "normalized_name": normalized_name,
                "duplicate_of_row": duplicate_of,
            }
        )

    if not groups:
        raise ValueError(f"{benchmark_name}: benchmark CSV has no valid model expectations")
    return {
        "raw_rows": len(raw_rows),
        "rows": rows,
        "groups": groups,
        "duplicates": duplicates,
        "invalid": invalid,
    }


def _build_observation_index(
    database: Database,
) -> dict[str, Any]:
    """Build one exact-name evidence index for the whole benchmark run.

    A benchmark can contain thousands of expectations and a registry can contain
    millions of models. Building the index once keeps evaluation linear in the
    registry size instead of scanning every model for every benchmark row.
    """

    active_revisions = {
        str(artifact["current_revision_id"]): str(artifact["source"])
        for artifact in database.table_rows("artifacts")
        if artifact.get("active")
        and artifact.get("current_revision_id")
    }
    model_names = {
        str(model["id"]): str(model["canonical_name"])
        for model in database.table_rows("models")
    }
    support: dict[str, dict[str, set[str]]] = {}
    for evidence in database.table_rows("evidence_provenance"):
        if evidence.get("subject_type") != "model":
            continue
        revision_id = str(evidence.get("artifact_revision_id", ""))
        source = active_revisions.get(revision_id)
        if source is None:
            continue
        try:
            value = json.loads(str(evidence.get("value_json", "null")))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        asserted_name = _asserted_normalized_name(evidence.get("predicate"), value)
        if not asserted_name:
            continue
        model_id = str(evidence.get("subject_id", ""))
        if model_id not in model_names:
            continue
        support.setdefault(asserted_name, {}).setdefault(model_id, set()).add(source)

    return {"model_names": model_names, "support": support}


def _observe(
    *,
    normalized_name: str,
    observation_index: Mapping[str, Any],
) -> dict[str, Any]:
    model_names = observation_index["model_names"]
    support = observation_index["support"].get(normalized_name, {})
    matches = [
        {
            "model_id": model_id,
            "canonical_name": model_names[model_id],
            "supporting_sources": sorted(sources),
        }
        for model_id, sources in support.items()
    ]

    matches.sort(key=lambda item: (item["canonical_name"].casefold(), item["model_id"]))
    return {
        "found": bool(matches),
        "matched_model_ids": [item["model_id"] for item in matches],
        "supporting_sources": sorted(
            {source for item in matches for source in item["supporting_sources"]}
        ),
        "matches": matches,
    }


def _asserted_normalized_name(predicate: Any, value: Any) -> str | None:
    if predicate == "alias" and isinstance(value, str):
        return normalize_name(value) or None
    if predicate == "documented_by" and isinstance(value, Mapping):
        name = value.get("name")
        return (normalize_name(name) or None) if isinstance(name, str) else None
    return None


def _retrieval_metadata(
    response: HttpResponse,
    *,
    requested_url: str,
    payload: bytes,
    retrieved_at: str,
    config_sha256: str,
) -> dict[str, Any]:
    headers = {str(key).casefold(): str(value) for key, value in response.headers.items()}
    result: dict[str, Any] = {
        "requested_url": requested_url,
        "response_url": response.url,
        "status": response.status,
        "retrieved_at": retrieved_at,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "config_sha256": config_sha256,
    }
    for header, key in (
        ("etag", "etag"),
        ("last-modified", "last_modified"),
        ("content-type", "content_type"),
        ("content-length", "content_length"),
        ("date", "response_date"),
    ):
        if value := headers.get(header):
            result[key] = value
    return result


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = ["evaluate_epoch_benchmark"]
