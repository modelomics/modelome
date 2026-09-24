"""First-party Medigan medical generative model package inventory."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpClient
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]
_REPOSITORY = "RichardObi/medigan"
_INDEX_PATH = "config/global.json"
_MAX_MODELS = 500


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MediganRegistrySourceAdapter:
    """Read Medigan's first-party model index and exact Zenodo package URLs.

    Each package is a ZIP bundle. The adapter records the declared bundle URL,
    not a guessed URL for an individual checkpoint inside it.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers models in Medigan's global.json index with declared Zenodo ZIP "
        "package links. Package contents are not enumerated or downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "medigan-first-party-model-index",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = _MAX_MODELS,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "medigan-first-party-model-index-v1",
            "repository": repository,
            "branch": branch,
            "index_path": _INDEX_PATH,
            "max_response_bytes": max_response_bytes,
            "max_entries": max_entries,
        })

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        index_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{self.branch}/{_INDEX_PATH}"
        )
        response = self.client.get(index_url, headers={"Accept": "application/json"})
        if response.status != 200:
            raise ValueError(f"{self.name}: model index returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: model index exceeds response limit")
        revision = hashlib.sha256(response.body).hexdigest()
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                (), {**state, "checked_at": checked}, True,
                upstream_count=state.get("checkpoint_count"),
            )
        entries = response.json()
        records = _parse_index(entries, self.max_entries, self.repository_url, self.branch)
        return SourcePage(
            tuple(records),
            {"completed_revision": revision, "checked_at": checked,
             "checkpoint_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_index(
    payload: Any, limit: int, repository_url: str, revision: str
) -> list[SourceRecord]:
    if not isinstance(payload, Mapping) or not payload or len(payload) > limit:
        raise ValueError("Medigan index must be a non-empty bounded object")
    records: list[SourceRecord] = []
    seen_urls: set[str] = set()
    index_url = f"{repository_url}/blob/{revision}/{_INDEX_PATH}"
    for model_id, value in sorted(payload.items()):
        if not isinstance(model_id, str) or not model_id or not isinstance(value, Mapping):
            raise ValueError("Medigan index contains an invalid model entry")
        execution = value.get("execution")
        description = value.get("description")
        if not isinstance(execution, Mapping) or not isinstance(description, Mapping):
            raise ValueError(f"Medigan entry {model_id} lacks execution or description metadata")
        package_url = execution.get("package_link")
        parts = urlsplit(package_url) if isinstance(package_url, str) else None
        if (
            parts is None or parts.scheme != "https" or parts.netloc != "zenodo.org"
            or not parts.path.startswith(("/record/", "/records/"))
            or not parts.path.lower().endswith(".zip") or package_url in seen_urls
        ):
            raise ValueError(f"Medigan entry {model_id} has an invalid or duplicate package link")
        seen_urls.add(package_url)
        title = description.get("title")
        package_name = execution.get("package_name")
        if (
            not isinstance(title, str) or not title.strip()
            or not isinstance(package_name, str) or not package_name.strip()
        ):
            raise ValueError(f"Medigan entry {model_id} lacks title or package name")
        version = description.get("version")
        doi_values = description.get("doi", [])
        if doi_values is None:
            doi_values = []
        if not isinstance(doi_values, list) or any(
            not isinstance(item, str) for item in doi_values
        ):
            raise ValueError(f"Medigan entry {model_id} has malformed DOI metadata")
        local_id = f"model:{model_id}"
        metadata: dict[str, Any] = {
            "package_name": package_name,
            "package_url": package_url,
            "package_format": "zip",
            "model_name": execution.get("model_name"),
            "checkpoint_extension": execution.get("extension"),
            "version": version,
            "doi": doi_values,
        }
        model = ModelHint(
            local_id, title,
            identifiers=(Identifier("medigan:model", model_id),),
            aliases=(package_name,), status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"package:{model_id}", local_id,
            identifiers=(Identifier("medigan:package-url", package_url),),
            version=version if isinstance(version, str) else None,
            metadata=metadata,
            locator="Medigan config/global.json execution.package_link",
        )
        records.append(SourceRecord(
            source_record_id=f"model:{model_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(package_url),
            title=title,
            raw={"model_id": model_id, "metadata": dict(value), "index_url": index_url,
                 "package_url": package_url},
            text=f"{title}; Medigan model id {model_id}; package {package_name}.",
            identifiers=(Identifier("medigan:model", model_id),),
            links=(
                Link(index_url, "model_card", crawl=False, model_local_ids=(local_id,)),
                Link(package_url, "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,), releases=(release,),
        ))
    return records


__all__ = ["MediganRegistrySourceAdapter"]
