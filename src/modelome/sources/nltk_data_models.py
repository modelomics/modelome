"""Pinned reader for model archives in the official NLTK data index."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PACKAGE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY = "nltk/nltk_data"
_INDEX = "index.xml"
_MODEL_DIRS = {"chunkers", "taggers"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class NltkDataModelIndexSourceAdapter:
    """Enumerate released tagger, chunker, and learned tokenizer bundles.

    NLTK's official data index supplies package IDs, archive URLs, sizes, and
    checksums. The adapter admits only named model packages in model directories
    and Punkt tokenizer models, excluding corpora and tagset metadata.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers model bundles identified by the official NLTK data index as "
        "taggers, chunkers, or Punkt tokenizer models. It excludes corpora, "
        "grammars, lexicons, and tagset mappings."
    )

    def __init__(
        self,
        *,
        name: str = "nltk-data-model-index",
        repository: str = _REPOSITORY,
        branch: str = "gh-pages",
        provider_namespace: str = "nltk:data-model",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 1000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if (
            not name.strip()
            or repository != _REPOSITORY
            or not branch.strip()
            or not provider_namespace.strip()
        ):
            raise ValueError("name, official repository, branch, and namespace are required")
        if max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("response and entry limits must be positive")
        self.name, self.repository, self.branch = name, repository, branch
        self.provider_namespace = provider_namespace
        self.max_response_bytes, self.max_entries = max_response_bytes, max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "nltk-data-model-index-v1",
                "repository": repository,
                "branch": branch,
                "index": _INDEX,
                "admission": "tagger/chunker and Punkt tokenizer packages",
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                (), {**state, "checked_at": checked}, True, upstream_count=state.get("model_count")
            )
        index_url = f"https://raw.githubusercontent.com/{self.repository}/{revision}/{_INDEX}"
        response = self.client.get(index_url, headers={"Accept": "application/xml"})
        if response.status != 200:
            raise ValueError(f"{self.name}: index returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: index exceeds response limit")
        try:
            root = ET.fromstring(response.body)
        except ET.ParseError as exc:
            raise ValueError(f"{self.name}: invalid XML index") from exc
        packages = root.find("packages")
        if packages is None:
            raise ValueError(f"{self.name}: index has no packages element")
        rows = _model_rows(packages, self.name)
        if len(rows) > self.max_entries:
            raise ValueError(f"{self.name}: model count exceeds configured limit")
        if not rows:
            raise ValueError(f"{self.name}: no supported model packages found")
        records = tuple(self._record(row, revision) for row in rows)
        return SourcePage(
            records,
            {
                "completed_revision": revision,
                "checked_at": checked,
                "index_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, row: Mapping[str, str], revision: str) -> SourceRecord:
        package_id = row["id"]
        model_id = f"model:{package_id}"
        package_url = row["url"]
        source_url = f"{self.repository_url}/blob/{revision}/{_INDEX}"
        model = ModelHint(
            model_id,
            row["name"],
            identifiers=(Identifier(self.provider_namespace, package_id),),
            aliases=(package_id,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{package_id}",
            model_id,
            version=package_id,
            identifiers=(Identifier(f"{self.provider_namespace}:package", package_id),),
            metadata={
                "package_id": package_id,
                "archive_url": package_url,
                "archive_size": int(row["size"]),
                "sha256": row["sha256_checksum"],
                "subdir": row["subdir"],
                "index_revision": revision,
            },
        )
        return SourceRecord(
            source_record_id=f"model:{package_id}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(package_url),
            title=f"NLTK {row['name']}",
            raw={**row, "repository": self.repository, "index_revision": revision},
            text=f"NLTK released model package {package_id}: {row['name']}",
            identifiers=(Identifier(self.provider_namespace, package_id),),
            links=(
                Link(package_url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(source_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(
                    self.repository_url,
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _model_rows(packages: ET.Element, source: str) -> tuple[Mapping[str, str], ...]:
    rows: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for package in packages.findall("package"):
        package_id = package.get("id", "")
        name = package.get("name", "")
        subdir = package.get("subdir", "")
        if subdir not in _MODEL_DIRS and not (
            subdir == "tokenizers" and "punkt tokenizer models" in name.lower()
        ):
            continue
        # Prevent tagset tables and other metadata in the taggers directory from
        # being represented as trained model releases.
        if subdir == "taggers" and "tagger" not in name.lower():
            continue
        url = package.get("url", "")
        size = package.get("size", "")
        sha256 = package.get("sha256_checksum", "")
        if (
            not _PACKAGE_ID.fullmatch(package_id)
            or not name.strip()
            or package_id in seen
            or not url.startswith(
                "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/"
            )
            or not url.endswith(f"/{subdir}/{package_id}.zip")
            or not size.isdecimal()
            or int(size) <= 0
            or not _SHA256.fullmatch(sha256)
        ):
            raise ValueError(f"{source}: invalid model package row {package_id!r}")
        seen.add(package_id)
        rows.append({**package.attrib})
    return tuple(sorted(rows, key=lambda row: row["id"]))
