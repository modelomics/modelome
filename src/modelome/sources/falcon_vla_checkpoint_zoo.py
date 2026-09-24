"""FALCON-VLA's official model zoo with revision-pinned weight objects."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from html import unescape
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient, HttpResponse
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

_REPOSITORY = "FALCON-VLA/FALCON"
_HF_REPOSITORY = "FALCON-VLA/FALCON-series"
_HEADING = "## 🤗 Model Zoo"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_PATH = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
_ROW = re.compile(
    r'<tr>\s*<td>(?P<name>.*?)</td>\s*<td>\s*<a\s+href="'
    r'https://huggingface\.co/FALCON-VLA/FALCON-series/tree/main/'
    r'(?P<directory>[A-Za-z0-9_.-]+)/ckpts">(?P<label>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]*>", " ", value))).strip()


def _parse_model_zoo(document: str, *, maximum: int) -> tuple[tuple[str, str], ...]:
    start = document.find(_HEADING)
    if start < 0:
        raise ValueError("FALCON Model Zoo section is missing")
    end = document.find("## 🏋️ Training", start)
    section = document[start : end if end >= 0 else len(document)]
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _ROW.finditer(section):
        name = _clean(match.group("name"))
        directory = match.group("directory")
        if not name.startswith("FALCON-") or not _PATH.fullmatch(directory):
            raise ValueError("unexpected FALCON model zoo row")
        if directory in seen:
            raise ValueError(f"duplicate FALCON checkpoint directory: {directory}")
        seen.add(directory)
        rows.append((name, directory))
        if len(rows) > maximum:
            raise ValueError(f"FALCON model zoo exceeds {maximum} entries")
    if not rows:
        raise ValueError("FALCON Model Zoo has no checkpoint rows")
    return tuple(rows)


class FalconVLACheckpointZooAdapter:
    """Enumerate the official FALCON model table and resolve exact weight files."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only weight checkpoints named by FALCON-VLA/FALCON's Model Zoo. "
        "Required ESM/VLM components, configuration files, and checkpoints outside "
        "that table are not indexed; no weights are downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "falcon-vla-checkpoint-zoo",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 32,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        for value, label in (
            (max_response_bytes, "max_response_bytes"),
            (max_entries, "max_entries"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "falcon-vla-checkpoint-zoo-v1",
                "repository": _REPOSITORY,
                "document": "README.md#model-zoo",
                "hub_repository": _HF_REPOSITORY,
                "admission": "one exact checkpoints/*.pt weight per first-party model-zoo row",
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{_REPOSITORY}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        commit = self.client.get(
            f"https://api.github.com/repos/{_REPOSITORY}/commits/main",
            headers={"Accept": "application/vnd.github+json"},
        )
        self._require(commit, "GitHub commit endpoint")
        commit_payload = commit.json()
        document_revision = (
            commit_payload.get("sha", "") if isinstance(commit_payload, Mapping) else ""
        )
        if not isinstance(document_revision, str) or not _SHA.fullmatch(document_revision):
            raise ValueError(f"{self.name}: missing full GitHub document revision")
        document_url = (
            f"https://raw.githubusercontent.com/{_REPOSITORY}/{document_revision}/README.md"
        )
        document = self.client.get(
            document_url, headers={"Accept": "text/markdown,text/plain"}
        )
        self._require(document, "pinned FALCON README")
        entries = _parse_model_zoo(document.text(), maximum=self.max_entries)

        info_response = self.client.get(
            f"https://huggingface.co/api/models/{_HF_REPOSITORY}",
            headers={"Accept": "application/json"},
        )
        self._require(info_response, "Hub repository metadata")
        info = info_response.json()
        revision = info.get("sha", "") if isinstance(info, Mapping) else ""
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: missing full Hub revision")
        tree_response = self.client.get(
            f"https://huggingface.co/api/models/{_HF_REPOSITORY}/tree/{revision}",
            params={"recursive": "true", "expand": "true"},
            headers={"Accept": "application/json"},
        )
        self._require(tree_response, "Hub file listing")
        tree = tree_response.json()
        if not isinstance(tree, list):
            raise ValueError(f"{self.name}: Hub file listing is malformed")

        records: list[SourceRecord] = []
        used_paths: set[str] = set()
        for name, directory in entries:
            prefix = f"{directory}/ckpts/"
            weights = [
                row for row in tree
                if isinstance(row, Mapping)
                and row.get("type") == "file"
                and isinstance(row.get("path"), str)
                and row["path"].startswith(prefix)
                and row["path"].endswith(".pt")
            ]
            if len(weights) != 1:
                raise ValueError(f"{self.name}: expected one checkpoint .pt file in {prefix}")
            weight = weights[0]
            path, oid, size = weight.get("path"), weight.get("oid"), weight.get("size")
            if not isinstance(path, str) or not _PATH.fullmatch(path) or path in used_paths:
                raise ValueError(f"{self.name}: invalid or duplicate weight path {path!r}")
            if not isinstance(oid, str) or not oid:
                raise ValueError(f"{self.name}: checkpoint has no object id: {path}")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"{self.name}: checkpoint has invalid size: {path}")
            used_paths.add(path)
            records.append(
                self._record(name, path, revision, oid, size, document_revision,
                             document.body, tree_response.body, document_url)
            )

        checked_at = self.clock()
        if checked_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return SourcePage(
            records=tuple(records),
            next_state={
                "checked_at": checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "document_revision": document_revision,
                "document_sha256": content_hash(document.body),
                "hub_revision": revision,
                "listing_sha256": content_hash(tree_response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self, name: str, path: str, revision: str, oid: str, size: int,
        document_revision: str, document_body: bytes, tree_body: bytes,
        document_url: str,
    ) -> SourceRecord:
        model_key = name.casefold()
        local_id = f"model:{model_key}"
        repo = _HF_REPOSITORY
        artifact = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(path, safe='/')}"
        identifier = Identifier("falcon:model", name)
        locator = f"README.md#model-zoo:{name}"
        metadata = {
            "hub_repository": repo,
            "revision": revision,
            "weight_path": path,
            "weight_oid": oid,
            "weight_size_bytes": size,
            "model_zoo_document_revision": document_revision,
            "model_zoo_document_sha256": content_hash(document_body),
            "hub_tree_sha256": content_hash(tree_body),
        }
        model = ModelHint(
            local_id=local_id,
            name=name,
            identifiers=(identifier,),
            aliases=(name,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{model_key}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("falcon:checkpoint", f"{name}@{revision}:{path}"),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"falcon-vla:{model_key}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact),
            title=name,
            raw=metadata,
            text=f"{name}; official FALCON model zoo checkpoint {path}",
            identifiers=(identifier,),
            links=(
                Link(artifact, relation="model_artifact", locator=path, crawl=False,
                     model_local_ids=(local_id,)),
                Link(f"https://huggingface.co/{repo}", relation="model_repository", crawl=False),
                Link(document_url, relation="model_catalog", locator=locator, crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )

    def _require(self, response: HttpResponse, label: str) -> None:
        if response.status != 200:
            raise ValueError(f"{self.name}: {label} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: {label} exceeds {self.max_response_bytes} bytes")
