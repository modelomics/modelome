"""Zatom-1 checkpoint URLs documented by its first-party repository."""

from __future__ import annotations

import re
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
_REPOSITORY = "Zatom-AI/zatom"
_PATH = "README.md"
_BRANCH = "main"
_ZENODO_RECORD = "19766997"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_CHECKPOINT_LINE = re.compile(
    r"^\s*wget\s+-P\s+checkpoints/\s+"
    r"(https://zenodo\.org/records/19766997/files/([A-Za-z0-9_.-]+\.ckpt))\s*$"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ZatomCheckpointRegistrySourceAdapter:
    """Index literal Zatom checkpoint downloads from the official README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only .ckpt download commands in Zatom-AI/zatom's Checkpoints README section. "
        "Other files and Zenodo records are excluded; checkpoint binaries are never fetched."
    )

    def __init__(
        self,
        *,
        name: str = "zatom-checkpoints",
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 100,
    ) -> None:
        if not name.strip() or max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("name and positive response/entry limits are required")
        self.name = name
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.checkpoint_signature = content_hash(
            {
                "adapter": "zatom-checkpoint-registry-v1",
                "repository": _REPOSITORY,
                "branch": _BRANCH,
                "source_path": _PATH,
                "zenodo_record": _ZENODO_RECORD,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
            }
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/commits/{_BRANCH}"

    def _raw_url(self, revision: str) -> str:
        return f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/{_PATH}"

    def _blob_url(self, revision: str) -> str:
        return f"https://github.com/{_REPOSITORY}/blob/{revision}/{_PATH}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(self.commit_url, headers={"Accept": "application/vnd.github+json"})
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage((), next_state, True, upstream_count=state.get("model_count"))

        response = self.client.get(self._raw_url(revision), headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds {self.max_response_bytes} bytes")
        checkpoints = _parse_readme(response.text(), self.name, self.max_entries)
        records = tuple(
            self._record(filename, url, revision, response.body)
            for filename, url in checkpoints
        )
        return SourcePage(
            records,
            {
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": self._raw_url(revision),
                "source_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, filename: str, url: str, revision: str, body: bytes) -> SourceRecord:
        model_id = f"model:{filename.removesuffix('.ckpt')}"
        handle = filename.removesuffix(".ckpt")
        identity = Identifier("zatom:checkpoint", handle)
        source_url = self._blob_url(revision)
        model = ModelHint(
            model_id,
            f"Zatom-1 {handle.replace('_', ' ')}",
            aliases=(filename,),
            identifiers=(identity,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            revision=revision,
            identifiers=(Identifier("zatom:checkpoint:release", handle),),
            metadata={
                "repository": _REPOSITORY,
                "revision": revision,
                "source_path": _PATH,
                "checkpoint_filename": filename,
                "checkpoint_url": url,
                "source_sha256": content_hash(body),
                "binary_reachability_checked": False,
            },
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{handle}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=model.name,
            raw={
                "checkpoint_filename": filename,
                "checkpoint_url": url,
                "repository": _REPOSITORY,
                "revision": revision,
            },
            text=f"Zatom-1 documented checkpoint: {filename}",
            identifiers=(identity,),
            links=(
                Link(source_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(f"https://github.com/{_REPOSITORY}", "source_implementation", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_readme(document: str, source: str, maximum: int) -> tuple[tuple[str, str], ...]:
    section = re.search(r"^### Checkpoints\s*$([\s\S]*?)(?=^## Training\s*$)", document, re.M)
    if section is None:
        raise ValueError(f"{source}: README has no bounded Checkpoints section")
    rows: list[tuple[str, str]] = []
    names: set[str] = set()
    for line in section.group(1).splitlines():
        if "zenodo.org/records/" not in line and "wget" not in line:
            continue
        match = _CHECKPOINT_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"{source}: unsupported or ambiguous checkpoint command")
        url, filename = match.groups()
        if urlsplit(url).hostname != "zenodo.org" or filename in names:
            raise ValueError(f"{source}: duplicate or invalid checkpoint URL")
        names.add(filename)
        rows.append((filename, canonicalize_url(url)))
        if len(rows) > maximum:
            raise ValueError(f"{source}: checkpoint list exceeds {maximum} entries")
    if not rows:
        raise ValueError(f"{source}: Checkpoints section contains no .ckpt downloads")
    return tuple(rows)


__all__ = ["ZatomCheckpointRegistrySourceAdapter"]
