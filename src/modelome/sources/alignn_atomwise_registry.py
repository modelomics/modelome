"""ALIGNN's first-party atomwise checkpoint registry."""

from __future__ import annotations

import json
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
_REPOSITORY = "usnistgov/alignn"
_BRANCH = "main"
_PATH = "alignn/ff/all_models_alignn_atomwise.json"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_FILE_URL = re.compile(r"^https://figshare\.com/ndownloader/files/[0-9]+$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AlignnAtomwiseRegistrySourceAdapter:
    """Read exact ALIGNN handle-to-Figshare-file rows at the current Git revision."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only literal checkpoint handles in ALIGNN's first-party atomwise JSON map. "
        "Each Figshare file ID is exact; filenames and model bytes are not inferred or fetched."
    )

    def __init__(
        self,
        *,
        name: str = "alignn-atomwise-checkpoints",
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 4 * 1024 * 1024,
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
                "adapter": "alignn-atomwise-registry-v1",
                "repository": _REPOSITORY,
                "branch": _BRANCH,
                "path": _PATH,
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
            return SourcePage(
                (), next_state, True, upstream_count=state.get("model_count")
            )

        response = self.client.get(self._raw_url(revision), headers={"Accept": "application/json"})
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: registry exceeds {self.max_response_bytes} bytes")
        registry = _parse_registry(response.text(), self.name, self.max_entries)
        records = tuple(
            self._record(handle, url, revision, response.body) for handle, url in registry
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

    def _record(self, handle: str, url: str, revision: str, body: bytes) -> SourceRecord:
        model_id = f"model:{handle}"
        identity = Identifier("alignn:atomwise-checkpoint", handle)
        source_url = self._blob_url(revision)
        model = ModelHint(
            model_id,
            f"ALIGNN atomwise checkpoint {handle}",
            aliases=(handle,),
            identifiers=(identity,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            revision=revision,
            identifiers=(Identifier("alignn:atomwise-checkpoint:release", handle),),
            metadata={
                "repository": _REPOSITORY,
                "revision": revision,
                "source_path": _PATH,
                "checkpoint_handle": handle,
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
            raw={"checkpoint_handle": handle, "checkpoint_url": url},
            text=f"ALIGNN atomwise checkpoint handle: {handle}",
            identifiers=(identity,),
            links=(
                Link(source_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(f"https://github.com/{_REPOSITORY}", "source_implementation", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_registry(document: str, source: str, maximum: int) -> tuple[tuple[str, str], ...]:
    try:
        payload = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: registry is not valid JSON") from error
    if not isinstance(payload, dict) or not payload or len(payload) > maximum:
        raise ValueError(f"{source}: registry must be a non-empty bounded JSON object")
    rows = []
    for handle, url in payload.items():
        if not isinstance(handle, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", handle):
            raise ValueError(f"{source}: invalid checkpoint handle {handle!r}")
        if not isinstance(url, str) or not _FILE_URL.fullmatch(url):
            raise ValueError(f"{source}: invalid Figshare checkpoint URL for {handle!r}")
        if urlsplit(url).hostname != "figshare.com":
            raise ValueError(f"{source}: checkpoint URL host must be figshare.com")
        rows.append((handle, url))
    return tuple(rows)


__all__ = ["AlignnAtomwiseRegistrySourceAdapter"]
