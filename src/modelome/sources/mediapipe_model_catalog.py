"""First-party MediaPipe legacy TFLite model bundle inventory."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

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

_REPOSITORY = "google-ai-edge/mediapipe"
_DOCUMENT = "docs/solutions/models.md"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_SAFE_ID = re.compile(r"[^a-z0-9]+")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MediaPipeModelCatalogSourceAdapter:
    """Enumerate exact binary links declared in Google's MediaPipe model list.

    The upstream document is explicitly for legacy solutions, whose support
    ended in 2023. This adapter catalogs only exact `.tflite`/`.task` links in
    that first-party Markdown file; it does not infer models from docs or
    download binary contents.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only exact .tflite and .task URLs listed in "
        "google-ai-edge/mediapipe/docs/solutions/models.md at the observed Git "
        "revision. The document describes legacy MediaPipe Solutions, whose "
        "support ended in 2023; newer task bundles are not comprehensively listed."
    )

    def __init__(
        self,
        *,
        name: str = "google-mediapipe-model-catalog",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 200,
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
                "adapter": "mediapipe-model-catalog-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "exact .tflite/.task links in first-party model-list Markdown",
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{_REPOSITORY}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response: HttpResponse = self.client.get(
            f"https://api.github.com/repos/{_REPOSITORY}/commits/master",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        payload = commit_response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")

        document_url = (
            f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/"
            f"{quote(_DOCUMENT, safe='/')}"
        )
        response: HttpResponse = self.client.get(
            document_url,
            headers={"Accept": "text/markdown,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: model list returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: model list exceeds {self.max_response_bytes} bytes")

        entries = _parse_entries(response.text(), maximum=self.max_entries, source=self.name)
        if not entries:
            raise ValueError(f"{self.name}: model list contains no .tflite or .task links")
        records = tuple(
            self._record(entry, revision, response.body, document_url) for entry in entries
        )
        checked_at = self.clock()
        if checked_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "document_url": document_url,
                "document_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        entry: tuple[str, str, str, int],
        revision: str,
        document: bytes,
        document_url: str,
    ) -> SourceRecord:
        family, label, artifact_url, line = entry
        filename = urlsplit(artifact_url).path.rsplit("/", 1)[-1]
        slug = _SAFE_ID.sub("-", filename.casefold()).strip("-")
        local_id = f"model:{slug}"
        identity = Identifier("mediapipe:artifact", artifact_url)
        locator = f"{_DOCUMENT}:line:{line}"
        model = ModelHint(
            local_id=local_id,
            name=f"MediaPipe {family}: {label}",
            identifiers=(identity,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        metadata = {
            "repository": _REPOSITORY,
            "revision": revision,
            "document_path": _DOCUMENT,
            "family": family,
            "upstream_label": label,
            "artifact_url": artifact_url,
            "filename": filename,
        }
        release = ReleaseHint(
            local_id=f"release:{slug}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("mediapipe:release", artifact_url),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"mediapipe:{slug}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"MediaPipe {family}: {label}",
            raw=metadata | {"document_sha256": content_hash(document)},
            text=f"{family} — {label}",
            identifiers=(identity,),
            links=(
                Link(
                    artifact_url,
                    relation="model_artifact",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
                Link(document_url, relation="model_catalog", locator=locator, crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_entries(
    document: str, *, maximum: int, source: str
) -> tuple[tuple[str, str, str, int], ...]:
    family = ""
    entries: dict[str, tuple[str, str, str, int]] = {}
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            title = heading.group(1).strip()
            family = re.sub(r"\]\(https?://[^)]+\)$", "", title).lstrip("[").strip()
            continue
        for link in _LINK.finditer(line):
            label, url = link.group("label").strip(), link.group("url")
            path = urlsplit(url).path.casefold()
            if not path.endswith((".tflite", ".task")):
                continue
            if urlsplit(url).scheme != "https" or not family:
                raise ValueError(f"{source}: invalid asset link on line {line_number}")
            candidate = (family, label, url, line_number)
            previous = entries.get(url)
            if previous is not None and previous[:3] != candidate[:3]:
                raise ValueError(f"{source}: duplicate artifact URL has conflicting labels: {url}")
            entries[url] = candidate
            if len(entries) > maximum:
                raise ValueError(f"{source}: model list exceeds {maximum} assets")
    return tuple(entries.values())
