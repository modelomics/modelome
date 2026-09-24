"""First-party Pelican-VLA checkpoint repository index."""

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

_REPOSITORY = "Open-X-Humanoid/Pelican-VLA05"
_DOCUMENT = "readme.md"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_HANDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PelicanVLACheckpointRegistrySourceAdapter:
    """Read the two explicit Hugging Face checkpoint rows from the official README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only Hugging Face checkpoint repositories explicitly listed in "
        "the Model Download section of Open-X-Humanoid/Pelican-VLA05/README.md. "
        "It does not inventory files within those repositories, fetch weights, or "
        "include required third-party backbone/tokenizer assets."
    )

    def __init__(
        self,
        *,
        name: str = "pelican-vla-checkpoint-registry",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 20,
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
                "adapter": "pelican-vla-checkpoint-registry-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "exact X-Humanoid Hugging Face links in Model Download table",
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{_REPOSITORY}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response: HttpResponse = self.client.get(
            f"https://api.github.com/repos/{_REPOSITORY}/commits/main",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        payload = commit_response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a full commit SHA")

        document_url = (
            f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/"
            f"{quote(_DOCUMENT, safe='/')}"
        )
        response: HttpResponse = self.client.get(
            document_url,
            headers={"Accept": "text/markdown,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds {self.max_response_bytes} bytes")

        entries = _parse_registry(response.text(), source=self.name, maximum=self.max_entries)
        if not entries:
            raise ValueError(f"{self.name}: Model Download table contains no matching checkpoints")
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
        name, description, model_url, line = entry
        handle = "/".join(part for part in urlsplit(model_url).path.split("/") if part)
        local_id = f"model:{handle}"
        identifier = Identifier("huggingface:model", handle)
        locator = f"{_DOCUMENT}:line:{line}"
        model = ModelHint(
            local_id=local_id,
            name=name,
            identifiers=(identifier,),
            aliases=(handle,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        metadata = {
            "repository": _REPOSITORY,
            "revision": revision,
            "document_path": _DOCUMENT,
            "checkpoint_repository": handle,
            "description": description,
        }
        release = ReleaseHint(
            local_id=f"release:{handle}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("pelican-vla:checkpoint-repository", handle),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"pelican-vla:{handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(model_url),
            title=name,
            raw=metadata | {"source_document_sha256": content_hash(document)},
            text=f"{name}: {description}",
            identifiers=(identifier,),
            links=(
                Link(model_url, relation="model_card", locator=locator, crawl=False),
                Link(document_url, relation="model_catalog", locator=locator, crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_registry(
    document: str,
    *,
    source: str,
    maximum: int,
) -> tuple[tuple[str, str, str, int], ...]:
    active = False
    section_level: int | None = None
    entries: dict[str, tuple[str, str, str, int]] = {}
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            title = heading.group("title").strip().casefold()
            level = len(heading.group("level"))
            if active and section_level is not None and level <= section_level:
                active = False
            if title == "model download":
                active = True
                section_level = level
            continue
        if not active or not line.lstrip().startswith("|"):
            continue
        links = tuple(_LINK.finditer(line))
        if not links:
            continue
        model_links = [
            link for link in links
            if link.group("url").startswith("https://huggingface.co/X-Humanoid/")
        ]
        if not model_links:
            continue
        if len(model_links) != 1:
            raise ValueError(f"{source}: expected one checkpoint repository on line {line_number}")
        link = model_links[0]
        url = link.group("url").rstrip("/")
        handle = "/".join(part for part in urlsplit(url).path.split("/") if part)
        if urlsplit(url).query or not _HANDLE.fullmatch(handle):
            raise ValueError(f"{source}: invalid checkpoint repository URL on line {line_number}")
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            raise ValueError(f"{source}: malformed checkpoint row on line {line_number}")
        name = cells[0].strip(" `*_\t")
        description = cells[2].strip(" `*_\t")
        candidate = (name, description, url, line_number)
        if url in entries and entries[url][:3] != candidate[:3]:
            raise ValueError(f"{source}: conflicting metadata for checkpoint repository {url}")
        entries[url] = candidate
        if len(entries) > maximum:
            raise ValueError(f"{source}: checkpoint table exceeds {maximum} entries")
    return tuple(entries.values())
