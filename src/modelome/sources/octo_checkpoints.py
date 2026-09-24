"""First-party Octo checkpoint inventory."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
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

_REPOSITORY = "octo-models/octo"
_DOCUMENT = "README.md"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_MODEL_URL = re.compile(
    r"^https://huggingface\.co/rail-berkeley/(?P<slug>octo-(?:base|small))$"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OctoCheckpointSourceAdapter:
    """Enumerate the Octo-Base and Octo-Small refs in the official README table."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only Octo-Base and Octo-Small in the official repository's "
        "Checkpoints table; it does not include community fine-tunes or other "
        "Octo versions."
    )

    def __init__(
        self,
        *,
        name: str = "octo-checkpoints",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 10,
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
                "adapter": "octo-checkpoints-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "exact HF links from the Checkpoints table",
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
            document_url, headers={"Accept": "text/markdown,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds {self.max_response_bytes} bytes")
        entries = _parse_checkpoints(response.text(), source=self.name, maximum=self.max_entries)
        if not entries:
            raise ValueError(f"{self.name}: Checkpoints table contains no Octo models")
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
        name, size, artifact_url, line = entry
        match = _MODEL_URL.fullmatch(artifact_url)
        assert match is not None
        slug = match.group("slug")
        local_id = f"model:{slug}"
        locator = f"{_DOCUMENT}:line:{line}"
        identifier = Identifier("huggingface:model", f"rail-berkeley/{slug}")
        model = ModelHint(
            local_id=local_id,
            name=name,
            identifiers=(identifier,),
            aliases=(slug,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        metadata = {
            "repository": _REPOSITORY,
            "revision": revision,
            "document_path": _DOCUMENT,
            "checkpoint_name": name,
            "checkpoint_repo": f"rail-berkeley/{slug}",
            "parameter_count": size,
            "source_document_sha256": content_hash(document),
        }
        release = ReleaseHint(
            local_id=f"release:{slug}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("octo:checkpoint", slug),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"octo:{slug}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"Octo {name}",
            raw=metadata,
            text=f"{name}; {size}",
            identifiers=(identifier,),
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


def _parse_checkpoints(
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
            level = len(heading.group("level"))
            title = heading.group("title").strip().casefold()
            if active and section_level is not None and level <= section_level:
                active = False
            if title == "checkpoints":
                active = True
                section_level = level
            continue
        if not active or "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        valid_links = [
            link for link in _LINK.finditer(cells[0]) if _MODEL_URL.fullmatch(link.group("url"))
        ]
        if not valid_links:
            continue
        if len(valid_links) != 1:
            raise ValueError(f"{source}: expected one Octo checkpoint link on line {line_number}")
        link = valid_links[0]
        entry = (link.group("label").strip(), cells[2].strip(), link.group("url"), line_number)
        entries[entry[2]] = entry
        if len(entries) > maximum:
            raise ValueError(f"{source}: checkpoint table exceeds {maximum} entries")
    return tuple(entries.values())
