"""First-party Galaxea G0.5 and legacy VLA checkpoint inventory."""

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

_REPOSITORY = "OpenGalaxea/GalaxeaVLA"
_LEGACY_REPOSITORY = "OpenGalaxea/G0-VLA"
_LEGACY_REVISION = "13a16a9049aee8f1d799b56fccc0c5832a75fc2f"
_DOCUMENT = "README.md"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_CHECKPOINT_URL = re.compile(
    r"^https://huggingface\.co/OpenGalaxea/G05/tree/main/(?P<slug>[a-z0-9][a-z0-9-]*)$"
)
_LEGACY_URL = re.compile(
    r"^https://huggingface\.co/OpenGalaxea/G0-VLA/(?P<mode>blob|tree)/main/(?P<path>[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*)$"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GalaxeaVLACheckpointSourceAdapter:
    """Enumerate current G0.5 and pinned legacy checkpoint refs."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers exact checkpoint refs listed in the Model Checkpoints tables "
        "of the current README and its pinned legacy "
        "README revision. It does not enumerate individual files within Hub "
        "folders. Hub access requires accepting applicable model terms."
    )

    def __init__(
        self,
        *,
        name: str = "galaxea-g05-checkpoints",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 50,
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
                "adapter": "galaxea-g05-checkpoints-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "exact checkpoint links in current and pinned legacy README tables",
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

        entries = _parse_checkpoints(response.text(), source=self.name, maximum=self.max_entries)
        if not entries:
            raise ValueError(f"{self.name}: Model Checkpoints table contains no G0.5 entries")
        legacy_document_url = (
            f"https://raw.githubusercontent.com/{_REPOSITORY}/"
            f"{_LEGACY_REVISION}/{quote(_DOCUMENT, safe='/')}"
        )
        legacy_response: HttpResponse = self.client.get(
            legacy_document_url,
            headers={"Accept": "text/markdown,text/plain"},
        )
        if legacy_response.status != 200:
            raise ValueError(
                f"{self.name}: pinned legacy README returned HTTP {legacy_response.status}"
            )
        if len(legacy_response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: legacy README exceeds {self.max_response_bytes} bytes")
        legacy_entries = _parse_legacy_checkpoints(
            legacy_response.text(), source=self.name, maximum=self.max_entries
        )
        all_entries = tuple(("g05", entry) for entry in entries) + tuple(
            ("legacy", entry) for entry in legacy_entries
        )
        if len(all_entries) > self.max_entries:
            raise ValueError(
                f"{self.name}: combined checkpoint inventory exceeds {self.max_entries}"
            )
        records = tuple(
            self._record(entry, revision, response.body, document_url) for entry in entries
        ) + tuple(
            self._record_legacy(
                entry,
                _LEGACY_REVISION,
                legacy_response.body,
                legacy_document_url,
            )
            for entry in legacy_entries
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
                "legacy_revision": _LEGACY_REVISION,
                "legacy_document_url": legacy_document_url,
                "legacy_document_sha256": content_hash(legacy_response.body),
                "document_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        entry: tuple[str, str, str, str, int],
        revision: str,
        document: bytes,
        document_url: str,
    ) -> SourceRecord:
        name, use_case, checkpoint_path, artifact_url, line = entry
        match = _CHECKPOINT_URL.fullmatch(artifact_url)
        assert match is not None
        slug = match.group("slug")
        local_id = f"model:{slug}"
        locator = f"{_DOCUMENT}:line:{line}"
        identifier = Identifier("galaxea:g05-checkpoint", slug)
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
            "checkpoint_folder": slug,
            "checkpoint_path": checkpoint_path,
            "use_case": use_case,
            "hub_repo": "OpenGalaxea/G05",
            "requires_model_terms_acceptance": True,
        }
        release = ReleaseHint(
            local_id=f"release:{slug}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("galaxea:g05-release", slug),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"galaxea:g05:{slug}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"Galaxea G0.5 {name}",
            raw=metadata | {"source_document_sha256": content_hash(document)},
            text=f"{name}: {use_case}\nCheckpoint path: {checkpoint_path}",
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
                Link(
                    "https://huggingface.co/OpenGalaxea/G05",
                    relation="source_model_repository",
                    crawl=False,
                ),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )

    def _record_legacy(
        self,
        entry: tuple[str, str, str, str, int],
        revision: str,
        document: bytes,
        document_url: str,
    ) -> SourceRecord:
        name, use_case, description, artifact_url, line = entry
        match = _LEGACY_URL.fullmatch(artifact_url)
        assert match is not None
        hub_path = match.group("path")
        slug = hub_path.split("/")[-1]
        local_id = f"model:{hub_path}"
        locator = f"{_DOCUMENT}@{revision}:line:{line}"
        identifier = Identifier("galaxea:g0-checkpoint", hub_path)
        model = ModelHint(
            local_id=local_id,
            name=name,
            identifiers=(identifier,),
            aliases=(hub_path,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        metadata = {
            "repository": _LEGACY_REPOSITORY,
            "revision": revision,
            "document_path": _DOCUMENT,
            "checkpoint_name": name,
            "checkpoint_path": hub_path,
            "checkpoint_form": "file" if match.group("mode") == "blob" else "folder",
            "use_case": use_case,
            "description": description,
            "hub_repo": "OpenGalaxea/G0-VLA",
            "source_document_sha256": content_hash(document),
        }
        release = ReleaseHint(
            local_id=f"release:{hub_path}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("galaxea:g0-release", hub_path),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"galaxea:g0:{slug}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"Galaxea {name}",
            raw=metadata,
            text=f"{name}: {use_case}. {description}",
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
                Link(
                    f"https://huggingface.co/{_LEGACY_REPOSITORY}",
                    relation="source_model_repository",
                    crawl=False,
                ),
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
) -> tuple[tuple[str, str, str, str, int], ...]:
    active = False
    section_level: int | None = None
    entries: dict[str, tuple[str, str, str, str, int]] = {}
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            level = len(heading.group("level"))
            title = heading.group("title").strip().casefold()
            if active and section_level is not None and level <= section_level:
                active = False
            if title == "model checkpoints":
                active = True
                section_level = level
            continue
        if not active or not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        links = tuple(_LINK.finditer(cells[0]))
        asset_links = [link for link in links if _CHECKPOINT_URL.fullmatch(link.group("url"))]
        if not asset_links:
            continue
        if len(asset_links) != 1:
            raise ValueError(f"{source}: expected one checkpoint link on line {line_number}")
        link = asset_links[0]
        label = link.group("label").strip()
        artifact_url = link.group("url")
        checkpoint_path = cells[-1].strip("` \t")
        use_case = cells[1].strip("` \t")
        candidate = (label, use_case, checkpoint_path, artifact_url, line_number)
        if artifact_url in entries and entries[artifact_url][:4] != candidate[:4]:
            raise ValueError(f"{source}: conflicting checkpoint rows for {artifact_url}")
        entries[artifact_url] = candidate
        if len(entries) > maximum:
            raise ValueError(f"{source}: checkpoint table exceeds {maximum} entries")
    return tuple(entries.values())


def _parse_legacy_checkpoints(
    document: str,
    *,
    source: str,
    maximum: int,
) -> tuple[tuple[str, str, str, str, int], ...]:
    active = False
    section_level: int | None = None
    entries: dict[str, tuple[str, str, str, str, int]] = {}
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            level = len(heading.group("level"))
            title = heading.group("title").strip().casefold()
            if active and section_level is not None and level <= section_level:
                active = False
            if title == "model checkpoints":
                active = True
                section_level = level
            continue
        if not active or "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        refs = [
            link for link in _LINK.finditer(cells[3]) if _LEGACY_URL.fullmatch(link.group("url"))
        ]
        if not refs:
            continue
        if len(refs) != 1:
            raise ValueError(f"{source}: expected one legacy checkpoint ref on line {line_number}")
        url = refs[0].group("url")
        match = _LEGACY_URL.fullmatch(url)
        assert match is not None
        name = cells[0].strip(" `*_")
        candidate = (
            name,
            cells[1].strip(" `*_	"),
            cells[2].strip(" `*_	"),
            url,
            line_number,
        )
        if url in entries and entries[url][:4] != candidate[:4]:
            raise ValueError(f"{source}: conflicting legacy checkpoint row for {url}")
        entries[url] = candidate
        if len(entries) > maximum:
            raise ValueError(f"{source}: legacy checkpoint table exceeds {maximum} entries")
    return tuple(entries.values())
