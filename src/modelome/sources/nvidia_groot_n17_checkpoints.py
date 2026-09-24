"""First-party NVIDIA GR00T N1.7 checkpoints from the official README table."""

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

_REPOSITORY = "NVIDIA/Isaac-GR00T"
_DOCUMENT = "README.md"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_TABLE_MODELS = frozenset(
    {
        "GR00T-N1.7-3B",
        "GR00T-N1.7-LIBERO",
        "GR00T-N1.7-DROID",
        "GR00T-N1.7-SimplerEnv-Bridge",
        "GR00T-N1.7-SimplerEnv-Fractal",
    }
)
_MODEL_URL = re.compile(r"^https://huggingface\.co/nvidia/(?P<slug>GR00T-N1\.7-[A-Za-z0-9-]+)$")
_TABLE_HEADING = "model checkpoints & embodiment tags"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class NvidiaGR00TN17CheckpointSourceAdapter:
    """Enumerate only the five exact N1.7 refs in NVIDIA's checkpoint table."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the five direct Hugging Face model refs in the official Isaac-GR00T "
        "README's N1.7 Checkpoints table. It does not enumerate previous N1.6/N1.5 "
        "releases, user fine-tunes, or gated backbone dependencies."
    )

    def __init__(
        self,
        *,
        name: str = "nvidia-groot-n17-checkpoints",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 8,
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
                "adapter": "nvidia-groot-n17-readme-checkpoints-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "models": sorted(_TABLE_MODELS),
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
            raise ValueError(f"{self.name}: README contains no N1.7 checkpoints")
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
        entry: tuple[str, str, str, str, int],
        revision: str,
        document: bytes,
        document_url: str,
    ) -> SourceRecord:
        slug, checkpoint_type, embodiment_tag, description, line = entry
        model_id = f"nvidia/{slug}"
        artifact_url = f"https://huggingface.co/{model_id}"
        local_id = f"model:{slug.casefold()}"
        locator = f"{_DOCUMENT}:line:{line}"
        identifier = Identifier("huggingface:model", model_id)
        metadata = {
            "repository": _REPOSITORY,
            "revision": revision,
            "document_path": _DOCUMENT,
            "checkpoint_name": slug,
            "checkpoint_repo": model_id,
            "checkpoint_type": checkpoint_type,
            "embodiment_tag": embodiment_tag,
            "description": description,
            "source_document_sha256": content_hash(document),
        }
        model = ModelHint(
            local_id=local_id,
            name=f"NVIDIA GR00T {slug.removeprefix('GR00T-')}",
            identifiers=(identifier,),
            aliases=(slug,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{slug.casefold()}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("nvidia:groot-checkpoint", slug),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"nvidia-groot:{slug.casefold()}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"NVIDIA GR00T {slug.removeprefix('GR00T-')}",
            raw=metadata,
            text=f"{slug}; {checkpoint_type}; {embodiment_tag}; {description}",
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
) -> tuple[tuple[str, str, str, str, int], ...]:
    in_section = False
    active_level: int | None = None
    entries: list[tuple[str, str, str, str, int]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            level = len(heading.group("level"))
            title = heading.group("title").strip().casefold()
            if active_level is not None and level <= active_level:
                in_section = False
                active_level = None
            if title == _TABLE_HEADING:
                in_section = True
                active_level = level
            continue
        if not in_section or not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4 or cells[0].casefold() in {"checkpoint", "---"}:
            continue
        links = tuple(_LINK.finditer(cells[0]))
        for link in links:
            url = link.group("url")
            match = _MODEL_URL.fullmatch(url)
            if match is None:
                continue
            slug = match.group("slug")
            label = link.group("label").strip().strip("`")
            if slug not in _TABLE_MODELS or label != f"nvidia/{slug}":
                continue
            if slug in seen:
                continue
            checkpoint_type = cells[1].strip(" `")
            embodiment_tag = _LINK.sub(lambda item: item.group("label"), cells[2]).strip(" `")
            description = " ".join(_LINK.sub(lambda item: item.group("label"), cells[3]).split())
            entries.append((slug, checkpoint_type, embodiment_tag, description, line_number))
            seen.add(slug)
            if len(entries) > maximum:
                raise ValueError(f"{source}: checkpoint list exceeds {maximum} entries")
    return tuple(entries)
