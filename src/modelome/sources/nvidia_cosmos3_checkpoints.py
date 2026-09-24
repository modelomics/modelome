"""NVIDIA Cosmos 3 checkpoint refs from the first-party README tables."""

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
    ModelRelationHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_REPOSITORY = "NVIDIA/cosmos"
_DOCUMENT = "README.md"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_MODEL_URL = re.compile(r"^https://huggingface\.co/nvidia/(?P<slug>Cosmos3-[A-Za-z0-9-]+)$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_ALLOWED = frozenset(
    {
        "Cosmos3-Super",
        "Cosmos3-Nano",
        "Cosmos3-Edge",
        "Cosmos3-Super-Text2Image",
        "Cosmos3-Super-Text2Image-4Step",
        "Cosmos3-Super-Image2Video",
        "Cosmos3-Super-Image2Video-4Step",
        "Cosmos3-Nano-Policy-DROID",
        "Cosmos3-Edge-Policy-DROID",
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class NvidiaCosmos3CheckpointSourceAdapter:
    """Enumerate the base and example checkpoint tables in NVIDIA/cosmos."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the three Cosmos 3 base models and six example checkpoint refs in "
        "the official README Models tables; it does not enumerate older Cosmos "
        "families, gated dependencies, or community fine-tunes."
    )

    def __init__(
        self,
        *,
        name: str = "nvidia-cosmos3-checkpoints",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 12,
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
                "adapter": "nvidia-cosmos3-readme-checkpoints-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "models": sorted(_ALLOWED),
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{_REPOSITORY}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
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
            raise ValueError(f"{self.name}: README Models tables contain no Cosmos 3 models")
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
        category, slug, detail, base_slug, line = entry
        artifact_url = f"https://huggingface.co/nvidia/{slug}"
        local_id = f"model:{slug.casefold()}"
        locator = f"{_DOCUMENT}:line:{line}"
        identifier = Identifier("huggingface:model", f"nvidia/{slug}")
        model = ModelHint(
            local_id=local_id,
            name=slug,
            identifiers=(identifier,),
            aliases=(slug,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        metadata = {
            "repository": _REPOSITORY,
            "revision": revision,
            "document_path": _DOCUMENT,
            "checkpoint_name": slug,
            "checkpoint_category": category,
            "checkpoint_detail": detail,
            "base_model": base_slug or None,
            "source_document_sha256": content_hash(document),
        }
        release = ReleaseHint(
            local_id=f"release:{slug.casefold()}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("nvidia:cosmos3-checkpoint", slug),),
            metadata=metadata,
            locator=locator,
        )
        relations: tuple[ModelRelationHint, ...] = ()
        if base_slug:
            relations = (
                ModelRelationHint(
                    subject_local_id=local_id,
                    predicate="fine_tuned_from",
                    target=ModelHint(
                        local_id=f"model:{base_slug.casefold()}",
                        name=base_slug,
                        identifiers=(Identifier("huggingface:model", f"nvidia/{base_slug}"),),
                        status=ModelStatus.RELEASED,
                    ),
                    locator=locator,
                ),
            )
        return SourceRecord(
            source_record_id=f"nvidia-cosmos3:{slug.casefold()}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"NVIDIA {slug}",
            raw=metadata,
            text=f"{slug}; {category}; {detail}",
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
            model_relations=relations,
            releases=(release,),
        )


def _parse_checkpoints(
    document: str,
    *,
    source: str,
    maximum: int,
) -> tuple[tuple[str, str, str, str, int], ...]:
    in_models = False
    section_level: int | None = None
    category = ""
    entries: dict[str, tuple[str, str, str, str, int]] = {}
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            level = len(heading.group("level"))
            title = heading.group("title").strip().casefold()
            if in_models and section_level is not None and level <= section_level:
                in_models = False
                category = ""
            if title == "models":
                in_models = True
                section_level = level
            continue
        if not in_models:
            continue
        if "example checkpoint" in line.casefold():
            category = "example"
            continue
        if "base model" in line.casefold() and "|" in line:
            category = "base"
        if not category or "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        matches = [
            match
            for match in _LINK.finditer(cells[0])
            if (model := _MODEL_URL.fullmatch(match.group("url")))
            and model.group("slug") in _ALLOWED
        ]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f"{source}: expected one Cosmos 3 model ref on line {line_number}")
        model_match = _MODEL_URL.fullmatch(matches[0].group("url"))
        assert model_match is not None
        slug = model_match.group("slug")
        if category == "base":
            detail = " ".join(_plain(cell) for cell in cells[1:])
            base_slug = ""
        else:
            base_name = _plain(cells[1]).strip().casefold()
            base_slug = {
                "super": "Cosmos3-Super",
                "nano": "Cosmos3-Nano",
                "edge": "Cosmos3-Edge",
            }.get(base_name, "")
            if not base_slug:
                raise ValueError(f"{source}: unknown Cosmos 3 base model on line {line_number}")
            detail = " ".join(_plain(cell) for cell in cells[1:])
        if slug in entries:
            raise ValueError(f"{source}: duplicate checkpoint ref {slug}")
        entries[slug] = (category, slug, detail, base_slug, line_number)
        if len(entries) > maximum:
            raise ValueError(f"{source}: checkpoint inventory exceeds {maximum} entries")
    return tuple(entries.values())


def _plain(value: str) -> str:
    value = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[*_~`]", "", value)
    return " ".join(value.split())
