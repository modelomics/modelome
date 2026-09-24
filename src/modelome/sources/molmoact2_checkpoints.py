"""First-party MolmoAct2 policy checkpoints listed in its official README."""

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

_REPOSITORY = "allenai/molmoact2"
_DOCUMENT = "README.md"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_CHECKPOINTS = frozenset(
    {
        "MolmoAct2",
        "MolmoAct2-Think",
        "MolmoAct2-Pretrain",
        "MolmoAct2-DROID",
        "MolmoAct2-BimanualYAM",
        "MolmoAct2-SO100_101",
        "MolmoAct2-LIBERO",
        "MolmoAct2-Think-LIBERO",
    }
)
_MODEL_URL = re.compile(r"^https://huggingface\.co/allenai/(?P<slug>[^/?#]+)$")
_SECTIONS = {"base models", "finetuned models"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MolmoAct2CheckpointSourceAdapter:
    """Enumerate only direct MolmoAct2 policy refs in the official README tables."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the eight MolmoAct2 action-policy checkpoint refs in the official "
        "README's Base Models and Finetuned Models tables. It excludes the separately "
        "listed Molmo2-ER VLM backbone and tokenizer, and does not enumerate third-party "
        "fine-tunes or files within model repositories."
    )

    def __init__(
        self,
        *,
        name: str = "molmoact2-checkpoints",
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
                "adapter": "molmoact2-readme-checkpoints-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "checkpoints": sorted(_CHECKPOINTS),
                "admission": sorted(_SECTIONS),
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
            raise ValueError(f"{self.name}: README contains no listed MolmoAct2 checkpoints")
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
        slug, section, use_case, description, line = entry
        model_id = f"allenai/{slug}"
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
            "declared_section": section,
            "use_case": use_case,
            "description": description,
            "source_document_sha256": content_hash(document),
        }
        model = ModelHint(
            local_id=local_id,
            name=slug,
            identifiers=(identifier,),
            aliases=(slug,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{slug.casefold()}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("allenai:molmoact2-checkpoint", slug),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"molmoact2:{slug.casefold()}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"Ai2 {slug}",
            raw=metadata,
            text=f"{slug}; {use_case}; {description}",
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
    active: str | None = None
    active_level: int | None = None
    entries: dict[str, tuple[str, str, str, str, int]] = {}
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            level = len(heading.group("level"))
            title = heading.group("title").strip().casefold()
            if active_level is not None and level <= active_level:
                active = None
                active_level = None
            if title in _SECTIONS:
                active = title
                active_level = level
            continue
        if active is None or not line.lstrip().startswith("|"):
            continue
        links = tuple(_LINK.finditer(line))
        for link in links:
            url = link.group("url").rstrip("/")
            match = _MODEL_URL.fullmatch(url)
            if match is None:
                continue
            slug = match.group("slug")
            if slug not in _CHECKPOINTS or link.group("label") != url:
                continue
            if urlsplit(url).query or urlsplit(url).fragment:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 4:
                raise ValueError(f"{source}: malformed checkpoint row on line {line_number}")
            candidate = (
                slug,
                active,
                cells[1],
                " ".join(cells[2].split()),
                line_number,
            )
            if slug in entries:
                if entries[slug][:4] != candidate[:4]:
                    raise ValueError(f"{source}: conflicting metadata for {slug}")
                continue
            entries[slug] = candidate
            if len(entries) > maximum:
                raise ValueError(f"{source}: checkpoint table exceeds {maximum} entries")
    return tuple(entries.values())
