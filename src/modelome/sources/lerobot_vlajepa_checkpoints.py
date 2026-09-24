"""LeRobot VLA-JEPA checkpoint table from its first-party guide."""

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

_REPOSITORY = "huggingface/lerobot"
_DOCUMENT = "docs/source/vla_jepa.mdx"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_CHECKPOINT = re.compile(r"^lerobot/(?P<slug>VLA-JEPA-[A-Za-z0-9-]+)$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class LeRobotVLAJEPACheckpointSourceAdapter:
    """Enumerate only exact model refs in the official Pretrained Checkpoints table."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the three LeRobot VLA-JEPA checkpoint refs and descriptive dataset, "
        "camera, world-model, and action-dimension cells in the official guide table. "
        "It does not infer dataset repository IDs or enumerate user fine-tunes."
    )

    def __init__(
        self,
        *,
        name: str = "lerobot-vlajepa-checkpoints",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 5,
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
                "adapter": "lerobot-vlajepa-checkpoints-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "exact IDs in Pretrained Checkpoints table",
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
            raise ValueError(f"{self.name}: source document returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: source document exceeds {self.max_response_bytes} bytes"
            )
        entries = _parse_checkpoints(response.text(), source=self.name, maximum=self.max_entries)
        if not entries:
            raise ValueError(f"{self.name}: no VLA-JEPA checkpoint refs found")
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
        entry: tuple[str, str, str, str, str, int],
        revision: str,
        document: bytes,
        document_url: str,
    ) -> SourceRecord:
        repo, dataset, cameras, world_model, action_dim, line = entry
        slug = repo.split("/", 1)[1]
        artifact_url = f"https://huggingface.co/{repo}"
        local_id = f"model:{slug.casefold()}"
        locator = f"{_DOCUMENT}:line:{line}"
        identifier = Identifier("huggingface:model", repo)
        metadata = {
            "repository": _REPOSITORY,
            "revision": revision,
            "document_path": _DOCUMENT,
            "checkpoint_repo": repo,
            "dataset_label": dataset,
            "camera_configuration": cameras,
            "world_model": world_model,
            "action_dimension": action_dim,
            "source_document_sha256": content_hash(document),
        }
        model = ModelHint(
            local_id=local_id,
            name=f"LeRobot {slug}",
            identifiers=(identifier,),
            aliases=(slug,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{slug.casefold()}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("lerobot:checkpoint", repo),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"lerobot-vlajepa:{slug.casefold()}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"LeRobot {slug}",
            raw=metadata,
            text=f"{repo}; dataset: {dataset}; cameras: {cameras}; action dim: {action_dim}",
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
) -> tuple[tuple[str, str, str, str, str, int], ...]:
    active = False
    section_level: int | None = None
    entries: dict[str, tuple[str, str, str, str, str, int]] = {}
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            level = len(heading.group("level"))
            title = heading.group("title").strip().casefold()
            if active and section_level is not None and level <= section_level:
                active = False
            if title == "pretrained checkpoints":
                active = True
                section_level = level
            continue
        if not active or "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        match = _CHECKPOINT.fullmatch(cells[0].strip("`"))
        if match is None:
            continue
        repo = f"lerobot/{match.group('slug')}"
        if repo in entries:
            raise ValueError(f"{source}: duplicate checkpoint ref {repo}")
        entry = (repo, cells[1], cells[2], cells[3], cells[4], line_number)
        entries[repo] = entry
        if len(entries) > maximum:
            raise ValueError(f"{source}: checkpoint table exceeds {maximum} entries")
    return tuple(entries.values())
