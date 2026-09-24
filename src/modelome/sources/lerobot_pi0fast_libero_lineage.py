"""LeRobot's exact π₀-FAST base-to-LIBERO checkpoint lineage."""

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

_REPOSITORY = "huggingface/lerobot"
_DOCUMENT = "docs/source/pi0fast.mdx"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_BASE = "lerobot/pi0fast-base"
_TUNED = "lerobot/pi0fast-libero"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class LeRobotPi0FastLiberoLineageSourceAdapter:
    """Capture only the base and fine-tuned refs linked in Reproducing results."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the π₀-FAST base and LIBERO checkpoint refs in LeRobot's "
        "Reproducing Results section. It does not infer other user checkpoints "
        "or assert dataset identity from the inconsistent sample command."
    )

    def __init__(
        self,
        *,
        name: str = "lerobot-pi0fast-libero-lineage",
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "lerobot-pi0fast-libero-lineage-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "base_model": _BASE,
                "finetuned_model": _TUNED,
                "max_response_bytes": max_response_bytes,
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
        locator = _parse_lineage(response.text())
        if locator is None:
            raise ValueError(f"{self.name}: no exact π₀-FAST checkpoint lineage is documented")
        checked_at = self.clock()
        if checked_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        record = self._record(revision, response.body, document_url, locator)
        return SourcePage(
            records=(record,),
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "document_url": document_url,
                "document_sha256": content_hash(response.body),
                "model_count": 2,
            },
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _record(
        self,
        revision: str,
        document: bytes,
        document_url: str,
        locator: str,
    ) -> SourceRecord:
        base_local_id = "model:pi0fast-base"
        tuned_local_id = "model:pi0fast-libero"
        base_id = Identifier("huggingface:model", _BASE)
        tuned_id = Identifier("huggingface:model", _TUNED)
        base_url = f"https://huggingface.co/{_BASE}"
        tuned_url = f"https://huggingface.co/{_TUNED}"
        models = (
            ModelHint(
                local_id=base_local_id,
                name="LeRobot π₀-FAST base checkpoint",
                identifiers=(base_id,),
                status=ModelStatus.RELEASED,
                locator=locator,
            ),
            ModelHint(
                local_id=tuned_local_id,
                name="LeRobot π₀-FAST LIBERO fine-tuned checkpoint",
                identifiers=(tuned_id,),
                status=ModelStatus.RELEASED,
                locator=locator,
            ),
        )
        relation = ModelRelationHint(
            subject_local_id=tuned_local_id,
            predicate="fine_tuned_from",
            target=models[0],
            locator=locator,
        )
        releases = tuple(
            ReleaseHint(
                local_id=f"release:{repo_id.rsplit('/', 1)[-1]}",
                model_local_id=local_id,
                revision=revision,
                identifiers=(Identifier("lerobot:checkpoint", repo_id),),
                metadata={
                    "repository": _REPOSITORY,
                    "revision": revision,
                    "document_path": _DOCUMENT,
                    "checkpoint_repo": repo_id,
                    "lineage_role": role,
                    "base_checkpoint_repo": _BASE,
                    "source_document_sha256": content_hash(document),
                },
                locator=locator,
            )
            for repo_id, local_id, role in (
                (_BASE, base_local_id, "base"),
                (_TUNED, tuned_local_id, "fine_tuned_libero"),
            )
        )
        return SourceRecord(
            source_record_id="lerobot:pi0fast-libero-lineage",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(tuned_url),
            title="LeRobot π₀-FAST base and LIBERO checkpoints",
            raw={
                "repository": _REPOSITORY,
                "revision": revision,
                "document_path": _DOCUMENT,
                "base_checkpoint_repo": _BASE,
                "finetuned_checkpoint_repo": _TUNED,
                "source_document_sha256": content_hash(document),
            },
            text=f"{_TUNED} fine-tuned from {_BASE} for LIBERO",
            identifiers=(base_id, tuned_id),
            links=(
                Link(
                    base_url,
                    relation="model_artifact",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(base_local_id,),
                ),
                Link(
                    tuned_url,
                    relation="model_artifact",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(tuned_local_id,),
                ),
                Link(document_url, relation="model_catalog", locator=locator, crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=models,
            model_relations=(relation,),
            releases=releases,
        )


def _parse_lineage(document: str) -> str | None:
    in_section = False
    level: int | None = None
    lines: list[tuple[int, str]] = []
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            if in_section:
                locator = _matching_lines(lines)
                if locator:
                    return locator
            current_level = len(heading.group("level"))
            title = heading.group("title").strip().casefold()
            if level is not None and current_level <= level:
                in_section = False
                level = None
            if title == "reproducing π₀fast results":
                in_section = True
                level = current_level
                lines = []
            continue
        if in_section:
            lines.append((line_number, line))
    if in_section:
        return _matching_lines(lines)
    return None


def _matching_lines(lines: list[tuple[int, str]]) -> str | None:
    text = " ".join(line for _, line in lines).casefold()
    required = (
        "take the lerobot pifast base model",
        "finetune for an additional 40k steps",
        "the finetuned model can be found here",
    )
    if not all(phrase in text for phrase in required):
        return None
    if not all(repo.casefold() in text for repo in (_BASE, _TUNED)):
        return None
    first = min(number for number, line in lines if any(repo in line for repo in (_BASE, _TUNED)))
    last = max(number for number, line in lines if any(repo in line for repo in (_BASE, _TUNED)))
    return f"{_DOCUMENT}:line:{first}-{last}"
