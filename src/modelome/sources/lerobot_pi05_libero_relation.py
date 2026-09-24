"""Exact Pi0.5 LIBERO checkpoint-to-dataset relation from LeRobot docs."""

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
_DOCUMENT = "docs/source/pi05.mdx"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_MODEL = "lerobot/pi05_libero_finetuned_v044"
_MODEL_URL = f"https://huggingface.co/{_MODEL}"
_DATASET = "lerobot/libero"
_DATASET_URL = f"https://huggingface.co/datasets/{_DATASET}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class LeRobotPi05LiberoRelationSourceAdapter:
    """Capture the explicit Pi0.5 LIBERO checkpoint-to-dataset relationship."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the Pi0.5 LIBERO checkpoint and LeRobot LIBERO dataset explicitly "
        "connected in the first-party Quickstart on LIBERO documentation. It does not "
        "infer relations for other LeRobot Hub models or datasets."
    )

    def __init__(
        self,
        *,
        name: str = "lerobot-pi05-libero-relation",
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
                "adapter": "lerobot-pi05-libero-relation-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "model": _MODEL,
                "dataset": _DATASET,
                "max_response_bytes": max_response_bytes,
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
            raise ValueError(f"{self.name}: source document returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: source document exceeds {self.max_response_bytes} bytes"
            )
        locator = _parse_relation(response.text())
        if locator is None:
            raise ValueError(f"{self.name}: no exact Pi0.5 LIBERO pair is documented")
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
                "model_count": 1,
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
        model_local_id = "model:pi05-libero-finetuned-v044"
        model_id = Identifier("huggingface:model", _MODEL)
        metadata = {
            "repository": _REPOSITORY,
            "revision": revision,
            "document_path": _DOCUMENT,
            "checkpoint_repo": _MODEL,
            "training_dataset_repo": _DATASET,
            "source_document_sha256": content_hash(document),
        }
        model = ModelHint(
            local_id=model_local_id,
            name="LeRobot Pi0.5 LIBERO fine-tuned checkpoint",
            identifiers=(model_id,),
            aliases=("pi05_libero_finetuned_v044",),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id="release:pi05-libero-finetuned-v044",
            model_local_id=model_local_id,
            revision=revision,
            identifiers=(Identifier("lerobot:checkpoint", _MODEL),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id="lerobot:pi05-libero-training-lineage",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(_MODEL_URL),
            title="LeRobot Pi0.5 LIBERO fine-tuned checkpoint",
            raw=metadata,
            text=f"{_MODEL} fine-tuned on {_DATASET}",
            identifiers=(model_id,),
            links=(
                Link(
                    _MODEL_URL,
                    relation="model_artifact",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(model_local_id,),
                ),
                Link(
                    _DATASET_URL,
                    relation="trained_on_dataset",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(model_local_id,),
                ),
                Link(document_url, relation="model_catalog", locator=locator, crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_relation(document: str) -> str | None:
    in_quickstart = False
    section_level: int | None = None
    lines: list[tuple[int, str]] = []
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            if in_quickstart:
                locator = _matching_quickstart(lines)
                if locator is not None:
                    return locator
            lines = []
            level = len(heading.group("level"))
            title = heading.group("title").strip().casefold()
            if section_level is not None and level <= section_level:
                in_quickstart = False
                section_level = None
            if title == "quickstart on libero":
                in_quickstart = True
                section_level = level
            continue
        if in_quickstart:
            lines.append((line_number, line))
    if in_quickstart:
        return _matching_quickstart(lines)
    return None


def _matching_quickstart(lines: list[tuple[int, str]]) -> str | None:
    text = " ".join(line for _, line in lines)
    normalized = text.casefold()
    if (
        "finetune the libero base model on" not in normalized
        or "checkpoint the results below were measured on" not in normalized
    ):
        return None
    refs = {link.group("url").rstrip("/") for link in _LINK.finditer(text)}
    if not {_MODEL_URL, _DATASET_URL}.issubset(refs):
        return None
    first_line = next(
        number for number, line in lines if _MODEL_URL in line or _DATASET_URL in line
    )
    last_line = max(number for number, line in lines if _MODEL_URL in line or _DATASET_URL in line)
    return f"{_DOCUMENT}:line:{first_line}-{last_line}"
