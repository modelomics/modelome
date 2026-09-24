"""First-party LeRobot documentation for an exact MolmoAct2 checkpoint lineage."""

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
_DOCUMENT = "docs/source/molmoact2.mdx"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_ALLOWED_MODEL = "allenai/MolmoAct2-LIBERO-LeRobot"
_ALLOWED_DATASET = "allenai/MolmoAct2-LIBERO-Dataset"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class LeRobotMolmoAct2RelationSourceAdapter:
    """Capture the explicit LeRobot LIBERO checkpoint-to-training-dataset edge."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the exact LeRobot-format MolmoAct2 LIBERO checkpoint and dataset "
        "linked together in LeRobot's MolmoAct2 performance section. It does not infer "
        "lineage for other model cards, enumerate the four camera-specific checkpoint "
        "names, or scrape general Hugging Face model and dataset repositories."
    )

    def __init__(
        self,
        *,
        name: str = "lerobot-molmoact2-libero-relation",
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
                "adapter": "lerobot-molmoact2-libero-relation-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "model": _ALLOWED_MODEL,
                "dataset": _ALLOWED_DATASET,
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
        locator = _parse_relation(response.text(), source=self.name)
        if locator is None:
            raise ValueError(f"{self.name}: no exact LIBERO model/dataset pair is documented")
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
        model_id = Identifier("huggingface:model", _ALLOWED_MODEL)
        model_local_id = "model:molmoact2-libero-lerobot"
        model_url = f"https://huggingface.co/{_ALLOWED_MODEL}"
        dataset_url = f"https://huggingface.co/datasets/{_ALLOWED_DATASET}"
        metadata = {
            "repository": _REPOSITORY,
            "revision": revision,
            "document_path": _DOCUMENT,
            "checkpoint_repo": _ALLOWED_MODEL,
            "trained_on_dataset": _ALLOWED_DATASET,
            "source_document_sha256": content_hash(document),
        }
        model = ModelHint(
            local_id=model_local_id,
            name="MolmoAct2 LIBERO LeRobot checkpoint",
            identifiers=(model_id,),
            aliases=("MolmoAct2-LIBERO-LeRobot",),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id="release:molmoact2-libero-lerobot",
            model_local_id=model_local_id,
            revision=revision,
            identifiers=(Identifier("lerobot:checkpoint", _ALLOWED_MODEL),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id="lerobot:molmoact2-libero-lineage",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(model_url),
            title="LeRobot MolmoAct2 LIBERO checkpoint",
            raw=metadata,
            text=f"{_ALLOWED_MODEL} trained on {_ALLOWED_DATASET}",
            identifiers=(model_id,),
            links=(
                Link(
                    model_url,
                    relation="model_artifact",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(model_local_id,),
                ),
                Link(
                    dataset_url,
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


def _parse_relation(document: str, *, source: str) -> str | None:
    """Require both exact HF refs and explicit lineage wording in one paragraph."""

    in_results = False
    result_level: int | None = None
    paragraph: list[tuple[int, str]] = []
    for line_number, line in enumerate(document.splitlines(), start=1):
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            if _matches_lineage(paragraph):
                return f"{_DOCUMENT}:line:{paragraph[0][0]}-{paragraph[-1][0]}"
            paragraph = []
            level = len(heading.group(1))
            title = heading.group(2).strip().casefold()
            if result_level is not None and level <= result_level:
                in_results = False
                result_level = None
            if title == "libero benchmark results":
                in_results = True
                result_level = level
            continue
        if not in_results:
            continue
        if not line.strip():
            if _matches_lineage(paragraph):
                return f"{_DOCUMENT}:line:{paragraph[0][0]}-{paragraph[-1][0]}"
            paragraph = []
            continue
        paragraph.append((line_number, line))
    if _matches_lineage(paragraph):
        return f"{_DOCUMENT}:line:{paragraph[0][0]}-{paragraph[-1][0]}"
    return None


def _matches_lineage(paragraph: list[tuple[int, str]]) -> bool:
    text = " ".join(line for _, line in paragraph)
    if "fine-tuned checkpoint" not in text.casefold() or "trained on" not in text.casefold():
        return False
    refs = {link.group("url").rstrip("/") for link in _LINK.finditer(text)}
    return refs == {
        f"https://huggingface.co/{_ALLOWED_MODEL}",
        f"https://huggingface.co/{_ALLOWED_DATASET}",
    }
