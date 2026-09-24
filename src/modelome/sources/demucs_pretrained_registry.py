"""Enumerate official Demucs checkpoint files from its remote manifest."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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

_REPOSITORY = "facebookresearch/demucs"
_BRANCH = "main"
_DOCUMENT_PATH = "demucs/remote/files.txt"
_COMMIT_URL = f"https://api.github.com/repos/{_REPOSITORY}/commits/{_BRANCH}"
_RAW_PREFIX = f"https://raw.githubusercontent.com/{_REPOSITORY}/"
_REPOSITORY_URL = f"https://github.com/{_REPOSITORY}"
_STORAGE_ROOT = "https://dl.fbaipublicfiles.com/demucs/"
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_WEIGHT_FILENAME = re.compile(r"^(?P<signature>[0-9a-f]{8})-[0-9a-f]{8}\.th$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class _Checkpoint:
    signature: str
    filename: str
    root: str
    family: str
    line_number: int

    @property
    def url(self) -> str:
        return f"{_STORAGE_ROOT}{self.root}{self.filename}"


class DemucsPretrainedRegistrySourceAdapter:
    """Read the finite model-file registry used by the official Demucs loader."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers individual checkpoint files in demucs/remote/files.txt. It excludes "
        "bag definitions that compose these files and any local Dora experiments not "
        "published in the remote manifest."
    )

    def __init__(
        self,
        *,
        name: str = "demucs-pretrained-checkpoints",
        max_response_bytes: int = 1024 * 1024,
        max_models: int = 100,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        for label, value in (
            ("max_response_bytes", max_response_bytes),
            ("max_models", max_models),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_models = max_models
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "demucs-pretrained-registry-v1",
                "repository": _REPOSITORY,
                "branch": _BRANCH,
                "document_path": _DOCUMENT_PATH,
                "storage_root": _STORAGE_ROOT,
                "max_response_bytes": max_response_bytes,
                "max_models": max_models,
            }
        )

    @property
    def commit_url(self) -> str:
        return _COMMIT_URL

    def raw_url(self, revision: str) -> str:
        return f"{_RAW_PREFIX}{revision}/{_DOCUMENT_PATH}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _SHA1.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = _isoformat(self.clock())
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        document_url = self.raw_url(revision)
        document_response: HttpResponse = self.client.get(
            document_url, headers={"Accept": "text/plain"}
        )
        if document_response.status != 200:
            raise ValueError(
                f"{self.name}: remote file list returned HTTP {document_response.status}"
            )
        if len(document_response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: remote file list exceeds response limit")
        checkpoints = _parse_file_list(
            document_response.text(), maximum=self.max_models, source=self.name
        )
        records = tuple(self._record(checkpoint, revision) for checkpoint in checkpoints)
        if not records:
            raise ValueError(f"{self.name}: remote file list contains no model checkpoints")
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "document_url": document_url,
                "document_sha256": content_hash(document_response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, checkpoint: _Checkpoint, revision: str) -> SourceRecord:
        model_name = checkpoint.signature
        local_id = f"model:{model_name}"
        identifier = Identifier("demucs:pretrained-signature", model_name)
        locator = f"{_DOCUMENT_PATH}:line-{checkpoint.line_number}"
        model = ModelHint(
            local_id=local_id,
            name=model_name,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{model_name}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("demucs:checkpoint-file", checkpoint.filename),),
            metadata={
                "repository": _REPOSITORY,
                "manifest_revision": revision,
                "manifest_path": _DOCUMENT_PATH,
                "manifest_line": checkpoint.line_number,
                "signature": checkpoint.signature,
                "filename": checkpoint.filename,
                "storage_root": checkpoint.root,
                "weight_url": checkpoint.url,
                "family": checkpoint.family,
            },
            locator=locator,
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(checkpoint.url),
            title=model_name,
            raw={
                "repository": _REPOSITORY,
                "manifest_revision": revision,
                "manifest_path": _DOCUMENT_PATH,
                "manifest_line": checkpoint.line_number,
                "signature": checkpoint.signature,
                "filename": checkpoint.filename,
                "storage_root": checkpoint.root,
                "weight_url": checkpoint.url,
                "family": checkpoint.family,
            },
            text=(
                f"Official Demucs pretrained checkpoint {model_name}\n"
                f"family: {checkpoint.family}\n"
                f"filename: {checkpoint.filename}"
            ),
            identifiers=(identifier,),
            links=(
                Link(
                    f"{_REPOSITORY_URL}/blob/{revision}/{_DOCUMENT_PATH}",
                    "model_card",
                    locator=locator,
                    crawl=False,
                ),
                Link(_REPOSITORY_URL, "source_implementation", crawl=False),
                Link(checkpoint.url, "weights", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_file_list(document: str, *, maximum: int, source: str) -> tuple[_Checkpoint, ...]:
    checkpoints: list[_Checkpoint] = []
    seen: set[str] = set()
    root: str | None = None
    family = ""
    for line_number, line in enumerate(document.splitlines(), start=1):
        value = line.strip()
        if not value:
            continue
        if value.startswith("#"):
            family = value.removeprefix("#").strip()
            continue
        if value.startswith("root:"):
            candidate = value.partition(":")[2].strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]+/", candidate):
                raise ValueError(f"{source}: invalid storage root at line {line_number}")
            root = candidate
            continue
        match = _WEIGHT_FILENAME.fullmatch(value)
        if match is None or root is None or not family:
            raise ValueError(f"{source}: invalid remote checkpoint entry at line {line_number}")
        signature = match.group("signature")
        if signature in seen:
            raise ValueError(f"{source}: duplicate model signature {signature!r}")
        seen.add(signature)
        checkpoints.append(_Checkpoint(signature, value, root, family, line_number))
        if len(checkpoints) > maximum:
            raise ValueError(f"{source}: remote file list exceeds {maximum} models")
    return tuple(checkpoints)


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = ["DemucsPretrainedRegistrySourceAdapter"]
