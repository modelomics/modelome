"""Pinned index reader for sherpa-onnx's source-separation model links."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
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

Clock = Callable[[], datetime]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LINK = re.compile(r"`(?P<label>[^`<>]+)\s+<(?P<url>https://[^>]+)>`_")
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,240}\.(?:onnx|tar\.bz2)$", re.I)
_ALLOWED_DOWNLOAD_PREFIX = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "source-separation-models/"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SherpaSourceSeparationSourceAdapter:
    """Read model-only links from sherpa's first-party source-separation doc.

    The document enumerates supported UVR ONNX files and Spleeter archive
    variants. It also mentions sample audio and generated outputs; those are
    deliberately excluded by the strict extension and release-path checks.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only model links declared in sherpa's source-separation documentation "
        "at one pinned commit. It does not crawl other sherpa-onnx model categories, "
        "follow the broader release asset list, or fetch model bytes."
    )

    def __init__(
        self,
        *,
        name: str = "sherpa-source-separation-models",
        repository: str = "k2-fsa/sherpa",
        branch: str = "master",
        document_path: str = "docs/source/onnx/source-separation/models.rst",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_models: int = 500,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        if repository != "k2-fsa/sherpa":
            raise ValueError("repository must be k2-fsa/sherpa")
        if branch != "master":
            raise ValueError("branch must be master")
        if document_path != "docs/source/onnx/source-separation/models.rst":
            raise ValueError("document_path must be the source-separation model index")
        for field, value in (
            ("max_response_bytes", max_response_bytes),
            ("max_models", max_models),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        self.name = name.strip()
        self.repository = repository
        self.branch = branch
        self.document_path = document_path
        self.max_response_bytes = max_response_bytes
        self.max_models = max_models
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "sherpa-source-separation-v1",
                "repository": repository,
                "branch": branch,
                "document_path": document_path,
                "max_response_bytes": max_response_bytes,
                "max_models": max_models,
                "download_prefix": _ALLOWED_DOWNLOAD_PREFIX,
            }
        )

    @property
    def repository_url(self) -> str:
        return "https://github.com/k2-fsa/sherpa"

    @property
    def commit_url(self) -> str:
        return "https://api.github.com/repos/k2-fsa/sherpa/commits/master"

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/k2-fsa/sherpa/{quote(revision, safe='')}/"
            f"{quote(self.document_path, safe='/')}"
        )

    def blob_url(self, revision: str) -> str:
        return f"{self.repository_url}/blob/{revision}/{self.document_path}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        payload = commit_response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
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

        response: HttpResponse = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: model index returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: model index exceeds {self.max_response_bytes} bytes")
        links = _parse_model_links(response.text(), maximum=self.max_models)
        records = tuple(self._record(filename, url, revision) for filename, url in links)
        if not records:
            raise ValueError(f"{self.name}: model index contains no supported model links")
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "document_url": self.raw_url(revision),
                "document_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, filename: str, url: str, revision: str) -> SourceRecord:
        model_name = filename.removesuffix(".tar.bz2").removesuffix(".onnx")
        local_id = f"model:{filename}"
        identifier = Identifier("sherpa:source-separation-model", model_name)
        locator = f"{self.document_path}:{filename}"
        model = ModelHint(
            local_id=local_id,
            name=model_name,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{filename}",
            model_local_id=local_id,
            version=filename,
            revision=revision,
            identifiers=(Identifier("sherpa:source-separation-artifact", filename),),
            metadata={
                "repository": "k2-fsa/sherpa-onnx",
                "documentation_revision": revision,
                "task": "audio-source-separation",
                "archive_filename": filename,
                "declared_download_url": url,
            },
            locator=locator,
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(self.blob_url(revision)),
            title=model_name,
            raw={
                "repository": "k2-fsa/sherpa-onnx",
                "documentation_repository": self.repository,
                "revision": revision,
                "document_path": self.document_path,
                "archive_filename": filename,
                "declared_download_url": url,
            },
            text=f"{model_name}\ntask: audio-source-separation",
            identifiers=(identifier,),
            links=(
                Link(self.blob_url(revision), relation="model_card", locator=locator, crawl=False),
                Link(
                    "https://github.com/k2-fsa/sherpa-onnx",
                    relation="source_implementation",
                    crawl=False,
                ),
                Link(url, relation="weights", locator=locator, crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_model_links(document: str, *, maximum: int) -> tuple[tuple[str, str], ...]:
    found: dict[str, str] = {}
    for match in _LINK.finditer(document):
        url = match.group("url")
        if not url.startswith(_ALLOWED_DOWNLOAD_PREFIX):
            continue
        filename = url.removeprefix(_ALLOWED_DOWNLOAD_PREFIX)
        label = match.group("label").strip()
        if (
            "/" in filename
            or not _FILENAME.fullmatch(filename)
            or label != filename
        ):
            raise ValueError("source-separation model link has an unexpected filename")
        existing = found.get(filename)
        if existing is not None and existing != url:
            raise ValueError(f"duplicate model identity with conflicting URLs: {filename}")
        found[filename] = url
        if len(found) > maximum:
            raise ValueError(f"model index exceeds {maximum} rows")
    return tuple(sorted(found.items()))


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
