"""Pinned ingestion of literal checkpoint registries published as JSON maps.

Many framework projects preserve a small, source-controlled mapping from an
exact public handle to a direct checkpoint URL.  This reader deliberately
accepts only that narrow shape: a JSON object with unique literal string keys
and recognised checkpoint-file URLs.  It resolves the repository revision
first, never imports a package, and never transfers a model binary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_NAMESPACE = re.compile(r"^[a-z][a-z0-9_-]*(?::[a-z][a-z0-9_-]*)+$")
_HANDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+/-]*$")
_CHECKPOINT_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".onnx",
    ".pdparams",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    handle: str
    url: str
    locator: str


class StaticJsonCheckpointRegistrySourceAdapter:
    """Enumerate literal public checkpoint handles at a pinned Git revision.

    The JSON document must be an object mapping one source-declared handle to
    one direct recognised checkpoint URL.  The handle is a source-native model
    identity rather than a guessed architecture equivalence.  Each emitted
    record contains exactly one model and one release, making every source,
    implementation, and artifact link model-scoped by construction.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers literal handle-to-checkpoint rows in one first-party JSON registry "
        "at a pinned commit. It does not execute package code, infer papers or "
        "architectures, follow artifact URLs, or download checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str,
        repository: str,
        branch: str,
        source_path: str,
        provider_namespace: str,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 100_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.source_path = _safe_path(source_path)
        self.provider_namespace = _namespace(provider_namespace)
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "static-json-checkpoint-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "literal JSON object of direct checkpoint file URLs",
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
        )

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(self.source_path, safe='/')}"
        )

    def blob_url(self, revision: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            if etag := _header(commit_response.headers, "etag"):
                next_state["commit_etag"] = etag
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision),
            headers={"Accept": "application/json,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: registry exceeds {self.max_response_bytes} bytes")
        checkpoints = _parse_registry(
            response.text(),
            source=self.name,
            path=self.source_path,
            maximum=self.max_entries,
        )
        records = tuple(
            self._record(checkpoint, revision, response.body) for checkpoint in checkpoints
        )
        if not records:
            raise ValueError(f"{self.name}: registry contains no checkpoint entries")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _revision(self) -> tuple[str, HttpResponse]:
        response: HttpResponse = self.client.get(
            self.commit_url,
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _record(
        self,
        checkpoint: _Checkpoint,
        revision: str,
        source: bytes,
    ) -> SourceRecord:
        identifier = Identifier(self.provider_namespace, checkpoint.handle)
        model = ModelHint(
            local_id=f"model:{checkpoint.handle}",
            name=self._model_name(checkpoint.handle),
            aliases=(checkpoint.handle,),
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=checkpoint.locator,
        )
        source_url = self.blob_url(revision)
        release = ReleaseHint(
            local_id=f"release:{checkpoint.handle}",
            model_local_id=model.local_id,
            revision=revision,
            identifiers=(
                Identifier(f"{self.provider_namespace}:release", checkpoint.handle),
            ),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "checkpoint_handle": checkpoint.handle,
                "weight_url": checkpoint.url,
            },
            locator=checkpoint.locator,
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{checkpoint.handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(source_url),
            title=model.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "checkpoint_handle": checkpoint.handle,
                "weight_url": checkpoint.url,
            },
            text=f"{self.name} checkpoint handle: {checkpoint.handle}",
            identifiers=(identifier,),
            links=(
                Link(
                    source_url,
                    relation="model_card",
                    locator=checkpoint.locator,
                    crawl=False,
                    model_local_ids=(model.local_id,),
                ),
                Link(
                    self.repository_url,
                    relation="source_implementation",
                    crawl=False,
                    model_local_ids=(model.local_id,),
                ),
                Link(
                    checkpoint.url,
                    relation="weights",
                    locator=checkpoint.locator,
                    crawl=False,
                    model_local_ids=(model.local_id,),
                ),
            ),
            models=(model,),
            releases=(release,),
        )

    def _model_name(self, handle: str) -> str:
        """Return the source-native display name for one registry key."""

        return handle.rsplit("/", 1)[-1]


def _parse_registry(
    document: str,
    *,
    source: str,
    path: str,
    maximum: int,
) -> tuple[_Checkpoint, ...]:
    try:
        payload = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: registry is not valid JSON: {error.msg}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: registry must be a JSON object")
    if not payload:
        raise ValueError(f"{source}: registry is empty")
    if len(payload) > maximum:
        raise ValueError(f"{source}: registry exceeds {maximum} entries")

    checkpoints = []
    for handle, raw_url in payload.items():
        if (
            not isinstance(handle, str)
            or not _HANDLE.fullmatch(handle)
            or ".." in handle.split("/")
        ):
            raise ValueError(f"{source}: invalid checkpoint handle {handle!r}")
        url = _checkpoint_url(raw_url, source, handle)
        checkpoints.append(
            _Checkpoint(
                handle=handle,
                url=url,
                locator=f"{path}:$[{json.dumps(handle, ensure_ascii=False)}]",
            )
        )
    return tuple(checkpoints)


def _checkpoint_url(value: Any, source: str, handle: str) -> str:
    url = _text(value)
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{source}: {handle!r} does not declare an absolute checkpoint URL")
    if not parsed.path.casefold().endswith(_CHECKPOINT_SUFFIXES):
        raise ValueError(f"{source}: {handle!r} URL is not a recognised checkpoint file")
    return canonicalize_url(url)


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _namespace(value: str) -> str:
    namespace = _required_text(value, "provider namespace")
    if not _NAMESPACE.fullmatch(namespace):
        raise ValueError("provider namespace must have one colon-separated suffix")
    return namespace


def _safe_path(value: str) -> str:
    path = _required_text(value, "source path")
    invalid_part = any(part in {"", ".", ".."} for part in path.split("/"))
    if path.startswith("/") or "\\" in path or invalid_part:
        raise ValueError("source path is not a safe relative path")
    return path


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _header(headers: Mapping[str, Any], key: str) -> str:
    value = headers.get(key) or headers.get(key.casefold()) or headers.get(key.title())
    return _text(value)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
