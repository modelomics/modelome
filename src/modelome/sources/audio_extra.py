"""Pinned ingestion of Coqui TTS's first-party pretrained-model index."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

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
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _header,
    _isoformat,
    _nonnegative_int,
    _positive_int,
    _text,
)

Clock = Callable[[], datetime]
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}$")
_ARTIFACT_SUFFIXES = (".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors", ".zip")
_URL_FIELDS = ("github_rls_url", "hf_url")
_METADATA_FIELDS = ("description", "license", "commit", "author", "model_hash", "tos_required")
_ROOTS = frozenset({"tts_models", "vocoder_models", "voice_conversion_models"})
_ARTIFACT_HOSTS = frozenset({"coqui.gateway.scarf.sh"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _ModelEntry:
    handle: str
    artifacts: tuple[str, ...]
    metadata: Mapping[str, Any]
    locator: str


class CoquiTtsRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate model leaves and direct checkpoint files from ``TTS/.models.json``.

    The registry is read as JSON at a pinned repository commit. The adapter only
    admits leaves under the three declared model-family roots that contain an
    explicit artifact URL to Coqui's download gateway. It neither imports Coqui
    code nor follows URLs or downloads checkpoint bytes.
    """

    coverage_limitation = (
        "Covers artifact-bearing leaves in Coqui TTS's TTS/.models.json at one "
        "pinned commit. It does not include checkpoints omitted from that index, "
        "follow gateway URLs, or download artifacts."
    )

    def __init__(
        self,
        *,
        name: str = "coqui-tts-model-registry",
        repository: str = "coqui-ai/TTS",
        branch: str = "dev",
        source_path: str = "TTS/.models.json",
        provider_namespace: str = "coqui:tts-model",
        max_response_bytes: int = 8 * 1024 * 1024,
        max_entries: int = 20_000,
        max_depth: int = 8,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.max_depth = _positive_int(max_depth, "max_depth")
        super().__init__(
            name=name,
            repository=repository,
            branch=branch,
            source_path=source_path,
            provider_namespace=provider_namespace,
            max_response_bytes=max_response_bytes,
            max_entries=max_entries,
            client=client,
            clock=clock,
        )
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "coqui-tts-model-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "max_depth": self.max_depth,
                "admission": "known model-family leaf with direct Coqui gateway checkpoint URL",
            }
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
            self.raw_url(revision), headers={"Accept": "application/json,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: registry exceeds {self.max_response_bytes} bytes")
        entries = _parse_registry(
            response.text(),
            source=self.name,
            path=self.source_path,
            maximum=self.max_entries,
            max_depth=self.max_depth,
        )
        records = tuple(self._record(entry, revision, response.body) for entry in entries)
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

    def _record(self, entry: _ModelEntry, revision: str, source: bytes) -> SourceRecord:
        source_url = self.blob_url(revision)
        local_id = f"model:{entry.handle}"
        identifier = Identifier(self.provider_namespace, entry.handle)
        model = ModelHint(
            local_id=local_id,
            name=entry.handle.rsplit("/", 1)[-1],
            aliases=(entry.handle,),
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=entry.locator,
        )
        release = ReleaseHint(
            local_id=f"release:{entry.handle}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier(f"{self.provider_namespace}:release", entry.handle),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "checkpoint_handle": entry.handle,
                "artifact_urls": list(entry.artifacts),
                **entry.metadata,
            },
            locator=entry.locator,
        )
        links = [
            Link(source_url, relation="model_card", locator=entry.locator, crawl=False,
                 model_local_ids=(local_id,)),
            Link(self.repository_url, relation="source_implementation", crawl=False,
                 model_local_ids=(local_id,)),
        ]
        links.extend(
            Link(url, relation="weights", locator=entry.locator, crawl=False,
                 model_local_ids=(local_id,))
            for url in entry.artifacts
        )
        return SourceRecord(
            source_record_id=f"model:{entry.handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(source_url),
            title=entry.handle,
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "checkpoint_handle": entry.handle,
                "artifact_urls": list(entry.artifacts),
                **entry.metadata,
            },
            text=f"Coqui TTS checkpoint: {entry.handle}",
            identifiers=(identifier,),
            links=tuple(links),
            models=(model,),
            releases=(release,),
        )


def _parse_registry(
    document: str,
    *,
    source: str,
    path: str,
    maximum: int,
    max_depth: int,
) -> tuple[_ModelEntry, ...]:
    try:
        payload = json.loads(document)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: registry is not valid JSON: {error.msg}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: registry must be a JSON object")
    unknown_roots = set(payload) - _ROOTS
    if unknown_roots:
        raise ValueError(f"{source}: registry has unrecognised model family roots")

    result: list[_ModelEntry] = []
    visited = 0

    def walk(node: Any, parts: tuple[str, ...], depth: int) -> None:
        nonlocal visited
        visited += 1
        if visited > maximum * 32:
            raise ValueError(f"{source}: registry exceeds traversal limit")
        if depth > max_depth:
            raise ValueError(f"{source}: registry exceeds maximum nesting depth {max_depth}")
        if not isinstance(node, Mapping):
            raise ValueError(f"{source}: expected an object at {'/'.join(parts)}")
        if any(field in node for field in (*_URL_FIELDS, *_METADATA_FIELDS)):
            handle = "/".join(parts)
            if len(parts) < 2 or any(not _KEY.fullmatch(part) for part in parts):
                raise ValueError(f"{source}: invalid model handle {handle!r}")
            artifacts = _artifact_urls(node, source=source, handle=handle)
            if artifacts:
                metadata = {
                    key: node[key]
                    for key in _METADATA_FIELDS
                    if key in node and isinstance(node[key], (str, int, bool))
                }
                result.append(
                    _ModelEntry(
                        handle=handle,
                        artifacts=artifacts,
                        metadata=metadata,
                        locator=f"{path}:$.{'/'.join(parts)}",
                    )
                )
                if len(result) > maximum:
                    raise ValueError(f"{source}: registry exceeds {maximum} model entries")
            return
        for key, child in node.items():
            if not isinstance(key, str) or not _KEY.fullmatch(key):
                raise ValueError(f"{source}: invalid registry key {key!r}")
            walk(child, (*parts, key), depth + 1)

    for root in sorted(_ROOTS & set(payload)):
        walk(payload[root], (root,), 1)
    return tuple(result)


def _artifact_urls(node: Mapping[str, Any], *, source: str, handle: str) -> tuple[str, ...]:
    urls: list[str] = []
    for field in _URL_FIELDS:
        value = node.get(field)
        if value is None:
            continue
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if not isinstance(candidate, str):
                raise ValueError(f"{source}: {handle!r} has a non-string artifact URL")
            parsed = urlsplit(candidate)
            if (
                parsed.scheme != "https"
                or parsed.hostname not in _ARTIFACT_HOSTS
                or parsed.port is not None
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError(f"{source}: {handle!r} has an untrusted artifact URL")
            if parsed.path.casefold().endswith(_ARTIFACT_SUFFIXES):
                normalized = canonicalize_url(candidate)
                if normalized not in urls:
                    urls.append(normalized)
    return tuple(urls)
