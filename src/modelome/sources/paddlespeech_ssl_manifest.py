"""PaddleSpeech's source-declared WavLM SSL checkpoint package."""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable, Mapping
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
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MODEL = "wavlmASR_librispeech-en-16k"
_VERSION = "1.0"
_URL = "https://paddlespeech.cdn.bcebos.com/wavlm/wavlm_baseplus_libriclean_100h.tar.gz"
_MD5_VALUE = "f2238e982bb8bcf046e536201f5ea629"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_manifest(document: str) -> tuple[str, str, str]:
    """Extract one literal model/version from the official SSL registry."""
    try:
        module = ast.parse(document)
    except SyntaxError as exc:
        raise ValueError("paddlespeech-ssl-manifest: invalid Python source") from exc
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "ssl_dynamic_pretrained_models"
    ]
    if len(assignments) != 1:
        raise ValueError("paddlespeech-ssl-manifest: expected one SSL mapping")
    try:
        registry = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError("paddlespeech-ssl-manifest: SSL mapping is not literal") from exc
    if not isinstance(registry, dict):
        raise ValueError("paddlespeech-ssl-manifest: SSL mapping is not a dictionary")
    versions = registry.get(_MODEL)
    values = versions.get(_VERSION) if isinstance(versions, dict) else None
    if not isinstance(values, dict):
        raise ValueError("paddlespeech-ssl-manifest: WavLM version 1.0 is missing")
    url, md5 = values.get("url"), values.get("md5")
    parsed = urlsplit(url) if isinstance(url, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or parsed.hostname != "paddlespeech.cdn.bcebos.com"
        or parsed.query
        or parsed.fragment
        or parsed.path != "/wavlm/wavlm_baseplus_libriclean_100h.tar.gz"
        or not isinstance(md5, str)
        or not _MD5.fullmatch(md5)
    ):
        raise ValueError("paddlespeech-ssl-manifest: WavLM artifact URL or MD5 is invalid")
    return _MODEL, url, md5


class PaddleSpeechSslManifestSourceAdapter:
    """Read PaddleSpeech's WavLM package URL and checksum without executing code."""

    disable_derived_extraction = True
    coverage_limitation = "Covers only the literal WavLM ASR Librispeech v1.0 package row."

    def __init__(
        self,
        *,
        name: str = "paddlespeech-wavlm-ssl-manifest",
        repository: str = "PaddlePaddle/PaddleSpeech",
        branch: str = "develop",
        source_path: str = "paddlespeech/resource/pretrained_models.py",
        max_response_bytes: int = 8 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name is required")
        if not _REPOSITORY.fullmatch(repository):
            raise ValueError("repository must use owner/name form")
        if (
            not branch.strip()
            or not source_path
            or source_path.startswith("/")
            or ".." in source_path.split("/")
        ):
            raise ValueError("branch and relative source_path are required")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be positive")
        self.name = name.strip()
        self.repository = repository
        self.branch = branch.strip()
        self.source_path = source_path
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paddlespeech-ssl-manifest-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "model": _MODEL,
                "version": _VERSION,
            }
        )

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = (
            payload.get("sha", "").strip()
            if isinstance(payload, Mapping) and isinstance(payload.get("sha"), str)
            else ""
        )
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                records=(),
                next_state={**state, "checked_at": checked_at},
                complete=True,
                upstream_count=1,
            )
        source_url = self.raw_url(revision)
        response = self.client.get(
            source_url, headers={"Accept": "text/plain,application/octet-stream"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source file returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source file exceeds response limit")
        model, artifact_url, md5 = _parse_manifest(response.text())
        revision_hash = hashlib.sha256(response.body).hexdigest()
        identifier = Identifier("paddlespeech:ssl-model", model)
        record = SourceRecord(
            source_record_id=f"model:{model}:{_VERSION}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"{model} v{_VERSION} pretrained package",
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": revision_hash,
                "model_name": model,
                "version": _VERSION,
                "artifact_url": artifact_url,
                "md5": md5,
            },
            text=f"PaddleSpeech WavLM SSL model: {model}",
            identifiers=(identifier,),
            links=(
                Link(artifact_url, relation="weights", locator="model.tar.gz", crawl=False),
                Link(
                    f"https://github.com/{self.repository}/blob/{quote(revision, safe='')}/"
                    f"{quote(self.source_path, safe='/')}",
                    relation="model_card",
                    locator="ssl_dynamic_pretrained_models",
                    crawl=False,
                ),
                Link(
                    f"https://github.com/{self.repository}",
                    relation="source_repository",
                    crawl=False,
                ),
            ),
            models=(
                ModelHint(
                    local_id=f"model:{model}",
                    name=model,
                    identifiers=(identifier,),
                    status=ModelStatus.RELEASED,
                    locator=f"ssl_dynamic_pretrained_models:{model}",
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{model}:{_VERSION}",
                    model_local_id=f"model:{model}",
                    revision=_VERSION,
                    identifiers=(Identifier("paddlespeech:ssl-release", f"{model}@{_VERSION}"),),
                    metadata={
                        "artifact_url": artifact_url,
                        "md5": md5,
                        "source_revision": revision,
                    },
                    locator=artifact_url,
                ),
            ),
        )
        return SourcePage(
            records=(record,),
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": revision_hash,
                "checkpoint_count": 1,
            },
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )


__all__ = ["PaddleSpeechSslManifestSourceAdapter"]
