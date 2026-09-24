"""PaddleHelix's explicitly linked ChemRL GEM pretrained model package."""

from __future__ import annotations

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
_URL = "https://baidu-nlp.bj.bcebos.com/PaddleHelix/pretrained_models/compound/pretrain_models-chemrl_gem.tgz"
_REPOSITORY = "PaddlePaddle/PaddleHelix"
_PATH = "apps/pretrained_compound/ChemRL/GEM/README.md"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_readme(document: str) -> str:
    """Require the exact first-party GEM model package download command."""
    matches = [
        line.strip().strip("`").strip()
        for line in document.splitlines()
        if "wget" in line and "pretrain_models-chemrl_gem.tgz" in line
    ]
    if matches != [f"wget {_URL}"]:
        raise ValueError("paddlehelix-gem: expected one exact GEM pretrained-model URL")
    parsed = urlsplit(_URL)
    if parsed.scheme != "https" or parsed.hostname != "baidu-nlp.bj.bcebos.com":
        raise ValueError("paddlehelix-gem: invalid artifact origin")
    return _URL


class PaddleHelixGemCheckpointSourceAdapter:
    """Enumerate the GEM pretrained compound model archive."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the exact ChemRL GEM pretrained-model package linked by its README."
    )

    def __init__(
        self,
        *,
        name: str = "paddlehelix-chemrl-gem-checkpoint",
        repository: str = _REPOSITORY,
        branch: str = "dev",
        source_path: str = _PATH,
        max_response_bytes: int = 1 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name is required")
        if repository != _REPOSITORY or branch.strip() == "" or source_path != _PATH:
            raise ValueError("repository and source_path must identify the official GEM README")
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
                "adapter": "paddlehelix-gem-checkpoint-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "artifact": _URL,
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
        response = self.client.get(source_url, headers={"Accept": "text/markdown,text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        artifact_url = _parse_readme(response.text())
        source_hash = hashlib.sha256(response.body).hexdigest()
        identifier = Identifier("paddlehelix:pretrained-compound", "chemrl-gem")
        record = SourceRecord(
            source_record_id="model:chemrl-gem",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title="PaddleHelix ChemRL GEM pretrained model package",
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": source_hash,
                "artifact_url": artifact_url,
            },
            text=(
                "PaddleHelix ChemRL GEM pretrained model package for molecular property prediction"
            ),
            identifiers=(identifier,),
            links=(
                Link(
                    artifact_url,
                    relation="weights",
                    locator="pretrain_models-chemrl_gem.tgz",
                    crawl=False,
                ),
                Link(
                    f"https://github.com/{self.repository}/blob/{quote(revision, safe='')}/"
                    f"{quote(self.source_path, safe='/')}",
                    relation="model_card",
                    locator="Model link",
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
                    local_id="model:chemrl-gem",
                    name="ChemRL GEM",
                    identifiers=(identifier,),
                    status=ModelStatus.RELEASED,
                    locator="apps/pretrained_compound/ChemRL/GEM",
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id="release:chemrl-gem",
                    model_local_id="model:chemrl-gem",
                    revision=revision,
                    identifiers=(
                        Identifier("paddlehelix:pretrained-compound-release", "chemrl-gem"),
                    ),
                    metadata={"artifact_url": artifact_url, "source_revision": revision},
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
                "source_sha256": source_hash,
                "checkpoint_count": 1,
            },
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )


__all__ = ["PaddleHelixGemCheckpointSourceAdapter"]
