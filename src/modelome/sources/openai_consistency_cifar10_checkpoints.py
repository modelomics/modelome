"""OpenAI's exact CIFAR-10 consistency checkpoint links from its JAX README."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpClient
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

_REPOSITORY = "openai/consistency_models_cifar10"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_URL = re.compile(
    r"https://openaipublic\.blob\.core\.windows\.net/consistency/"
    r"jcm_checkpoints/[A-Za-z0-9_-]+(?:/checkpoints/checkpoint_\d+)?"
)
_ROW = re.compile(
    r"^\s*\*\s+(?P<label>[^:]+):\s+\[(?P<name>[^\]]+)\]\((?P<url>https://[^)]+)\)\s*$"
)


class OpenAIConsistencyCIFAR10CheckpointSourceAdapter:
    """Index the ten explicitly linked CIFAR-10 consistency model snapshots."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the ten direct links in openai/consistency_models_cifar10 README's "
        "pre-trained models list; it does not enumerate additional run checkpoints."
    )

    def __init__(
        self,
        *,
        name: str = "openai-consistency-cifar10-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "main",
        document_path: str = "README.md",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_checkpoints: int = 100,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY or branch != "main" or document_path != "README.md":
            raise ValueError("repository, branch, and document_path must identify official README")
        if not name.strip() or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (max_response_bytes, max_checkpoints)
        ):
            raise ValueError("name and positive limits are required")
        self.name, self.repository, self.branch = name, repository, branch
        self.document_path = document_path
        self.max_response_bytes, self.max_checkpoints = max_response_bytes, max_checkpoints
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openai-consistency-cifar10-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "document_path": document_path,
                "max_response_bytes": max_response_bytes,
                "max_checkpoints": max_checkpoints,
                "admission": "literal checkpoint links on official OpenAI blob host",
            }
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{self.repository}/commits/{self.branch}"

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/{revision}/{self.document_path}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(self.commit_url, headers={"Accept": "application/vnd.github+json"})
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid repository revision")
        response = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/markdown,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response byte limit")
        try:
            readme = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: README is not UTF-8") from exc
        rows = _parse_rows(readme, maximum=self.max_checkpoints)
        if not rows:
            raise ValueError(f"{self.name}: no direct checkpoint rows in README")
        digest = content_hash(response.body)
        if revision == state.get("completed_revision") and digest == state.get("source_digest"):
            count = state.get("record_count")
            return SourcePage(
                (),
                dict(state),
                True,
                upstream_count=count
                if isinstance(count, int) and not isinstance(count, bool)
                else 0,
            )
        records = tuple(self._record(row, revision, digest) for row in rows)
        next_state = {
            "completed_revision": revision,
            "source_digest": digest,
            "record_count": len(records),
        }
        return SourcePage(
            records, next_state, True, upstream_count=len(records), authoritative_snapshot=True
        )

    def _record(self, row: Mapping[str, str], revision: str, digest: str) -> SourceRecord:
        label, name, url = row["label"], row["name"], row["url"]
        model_id = urlsplit(url).path.rsplit("/", 1)[-1]
        identifier = urlsplit(url).path.removeprefix("/consistency/")
        local_id = f"model:{identifier}"
        locator = f"{self.document_path}: {label}"
        return SourceRecord(
            source_record_id=f"openai-consistency-cifar10:{identifier}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"{label} checkpoint",
            identifiers=(Identifier("openai:consistency-cifar10-checkpoint", identifier),),
            links=(
                Link(url, relation="weights", crawl=False),
                Link(f"https://github.com/{self.repository}", relation="repository", crawl=False),
                Link(self.raw_url(revision), relation="model_card", crawl=False, locator=locator),
            ),
            raw={
                "record_type": "openai_consistency_cifar10_checkpoint",
                "checkpoint_name": name,
                "checkpoint_url": url,
                "display_label": label,
                "category": "unconditional-image-generation",
                "source_revision": revision,
                "source_sha256": digest,
                "checkpoint_bytes_fetched": False,
            },
            models=(
                ModelHint(
                    local_id=local_id,
                    name=label,
                    aliases=(name, model_id),
                    identifiers=(Identifier("openai:consistency-cifar10-model", identifier),),
                    status=ModelStatus.RELEASED,
                    locator=locator,
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{identifier}",
                    model_local_id=local_id,
                    version=model_id,
                    identifiers=(Identifier("openai:consistency-cifar10-checkpoint", identifier),),
                    metadata={"checkpoint_url": url, "category": "unconditional-image-generation"},
                    locator=locator,
                ),
            ),
        )


def _parse_rows(readme: str, *, maximum: int) -> tuple[Mapping[str, str], ...]:
    start = readme.find("# Pre-trained models")
    end = readme.find("# Dependencies", start + 1)
    if start < 0 or end < 0:
        return ()
    rows: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for line in readme[start:end].splitlines():
        match = _ROW.fullmatch(line)
        if match is None:
            continue
        url = match.group("url")
        parsed = urlsplit(url)
        if parsed.hostname != "openaipublic.blob.core.windows.net" or _URL.fullmatch(url) is None:
            continue
        if url in seen:
            raise ValueError(f"duplicate CIFAR-10 consistency checkpoint {url}")
        seen.add(url)
        rows.append(
            {"label": match.group("label").strip(), "name": match.group("name").strip(), "url": url}
        )
        if len(rows) > maximum:
            raise ValueError(f"CIFAR-10 consistency checkpoint count exceeds {maximum}")
    return tuple(rows)


__all__ = ["OpenAIConsistencyCIFAR10CheckpointSourceAdapter"]
