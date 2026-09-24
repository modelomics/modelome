"""Official OpenAI consistency-model checkpoints from its README inventory."""

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

_REPOSITORY = "openai/consistency_models"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_ROW = re.compile(
    r"^\s*\*\s+(?P<label>[^:]+):\s+"
    r"\[(?P<filename>[A-Za-z0-9_.+-]+\.pt)\]"
    r"\((?P<url>https://openaipublic\.blob\.core\.windows\.net/"
    r"consistency/[A-Za-z0-9_.+-]+\.pt)\)\s*$"
)


class OpenAIConsistencyCheckpointSourceAdapter:
    """Index direct links for OpenAI's released consistency checkpoints."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the twelve .pt links in openai/consistency_models README's "
        "pre-trained models list; it excludes the separate CIFAR-10 JAX repository."
    )

    def __init__(
        self,
        *,
        name: str = "openai-consistency-checkpoints",
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
                "adapter": "openai-consistency-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "document_path": document_path,
                "max_response_bytes": max_response_bytes,
                "max_checkpoints": max_checkpoints,
                "admission": "official README checkpoint list and exact blob URL",
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
        commit_response = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        commit = commit_response.json()
        revision = commit.get("sha") if isinstance(commit, Mapping) else None
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
            raise ValueError(f"{self.name}: no checkpoint rows in README")
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
        label, filename, url = row["label"], row["filename"], row["url"]
        local_id = f"model:{filename.removesuffix('.pt')}"
        release_id = f"consistency/{filename}"
        task = _category(label)
        locator = f"{self.document_path}: {label}"
        return SourceRecord(
            source_record_id=f"openai-consistency:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"{label} checkpoint",
            identifiers=(Identifier("openai:consistency-release-asset", release_id),),
            links=(
                Link(url, relation="weights", crawl=False),
                Link(f"https://github.com/{self.repository}", relation="repository", crawl=False),
                Link(self.raw_url(revision), relation="model_card", crawl=False, locator=locator),
            ),
            raw={
                "record_type": "openai_consistency_checkpoint",
                "checkpoint_name": filename,
                "display_label": label,
                "checkpoint_url": url,
                "category": task,
                "source_revision": revision,
                "source_sha256": digest,
                "checkpoint_bytes_fetched": False,
            },
            models=(
                ModelHint(
                    local_id=local_id,
                    name=label,
                    aliases=(filename, filename.removesuffix(".pt")),
                    identifiers=(Identifier("openai:consistency-checkpoint", filename),),
                    status=ModelStatus.RELEASED,
                    locator=locator,
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{release_id}",
                    model_local_id=local_id,
                    version="released",
                    identifiers=(Identifier("openai:consistency-release-asset", release_id),),
                    metadata={"filename": filename, "checkpoint_url": url, "category": task},
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
        label, filename, url = (match.group(key).strip() for key in ("label", "filename", "url"))
        parsed = urlsplit(url)
        if (
            parsed.hostname != "openaipublic.blob.core.windows.net"
            or parsed.path.rsplit("/", 1)[-1] != filename
        ):
            continue
        if filename in seen:
            raise ValueError(f"duplicate consistency checkpoint {filename}")
        seen.add(filename)
        rows.append({"label": label, "filename": filename, "url": url})
        if len(rows) > maximum:
            raise ValueError(f"consistency checkpoint count exceeds {maximum}")
    return tuple(rows)


def _category(label: str) -> str:
    value = label.casefold()
    if "imagenet" in value:
        return "class-conditional-image-generation"
    return "unconditional-image-generation"


__all__ = ["OpenAIConsistencyCheckpointSourceAdapter"]
