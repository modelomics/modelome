"""Metadata-only index of NVIDIA EDM checkpoint URLs named in its README."""

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

_REPOSITORY = "NVlabs/edm"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_URL = re.compile(
    r"https://nvlabs-fi-cdn\.nvidia\.com/edm/pretrained/(?:baseline/)?[A-Za-z0-9_.+-]+\.pkl"
)


class NVlabsEDMCheckpointSourceAdapter:
    """Index exact `.pkl` network URLs in NVlabs/edm's pretrained-model examples."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only exact model URLs in the README's Pre-trained models section. It does "
        "not enumerate the linked CDN directories or claim those are the complete archive."
    )

    def __init__(
        self,
        *,
        name: str = "nvlabs-edm-checkpoints",
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
                "adapter": "nvlabs-edm-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "document_path": document_path,
                "max_response_bytes": max_response_bytes,
                "max_checkpoints": max_checkpoints,
                "admission": "literal first-party pretrained example URLs",
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
        response = self.client.get(self.raw_url(revision), headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response byte limit")
        try:
            readme = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: README is not UTF-8") from exc
        rows = _parse_urls(readme, maximum=self.max_checkpoints)
        if not rows:
            raise ValueError(f"{self.name}: no checkpoint URLs in README section")
        digest = content_hash(response.body)
        if revision == state.get("completed_revision") and digest == state.get("source_digest"):
            count = state.get("record_count")
            return SourcePage(
                (), dict(state), True, upstream_count=count if isinstance(count, int) else 0
            )
        records = tuple(self._record(url, revision, digest) for url in rows)
        next_state = {
            "completed_revision": revision,
            "source_digest": digest,
            "record_count": len(records),
        }
        return SourcePage(
            records, next_state, True, upstream_count=len(records), authoritative_snapshot=True
        )

    def _record(self, url: str, revision: str, digest: str) -> SourceRecord:
        filename = urlsplit(url).path.rsplit("/", 1)[-1]
        model_id = filename.removesuffix(".pkl")
        category = _category(filename)
        release_id = f"edm-readme/{filename}"
        local_id = f"model:{model_id}"
        locator = f"{self.document_path}: {filename}"
        return SourceRecord(
            source_record_id=f"nvlabs-edm:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"EDM {filename} network checkpoint",
            identifiers=(Identifier("nvlabs:edm-checkpoint", filename),),
            links=(
                Link(url, relation="weights", crawl=False),
                Link(f"https://github.com/{self.repository}", relation="repository", crawl=False),
                Link(self.raw_url(revision), relation="model_card", crawl=False, locator=locator),
            ),
            raw={
                "record_type": "nvlabs_edm_checkpoint",
                "checkpoint_name": filename,
                "checkpoint_url": url,
                "category": category,
                "source_revision": revision,
                "source_sha256": digest,
                "checkpoint_bytes_fetched": False,
            },
            models=(
                ModelHint(
                    local_id=local_id,
                    name=f"EDM {model_id}",
                    aliases=(filename, model_id),
                    identifiers=(Identifier("nvlabs:edm-model", model_id),),
                    status=ModelStatus.RELEASED,
                    locator=locator,
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{release_id}",
                    model_local_id=local_id,
                    version="README-listed",
                    identifiers=(Identifier("nvlabs:edm-checkpoint", filename),),
                    metadata={"filename": filename, "checkpoint_url": url, "category": category},
                    locator=locator,
                ),
            ),
        )


def _parse_urls(readme: str, *, maximum: int) -> tuple[str, ...]:
    start = readme.find("## Pre-trained models")
    end = readme.find("## Calculating FID", start + 1)
    if start < 0 or end < 0:
        return ()
    rows: list[str] = []
    seen: set[str] = set()
    for url in _URL.findall(readme[start:end]):
        parsed = urlsplit(url)
        if parsed.hostname != "nvlabs-fi-cdn.nvidia.com" or not parsed.path.endswith(".pkl"):
            continue
        if url in seen:
            continue
        seen.add(url)
        rows.append(url)
        if len(rows) > maximum:
            raise ValueError(f"EDM checkpoint count exceeds {maximum}")
    return tuple(rows)


def _category(filename: str) -> str:
    return (
        "class-conditional-image-generation"
        if "-cond-" in filename
        else "unconditional-image-generation"
    )


__all__ = ["NVlabsEDMCheckpointSourceAdapter"]
