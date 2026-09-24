"""NVIDIA EDM2 checkpoint URLs explicitly expanded in its official README."""

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

_REPOSITORY = "NVlabs/edm2"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HOST = "nvlabs-fi-cdn.nvidia.com"
_PREFIX = "/edm2/posthoc-reconstructions/"
_URL = re.compile(
    r"https://nvlabs-fi-cdn\.nvidia\.com/edm2/posthoc-reconstructions/[A-Za-z0-9_.-]+\.pkl"
)


class NVlabsEDM2CheckpointSourceAdapter:
    """Index exact conditional and guidance EDM2 networks linked by NVlabs."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only literal network URLs in the README's expanded preset example and "
        "FLOPs example; it does not enumerate the broader preset-to-file matrix."
    )

    def __init__(
        self,
        *,
        name: str = "nvlabs-edm2-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "main",
        document_path: str = "README.md",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_checkpoints: int = 20,
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
                "adapter": "nvlabs-edm2-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "document_path": document_path,
                "max_response_bytes": max_response_bytes,
                "max_checkpoints": max_checkpoints,
                "admission": "literal first-party EDM2 network URLs in README model section",
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
                (),
                dict(state),
                True,
                upstream_count=count
                if isinstance(count, int) and not isinstance(count, bool)
                else 0,
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
        role = "guidance" if "-uncond-" in filename else "conditional"
        local_id = f"model:{model_id}"
        locator = f"{self.document_path}: Using pre-trained models"
        release_id = f"posthoc-reconstructions/{filename}"
        return SourceRecord(
            source_record_id=f"nvlabs-edm2:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"EDM2 {role} checkpoint {filename}",
            identifiers=(Identifier("nvlabs:edm2-checkpoint", release_id),),
            links=(
                Link(url, relation="weights", crawl=False),
                Link(f"https://github.com/{self.repository}", relation="repository", crawl=False),
                Link(self.raw_url(revision), relation="model_card", crawl=False, locator=locator),
            ),
            raw={
                "record_type": "nvlabs_edm2_checkpoint",
                "checkpoint_name": filename,
                "checkpoint_url": url,
                "role": role,
                "category": "unconditional-image-generation"
                if role == "guidance"
                else "class-conditional-image-generation",
                "source_revision": revision,
                "source_sha256": digest,
                "checkpoint_bytes_fetched": False,
            },
            models=(
                ModelHint(
                    local_id=local_id,
                    name=f"EDM2 {model_id}",
                    aliases=(filename, model_id),
                    identifiers=(Identifier("nvlabs:edm2-model", model_id),),
                    status=ModelStatus.RELEASED,
                    locator=locator,
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{release_id}",
                    model_local_id=local_id,
                    version="README-listed",
                    identifiers=(Identifier("nvlabs:edm2-checkpoint", release_id),),
                    metadata={"filename": filename, "checkpoint_url": url, "role": role},
                    locator=locator,
                ),
            ),
        )


def _parse_urls(readme: str, *, maximum: int) -> tuple[str, ...]:
    sections: list[str] = []
    for heading in ("## Using pre-trained models", "## Calculating FLOPs and metrics"):
        start = readme.find(heading)
        if start < 0:
            continue
        content_start = start + len(heading)
        end_match = re.search(r"(?m)^## (?!#)", readme[content_start:])
        if end_match is not None:
            sections.append(readme[content_start : content_start + end_match.start()])
    rows: list[str] = []
    seen: set[str] = set()
    for section in sections:
        for url in _URL.findall(section):
            parsed = urlsplit(url)
            if parsed.hostname != _HOST or not parsed.path.startswith(_PREFIX):
                continue
            if url not in seen:
                seen.add(url)
                rows.append(url)
                if len(rows) > maximum:
                    raise ValueError(f"EDM2 checkpoint count exceeds {maximum}")
    return tuple(rows)


__all__ = ["NVlabsEDM2CheckpointSourceAdapter"]
