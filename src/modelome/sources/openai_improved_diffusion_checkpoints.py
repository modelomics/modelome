"""Official OpenAI improved-diffusion checkpoint URLs from its README."""

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

_REPOSITORY = "openai/improved-diffusion"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_ROW = re.compile(
    r"^(?P<label>.+?)\s+\[\[checkpoint\]\((?P<url>https://"
    r"openaipublic\.blob\.core\.windows\.net/diffusion/march-2021/"
    r"(?P<filename>[A-Za-z0-9_.+-]+\.pt))\)\]:$"
)


class OpenAIImprovedDiffusionCheckpointSourceAdapter:
    """Index the eight paper checkpoints named by OpenAI's project README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only direct checkpoint links in the Models and Hyperparameters section "
        "of openai/improved-diffusion; it excludes training outputs and other repos."
    )

    def __init__(
        self,
        *,
        name: str = "openai-improved-diffusion-checkpoints",
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
        self.name = name
        self.repository = repository
        self.branch = branch
        self.document_path = document_path
        self.max_response_bytes = max_response_bytes
        self.max_checkpoints = max_checkpoints
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openai-improved-diffusion-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "document_path": document_path,
                "max_response_bytes": max_response_bytes,
                "max_checkpoints": max_checkpoints,
                "admission": "named README checkpoint links on official blob host",
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
            raise ValueError(f"{self.name}: no direct checkpoint rows in README")
        digest = content_hash(response.body)
        if revision == state.get("completed_revision") and digest == state.get("source_digest"):
            return SourcePage(
                (), dict(state), True, upstream_count=_nonnegative(state.get("record_count"))
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
        filename, label, url = row["filename"], row["label"], row["url"]
        model_id = filename.removesuffix(".pt")
        local_id = f"model:{model_id}"
        release = f"march-2021/{filename}"
        category = _category(label)
        return SourceRecord(
            source_record_id=f"openai-improved-diffusion:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"{label} checkpoint",
            identifiers=(Identifier("openai:improved-diffusion-release-asset", release),),
            links=(
                Link(url, relation="weights", crawl=False),
                Link(f"https://github.com/{self.repository}", relation="repository", crawl=False),
                Link(
                    self.raw_url(revision),
                    relation="model_card",
                    crawl=False,
                    locator=f"{self.document_path}: {label}",
                ),
            ),
            raw={
                "record_type": "openai_improved_diffusion_checkpoint",
                "checkpoint_name": filename,
                "display_label": label,
                "checkpoint_url": url,
                "category": category,
                "release_tag": "march-2021",
                "source_revision": revision,
                "source_sha256": digest,
                "checkpoint_bytes_fetched": False,
            },
            models=(
                ModelHint(
                    local_id=local_id,
                    name=label,
                    aliases=(filename, model_id),
                    identifiers=(Identifier("openai:improved-diffusion-checkpoint", filename),),
                    status=ModelStatus.RELEASED,
                    locator=f"{self.document_path}: {label}",
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{release}",
                    model_local_id=local_id,
                    version="march-2021",
                    identifiers=(Identifier("openai:improved-diffusion-release-asset", release),),
                    metadata={"filename": filename, "checkpoint_url": url, "category": category},
                    locator=f"{self.document_path}: {label}",
                ),
            ),
        )


def _parse_rows(readme: str, *, maximum: int) -> tuple[Mapping[str, str], ...]:
    start = readme.find("## Models and Hyperparameters")
    if start < 0:
        return ()
    rows: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for line in readme[start:].splitlines()[1:]:
        match = _ROW.fullmatch(line.strip())
        if match is None:
            continue
        label = re.sub(r"`([^`]*)`", r"\1", match.group("label")).strip()
        filename, url = match.group("filename"), match.group("url")
        parsed = urlsplit(url)
        if (
            parsed.hostname != "openaipublic.blob.core.windows.net"
            or parsed.path.rsplit("/", 1)[-1] != filename
        ):
            continue
        if filename in seen:
            raise ValueError(f"duplicate improved-diffusion checkpoint {filename}")
        seen.add(filename)
        rows.append({"label": label, "filename": filename, "url": url})
        if len(rows) > maximum:
            raise ValueError(f"improved-diffusion checkpoint count exceeds {maximum}")
    return tuple(rows)


def _category(label: str) -> str:
    value = label.casefold()
    if "upsampling" in value:
        return "super-resolution-diffusion"
    if "class-conditional" in value:
        return "class-conditional-image-generation"
    return "unconditional-image-generation"


def _nonnegative(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = ["OpenAIImprovedDiffusionCheckpointSourceAdapter"]
