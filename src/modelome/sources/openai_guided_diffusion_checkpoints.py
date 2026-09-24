"""Official OpenAI guided-diffusion checkpoint URLs from its README manifest."""

from __future__ import annotations

import re
from collections.abc import Mapping
from html import unescape
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

_REPOSITORY = "openai/guided-diffusion"
_BRANCH = "main"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_ROW = re.compile(
    r"^\s*\*\s+(?P<label>[^:]+):\s+"
    r"\[(?P<filename>[A-Za-z0-9_.+-]+\.pt)\]"
    r"\((?P<url>https://openaipublic\.blob\.core\.windows\.net/"
    r"diffusion/jul-2021/[A-Za-z0-9_.+-]+\.pt)\)\s*$"
)


class OpenAIGuidedDiffusionCheckpointSourceAdapter:
    """Index pretrained guided-diffusion checkpoints with direct official URLs."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the `.pt` model rows in openai/guided-diffusion's README pretrained "
        "model section. It includes image classifiers and upsamplers used with the "
        "diffusion models, and excludes training outputs and other OpenAI research repos."
    )

    def __init__(
        self,
        *,
        name: str = "openai-guided-diffusion-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = _BRANCH,
        document_path: str = "README.md",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_checkpoints: int = 100,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY or branch != _BRANCH or document_path != "README.md":
            raise ValueError(
                "repository, branch, and document_path must identify the official README"
            )
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
                "adapter": "openai-guided-diffusion-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "document_path": document_path,
                "max_response_bytes": max_response_bytes,
                "max_checkpoints": max_checkpoints,
                "admission": "README download-list rows and exact first-party blob URL",
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
        rows = _parse_download_rows(readme, maximum=self.max_checkpoints)
        if not rows:
            raise ValueError(f"{self.name}: no direct checkpoint rows in README")
        digest = content_hash(response.body)
        if revision == state.get("completed_revision") and digest == state.get("source_digest"):
            return SourcePage(
                (),
                dict(state),
                True,
                upstream_count=_nonnegative(state.get("record_count")),
            )
        records = tuple(self._record(row, revision, digest) for row in rows)
        next_state = {
            "completed_revision": revision,
            "source_digest": digest,
            "record_count": len(records),
        }
        return SourcePage(
            records,
            next_state,
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, row: Mapping[str, str], revision: str, digest: str) -> SourceRecord:
        label = row["label"].strip()
        filename = row["filename"]
        url = row["url"]
        category = _category(label)
        model_id = filename.removesuffix(".pt")
        model_local_id = f"model:{model_id}"
        model = ModelHint(
            local_id=model_local_id,
            name=label,
            aliases=(filename, model_id),
            identifiers=(Identifier("openai:guided-diffusion-checkpoint", filename),),
            status=ModelStatus.RELEASED,
            locator=f"{self.document_path}: {label} -> {filename}",
        )
        release_value = f"jul-2021/{filename}"
        return SourceRecord(
            source_record_id=f"openai-guided-diffusion:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"{label} checkpoint",
            identifiers=(Identifier("openai:guided-diffusion-release-asset", release_value),),
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
                "record_type": "openai_guided_diffusion_checkpoint",
                "checkpoint_name": filename,
                "display_label": label,
                "checkpoint_url": url,
                "category": category,
                "release_tag": "jul-2021",
                "source_revision": revision,
                "source_sha256": digest,
                "checkpoint_bytes_fetched": False,
            },
            models=(model,),
            releases=(
                ReleaseHint(
                    local_id=f"release:{release_value}",
                    model_local_id=model_local_id,
                    version="jul-2021",
                    identifiers=(
                        Identifier("openai:guided-diffusion-release-asset", release_value),
                    ),
                    metadata={
                        "filename": filename,
                        "checkpoint_url": url,
                        "category": category,
                    },
                    locator=f"{self.document_path}: {label}",
                ),
            ),
        )


def _parse_download_rows(readme: str, *, maximum: int) -> tuple[Mapping[str, str], ...]:
    start = readme.find("# Download pre-trained models")
    end = readme.find("# Sampling from pre-trained models", start + 1)
    if start < 0 or end < 0:
        return ()
    rows: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for line in readme[start:end].splitlines():
        match = _ROW.fullmatch(line)
        if match is None:
            continue
        label = unescape(match.group("label").strip())
        filename = match.group("filename")
        url = match.group("url")
        parsed = urlsplit(url)
        if (
            parsed.hostname != "openaipublic.blob.core.windows.net"
            or parsed.path.rsplit("/", 1)[-1] != filename
        ):
            continue
        if filename in seen:
            raise ValueError(f"duplicate guided-diffusion checkpoint {filename}")
        seen.add(filename)
        rows.append({"label": label, "filename": filename, "url": url})
        if len(rows) > maximum:
            raise ValueError(f"guided-diffusion checkpoint count exceeds {maximum}")
    return tuple(rows)


def _category(label: str) -> str:
    value = label.casefold()
    if "classifier" in value:
        return "image-classifier"
    if "upsampler" in value:
        return "super-resolution-diffusion"
    if value.startswith("lsun"):
        return "class-unconditional-image-generation"
    if "unconditional" in value or "not class conditional" in value:
        return "unconditional-image-generation"
    return "class-conditional-image-generation"


def _nonnegative(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = ["OpenAIGuidedDiffusionCheckpointSourceAdapter"]
