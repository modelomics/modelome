"""Yandex Research's separately trained FFHQ-256 DDPM checkpoint."""

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

_REPOSITORY = "yandex-research/ddpm-segmentation"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_ROW = re.compile(
    r"^\*(?P<label>[^*]+):\*\s*\[(?P<filename>[^\]]+\.pt)\]\((?P<url>https://[^)]+)\).*$"
)
_URL = "https://storage.yandexcloud.net/yandex-research/ddpm-segmentation/models/ddpm_checkpoints/ffhq.pt"


class YandexDDPMFFHQCheckpointSourceAdapter:
    """Index the exact Yandex-hosted FFHQ checkpoint named in its paper repo."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the separately trained FFHQ-256 checkpoint documented by the "
        "repository; the LSUN checkpoints are upstream guided-diffusion assets."
    )

    def __init__(
        self,
        *,
        name: str = "yandex-ddpm-ffhq-checkpoint",
        repository: str = _REPOSITORY,
        branch: str = "master",
        document_path: str = "README.md",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_checkpoints: int = 10,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY or branch != "master" or document_path != "README.md":
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
                "adapter": "yandex-ddpm-ffhq-v1",
                "repository": repository,
                "branch": branch,
                "document_path": document_path,
                "max_response_bytes": max_response_bytes,
                "max_checkpoints": max_checkpoints,
                "admission": "FFHQ direct URL in the official pretrained DDPM list",
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
        rows = _parse_rows(readme, maximum=self.max_checkpoints)
        if not rows:
            raise ValueError(f"{self.name}: FFHQ checkpoint mapping missing from README")
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
        locator = f"{self.document_path}: {label}"
        release_id = f"ffhq-256/{filename}"
        local_id = "model:ffhq-256-ddpm"
        return SourceRecord(
            source_record_id=f"yandex-ddpm:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"Yandex DDPM {label} checkpoint",
            identifiers=(Identifier("yandex-research:ddpm-checkpoint", release_id),),
            links=(
                Link(url, relation="weights", crawl=False),
                Link(f"https://github.com/{self.repository}", relation="repository", crawl=False),
                Link(self.raw_url(revision), relation="model_card", crawl=False, locator=locator),
            ),
            raw={
                "record_type": "yandex_ddpm_checkpoint",
                "checkpoint_name": filename,
                "display_label": label,
                "checkpoint_url": url,
                "category": "unconditional-image-generation",
                "dataset": "FFHQ-256",
                "source_revision": revision,
                "source_sha256": digest,
                "checkpoint_bytes_fetched": False,
            },
            models=(
                ModelHint(
                    local_id=local_id,
                    name="Yandex DDPM FFHQ-256",
                    aliases=(filename, "ffhq"),
                    identifiers=(Identifier("yandex-research:ddpm-model", "ffhq-256"),),
                    status=ModelStatus.RELEASED,
                    locator=locator,
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{release_id}",
                    model_local_id=local_id,
                    version="updated-2022-03-08",
                    identifiers=(Identifier("yandex-research:ddpm-checkpoint", release_id),),
                    metadata={
                        "filename": filename,
                        "checkpoint_url": url,
                        "dataset": "FFHQ-256",
                        "category": "unconditional-image-generation",
                    },
                    locator=locator,
                ),
            ),
        )


def _parse_rows(readme: str, *, maximum: int) -> tuple[Mapping[str, str], ...]:
    start = readme.find("### Pretrained DDPMs")
    end = readme.find("### Run", start + 1)
    if start < 0 or end < 0:
        return ()
    rows: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for line in readme[start:end].splitlines():
        match = _ROW.fullmatch(line.strip())
        if match is None:
            continue
        label, filename, url = (match.group(key).strip() for key in ("label", "filename", "url"))
        parsed = urlsplit(url)
        if parsed.hostname != "storage.yandexcloud.net" or url != _URL or filename != "ffhq.pt":
            continue
        if url in seen:
            raise ValueError("duplicate Yandex DDPM FFHQ checkpoint mapping")
        seen.add(url)
        rows.append({"label": label, "filename": filename, "url": url})
        if len(rows) > maximum:
            raise ValueError(f"Yandex DDPM checkpoint count exceeds {maximum}")
    return tuple(rows)


__all__ = ["YandexDDPMFFHQCheckpointSourceAdapter"]
