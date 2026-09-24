"""Enumerate RFdiffusion's first-party checkpoint URLs from its README."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

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

Clock = Callable[[], datetime]
_REPOSITORY = "RosettaCommons/RFdiffusion"
_README_PATH = "README.md"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DOWNLOAD_COMMAND = re.compile(r"^\s*wget\s+(?P<url>https?://\S+)\s*$")
_WEIGHT_PATH = re.compile(
    r"^/pub/RFdiffusion/(?P<opaque>[0-9a-f]{32})/(?P<filename>[A-Za-z0-9_.-]+\.pt)$"
)
_MAX_ENTRIES = 100


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RFDiffusionCheckpointSourceAdapter:
    """Read exact RFdiffusion weight URLs listed in the authors' README.

    The source enumerates a small, static collection of direct public weights.
    This adapter records those links and does not fetch the checkpoint bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers .pt weight URLs listed as wget commands in the first-party "
        "RFdiffusion README. It does not infer undocumented files in the hosting "
        "directory or fetch checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "rfdiffusion-first-party-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = _MAX_ENTRIES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "rfdiffusion-first-party-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "readme_path": _README_PATH,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                (), {**state, "checked_at": checked}, True,
                upstream_count=state.get("checkpoint_count"),
            )
        readme_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{revision}/{_README_PATH}"
        )
        response = self.client.get(readme_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        checkpoints = _parse_weight_commands(response.text(), self.max_entries)
        record = self._record(checkpoints, readme_url, revision)
        return SourcePage(
            (record,),
            {"completed_revision": revision, "checked_at": checked,
             "checkpoint_count": len(checkpoints)},
            True,
            upstream_count=len(checkpoints),
            authoritative_snapshot=True,
        )

    def _record(
        self, checkpoints: list[dict[str, str]], readme_url: str, revision: str
    ) -> SourceRecord:
        namespace = "rfdiffusion:checkpoint"
        models: list[ModelHint] = []
        releases: list[ReleaseHint] = []
        links: list[Link] = []
        for item in checkpoints:
            name, filename, url = item["name"], item["filename"], item["url"]
            local_id = f"model:{name}"
            models.append(
                ModelHint(
                    local_id,
                    f"RFdiffusion {name}",
                    identifiers=(Identifier(namespace, name),),
                    aliases=(filename,),
                    status=ModelStatus.RELEASED,
                )
            )
            releases.append(
                ReleaseHint(
                    f"release:{name}", local_id,
                    identifiers=(Identifier(f"{namespace}:url", url),),
                    metadata={"checkpoint_filename": filename, "checkpoint_url": url},
                    locator=f"README.md: wget {url}",
                )
            )
            links.append(Link(url, "weights", crawl=False, model_local_ids=(local_id,)))
        page_url = f"{self.repository_url}/blob/{revision}/{_README_PATH}"
        return SourceRecord(
            source_record_id=f"{self.name}:{revision}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(page_url),
            title="RFdiffusion pretrained checkpoint links",
            raw={"repository": self.repository, "revision": revision,
                 "readme_url": readme_url, "checkpoints": checkpoints},
            text="\n".join(f"{item['name']} {item['url']}" for item in checkpoints),
            identifiers=(Identifier("rfdiffusion:registry-revision", revision),),
            links=(Link(page_url, "model_card", crawl=False), *links),
            models=tuple(models),
            releases=tuple(releases),
        )


def _parse_weight_commands(readme: str, limit: int) -> list[dict[str, str]]:
    checkpoints: dict[str, dict[str, str]] = {}
    urls: set[str] = set()
    for line in readme.splitlines():
        match = _DOWNLOAD_COMMAND.fullmatch(line)
        if not match:
            continue
        url = match.group("url")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or parts.netloc != "files.ipd.uw.edu":
            continue
        resource = _WEIGHT_PATH.fullmatch(parts.path)
        if not resource:
            continue
        filename = resource.group("filename")
        name = filename.removesuffix(".pt")
        if name in checkpoints or url in urls:
            raise ValueError("RFdiffusion README repeats a checkpoint name or URL")
        urls.add(url)
        checkpoints[name] = {"name": name, "filename": filename, "url": url}
        if len(checkpoints) > limit:
            raise ValueError("RFdiffusion checkpoint list exceeds entry limit")
    if not checkpoints:
        raise ValueError("RFdiffusion README contains no admitted checkpoint commands")
    return [checkpoints[name] for name in sorted(checkpoints)]


__all__ = ["RFDiffusionCheckpointSourceAdapter"]
