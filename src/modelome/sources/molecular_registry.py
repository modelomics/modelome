"""First-party OpenFold checkpoint names documented by its inference guides."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_CHECKPOINT = re.compile(
    r"(?<![A-Za-z0-9_-])([A-Za-z0-9_][A-Za-z0-9_.-]*\.pt)(?![A-Za-z0-9_-])"
)
_DOCS = (
    "docs/source/original_readme.md",
    "docs/source/Single_Sequence_Inference.md",
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OpenFoldCheckpointRegistrySourceAdapter:
    """Index checkpoint filenames explicitly named in OpenFold's own guides.

    The guides document the downloadable OpenFold parameter sets and the
    SoloSeq checkpoint. This is deliberately a bounded documentation catalog:
    it does not infer model names from code/configuration or fetch weight files.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers .pt checkpoint filenames explicitly shown in the two first-party "
        "OpenFold inference guides. This is a documented example inventory, not "
        "a complete index of every downloadable parameter archive; weights are "
        "not fetched."
    )

    def __init__(
        self,
        *,
        name: str = "openfold-documented-checkpoints",
        repository: str = "aqlaboratory/openfold",
        branch: str = "main",
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not name.strip() or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("name and owner/repository are required")
        if not branch.strip() or max_response_bytes <= 0:
            raise ValueError("branch and positive max_response_bytes are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "openfold-documentation-checkpoints-v1",
            "repository": repository,
            "branch": branch,
            "documents": _DOCS,
            "max_response_bytes": max_response_bytes,
        })

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
            return SourcePage((), {**state, "checked_at": checked}, True,
                              upstream_count=state.get("model_count"))

        found: dict[str, tuple[str, str]] = {}
        for path in _DOCS:
            url = f"https://raw.githubusercontent.com/{self.repository}/{revision}/{path}"
            response = self.client.get(url, headers={"Accept": "text/plain"})
            if response.status != 200:
                raise ValueError(f"{self.name}: guide {path} returned HTTP {response.status}")
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: guide {path} exceeds response limit")
            for match in _CHECKPOINT.finditer(response.text()):
                found.setdefault(match.group(1), (path, url))
        if not found:
            raise ValueError(f"{self.name}: no documented .pt checkpoints found")
        records = tuple(
            self._record(filename, path, url, revision)
            for filename, (path, url) in sorted(found.items())
        )
        return SourcePage(
            records,
            {"completed_revision": revision, "checked_at": checked,
             "model_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, filename: str, path: str, doc_url: str, revision: str) -> SourceRecord:
        handle = filename.removesuffix(".pt")
        model_id = f"model:{handle}"
        namespace = "openfold:checkpoint"
        page_url = f"{self.repository_url}/blob/{revision}/{quote(path, safe='/')}"
        model = ModelHint(
            model_id,
            f"OpenFold {handle}",
            identifiers=(Identifier(namespace, handle),),
            aliases=(filename,),
            status=ModelStatus.DOCUMENTED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            revision=revision,
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={"repository": self.repository, "guide_path": path,
                      "checkpoint_filename": filename},
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(page_url),
            title=f"OpenFold {handle}",
            raw={"repository": self.repository, "revision": revision,
                 "guide_path": path, "guide_url": doc_url,
                 "checkpoint_filename": filename},
            text=f"OpenFold checkpoint documented in {path}: {filename}",
            identifiers=(Identifier(namespace, handle),),
            links=(
                Link(page_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(self.repository_url, "source_implementation", crawl=False,
                     model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


__all__ = ["OpenFoldCheckpointRegistrySourceAdapter"]
