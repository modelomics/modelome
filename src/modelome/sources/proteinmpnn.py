"""Enumerate official ProteinMPNN checkpoints from its first-party Git tree."""

from __future__ import annotations

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
_WEIGHT_DIRS = {"vanilla_model_weights", "soluble_model_weights", "ca_model_weights"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ProteinMpnSourceAdapter:
    """Read checkpoint filenames in the official ``dauparas/ProteinMPNN`` tree.

    GitHub's tree is an immutable, source-authored manifest. The adapter admits
    only ``.pt`` files directly inside the three checkpoint directories named
    by the official README; it never downloads weight bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers .pt checkpoints in vanilla_model_weights, soluble_model_weights, "
        "and ca_model_weights in the official dauparas/ProteinMPNN repository. "
        "It does not cover forks, external fine-tunes, or download checkpoint bytes."
    )

    def __init__(
        self, *, name: str = "proteinmpnn-checkpoints",
        repository: str = "dauparas/ProteinMPNN", branch: str = "main",
        max_response_bytes: int = 16 *  1024 * 1024,
        client: HttpClient | Any | None = None, clock: Clock = _utcnow,
    ) -> None:
        self.name, self.repository, self.branch = name, repository, branch
        if not name.strip() or "/" not in repository or not branch.strip():
            raise ValueError("name, owner/repository, and branch are required")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "proteinmpnn-tree-v1", "repository": repository,
            "branch": branch, "max_response_bytes": max_response_bytes,
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
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
        ):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage((), {**state, "checked_at": checked}, True,
                              upstream_count=state.get("model_count"))
        tree = self.client.get(
            f"https://api.github.com/repos/{self.repository}/git/trees/{revision}?recursive=1",
            headers={"Accept": "application/vnd.github+json"},
        )
        if tree.status != 200:
            raise ValueError(f"{self.name}: tree endpoint returned HTTP {tree.status}")
        if len(tree.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: tree exceeds response limit")
        tree_data = tree.json()
        if not isinstance(tree_data, Mapping) or tree_data.get("truncated") is True:
            raise ValueError(f"{self.name}: tree is missing or truncated")
        entries = tree_data.get("tree")
        if not isinstance(entries, list):
            raise ValueError(f"{self.name}: tree has no entry list")
        paths = sorted({
            item["path"] for item in entries
            if isinstance(item, Mapping) and isinstance(item.get("path"), str)
            and item.get("type") == "blob" and item["path"].endswith(".pt")
            and item["path"].split("/", 1)[0] in _WEIGHT_DIRS
            and "/" not in item["path"].split("/", 1)[-1]
        })
        if not paths:
            raise ValueError(f"{self.name}: no official checkpoints found")
        records = tuple(self._record(path, revision) for path in paths)
        return SourcePage(records, {"completed_revision": revision,
                                    "checked_at": checked, "model_count": len(records)},
                          True, upstream_count=len(records), authoritative_snapshot=True)

    def _record(self, path: str, revision: str) -> SourceRecord:
        handle = path[:-3]
        name = path.rsplit("/", 1)[-1][:-3]
        weight_url = (
            f"https://raw.githubusercontent.com/{self.repository}/{revision}/"
            f"{quote(path, safe='/')}"
        )
        page_url = f"{self.repository_url}/blob/{revision}/{quote(path, safe='/')}"
        model_id = f"model:{handle}"
        namespace = "proteinmpnn:checkpoint"
        model = ModelHint(model_id, name, (Identifier(namespace, handle),),
                          aliases=(handle,), status=ModelStatus.RELEASED)
        release = ReleaseHint(f"release:{handle}", model_id, revision=revision,
                              identifiers=(Identifier(f"{namespace}:release", handle),),
                              metadata={"repository": self.repository, "path": path,
                                        "weight_url": weight_url})
        return SourceRecord(
            source_record_id=f"checkpoint:{handle}", kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(page_url), title=f"ProteinMPNN {name}",
            raw={"repository": self.repository, "revision": revision,
                 "checkpoint_path": path, "weight_url": weight_url},
            text=f"Official ProteinMPNN checkpoint: {handle}",
            identifiers=(Identifier(namespace, handle),),
            links=(Link(page_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                   Link(self.repository_url, "source_implementation", crawl=False,
                        model_local_ids=(model_id,)),
                   Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,))),
            models=(model,), releases=(release,),
        )


__all__ = ["ProteinMpnSourceAdapter"]
