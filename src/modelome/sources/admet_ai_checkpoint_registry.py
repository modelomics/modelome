"""First-party ADMET-AI Chemprop checkpoint files in its repository tree."""

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
_TREE_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = "swansonk14/admet_ai"
_MODEL_PATH = re.compile(
    r"^admet_ai/resources/models/(admet_classification|admet_regression)/model_([0-4])\.pt$"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ADMETAICheckpointRegistrySourceAdapter:
    """Enumerate the exact 5+5 released model files from ADMET-AI's source tree."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the five .pt files in each of ADMET-AI's classification and "
        "regression checkpoint directories. It excludes datasets, user checkpoints, "
        "older model releases, and unrelated repository files; weight bytes are not fetched."
    )

    def __init__(
        self,
        *,
        name: str = "admet-ai-chemprop-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "main",
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if repository != _REPOSITORY or not branch.strip() or not name.strip():
            raise ValueError("repository is fixed; name and branch are required")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "admet-ai-chemprop-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "model_path_pattern": _MODEL_PATH.pattern,
                "max_response_bytes": max_response_bytes,
                "admission": (
                    "exact model_0.pt through model_4.pt in classification/regression dirs"
                ),
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_url = (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )
        commit_response = self.client.get(
            commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        commit = commit_response.json()
        revision = commit.get("sha") if isinstance(commit, Mapping) else None
        tree = commit.get("commit", {}).get("tree", {}) if isinstance(commit, Mapping) else {}
        tree_sha = tree.get("sha") if isinstance(tree, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        if not isinstance(tree_sha, str) or not _TREE_SHA.fullmatch(tree_sha):
            raise ValueError(f"{self.name}: commit endpoint did not include a valid tree SHA")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                (),
                {**state, "checked_at": checked_at},
                True,
                upstream_count=state.get("model_count"),
            )

        tree_url = f"https://api.github.com/repos/{self.repository}/git/trees/{tree_sha}"
        tree_response = self.client.get(
            tree_url,
            params={"recursive": "1"},
            headers={"Accept": "application/vnd.github+json"},
        )
        if tree_response.status != 200:
            raise ValueError(f"{self.name}: Git tree returned HTTP {tree_response.status}")
        if len(tree_response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: Git tree exceeds {self.max_response_bytes} bytes")
        payload = tree_response.json()
        entries = payload.get("tree") if isinstance(payload, Mapping) else None
        if not isinstance(entries, list):
            raise ValueError(f"{self.name}: Git tree response has no entry list")
        if payload.get("truncated") is not False:
            raise ValueError(
                f"{self.name}: Git tree response is truncated or lacks completeness flag"
            )
        checkpoints: list[tuple[str, str, str]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            path = entry.get("path")
            if not isinstance(path, str):
                continue
            match = _MODEL_PATH.fullmatch(path)
            if not match:
                continue
            if entry.get("type") != "blob" or not isinstance(entry.get("sha"), str):
                raise ValueError(f"{self.name}: checkpoint path is not a Git blob: {path}")
            checkpoints.append((path, match.group(1), match.group(2)))
        checkpoints.sort()
        expected_count = 10
        if len(checkpoints) != expected_count:
            raise ValueError(
                f"{self.name}: expected {expected_count} recognized checkpoint files, "
                f"found {len(checkpoints)}"
            )
        records = tuple(
            self._record(revision, path, family, index) for path, family, index in checkpoints
        )
        return SourcePage(
            records,
            {
                "completed_revision": revision,
                "checked_at": checked_at,
                "tree_sha": tree_sha,
                "model_count": len(records),
                "source_sha256": content_hash(tree_response.body),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, revision: str, path: str, family: str, index: str) -> SourceRecord:
        handle = f"{family}/model_{index}"
        namespace = "admet-ai:checkpoint"
        model_id = f"model:{handle}"
        family_label = "classification" if family == "admet_classification" else "regression"
        model = ModelHint(
            model_id,
            f"ADMET-AI {family_label} Chemprop ensemble member {index}",
            identifiers=(Identifier(namespace, handle),),
            aliases=(f"{family}/model_{index}.pt",),
            status=ModelStatus.RELEASED,
        )
        weight_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(path, safe='/')}"
        )
        blob_url = f"{self.repository_url}/blob/{quote(revision, safe='')}/{quote(path, safe='/')}"
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            revision=revision,
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "checkpoint_path": path,
                "checkpoint_filename": f"model_{index}.pt",
                "ensemble_family": family_label,
                "ensemble_member": int(index),
                "weight_url": weight_url,
            },
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(blob_url),
            title=model.name,
            raw={
                "checkpoint_handle": handle,
                "checkpoint_path": path,
                "weight_url": weight_url,
                "ensemble_family": family_label,
                "ensemble_member": int(index),
            },
            text=(
                f"ADMET-AI's Chemprop {family_label} checkpoint member "
                f"{index}, listed in the repository's model resources."
            ),
            links=(
                Link(blob_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


__all__ = ["ADMETAICheckpointRegistrySourceAdapter"]
