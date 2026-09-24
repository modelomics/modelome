"""DeepModeling Uni-Mol's literal GitHub-release checkpoint links."""

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
_REPOSITORY = "deepmodeling/Uni-Mol"
_SOURCE_PATH = "unimol/README.md"
_CHECKPOINTS = {
    "mol_pre_no_h_220816.pt": "molecular-pretrain-no-hydrogen",
    "mol_pre_all_h_220816.pt": "molecular-pretrain-all-hydrogen",
    "pocket_pre_220816.pt": "pocket-pretrain",
    "qm9_220908.pt": "conformation-generation-qm9",
    "drugs_220908.pt": "conformation-generation-drugs",
    "binding_pose_220908.pt": "protein-ligand-binding-pose",
}
_URL_PATTERN = re.compile(
    r"https://github\.com/deepmodeling/Uni-Mol/releases/download/v0\.1/"
    r"([A-Za-z0-9_.-]+\.pt)"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class UniMolCheckpointSourceAdapter:
    """Index six exact checkpoint links published in Uni-Mol's first-party README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers six Uni-Mol v0.1 checkpoint links listed in the first-party README. "
        "It does not cover Uni-Mol2/Hugging Face checkpoints or fetch model bytes."
    )

    def __init__(
        self,
        *,
        name: str = "unimol-release-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "main",
        source_path: str = _SOURCE_PATH,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if (
            repository != _REPOSITORY
            or source_path != _SOURCE_PATH
            or not branch.strip()
            or not name.strip()
            or max_response_bytes <= 0
        ):
            raise ValueError(
                "repository/source path are fixed; name, branch, and limit are required"
            )
        self.name, self.repository, self.branch = name, repository, branch
        self.source_path = source_path
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "unimol-release-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "checkpoints": _CHECKPOINTS,
                "max_response_bytes": max_response_bytes,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_url = (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
        )
        commit_response = self.client.get(
            commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        payload = commit_response.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                (), {**state, "checked_at": checked_at}, True,
                upstream_count=state.get("model_count"),
            )

        source_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(self.source_path, safe='/')}"
        )
        response = self.client.get(source_url, headers={"Accept": "text/markdown,text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: source returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source exceeds {self.max_response_bytes} bytes")
        urls = _parse_checkpoint_urls(response.text(), self.name)
        source_hash = content_hash(response.body)
        records = tuple(
            self._record(revision, source_url, source_hash, filename, url)
            for filename, url in urls
        )
        return SourcePage(
            records,
            {
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": source_hash,
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self, revision: str, source_url: str, source_hash: str,
        filename: str, weight_url: str,
    ) -> SourceRecord:
        handle = _CHECKPOINTS[filename]
        namespace = "unimol:checkpoint"
        model_id = f"model:{handle}"
        name = handle.replace("-", " ").title()
        model = ModelHint(
            model_id,
            f"Uni-Mol {name} checkpoint",
            identifiers=(Identifier(namespace, handle),),
            aliases=(filename,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            version="v0.1",
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "checkpoint_filename": filename,
                "weight_url": weight_url,
                "source_sha256": source_hash,
            },
        )
        blob_url = (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(weight_url),
            title=model.name,
            raw={
                "checkpoint_handle": handle,
                "checkpoint_filename": filename,
                "weight_url": weight_url,
            },
            text=f"Uni-Mol's first-party README lists the {name} checkpoint.",
            links=(
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(blob_url, "source_implementation", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_checkpoint_urls(document: str, source: str) -> tuple[tuple[str, str], ...]:
    found: dict[str, str] = {}
    for filename in _URL_PATTERN.findall(document):
        if filename not in _CHECKPOINTS or filename in found:
            raise ValueError(f"{source}: unexpected or duplicate Uni-Mol checkpoint URL")
        found[filename] = (
            "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/" + filename
        )
    if set(found) != set(_CHECKPOINTS):
        raise ValueError(f"{source}: expected all six Uni-Mol checkpoint URLs")
    return tuple((filename, found[filename]) for filename in _CHECKPOINTS)


__all__ = ["UniMolCheckpointSourceAdapter"]
