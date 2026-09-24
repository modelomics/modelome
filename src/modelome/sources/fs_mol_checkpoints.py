"""Microsoft FS-Mol's first-party few-shot molecular checkpoint table."""

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
_REPOSITORY = "microsoft/FS-Mol"
_SOURCE_PATH = "README.md"
_CHECKPOINTS = {
    "GNN-MAML": ("MAML-Support16_best_validation.pkl", "31346701"),
    "GNN-MT": ("multitask_best_model.pt", "31338334"),
    "PN": ("PN-Support64_best_validation.pt", "31307479"),
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class FSMolCheckpointSourceAdapter:
    """Parse FS-Mol's literal pretrained checkpoint table without executing code."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the three pretrained model links in Microsoft's FS-Mol README. "
        "Artifacts are hosted on Figshare; checkpoint bytes are not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "fs-mol-checkpoints",
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
                "adapter": "fs-mol-checkpoints-v1",
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
        rows = _parse_checkpoint_table(response.text(), self.name)
        source_hash = content_hash(response.body)
        records = tuple(
            self._record(revision, source_url, source_hash, model, filename, weight_url)
            for model, filename, weight_url in rows
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
        model_name: str, filename: str, weight_url: str,
    ) -> SourceRecord:
        handle = model_name.lower()
        namespace = "fs-mol:checkpoint"
        model_id = f"model:{handle}"
        model = ModelHint(
            model_id,
            f"FS-Mol {model_name} pretrained checkpoint",
            identifiers=(Identifier(namespace, model_name),),
            aliases=(filename,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            identifiers=(Identifier(f"{namespace}:release", model_name),),
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
                "checkpoint_handle": model_name,
                "checkpoint_filename": filename,
                "weight_url": weight_url,
            },
            text=f"Microsoft's FS-Mol README lists the pretrained {model_name} checkpoint.",
            links=(
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(blob_url, "source_implementation", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_checkpoint_table(document: str, source: str) -> tuple[tuple[str, str, str], ...]:
    found: dict[str, tuple[str, str]] = {}
    for line in document.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] not in _CHECKPOINTS:
            continue
        match = re.fullmatch(
            r"\[([^\]]+)\]\((https://figshare\.com/ndownloader/files/(\d+))\)",
            cells[2],
        )
        if match is None:
            raise ValueError(f"{source}: invalid Figshare link for {cells[0]}")
        filename, expected_id = _CHECKPOINTS[cells[0]]
        label, url, file_id = match.groups()
        if label != filename or file_id != expected_id or cells[0] in found:
            raise ValueError(f"{source}: unexpected or duplicate checkpoint row for {cells[0]}")
        found[cells[0]] = (filename, url)
    if set(found) != set(_CHECKPOINTS):
        raise ValueError(f"{source}: expected exactly the three FS-Mol checkpoint rows")
    return tuple((model, *found[model]) for model in _CHECKPOINTS)


__all__ = ["FSMolCheckpointSourceAdapter"]
