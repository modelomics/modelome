"""Microsoft Research MoLeR's explicitly linked pretrained checkpoint."""

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
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = "microsoft/molecule-generation"
_SOURCE_PATH = "README.md"
_CHECKPOINT_FILENAME = "GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl"
_WEIGHT_URL = "https://figshare.com/ndownloader/files/34642724"
_CHECKPOINT_LINK = re.compile(
    r"A MoLeR checkpoint trained using the default hyperparameters is available "
    r"\[here\]\((https://[^)]+)\)"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MoLeRCheckpointSourceAdapter:
    """Read MoLeR's single pretrained checkpoint link from its first-party README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only Microsoft's explicitly linked default MoLeR checkpoint. "
        "The linked artifact is hosted on Figshare; model bytes are not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "moler-default-checkpoint",
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
                "adapter": "moler-default-checkpoint-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "expected_url": _WEIGHT_URL,
                "expected_filename": _CHECKPOINT_FILENAME,
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
        weight_url = _checkpoint_url(response.text(), self.name)
        record = self._record(revision, source_url, weight_url, content_hash(response.body))
        return SourcePage(
            (record,),
            {
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": content_hash(response.body),
                "model_count": 1,
            },
            True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _record(
        self, revision: str, source_url: str, weight_url: str, source_hash: str
    ) -> SourceRecord:
        handle = "moler-default-2022-02-24"
        namespace = "microsoft-moler:checkpoint"
        model_id = f"model:{handle}"
        model = ModelHint(
            model_id,
            "MoLeR default pretrained checkpoint",
            identifiers=(Identifier(namespace, handle),),
            aliases=(_CHECKPOINT_FILENAME, "MoLeR default checkpoint"),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            version="2022-02-24",
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "checkpoint_filename": _CHECKPOINT_FILENAME,
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
                "checkpoint_filename": _CHECKPOINT_FILENAME,
                "weight_url": weight_url,
            },
            text="Microsoft's MoLeR README links this default pretrained checkpoint.",
            links=(
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(blob_url, "source_implementation", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _checkpoint_url(document: str, source: str) -> str:
    matches = _CHECKPOINT_LINK.findall(document)
    if matches != [_WEIGHT_URL]:
        raise ValueError(f"{source}: expected the exact first-party MoLeR checkpoint link")
    parsed = urlsplit(matches[0])
    if parsed.scheme != "https" or parsed.hostname != "figshare.com":
        raise ValueError(f"{source}: checkpoint link must use HTTPS Figshare")
    return matches[0]


__all__ = ["MoLeRCheckpointSourceAdapter"]
