"""Exact checkpoint references documented by TensorFlow's AudioSet model READMEs."""

from __future__ import annotations

import re
from collections.abc import Mapping
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

_REPOSITORY = "tensorflow/models"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_AUDIOSET_URL = re.compile(r"https://storage\.googleapis\.com/audioset/[A-Za-z0-9_.-]+")
_DOCUMENTS = {
    "research/audioset/yamnet/README.md": (
        "yamnet",
        "YAMNet",
        ("https://storage.googleapis.com/audioset/yamnet.h5",),
        "task:audio-event-classification",
    ),
    "research/audioset/vggish/README.md": (
        "vggish",
        "VGGish",
        (
            "https://storage.googleapis.com/audioset/vggish_model.ckpt",
            "https://storage.googleapis.com/audioset/vggish_pca_params.npz",
        ),
        "task:audio-embedding",
    ),
}


class TensorFlowAudioSetCheckpointSourceAdapter:
    """Read an exact AudioSet model/checkpoint map from one first-party README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the YAMNet and VGGish artifacts explicitly linked by their "
        "TensorFlow Models AudioSet README files. It does not enumerate other "
        "TensorFlow research models or infer additional weights."
    )

    def __init__(
        self,
        *,
        name: str = "tensorflow-audioset-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "master",
        document_path: str,
        max_bytes: int = 2 * 1024 * 1024,
        client: Any | None = None,
    ) -> None:
        if document_path not in _DOCUMENTS:
            raise ValueError("document_path must select the YAMNet or VGGish README")
        if repository != _REPOSITORY:
            raise ValueError("repository must be tensorflow/models")
        if branch != "master":
            raise ValueError("branch must be master")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        self.name = name.strip()
        self.document_path = document_path
        self.max_bytes = max_bytes
        self.client = client or HttpClient(max_response_bytes=max_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "tensorflow-audioset-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "document_path": document_path,
                "max_bytes": max_bytes,
            }
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/commits/master"

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/"
            f"{quote(self.document_path, safe='/')}"
        )

    def blob_url(self, revision: str) -> str:
        return (
            f"https://github.com/{_REPOSITORY}/blob/{revision}/"
            f"{quote(self.document_path, safe='/')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=1,
            )

        response = self.client.get(self.raw_url(revision), headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: model README returned HTTP {response.status}")
        if len(response.body) > self.max_bytes:
            raise ValueError(f"{self.name}: model README exceeds {self.max_bytes} bytes")
        model_id, model_name, declared_weights, task = _DOCUMENTS[self.document_path]
        found = set(_AUDIOSET_URL.findall(response.text()))
        if not set(declared_weights).issubset(found):
            raise ValueError(f"{self.name}: README lacks its documented checkpoint map")
        record = self._record(model_id, model_name, declared_weights, task, revision)
        return SourcePage(
            records=(record,),
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "document_url": self.raw_url(revision),
                "document_sha256": content_hash(response.body),
                "model_count": 1,
            },
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _record(
        self,
        model_id: str,
        model_name: str,
        declared_weights: tuple[str, ...],
        task: str,
        revision: str,
    ) -> SourceRecord:
        model_local_id = f"model:{model_id}"
        locator = f"{self.document_path}:documented checkpoint links"
        doc_url = self.blob_url(revision)
        checkpoint_id = f"{model_id}@{revision}"
        weights = tuple(canonicalize_url(url) for url in declared_weights)
        model = ModelHint(
            local_id=model_local_id,
            name=model_name,
            identifiers=(Identifier("tensorflow-audioset:model", model_id),),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{checkpoint_id}",
            model_local_id=model_local_id,
            revision=revision,
            identifiers=(Identifier("tensorflow-audioset:checkpoint-map", checkpoint_id),),
            metadata={"task": task.removeprefix("task:"), "checkpoint_urls": list(weights)},
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"tensorflow-audioset:{model_id}:{revision}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(doc_url),
            title=model_name,
            raw={
                "repository": _REPOSITORY,
                "revision": revision,
                "document_path": self.document_path,
                "checkpoint_urls": list(weights),
            },
            text=f"{model_name}\ntask: {task.removeprefix('task:')}\n" + "\n".join(weights),
            identifiers=(Identifier("tensorflow-audioset:model", model_id),),
            links=(
                Link(doc_url, relation="model_card", locator=locator, crawl=False),
                *(Link(url, relation="weights", locator=locator, crawl=False) for url in weights),
            ),
            models=(model,),
            releases=(release,),
        )
