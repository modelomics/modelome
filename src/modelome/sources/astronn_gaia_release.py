"""Pinned astroNN Gaia DR2 paper model-directory releases.

The authors' paper repository names five trained model directories and documents
loading them with ``astroNN.models.load_folder``.  This adapter records those
directory locations at a pinned Git commit; it does not infer or list files inside
them and does not claim that a particular weight file is directly downloadable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from modelome.http import HttpResponse
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
from modelome.normalize import content_hash

Clock = Any
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MODEL_DIRECTORIES = (
    "astroNN_no_offset_model",
    "astroNN_constant_model",
    "astroNN_constant_model_reduced",
    "astroNN_multivariate_model",
    "astroNN_multivariate_model_reduced",
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AstroNNGaiaReleaseSourceAdapter:
    """Record five first-party paper-repository model directory releases."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the five model directories named in the astroNN Gaia DR2 "
        "paper repository README. Each record links to that directory at an "
        "observed commit. Directory contents and individual checkpoint files "
        "are not enumerated; the adapter makes no direct-download claim."
    )

    def __init__(
        self,
        *,
        name: str = "astronn-gaia-dr2-model-releases",
        repository: str = "henrysky/astroNN_gaia_dr2_paper",
        branch: str = "master",
        max_response_bytes: int = 8 * 1024 * 1024,
        client: Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        if client is None:
            from modelome.http import HttpClient

            client = HttpClient(max_response_bytes=self.max_response_bytes)
        self.client = client
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "astronn-gaia-release-v1",
                "repository": self.repository,
                "branch": self.branch,
                "directories": _MODEL_DIRECTORIES,
                "max_response_bytes": self.max_response_bytes,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
        )

    def tree_url(self, revision: str) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/git/trees/"
            f"{quote(revision, safe='')}?recursive=1"
        )

    def directory_url(self, revision: str, directory: str) -> str:
        return (
            f"{self.repository_url}/tree/{quote(revision, safe='')}/"
            f"{quote(directory, safe='/')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        _check_response(commit_response, self.name, "commit", self.max_response_bytes)
        commit_payload = commit_response.json()
        revision = (
            commit_payload.get("sha") if isinstance(commit_payload, Mapping) else None
        )
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = self.clock().astimezone(UTC).isoformat()
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=len(_MODEL_DIRECTORIES),
            )

        tree_response: HttpResponse = self.client.get(
            self.tree_url(revision),
            headers={"Accept": "application/vnd.github+json"},
        )
        _check_response(tree_response, self.name, "git tree", self.max_response_bytes)
        tree_payload = tree_response.json()
        if not isinstance(tree_payload, Mapping) or tree_payload.get("truncated") is True:
            raise ValueError(f"{self.name}: Git tree is missing or truncated")
        entries = tree_payload.get("tree")
        if not isinstance(entries, list):
            raise ValueError(f"{self.name}: Git tree has no entry list")
        directories = {
            item.get("path")
            for item in entries
            if isinstance(item, Mapping) and item.get("type") == "tree"
        }
        missing = set(_MODEL_DIRECTORIES) - directories
        if missing:
            raise ValueError(
                f"{self.name}: expected model directories absent at {revision}: "
                f"{', '.join(sorted(missing))}"
            )

        records = tuple(self._record(directory, revision) for directory in _MODEL_DIRECTORIES)
        next_state = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "tree_url": self.tree_url(revision),
            "model_count": len(records),
        }
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, directory: str, revision: str) -> SourceRecord:
        identifier = Identifier("astronn-gaia-dr2-paper:model-directory", directory)
        page_url = self.directory_url(revision, directory)
        model = ModelHint(
            local_id=f"model:{directory}",
            name=directory,
            aliases=(directory,),
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=f"repository tree:{directory}",
        )
        release = ReleaseHint(
            local_id=f"release:{directory}",
            model_local_id=model.local_id,
            revision=revision,
            identifiers=(
                Identifier("astronn-gaia-dr2-paper:release-directory", directory),
            ),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "model_directory": directory,
                "directory_page_url": page_url,
                "artifact_locator_kind": "repository_directory",
            },
            locator=f"repository tree:{directory}",
        )
        return SourceRecord(
            source_record_id=f"model-directory:{directory}",
            kind=ArtifactKind.PROVIDER_PAGE,
            canonical_url=page_url,
            title=directory,
            raw={
                "repository": self.repository,
                "revision": revision,
                "model_directory": directory,
                "artifact_locator_kind": "repository_directory",
            },
            text=(
                f"First-party astroNN Gaia DR2 paper model directory {directory}; "
                "directory contents are not enumerated."
            ),
            identifiers=(identifier,),
            links=(
                Link(
                    page_url,
                    relation="model_release_directory",
                    locator=f"repository tree:{directory}",
                    crawl=False,
                    model_local_ids=(model.local_id,),
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _check_response(response: HttpResponse, source: str, label: str, maximum: int) -> None:
    if response.status != 200:
        raise ValueError(f"{source}: {label} endpoint returned HTTP {response.status}")
    if len(response.body) > maximum:
        raise ValueError(f"{source}: {label} response exceeds {maximum} bytes")


def _repository(value: str) -> str:
    text = _required_text(value, "repository")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text):
        raise ValueError("repository must be owner/name")
    return text


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _positive_int(value: int, label: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""
