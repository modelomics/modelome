"""PaddleNLP Taskflow's explicit text-similarity checkpoint map."""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient, HttpResponse
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
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HOSTS = {"bj.bcebos.com", "paddlenlp.bj.bcebos.com"}
_ARTIFACT_PATHS = ("/taskflow/text_similarity/", "/paddlenlp/taskflow/text_similarity/")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_resource_map(document: str, *, max_entries: int) -> tuple[tuple[str, str, str], ...]:
    """Read TextSimilarityTask's literal map without executing Python."""
    try:
        module = ast.parse(document)
    except SyntaxError as exc:
        raise ValueError("paddlenlp-taskflow-text-similarity: source is not valid Python") from exc
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "TextSimilarityTask"
    ]
    if len(classes) != 1:
        raise ValueError(
            "paddlenlp-taskflow-text-similarity: expected one TextSimilarityTask class"
        )
    assignments = [
        node
        for node in classes[0].body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "resource_files_urls"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("paddlenlp-taskflow-text-similarity: expected one resource_files_urls map")
    try:
        resources_by_model = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError("paddlenlp-taskflow-text-similarity: resource map is not literal") from exc
    if not isinstance(resources_by_model, dict):
        raise ValueError("paddlenlp-taskflow-text-similarity: resource map is not a mapping")

    entries: dict[str, tuple[str, str]] = {}
    for model_name, resources in resources_by_model.items():
        if (
            not isinstance(model_name, str)
            or not _MODEL_NAME.fullmatch(model_name)
            or model_name.startswith("__internal_testing__/")
        ):
            continue
        if not isinstance(resources, dict):
            continue
        model_state = resources.get("model_state")
        if (
            not isinstance(model_state, list)
            or len(model_state) != 2
            or not all(isinstance(item, str) for item in model_state)
        ):
            continue
        artifact_url, md5 = model_state
        parsed = urlsplit(artifact_url)
        if (
            parsed.scheme != "https"
            or (parsed.hostname or "").casefold() not in _HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith(_ARTIFACT_PATHS)
            or not parsed.path.casefold().endswith(".pdparams")
            or not _MD5.fullmatch(md5)
        ):
            continue
        value = (artifact_url, md5)
        if model_name in entries and entries[model_name] != value:
            raise ValueError(
                f"paddlenlp-taskflow-text-similarity: conflicting entry for {model_name}"
            )
        entries[model_name] = value
        if len(entries) > max_entries:
            raise ValueError(
                f"paddlenlp-taskflow-text-similarity: source exceeds {max_entries} public models"
            )
    if not entries:
        raise ValueError("paddlenlp-taskflow-text-similarity: no public checkpoints found")
    return tuple((name, *entries[name]) for name in sorted(entries))


class PaddleNlpTaskflowTextSimilaritySourceAdapter:
    """Enumerate public Taskflow text-similarity model_state checkpoints."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public TextSimilarityTask model_state .pdparams URLs and MD5s. "
        "Excludes internal fixtures, auxiliary config files, and other Taskflow tasks."
    )

    def __init__(
        self,
        *,
        name: str = "paddlenlp-taskflow-text-similarity-checkpoints",
        repository: str = "PaddlePaddle/PaddleNLP",
        branch: str = "develop",
        source_path: str = "paddlenlp/taskflow/text_similarity.py",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 100,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name is required")
        if not isinstance(repository, str) or not _REPOSITORY.fullmatch(repository):
            raise ValueError("repository must use owner/name form")
        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("branch is required")
        if (
            not isinstance(source_path, str)
            or not source_path
            or source_path.startswith("/")
            or ".." in source_path.split("/")
        ):
            raise ValueError("source_path must be a relative repository path")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.name = name.strip()
        self.repository = repository
        self.branch = branch.strip()
        self.source_path = source_path
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paddlenlp-taskflow-text-similarity-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "literal public model_state .pdparams URL with source MD5",
            }
        )

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )

    def blob_url(self, revision: str) -> str:
        return (
            f"https://github.com/{self.repository}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = (
            payload.get("sha", "").strip()
            if isinstance(payload, Mapping) and isinstance(payload.get("sha"), str)
            else ""
        )
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                records=(),
                next_state={**state, "checked_at": checked_at},
                complete=True,
                upstream_count=state.get("checkpoint_count"),
            )
        source_url = self.raw_url(revision)
        response = self.client.get(
            source_url, headers={"Accept": "text/plain,application/octet-stream"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source file returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source file exceeds {self.max_response_bytes} bytes")
        source_hash = hashlib.sha256(response.body).hexdigest()
        records = tuple(
            self._record(model_name, url, md5, revision, source_hash)
            for model_name, url, md5 in _parse_resource_map(
                response.text(), max_entries=self.max_entries
            )
        )
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": source_hash,
                "checkpoint_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self, model_name: str, artifact_url: str, md5: str, revision: str, source_hash: str
    ) -> SourceRecord:
        identifier = Identifier("paddlenlp:taskflow-text-similarity", model_name)
        return SourceRecord(
            source_record_id=f"model:{model_name}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"{model_name} model_state.pdparams",
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": source_hash,
                "model_name": model_name,
                "artifact_url": artifact_url,
                "md5": md5,
            },
            text=f"PaddleNLP Taskflow text-similarity checkpoint: {model_name}",
            identifiers=(identifier,),
            links=(
                Link(artifact_url, relation="weights", locator="model_state.pdparams", crawl=False),
                Link(
                    self.blob_url(revision),
                    relation="model_card",
                    locator=f"resource_files_urls:{model_name}",
                    crawl=False,
                ),
                Link(
                    f"https://github.com/{self.repository}",
                    relation="source_repository",
                    crawl=False,
                ),
            ),
            models=(
                ModelHint(
                    local_id=f"model:{model_name}",
                    name=model_name,
                    identifiers=(identifier,),
                    status=ModelStatus.RELEASED,
                    locator=f"resource_files_urls:{model_name}.model_state",
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{model_name}",
                    model_local_id=f"model:{model_name}",
                    revision=revision,
                    identifiers=(
                        Identifier("paddlenlp:taskflow-text-similarity-release", model_name),
                    ),
                    metadata={
                        "artifact_url": artifact_url,
                        "md5": md5,
                        "source_revision": revision,
                    },
                    locator=artifact_url,
                ),
            ),
        )


__all__ = ["PaddleNlpTaskflowTextSimilaritySourceAdapter"]
