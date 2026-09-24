"""PaddleNLP Taskflow's explicit Chinese spelling correction checkpoint map."""

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
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HOSTS = {"bj.bcebos.com", "paddlenlp.bj.bcebos.com"}
_ARTIFACT_PATHS = ("/taskflow/text_correction/", "/paddlenlp/taskflow/text_correction/")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_resource_map(document: str) -> tuple[tuple[str, str, str], ...]:
    """Read only CSCTask's literal public model_state checkpoint."""
    try:
        module = ast.parse(document)
    except SyntaxError as exc:
        raise ValueError("paddlenlp-taskflow-text-correction: invalid Python source") from exc
    classes = [
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "CSCTask"
    ]
    if len(classes) != 1:
        raise ValueError("paddlenlp-taskflow-text-correction: expected one CSCTask class")
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
        raise ValueError("paddlenlp-taskflow-text-correction: expected one resource map")
    try:
        resources_by_model = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError("paddlenlp-taskflow-text-correction: resource map is not literal") from exc
    resources = (
        resources_by_model.get("ernie-csc") if isinstance(resources_by_model, dict) else None
    )
    model_state = resources.get("model_state") if isinstance(resources, dict) else None
    if (
        not isinstance(model_state, list)
        or len(model_state) != 2
        or not all(isinstance(item, str) for item in model_state)
    ):
        raise ValueError("paddlenlp-taskflow-text-correction: missing literal ernie-csc weights")
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
        or not parsed.path.endswith("/ernie-csc/model_state.pdparams")
        or not _MD5.fullmatch(md5)
    ):
        raise ValueError("paddlenlp-taskflow-text-correction: invalid ernie-csc artifact")
    return (("ernie-csc", artifact_url, md5),)


class PaddleNlpTaskflowTextCorrectionSourceAdapter:
    """Enumerate the official Taskflow ernie-csc checkpoint."""

    disable_derived_extraction = True
    coverage_limitation = "Covers the public CSCTask ernie-csc model_state checkpoint only."

    def __init__(
        self,
        *,
        name: str = "paddlenlp-taskflow-text-correction-checkpoints",
        repository: str = "PaddlePaddle/PaddleNLP",
        branch: str = "develop",
        source_path: str = "paddlenlp/taskflow/text_correction.py",
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name is required")
        if not _REPOSITORY.fullmatch(repository):
            raise ValueError("repository must use owner/name form")
        if (
            not branch.strip()
            or not source_path
            or source_path.startswith("/")
            or ".." in source_path.split("/")
        ):
            raise ValueError("branch and relative source_path are required")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be positive")
        self.name = name.strip()
        self.repository = repository
        self.branch = branch.strip()
        self.source_path = source_path
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paddlenlp-taskflow-text-correction-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
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
            raise ValueError(f"{self.name}: invalid commit revision")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                records=(),
                next_state={**state, "checked_at": checked_at},
                complete=True,
                upstream_count=1,
            )
        source_url = self.raw_url(revision)
        response = self.client.get(
            source_url, headers={"Accept": "text/plain,application/octet-stream"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source file returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source file exceeds response limit")
        source_hash = hashlib.sha256(response.body).hexdigest()
        records = (self._record(*_parse_resource_map(response.text())[0], revision, source_hash),)
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": source_hash,
                "checkpoint_count": 1,
            },
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _record(
        self, model_name: str, artifact_url: str, md5: str, revision: str, source_hash: str
    ) -> SourceRecord:
        identifier = Identifier("paddlenlp:taskflow-text-correction", model_name)
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
            text=f"PaddleNLP Taskflow text-correction checkpoint: {model_name}",
            identifiers=(identifier,),
            links=(
                Link(artifact_url, relation="weights", locator="model_state.pdparams", crawl=False),
                Link(
                    f"https://github.com/{self.repository}/blob/{quote(revision, safe='')}/"
                    f"{quote(self.source_path, safe='/')}",
                    relation="model_card",
                    locator="CSCTask.resource_files_urls:ernie-csc",
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
                    locator="CSCTask.resource_files_urls:ernie-csc",
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{model_name}",
                    model_local_id=f"model:{model_name}",
                    revision=revision,
                    identifiers=(
                        Identifier("paddlenlp:taskflow-text-correction-release", model_name),
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


__all__ = ["PaddleNlpTaskflowTextCorrectionSourceAdapter"]
