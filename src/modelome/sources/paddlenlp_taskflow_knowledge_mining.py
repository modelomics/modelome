"""PaddleNLP Taskflow's explicit knowledge-mining checkpoint resource maps."""

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
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MAP_CLASSES = {"WordTagTask", "NPTagTask"}
_ARTIFACT_PATH = "/paddlenlp/taskflow/knowledge_mining/"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_resource_maps(document: str, *, max_entries: int) -> tuple[tuple[str, str, str], ...]:
    """Read only literal model_state entries on the named public task classes."""
    try:
        module = ast.parse(document)
    except SyntaxError as exc:
        raise ValueError("paddlenlp-taskflow-knowledge-mining: source is not valid Python") from exc
    found: dict[str, tuple[str, str]] = {}
    classes = [
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name in _MAP_CLASSES
    ]
    if {node.name for node in classes} != _MAP_CLASSES:
        raise ValueError(
            "paddlenlp-taskflow-knowledge-mining: expected WordTagTask and NPTagTask maps"
        )
    for task_class in classes:
        assignments = [
            node
            for node in task_class.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "resource_files_urls"
                for target in node.targets
            )
        ]
        if len(assignments) != 1:
            raise ValueError(
                f"paddlenlp-taskflow-knowledge-mining: expected one map in {task_class.name}"
            )
        try:
            resource_map = ast.literal_eval(assignments[0].value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
            raise ValueError(
                "paddlenlp-taskflow-knowledge-mining: resource map is not literal"
            ) from exc
        if not isinstance(resource_map, dict):
            raise ValueError("paddlenlp-taskflow-knowledge-mining: resource map is not a mapping")
        for name, resources in resource_map.items():
            if (
                not isinstance(name, str)
                or not _NAME.fullmatch(name)
                or name.startswith("__internal_testing__/")
            ):
                continue
            if not isinstance(resources, dict):
                continue
            weight = resources.get("model_state")
            if (
                not isinstance(weight, list)
                or len(weight) != 2
                or not all(isinstance(v, str) for v in weight)
            ):
                continue
            url, md5 = weight
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.hostname != "bj.bcebos.com"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or not parsed.path.startswith(_ARTIFACT_PATH)
                or not parsed.path.endswith(".pdparams")
                or not _MD5.fullmatch(md5)
            ):
                continue
            candidate = (url, md5)
            if name in found and found[name] != candidate:
                raise ValueError(
                    f"paddlenlp-taskflow-knowledge-mining: conflicting entry for {name}"
                )
            found[name] = candidate
            if len(found) > max_entries:
                raise ValueError(
                    f"paddlenlp-taskflow-knowledge-mining: source exceeds {max_entries} entries"
                )
    if not found:
        raise ValueError("paddlenlp-taskflow-knowledge-mining: no public checkpoint entries found")
    return tuple((name, *found[name]) for name in sorted(found))


class PaddleNlpTaskflowKnowledgeMiningSourceAdapter:
    """Enumerate direct WordTag/NPTag Taskflow checkpoint resources."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public WordTagTask and NPTagTask model_state .pdparams resources. "
        "Excludes internal fixtures, auxiliary assets, and other Taskflow modules."
    )

    def __init__(
        self,
        *,
        name: str = "paddlenlp-taskflow-knowledge-mining-checkpoints",
        repository: str = "PaddlePaddle/PaddleNLP",
        branch: str = "develop",
        source_path: str = "paddlenlp/taskflow/knowledge_mining.py",
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
        self.name, self.repository, self.branch, self.source_path = (
            name.strip(),
            repository,
            branch.strip(),
            source_path,
        )
        self.max_response_bytes, self.max_entries = max_response_bytes, max_entries
        self.client, self.clock = client or HttpClient(max_response_bytes=max_response_bytes), clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paddlenlp-taskflow-knowledge-mining-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "public literal model_state .pdparams URL with source MD5",
            }
        )

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(self.source_path, safe='/')}"
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
        now = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                records=(),
                next_state={**state, "checked_at": now},
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
        digest = hashlib.sha256(response.body).hexdigest()
        records = tuple(
            self._record(name, url, md5, revision, digest)
            for name, url, md5 in _parse_resource_maps(
                response.text(), max_entries=self.max_entries
            )
        )
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": now,
                "source_url": source_url,
                "source_sha256": digest,
                "checkpoint_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, name: str, url: str, md5: str, revision: str, digest: str) -> SourceRecord:
        identifier = Identifier("paddlenlp:taskflow-knowledge-mining", name)
        return SourceRecord(
            source_record_id=f"model:{name}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"{name} model_state.pdparams",
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": digest,
                "model_name": name,
                "artifact_url": url,
                "md5": md5,
            },
            text=f"PaddleNLP Taskflow knowledge-mining checkpoint: {name}",
            identifiers=(identifier,),
            links=(
                Link(url, relation="weights", locator="model_state.pdparams", crawl=False),
                Link(
                    self.blob_url(revision),
                    relation="model_card",
                    locator=f"resource_files_urls:{name}",
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
                    local_id=f"model:{name}",
                    name=name,
                    identifiers=(identifier,),
                    status=ModelStatus.RELEASED,
                    locator=f"resource_files_urls:{name}.model_state",
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{name}",
                    model_local_id=f"model:{name}",
                    revision=revision,
                    identifiers=(Identifier("paddlenlp:taskflow-knowledge-mining-release", name),),
                    metadata={"artifact_url": url, "md5": md5, "source_revision": revision},
                    locator=url,
                ),
            ),
        )


__all__ = ["PaddleNlpTaskflowKnowledgeMiningSourceAdapter"]
