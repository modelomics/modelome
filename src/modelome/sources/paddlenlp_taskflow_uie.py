"""PaddleNLP Taskflow's explicit UIE checkpoint resource manifest."""

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
_HOST = "bj.bcebos.com"
_ARTIFACT_PATH = "/paddlenlp/taskflow/information_extraction/"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _isoformat(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise ValueError("clock must return a datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _parse_uie_resource_map(document: str, *, max_entries: int) -> tuple[tuple[str, str, str], ...]:
    """Read only the literal ``resource_files_urls`` mapping; never execute Python."""

    try:
        module = ast.parse(document)
    except SyntaxError as exc:
        raise ValueError("paddlenlp-taskflow-uie: source is not valid Python") from exc
    assignments = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "resource_files_urls"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("paddlenlp-taskflow-uie: expected one resource_files_urls mapping")
    try:
        resource_map = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError(
            "paddlenlp-taskflow-uie: resource_files_urls is not a literal mapping"
        ) from exc
    if not isinstance(resource_map, dict):
        raise ValueError("paddlenlp-taskflow-uie: resource_files_urls is not a mapping")

    entries: dict[str, tuple[str, str]] = {}
    for model_name, resources in resource_map.items():
        if not isinstance(model_name, str) or not _MODEL_NAME.fullmatch(model_name):
            continue
        # PaddleNLP keeps generated fixtures in this public Python map too. They
        # are tests, not released pretrained models.
        if model_name.startswith("__internal_testing__/"):
            continue
        if not isinstance(resources, dict):
            continue
        weight_info = resources.get("model_state")
        if (
            not isinstance(weight_info, list)
            or len(weight_info) != 2
            or not all(isinstance(item, str) for item in weight_info)
        ):
            continue
        artifact_url, md5 = weight_info
        parsed = urlsplit(artifact_url)
        if (
            parsed.scheme != "https"
            or (parsed.hostname or "").casefold() != _HOST
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith(_ARTIFACT_PATH)
            or not parsed.path.casefold().endswith(".pdparams")
            or not _MD5.fullmatch(md5)
        ):
            continue
        details = (artifact_url, md5)
        existing = entries.get(model_name)
        if existing is not None and existing != details:
            raise ValueError(f"paddlenlp-taskflow-uie: conflicting entry for {model_name}")
        entries[model_name] = details
        if len(entries) > max_entries:
            raise ValueError(f"paddlenlp-taskflow-uie: source exceeds {max_entries} public models")
    if not entries:
        raise ValueError("paddlenlp-taskflow-uie: no exact public checkpoint entries found")
    return tuple((name, *entries[name]) for name in sorted(entries))


class PaddleNlpTaskflowUieSourceAdapter:
    """Enumerate public UIE Taskflow models with declared direct weight URLs."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only public UIE model_state entries in PaddleNLP Taskflow's "
        "information-extraction resource map. It excludes internal test models, "
        "other Taskflow tasks, framework registry weights, Hugging Face mirrors, "
        "and auxiliary config/tokenizer resources."
    )

    def __init__(
        self,
        *,
        name: str = "paddlenlp-taskflow-uie-checkpoints",
        repository: str = "PaddlePaddle/PaddleNLP",
        branch: str = "develop",
        source_path: str = "paddlenlp/taskflow/information_extraction.py",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 1_000,
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
        for value, field in (
            (max_response_bytes, "max_response_bytes"),
            (max_entries, "max_entries"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
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
                "adapter": "paddlenlp-taskflow-uie-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "public literal model_state .pdparams URL with source-declared MD5",
            }
        )

    @property
    def commit_url(self) -> str:
        revision = quote(self.branch, safe="")
        return f"https://api.github.com/repos/{self.repository}/commits/{revision}"

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
        commit_response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        payload = commit_response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("checkpoint_count")),
            )

        source_url = self.raw_url(revision)
        response: HttpResponse = self.client.get(
            source_url, headers={"Accept": "text/plain,application/octet-stream"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source file returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source file exceeds {self.max_response_bytes} bytes")
        document_hash = hashlib.sha256(response.body).hexdigest()
        entries = _parse_uie_resource_map(response.text(), max_entries=self.max_entries)
        records = tuple(
            self._record(name, url, md5, revision, document_hash)
            for name, url, md5 in entries
        )
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": document_hash,
                "checkpoint_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        name: str,
        artifact_url: str,
        md5: str,
        revision: str,
        source_hash: str,
    ) -> SourceRecord:
        identifier = Identifier("paddlenlp:taskflow-uie", name)
        model = ModelHint(
            local_id=f"model:{name}",
            name=name,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=f"resource_files_urls:{name}.model_state",
        )
        return SourceRecord(
            source_record_id=f"model:{name}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"{name} model_state.pdparams",
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": source_hash,
                "model_name": name,
                "artifact_url": artifact_url,
                "md5": md5,
            },
            text=f"PaddleNLP Taskflow UIE checkpoint: {name}",
            identifiers=(identifier,),
            links=(
                Link(artifact_url, relation="weights", locator="model_state.pdparams", crawl=False),
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
            models=(model,),
            releases=(
                ReleaseHint(
                    local_id=f"release:{name}",
                    model_local_id=f"model:{name}",
                    revision=revision,
                    identifiers=(Identifier("paddlenlp:taskflow-uie-release", name),),
                    metadata={
                        "artifact_url": artifact_url,
                        "md5": md5,
                        "source_revision": revision,
                    },
                    locator=artifact_url,
                ),
            ),
        )


__all__ = ["PaddleNlpTaskflowUieSourceAdapter"]
