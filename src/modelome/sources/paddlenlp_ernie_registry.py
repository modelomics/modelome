"""PaddleNLP's explicit ERNIE-family transformer checkpoint map."""

from __future__ import annotations

import ast
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
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_REPOSITORY = "PaddlePaddle/PaddleNLP"
_SOURCE_PATH = "paddlenlp/transformers/ernie/configuration.py"
_INIT_NAME = "ERNIE_PRETRAINED_INIT_CONFIGURATION"
_RESOURCE_NAME = "ERNIE_PRETRAINED_RESOURCE_FILES_MAP"
_ALLOWED_HOSTS = frozenset({"bj.bcebos.com", "paddlenlp.bj.bcebos.com"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _literal_assignment(tree: ast.Module, name: str) -> tuple[dict[str, Any], ast.Dict]:
    matches = [
        target
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name) and target.id == name
    ]
    if len(matches) != 1:
        raise ValueError(f"paddlenlp-ernie: expected one {name} assignment")
    assignment = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in statement.targets)
    )
    if not isinstance(assignment.value, ast.Dict):
        raise ValueError(f"paddlenlp-ernie: {name} must be a literal dictionary")
    value = ast.literal_eval(assignment.value)
    if not isinstance(value, dict):
        raise ValueError(f"paddlenlp-ernie: {name} must evaluate to a dictionary")
    return value, assignment.value


def _parse_registry(document: str) -> tuple[tuple[str, str, int], ...]:
    try:
        tree = ast.parse(document)
        model_ids, _ = _literal_assignment(tree, _INIT_NAME)
        resource_map, map_node = _literal_assignment(tree, _RESOURCE_NAME)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"paddlenlp-ernie: invalid literal model registry: {exc}") from exc
    state_map = resource_map.get("model_state")
    if not isinstance(state_map, dict):
        raise ValueError("paddlenlp-ernie: model_state resource map is missing")

    entry_lines: dict[str, int] = {}
    for key_node, value_node in zip(map_node.keys, map_node.values, strict=True):
        if ast.literal_eval(key_node) != "model_state" or not isinstance(value_node, ast.Dict):
            continue
        for model_node, _url_node in zip(value_node.keys, value_node.values, strict=True):
            try:
                model_id = ast.literal_eval(model_node)
            except (TypeError, ValueError):
                continue
            if isinstance(model_id, str):
                entry_lines[model_id] = model_node.lineno

    rows: list[tuple[str, str, int]] = []
    for model_id, url in state_map.items():
        if (
            model_id not in model_ids
            or not isinstance(model_id, str)
            or not _MODEL_ID.fullmatch(model_id)
        ):
            continue
        if not isinstance(url, str):
            continue
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in _ALLOWED_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.casefold().endswith(".pdparams")
        ):
            continue
        rows.append((model_id, canonicalize_url(url), entry_lines.get(model_id, 0)))
    if not rows:
        raise ValueError("paddlenlp-ernie: no exact model-state checkpoint entries found")
    return tuple(sorted(rows))


class PaddleNlpErnieRegistrySourceAdapter:
    """Read literal ERNIE model IDs and direct checkpoint URLs without execution."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only PaddleNLP's literal ERNIE-family model_state URL map whose IDs also "
        "appear in its pretrained initialization map. It does not execute Python, resolve "
        "aliases through imports, or infer other transformer-family resources."
    )

    def __init__(
        self,
        *,
        name: str = "paddlenlp-ernie-pretrained-registry",
        repository: str = _REPOSITORY,
        branch: str = "develop",
        source_path: str = _SOURCE_PATH,
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name is required")
        if repository != _REPOSITORY or source_path != _SOURCE_PATH:
            raise ValueError("repository and source_path must identify PaddleNLP's ERNIE registry")
        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("branch is required")
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
                "adapter": "paddlenlp-ernie-literal-registry-v1",
                "repository": repository,
                "branch": self.branch,
                "source_path": source_path,
                "max_response_bytes": max_response_bytes,
                "admission": "model_state URLs intersecting literal pretrained model IDs",
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

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
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        payload = commit_response.json()
        revision = (
            payload.get("sha", "").strip()
            if isinstance(payload, Mapping) and isinstance(payload.get("sha"), str)
            else ""
        )
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = self.clock().astimezone(UTC).isoformat()
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=int(state.get("model_count", 0)),
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source document returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: source document exceeds {self.max_response_bytes} bytes"
            )
        rows = _parse_registry(response.text())
        records = tuple(self._record(row, revision, response.body) for row in rows)
        next_state = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
        }
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, row: tuple[str, str, int], revision: str, source: bytes) -> SourceRecord:
        model_id, artifact_url, line = row
        identifier = Identifier("paddlenlp:transformer-model", model_id)
        local_id = f"model:{content_hash(model_id)[:24]}"
        source_url = self.blob_url(revision)
        return SourceRecord(
            source_record_id=f"paddlenlp-ernie:{model_id}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=artifact_url,
            title=f"PaddleNLP {model_id} pretrained transformer weights",
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "model_id": model_id,
                "artifact_url": artifact_url,
            },
            text=f"PaddleNLP pretrained transformer model {model_id}",
            identifiers=(identifier,),
            links=(
                Link(
                    artifact_url, relation="weights", locator=f"model_state:{model_id}", crawl=False
                ),
                Link(source_url, relation="model_card", locator=f"line:{line}", crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=(
                ModelHint(
                    local_id=local_id,
                    name=model_id,
                    identifiers=(identifier,),
                    status=ModelStatus.RELEASED,
                    locator=f"model_state:{model_id}",
                ),
            ),
        )


__all__ = ["PaddleNlpErnieRegistrySourceAdapter"]
