"""Pinned ingestion of literal checkpoint maps embedded in Python source.

The reader intentionally parses a single assignment with :mod:`ast`; it never
imports or executes upstream package code.  This captures repositories that
publish their official model handles in a literal ``dict[str, str]`` while
retaining the same bounded, source-scoped checkpoint semantics as the JSON-map
reader.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpResponse
from modelome.models import SourcePage, SourceRecord
from modelome.normalize import content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _checkpoint_url,
    _header,
    _isoformat,
    _nonnegative_int,
    _required_text,
    _text,
)

_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HANDLE = re.compile(r"^[^\s\x00-\x1f\x7f](?:[^\x00-\x1f\x7f]{0,511})$")


class StaticPythonCheckpointRegistrySourceAdapter(
    StaticJsonCheckpointRegistrySourceAdapter
):
    """Read one literal public handle-to-checkpoint dictionary at a Git revision.

    ``mapping_variable`` must occur in exactly one simple assignment. Its value
    must be a literal dictionary whose unique string keys point to direct,
    recognised checkpoint-file URLs. Dynamic values, calls, imports, duplicate
    keys, and aliases are rejected instead of being evaluated or guessed.
    """

    coverage_limitation = (
        "Covers literal handle-to-checkpoint rows in one first-party Python source "
        "file at a pinned commit. It AST-parses exactly one static dictionary; it "
        "does not import or execute package code, infer papers or architectures, "
        "follow artifact URLs, or download checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str,
        repository: str,
        branch: str,
        source_path: str,
        mapping_variable: str,
        provider_namespace: str,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 100_000,
        **kwargs: Any,
    ) -> None:
        self.mapping_variable = _mapping_variable(mapping_variable)
        super().__init__(
            name=name,
            repository=repository,
            branch=branch,
            source_path=source_path,
            provider_namespace=provider_namespace,
            max_response_bytes=max_response_bytes,
            max_entries=max_entries,
            **kwargs,
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "static-python-checkpoint-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "mapping_variable": self.mapping_variable,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "one literal Python dictionary of direct checkpoint URLs",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            if etag := _header(commit_response.headers, "etag"):
                next_state["commit_etag"] = etag
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision),
            headers={"Accept": "text/x-python,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: registry exceeds {self.max_response_bytes} bytes")
        checkpoints = _parse_registry(
            response.text(),
            source=self.name,
            path=self.source_path,
            mapping_variable=self.mapping_variable,
            maximum=self.max_entries,
        )
        records = tuple(
            self._record(checkpoint, revision, response.body) for checkpoint in checkpoints
        )
        if not records:
            raise ValueError(f"{self.name}: registry contains no checkpoint entries")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _model_name(self, handle: str) -> str:
        return handle

    def _record(
        self,
        checkpoint: _Checkpoint,
        revision: str,
        source: bytes,
    ) -> SourceRecord:
        record = super()._record(checkpoint, revision, source)
        parts = urlsplit(checkpoint.url).path.rstrip("/").split("/")
        digest = parts[-2] if len(parts) >= 2 else ""
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            return record
        checksum = {
            "algorithm": "sha256",
            "value": digest.casefold(),
            "source": "url_path",
        }
        release = record.releases[0]
        return replace(
            record,
            raw={**record.raw, "weight_checksum": checksum},
            releases=(
                replace(
                    release,
                    metadata={**release.metadata, "weight_checksum": checksum},
                ),
            ),
        )


def _mapping_variable(value: str) -> str:
    variable = _required_text(value, "mapping variable")
    if not _VARIABLE.fullmatch(variable):
        raise ValueError("mapping variable must be a Python identifier")
    return variable


def _parse_registry(
    document: str,
    *,
    source: str,
    path: str,
    mapping_variable: str,
    maximum: int,
) -> tuple[_Checkpoint, ...]:
    try:
        module = ast.parse(document, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{source}: registry is not valid Python: {error.msg}") from error

    assignments = [
        node
        for node in module.body
        if _is_mapping_assignment(node, mapping_variable)
    ]
    if len(assignments) != 1:
        raise ValueError(
            f"{source}: expected exactly one simple assignment to {mapping_variable}"
        )
    assignment = assignments[0]
    value = _assignment_value(assignment)
    if not isinstance(value, ast.Dict):
        raise ValueError(f"{source}: {mapping_variable} must be a literal dictionary")
    if not value.keys:
        raise ValueError(f"{source}: {mapping_variable} is empty")
    if len(value.keys) > maximum:
        raise ValueError(f"{source}: registry exceeds {maximum} entries")

    checkpoints: list[_Checkpoint] = []
    handles: set[str] = set()
    for key, raw_url in zip(value.keys, value.values, strict=True):
        handle = _literal_string(key)
        url = _literal_string(raw_url)
        if handle is None or url is None:
            raise ValueError(
                f"{source}: {mapping_variable} entries must use literal string keys and values"
            )
        if (
            not _HANDLE.fullmatch(handle)
            or handle != handle.strip()
            or ".." in handle.split("/")
        ):
            raise ValueError(f"{source}: invalid checkpoint handle {handle!r}")
        if handle in handles:
            raise ValueError(f"{source}: {mapping_variable} contains duplicate handle {handle!r}")
        handles.add(handle)
        checkpoints.append(
            _Checkpoint(
                handle=handle,
                url=_checkpoint_url(url, source, handle),
                locator=f"{path}:L{key.lineno}",
            )
        )
    return tuple(checkpoints)


def _is_mapping_assignment(node: ast.stmt, mapping_variable: str) -> bool:
    if isinstance(node, ast.Assign):
        return len(node.targets) == 1 and _is_mapping_target(node.targets[0], mapping_variable)
    if isinstance(node, ast.AnnAssign):
        return _is_mapping_target(node.target, mapping_variable)
    return False


def _is_mapping_target(node: ast.expr, mapping_variable: str) -> bool:
    return isinstance(node, ast.Name) and node.id == mapping_variable


def _assignment_value(node: ast.stmt) -> ast.expr | None:
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, ast.AnnAssign):
        return node.value
    return None


def _literal_string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
