"""Pinned ingestion for first-party graph-ML checkpoint path registries.

Some graph libraries publish a literal handle-to-relative-path dictionary and
resolve those paths against their own artifact host at runtime. This adapter
reads exactly one such dictionary without importing or executing the package.
Only direct checkpoint paths under the configured HTTPS base URL are admitted.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpResponse
from modelome.models import SourcePage
from modelome.normalize import content_hash
from modelome.sources.static_json_checkpoint_registry import (
    _Checkpoint,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)
from modelome.sources.static_python_checkpoint_registry import (
    StaticPythonCheckpointRegistrySourceAdapter,
)


class GraphMLRegistrySourceAdapter(StaticPythonCheckpointRegistrySourceAdapter):
    """Read one literal graph-ML checkpoint map whose values are relative paths.

    The registry structure remains intentionally narrow: one Python literal
    dictionary with unique string keys and values. The configured artifact base
    must be an HTTPS origin, and each path must stay below its path prefix.
    """

    coverage_limitation = (
        "Covers literal handle-to-relative-checkpoint-path rows in one first-party "
        "Python source file at a pinned commit. It AST-parses one static dictionary, "
        "resolves paths under one configured HTTPS artifact prefix, and does not "
        "import code, infer architectures, follow links, or download checkpoint bytes."
    )

    def __init__(self, *, checkpoint_base_url: str, **kwargs: Any) -> None:
        self.checkpoint_base_url = _https_base(checkpoint_base_url)
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "graph-ml-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "mapping_variable": self.mapping_variable,
                "provider_namespace": self.provider_namespace,
                "checkpoint_base_url": self.checkpoint_base_url,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "literal Python map of relative checkpoint paths",
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
        checkpoints = _parse_relative_registry(
            response.text(),
            source=self.name,
            path=self.source_path,
            mapping_variable=self.mapping_variable,
            maximum=self.max_entries,
            base_url=self.checkpoint_base_url,
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


def _parse_relative_registry(
    document: str,
    *,
    source: str,
    path: str,
    mapping_variable: str,
    maximum: int,
    base_url: str,
) -> tuple[_Checkpoint, ...]:
    try:
        module = ast.parse(document, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{source}: registry is not valid Python: {error.msg}") from error

    from modelome.sources.static_python_checkpoint_registry import (
        _assignment_value,
        _is_mapping_assignment,
        _literal_string,
    )

    assignments = [
        node for node in module.body if _is_mapping_assignment(node, mapping_variable)
    ]
    if len(assignments) != 1:
        raise ValueError(
            f"{source}: expected exactly one simple assignment to {mapping_variable}"
        )
    mapping = _assignment_value(assignments[0])
    if not isinstance(mapping, ast.Dict) or not mapping.keys:
        raise ValueError(f"{source}: {mapping_variable} must be a non-empty literal dictionary")
    if len(mapping.keys) > maximum:
        raise ValueError(f"{source}: registry exceeds {maximum} entries")

    base = urlsplit(base_url)
    checkpoints: list[_Checkpoint] = []
    handles: set[str] = set()
    for key, value in zip(mapping.keys, mapping.values, strict=True):
        handle = _literal_string(key)
        relative_path = _literal_string(value)
        if handle is None or relative_path is None:
            raise ValueError(f"{source}: registry entries must use literal string keys and values")
        if not handle or handle != handle.strip() or handle in handles:
            raise ValueError(f"{source}: invalid or duplicate checkpoint handle {handle!r}")
        handles.add(handle)
        parsed_path = urlsplit(relative_path)
        path_part = parsed_path.path
        if (
            parsed_path.scheme
            or parsed_path.netloc
            or parsed_path.query
            or parsed_path.fragment
            or not path_part
            or path_part.startswith("/")
            or "\\" in path_part
            or any(part in {"", ".", ".."} for part in path_part.split("/"))
            or not path_part.casefold().endswith((".ckpt", ".pkl", ".pth", ".pt", ".safetensors"))
        ):
            raise ValueError(f"{source}: {handle!r} has an unsafe checkpoint path")
        url = f"{base.scheme}://{base.netloc}{base.path}{path_part}"
        checkpoints.append(
            _Checkpoint(
                handle=handle,
                url=url,
                locator=f"{path}:L{key.lineno}",
            )
        )
    return tuple(checkpoints)


def _https_base(value: str) -> str:
    base = value.strip() if isinstance(value, str) else ""
    parsed = urlsplit(base)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
        or ".." in parsed.path.split("/")
    ):
        raise ValueError("checkpoint_base_url must be an HTTPS origin/path prefix")
    return base.rstrip("/") + "/"
