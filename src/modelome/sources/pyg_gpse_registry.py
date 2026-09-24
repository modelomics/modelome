"""Pinned reader for PyG's literal GPSE pretrained checkpoint registry."""

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
    _checkpoint_url,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)
from modelome.sources.static_python_checkpoint_registry import (
    StaticPythonCheckpointRegistrySourceAdapter,
)


class PyGGPSECheckpointRegistrySourceAdapter(StaticPythonCheckpointRegistrySourceAdapter):
    """Read ``GPSE.url_dict`` as literal source identities and artifact URLs.

    PyG stores this mapping as a class attribute, so the generic module-level
    Python mapping reader cannot represent it. This adapter requires one class
    named ``GPSE`` and one direct literal ``url_dict`` assignment inside it.
    It never imports or executes PyG.
    """

    coverage_limitation = (
        "Covers only literal entries in GPSE.url_dict in the first-party PyG source "
        "file at a pinned commit. It does not import package code, infer other PyG "
        "models, crawl Zenodo records, or download checkpoint bytes."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("repository", "pyg-team/pytorch_geometric")
        kwargs.setdefault("branch", "master")
        kwargs.setdefault("source_path", "torch_geometric/nn/models/gpse.py")
        kwargs.setdefault("mapping_variable", "url_dict")
        kwargs.setdefault("provider_namespace", "pytorch-geometric:gpse-checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "pyg-gpse-checkpoint-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "literal GPSE.url_dict with direct checkpoint URLs",
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
        checkpoints = _parse_gpse_map(
            response.text(),
            source=self.name,
            path=self.source_path,
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


def _parse_gpse_map(
    document: str,
    *,
    source: str,
    path: str,
    maximum: int,
) -> tuple[_Checkpoint, ...]:
    try:
        module = ast.parse(document, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{source}: registry is not valid Python: {error.msg}") from error
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "GPSE"
    ]
    if len(classes) != 1:
        raise ValueError(f"{source}: expected exactly one GPSE class")
    assignments = [
        node
        for node in classes[0].body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "url_dict"
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise ValueError(f"{source}: expected one literal GPSE.url_dict mapping")
    mapping = assignments[0].value
    if not mapping.keys:
        raise ValueError(f"{source}: GPSE.url_dict is empty")
    if len(mapping.keys) > maximum:
        raise ValueError(f"{source}: registry exceeds {maximum} entries")

    checkpoints: list[_Checkpoint] = []
    handles: set[str] = set()
    for key, raw_url in zip(mapping.keys, mapping.values, strict=True):
        handle = _string(key)
        url = _string(raw_url)
        if handle is None or url is None:
            raise ValueError(f"{source}: GPSE.url_dict entries must be literal strings")
        if not handle or handle != handle.strip() or handle in handles:
            raise ValueError(f"{source}: invalid or duplicate checkpoint handle {handle!r}")
        handles.add(handle)
        checked_url = _checkpoint_url(url, source, handle)
        if not urlsplit(checked_url).path.casefold().endswith(".pt"):
            raise ValueError(f"{source}: {handle!r} is not a PyTorch checkpoint URL")
        checkpoints.append(
            _Checkpoint(handle=handle, url=checked_url, locator=f"{path}:L{key.lineno}")
        )
    return tuple(checkpoints)


def _string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
