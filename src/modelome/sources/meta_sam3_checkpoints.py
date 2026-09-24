"""Enumerate Meta's gated SAM 3 and SAM 3.1 checkpoint pointers.

The SAM 3 model builder maps each public version selector to a Hugging Face
repository and filename. This adapter statically parses that mapping and
records file pointers without requesting access or fetching checkpoint bytes.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from modelome.http import HttpResponse
from modelome.models import SourcePage
from modelome.normalize import content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)


class MetaSAM3CheckpointSourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Read exact SAM 3 Hugging Face checkpoint mappings from source code."""

    coverage_limitation = (
        "Covers only SAM 3 and SAM 3.1 checkpoint files selected by the official "
        "facebookresearch/sam3 model builder. The repositories require approved "
        "Hugging Face access; the adapter records gated pointers and fetches no weights."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "meta-sam3-checkpoints")
        kwargs.setdefault("repository", "facebookresearch/sam3")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", "sam3/model_builder.py")
        kwargs.setdefault("provider_namespace", "meta:sam3-checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "meta-sam3-checkpoints-v1",
                "repository": self.repository,
                "branch": self.branch,
                "paths": ["sam3/model_builder.py", "README.md"],
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "static download_ckpt_from_hf version/repository/filename map",
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

        builder = self._get_text(revision, "sam3/model_builder.py")
        readme = self._get_text(revision, "README.md")
        if "request access to the checkpoints" not in readme.casefold():
            raise ValueError(f"{self.name}: README no longer documents gated checkpoint access")
        mappings = _sam3_checkpoint_map(builder, source=self.name)
        if not mappings:
            raise ValueError(f"{self.name}: no supported checkpoint mappings found")
        if len(mappings) > self.max_entries:
            raise ValueError(f"{self.name}: model list exceeds {self.max_entries} entries")
        source = builder.encode() + b"\0" + readme.encode()
        records = []
        for handle, (repo_id, filename) in sorted(mappings.items()):
            if repo_id not in {"facebook/sam3", "facebook/sam3.1"}:
                raise ValueError(f"{self.name}: unexpected Hugging Face repo {repo_id!r}")
            if "/" in filename or not filename.endswith(".pt"):
                raise ValueError(f"{self.name}: unexpected checkpoint filename {filename!r}")
            checkpoint = _Checkpoint(
                handle=handle,
                url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
                locator=f"sam3/model_builder.py:download_ckpt_from_hf:{handle}",
            )
            record = self._record(checkpoint, revision, source)
            release = record.releases[0]
            metadata = {
                **release.metadata,
                "huggingface_repository": repo_id,
                "access_requirement": "Hugging Face approval and authentication",
                "access_restricted": True,
            }
            records.append(
                replace(
                    record,
                    raw={**record.raw, "access_restricted": True},
                    releases=(replace(release, metadata=metadata),),
                )
            )
        records_tuple = tuple(records)
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "document_url": self.raw_url(revision),
            "document_sha256": content_hash(source),
            "model_count": len(records_tuple),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records_tuple,
            next_state=next_state,
            complete=True,
            upstream_count=len(records_tuple),
            authoritative_snapshot=True,
        )

    def _get_text(self, revision: str, path: str) -> str:
        response: HttpResponse = self.client.get(
            f"https://raw.githubusercontent.com/{self.repository}/{revision}/{path}",
            headers={"Accept": "text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: {path} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: {path} exceeds {self.max_response_bytes} bytes")
        return response.text()


def _sam3_checkpoint_map(document: str, *, source: str) -> dict[str, tuple[str, str]]:
    try:
        module = ast.parse(document, filename="sam3/model_builder.py")
    except SyntaxError as error:
        raise ValueError(f"{source}: model builder is not valid Python") from error
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "download_ckpt_from_hf"
    ]
    if len(functions) != 1:
        raise ValueError(f"{source}: expected one download_ckpt_from_hf function")
    branches: dict[str, dict[str, str]] = {}
    function = functions[0]
    for statement in function.body:
        if not isinstance(statement, ast.If):
            continue
        test = statement.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "version"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "sam3.1"
        ):
            continue
        branches["sam3.1"] = _literal_assignments(statement.body)
        branches["sam3"] = _literal_assignments(statement.orelse)
    if set(branches) != {"sam3", "sam3.1"}:
        raise ValueError(f"{source}: expected explicit sam3/sam3.1 branches")
    result: dict[str, tuple[str, str]] = {}
    for handle, branch in branches.items():
        repo_id, filename = branch.get("repo_id"), branch.get("ckpt_name")
        if not repo_id or not filename:
            raise ValueError(f"{source}: {handle} checkpoint mapping is incomplete")
        result[handle] = (repo_id, filename)
    return result


def _literal_assignments(statements: list[ast.stmt]) -> dict[str, str]:
    values: dict[str, str] = {}
    for statement in statements:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        value = statement.value
        if (
            isinstance(target, ast.Name)
            and target.id in {"repo_id", "ckpt_name"}
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            values[target.id] = value.value
    return values


__all__ = ["MetaSAM3CheckpointSourceAdapter"]
