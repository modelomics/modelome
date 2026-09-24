"""Pinned ingestion of MLX-LM's literal model-type remapping registry.

This is an architecture compatibility registry, not a checkpoint catalog. It
records only the exact keys in Apple's first-party MLX-LM source map and the
implementation type to which each key is remapped.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import content_hash
from modelome.sources.static_json_checkpoint_registry import (
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_TYPE = re.compile(r"^[a-z][a-z0-9_]*$")


class MlxRegistrySourceAdapter:
    """Read ``MODEL_REMAPPING`` from MLX-LM at a resolved Git revision."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only keys explicitly listed in MLX-LM's MODEL_REMAPPING map at "
        "one observed commit. These are model-type aliases used for architecture "
        "loading, not pretrained checkpoint identities; this source asserts no "
        "weights, releases, or complete MLX model coverage."
    )

    def __init__(self, *, client: Any | None = None, clock: Any = None) -> None:
        self.name = "mlx-lm-model-type-remapping"
        self.repository = "ml-explore/mlx-lm"
        self.branch = "main"
        self.source_path = "mlx_lm/utils.py"
        self.client = client or HttpClient(max_response_bytes=4 * 1024 * 1024)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.checkpoint_signature = content_hash(
            {
                "adapter": "mlx-model-remapping-v1",
                "repo": self.repository,
                "branch": self.branch,
                "path": self.source_path,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _REVISION.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )
        url = f"https://raw.githubusercontent.com/{self.repository}/{revision}/{self.source_path}"
        response = self.client.get(url, headers={"Accept": "text/x-python,text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: source file returned HTTP {response.status}")
        if len(response.body) > 4 * 1024 * 1024:
            raise ValueError(f"{self.name}: source file exceeds 4194304 bytes")
        rows = _parse(response.text())
        blob = f"{self.repository_url}/blob/{revision}/{self.source_path}"
        records = tuple(self._record(key, target, blob, url) for key, target in rows)
        if not records:
            raise ValueError(f"{self.name}: remapping registry is empty")
        next_state = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "model_count": len(records),
            "source_url": url,
            "source_sha256": content_hash(response.body),
        }
        if etag := _header(commit.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, key: str, target: str, blob: str, raw_url: str) -> SourceRecord:
        model_id = f"model:{key}"
        model = ModelHint(
            local_id=model_id,
            name=key,
            aliases=(target,),
            identifiers=(Identifier("mlx-lm:model-type-alias", key),),
            status=ModelStatus.DOCUMENTED,
            locator="mlx_lm/utils.py:MODEL_REMAPPING",
        )
        return SourceRecord(
            source_record_id=f"mlx-lm-remapping:{key}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=blob,
            title=f"MLX-LM model type alias {key}",
            raw={"model_type": key, "implementation_type": target, "source_url": raw_url},
            text=f"MLX-LM remaps model type {key} to implementation type {target}.",
            links=(
                Link(self.repository_url, "source_repository", crawl=False),
                Link(
                    blob,
                    "registry_definition",
                    locator="mlx_lm/utils.py:MODEL_REMAPPING",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
            ),
            models=(model,),
        )


def _parse(source: str) -> tuple[tuple[str, str], ...]:
    try:
        module = ast.parse(source, filename="mlx_lm/utils.py")
    except SyntaxError as error:
        raise ValueError(f"{error.msg}") from error
    assignments = [
        n
        for n in module.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "MODEL_REMAPPING"
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise ValueError("expected exactly one literal MODEL_REMAPPING dictionary")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    node = assignments[0].value
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        key = (
            key_node.value
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)
            else None
        )
        value = (
            value_node.value
            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str)
            else None
        )
        if key is None or value is None or not _TYPE.fullmatch(key) or not _TYPE.fullmatch(value):
            raise ValueError("MODEL_REMAPPING must contain literal model-type strings")
        if key in seen:
            raise ValueError(f"duplicate model-type alias {key!r}")
        seen.add(key)
        result.append((key, value))
    return tuple(result)
