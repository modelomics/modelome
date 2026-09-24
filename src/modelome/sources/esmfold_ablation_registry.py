"""Exact first-party ESMFold ablation checkpoint identities."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
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
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]
_REPOSITORY = "facebookresearch/esm"
_PATH = "esm/esmfold/v1/pretrained.py"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_NAME = re.compile(r"^esmfold_structure_module_only_(?:8M|35M|150M|650M|3B|15B)(?:_270K)?$")
_MAX_ENTRIES = 32


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ESMFoldAblationRegistryAdapter:
    """Enumerate ESMFold ablations from literal first-party loader calls.

    The README publishes the family using a wildcard. This adapter obtains
    exact checkpoint names from its Python loader definitions and expands only
    the first-party URL template used by ``_load_model``. It does not fetch
    checkpoint bytes or infer variants from filenames.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the literal ESMFold structure-module-only loader functions in "
        "facebookresearch/esm. Other ESM weights and non-public/gated weights "
        "are outside this exact ablation inventory. Checkpoint bytes are not fetched."
    )

    def __init__(
        self,
        *,
        name: str = "esmfold-ablation-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_source_bytes: int = 512 * 1024,
        max_entries: int = _MAX_ENTRIES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or max_source_bytes <= 0 or max_entries <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_source_bytes = max_source_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_source_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "esmfold-ablation-registry-v1",
            "repository": repository,
            "branch": branch,
            "registry_path": _PATH,
            "max_source_bytes": max_source_bytes,
            "max_entries": max_entries,
        })

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
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage((), {**state, "checked_at": checked}, True,
                              upstream_count=state.get("model_count"))

        source_url = f"https://raw.githubusercontent.com/{self.repository}/{revision}/{_PATH}"
        response = self.client.get(source_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_source_bytes:
            raise ValueError(f"{self.name}: registry exceeds response limit")
        names = _parse_names(response.text(), max_entries=self.max_entries)
        records = tuple(self._record(item, revision, source_url) for item in names)
        return SourcePage(
            records,
            {"completed_revision": revision, "checked_at": checked,
             "model_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, name: str, revision: str, source_url: str) -> SourceRecord:
        model_id = f"model:{name}"
        weight_url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{name}.pt"
        source_page = f"{self.repository_url}/blob/{revision}/{_PATH}"
        model = ModelHint(
            model_id,
            f"ESMFold {name.removeprefix('esmfold_')}",
            identifiers=(Identifier("esmfold:checkpoint", name),),
            aliases=(f"{name}.pt",),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{name}", model_id, revision=revision,
            identifiers=(Identifier("esmfold:checkpoint-release", name),),
            metadata={"repository": self.repository, "checkpoint_filename": f"{name}.pt",
                      "loader_path": _PATH},
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{name}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(weight_url),
            title=f"ESMFold {name.removeprefix('esmfold_')}",
            raw={"repository": self.repository, "revision": revision,
                 "loader_path": _PATH, "loader_url": source_url,
                 "checkpoint_name": name, "checkpoint_url": weight_url},
            text=f"First-party ESMFold loader maps {name} to {weight_url}",
            identifiers=(Identifier("esmfold:checkpoint", name),),
            links=(
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(source_page, "model_card", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_names(source: str, *, max_entries: int) -> tuple[str, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("ESMFold loader source is invalid Python") from exc
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("esmfold_structure_module_only_"):
            continue
        returns = [child for child in ast.walk(node) if isinstance(child, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
            raise ValueError("ESMFold ablation loader must have one literal return call")
        call = returns[0].value
        if (not isinstance(call.func, ast.Name) or call.func.id != "_load_model"
                or len(call.args) != 1 or call.keywords):
            raise ValueError("ESMFold ablation loader must call _load_model with one literal")
        try:
            name = ast.literal_eval(call.args[0])
        except (ValueError, TypeError) as exc:
            raise ValueError("ESMFold checkpoint name must be a literal") from exc
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("ESMFold checkpoint name is outside the supported inventory")
        if node.name != name:
            raise ValueError("ESMFold loader name and checkpoint name disagree")
        names.append(name)
    if not names:
        raise ValueError("ESMFold ablation registry is empty")
    if len(names) > max_entries:
        raise ValueError("ESMFold ablation registry exceeds entry limit")
    if len(set(names)) != len(names):
        raise ValueError("ESMFold ablation registry contains duplicate checkpoint names")
    return tuple(sorted(names))


__all__ = ["ESMFoldAblationRegistryAdapter"]
