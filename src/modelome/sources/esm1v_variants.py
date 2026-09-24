"""Exact ESM-1v ensemble checkpoint variants from first-party loaders."""

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
_PATH = "esm/pretrained.py"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MODEL = re.compile(r"^esm1v_t33_650M_UR90S_([2-5])$")
_MAX_ENTRIES = 8


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ESM1vVariantRegistryAdapter:
    """Index exact ESM-1v ensemble variants omitted by the README's range row.

    The official README gives a ``[1-5]`` shorthand but only shows the model 1
    asset URL. The first-party loader declares variants 2–5 by literal name;
    the implementation's loader URL template maps each literal to its direct
    ``.pt`` object. Model 1 remains covered by the existing README-table source.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only ESM-1v loader variants 2 through 5 explicitly defined in "
        "facebookresearch/esm. Variant 1 is covered by the upstream README table; "
        "other ESM families are out of scope. Weight bytes are not fetched."
    )

    def __init__(
        self,
        *,
        name: str = "esm1v-ensemble-variants",
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
            "adapter": "esm1v-variant-registry-v1",
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
            f"ESM-1v ensemble member {name[-1]}",
            identifiers=(Identifier("esm:model", name),),
            aliases=(f"{name}.pt",),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{name}", model_id, revision=revision,
            identifiers=(Identifier("esm:checkpoint-release", name),),
            metadata={"repository": self.repository, "checkpoint_filename": f"{name}.pt",
                      "loader_path": _PATH, "ensemble": "ESM-1v", "ensemble_size": 5},
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{name}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(weight_url),
            title=f"ESM-1v ensemble member {name[-1]}",
            raw={"repository": self.repository, "revision": revision,
                 "loader_path": _PATH, "loader_url": source_url,
                 "checkpoint_name": name, "checkpoint_url": weight_url,
                 "ensemble": "ESM-1v", "ensemble_member": int(name[-1])},
            text=f"First-party ESM-1v loader maps ensemble member {name[-1]} to {weight_url}",
            identifiers=(Identifier("esm:model", name),),
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
        raise ValueError("ESM loader source is invalid Python") from exc
    found: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        match = _MODEL.fullmatch(node.name)
        if match is None:
            continue
        returns = [child for child in ast.walk(node) if isinstance(child, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
            raise ValueError("ESM-1v loader must have one literal return call")
        call = returns[0].value
        if (not isinstance(call.func, ast.Name) or call.func.id != "load_model_and_alphabet_hub"
                or len(call.args) != 1 or call.keywords):
            raise ValueError("ESM-1v loader must call the hub loader with one literal")
        try:
            value = ast.literal_eval(call.args[0])
        except (ValueError, TypeError) as exc:
            raise ValueError("ESM-1v checkpoint name must be a literal") from exc
        if value != node.name:
            raise ValueError("ESM-1v loader and checkpoint names disagree")
        found.append(value)
    if not found:
        raise ValueError("ESM-1v variant inventory is empty")
    if len(found) > max_entries:
        raise ValueError("ESM-1v variant inventory exceeds entry limit")
    if len(set(found)) != len(found):
        raise ValueError("ESM-1v variant inventory contains duplicate names")
    return tuple(sorted(found))


__all__ = ["ESM1vVariantRegistryAdapter"]
