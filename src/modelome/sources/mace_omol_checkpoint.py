"""First-party MACE-OMOL checkpoint declared in its loader-local manifest."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

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
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = "ACEsuit/mace"
_SOURCE_PATH = "mace/calculators/foundations_models.py"
_HANDLE = "extra_large"
_WEIGHT_URL = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/"
    "mace_omol_0/MACE-omol-0-extra-large-1024.model"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MaceOmolCheckpointSourceAdapter:
    """Parse the exact ``urls`` map local to MACE's ``mace_omol`` loader.

    This closes a gap in the existing MACE checkpoint adapters: the OMOL map
    is nested in a function, so the module-level MACE-MP/Polar maps do not see
    it. The source is AST-parsed and never imported or executed.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the single literal extra_large MACE-OMOL checkpoint in the "
        "mace_omol loader. It does not include other MACE families or download weights."
    )

    def __init__(
        self,
        *,
        name: str = "mace-omol-checkpoint",
        repository: str = _REPOSITORY,
        branch: str = "develop",
        source_path: str = _SOURCE_PATH,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if (
            repository != _REPOSITORY
            or source_path != _SOURCE_PATH
            or not branch.strip()
            or not name.strip()
            or max_response_bytes <= 0
        ):
            raise ValueError(
                "repository/source path are fixed; name, branch, and limit are required"
            )
        self.name, self.repository, self.branch = name, repository, branch
        self.source_path = source_path
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "mace-omol-checkpoint-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "function": "mace_omol",
                "mapping": "urls",
                "expected_url": _WEIGHT_URL,
                "max_response_bytes": max_response_bytes,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_url = (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
        )
        commit_response = self.client.get(
            commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        payload = commit_response.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                (), {**state, "checked_at": checked_at}, True,
                upstream_count=state.get("model_count"),
            )

        source_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(self.source_path, safe='/')}"
        )
        response = self.client.get(source_url, headers={"Accept": "text/x-python,text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: source returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source exceeds {self.max_response_bytes} bytes")
        weight_url = _parse_omol_url(response.text(), self.name)
        record = self._record(revision, source_url, weight_url, content_hash(response.body))
        return SourcePage(
            (record,),
            {
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": content_hash(response.body),
                "model_count": 1,
            },
            True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _record(
        self, revision: str, source_url: str, weight_url: str, source_hash: str
    ) -> SourceRecord:
        namespace = "mace:omol-checkpoint"
        model_id = f"model:{_HANDLE}"
        model = ModelHint(
            model_id,
            "MACE-OMOL extra-large checkpoint",
            identifiers=(Identifier(namespace, _HANDLE),),
            aliases=("MACE-omol-0-extra-large-1024.model",),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{_HANDLE}",
            model_id,
            version="omol-0",
            identifiers=(Identifier(f"{namespace}:release", "omol-0"),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_function": "mace_omol",
                "checkpoint_handle": _HANDLE,
                "weight_url": weight_url,
                "source_sha256": source_hash,
            },
        )
        blob_url = (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )
        return SourceRecord(
            source_record_id=f"checkpoint:mace-omol:{_HANDLE}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(weight_url),
            title=model.name,
            raw={
                "checkpoint_handle": _HANDLE,
                "weight_url": weight_url,
                "source_function": "mace_omol",
            },
            text="MACE's mace_omol loader declares this exact extra-large checkpoint URL.",
            links=(
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(blob_url, "source_implementation", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_omol_url(source_text: str, source: str) -> str:
    try:
        module = ast.parse(source_text, filename=_SOURCE_PATH)
    except SyntaxError as error:
        raise ValueError(f"{source}: source is not valid Python: {error.msg}") from error
    functions = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "mace_omol"
    ]
    if len(functions) != 1:
        raise ValueError(f"{source}: expected one mace_omol function")
    assignments = [
        node for node in functions[0].body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "urls"
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise ValueError(f"{source}: expected one literal urls map in mace_omol")
    mapping = assignments[0].value
    entries = [
        (key.value, value.value)
        for key, value in zip(mapping.keys, mapping.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
        and isinstance(value, ast.Constant) and isinstance(value.value, str)
    ]
    if len(mapping.keys) != 1 or entries != [(_HANDLE, _WEIGHT_URL)]:
        raise ValueError(f"{source}: unexpected mace_omol checkpoint map")
    parsed = urlsplit(entries[0][1])
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError(f"{source}: checkpoint URL is not first-party HTTPS")
    return entries[0][1]


__all__ = ["MaceOmolCheckpointSourceAdapter"]
