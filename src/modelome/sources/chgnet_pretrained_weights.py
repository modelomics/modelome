"""CHGNet's first-party pretrained checkpoint path registry."""

from __future__ import annotations

import ast
import posixpath
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import PurePosixPath
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
_REPOSITORY = "CederGroupHub/chgnet"
_BRANCH = "main"
_MODEL_PATH = "chgnet/model/model.py"
_PRETRAINED_PREFIX = "chgnet/pretrained/"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MODELS = frozenset({"0.2.0", "0.3.0", "r2scan"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CHGNetPretrainedWeightsSourceAdapter:
    """Resolve CHGNet's literal model-name-to-checkpoint paths at a pinned commit."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the three model_name-to-relative-path entries in CHGNet's first-party "
        "CHGNet.load implementation. The Git tree verifies file identities and sizes; "
        "checkpoint bytes are not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "chgnet-pretrained-weights",
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 12 * 1024 * 1024,
    ) -> None:
        if not name.strip() or max_response_bytes <= 0:
            raise ValueError("name and positive response limit are required")
        self.name = name
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "chgnet-pretrained-weights-v1",
                "repository": _REPOSITORY,
                "branch": _BRANCH,
                "model_path": _MODEL_PATH,
                "max_response_bytes": max_response_bytes,
            }
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/commits/{_BRANCH}"

    def _raw_url(self, revision: str, path: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{_REPOSITORY}/"
            f"{revision}/{quote(path, safe='/')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        commit = commit_response.json()
        revision = commit.get("sha", "") if isinstance(commit, Mapping) else ""
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage((), next_state, True, upstream_count=state.get("model_count"))

        source_response = self.client.get(
            self._raw_url(revision, _MODEL_PATH), headers={"Accept": "text/plain"}
        )
        tree_url = (
            f"https://api.github.com/repos/{_REPOSITORY}/git/trees/{revision}?recursive=1"
        )
        tree_response = self.client.get(tree_url, headers={"Accept": "application/vnd.github+json"})
        if source_response.status != 200:
            raise ValueError(f"{self.name}: model source returned HTTP {source_response.status}")
        if tree_response.status != 200:
            raise ValueError(f"{self.name}: file tree returned HTTP {tree_response.status}")
        for response in (source_response, tree_response):
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: metadata exceeds {self.max_response_bytes} bytes")
        model_paths = _parse_checkpoint_map(source_response.text(), self.name)
        blob_metadata = _parse_git_tree(tree_response.json(), model_paths, self.name)
        records = tuple(
            self._record(model_name, path, blob_metadata[path], revision, source_response.body)
            for model_name, path in model_paths
        )
        return SourcePage(
            records,
            {
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": self._raw_url(revision, _MODEL_PATH),
                "model_source_sha256": content_hash(source_response.body),
                "git_tree_sha256": content_hash(tree_response.body),
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        model_name: str,
        path: str,
        blob: tuple[str, int],
        revision: str,
        source: bytes,
    ) -> SourceRecord:
        blob_sha, size = blob
        filename = PurePosixPath(path).name
        model_id = f"model:{model_name}"
        identity = Identifier("chgnet:checkpoint", model_name)
        weight_url = self._raw_url(revision, path)
        model_source_url = self._raw_url(revision, _MODEL_PATH)
        model = ModelHint(
            model_id,
            f"CHGNet {model_name}",
            aliases=(filename,),
            identifiers=(identity,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{model_name}",
            model_id,
            revision=revision,
            identifiers=(Identifier("chgnet:checkpoint:release", model_name),),
            metadata={
                "repository": _REPOSITORY,
                "revision": revision,
                "model_name": model_name,
                "source_path": _MODEL_PATH,
                "checkpoint_path": path,
                "checkpoint_filename": filename,
                "git_blob_sha": blob_sha,
                "size_bytes": size,
                "source_sha256": content_hash(source),
                "binary_reachability_checked": False,
            },
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{model_name}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(weight_url),
            title=model.name,
            raw={
                "model_name": model_name,
                "checkpoint_path": path,
                "checkpoint_filename": filename,
                "checkpoint_url": weight_url,
                "git_blob_sha": blob_sha,
                "size_bytes": size,
            },
            text=f"CHGNet.load(model_name={model_name!r}) selects {path}",
            identifiers=(identity,),
            links=(
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(model_source_url, "source_implementation", crawl=False),
                Link(
                    f"https://github.com/{_REPOSITORY}",
                    "source_repository",
                    crawl=False,
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_checkpoint_map(source_text: str, source: str) -> tuple[tuple[str, str], ...]:
    match = re.search(
        r"checkpoint_path\s*=\s*(\{[\s\S]*?\})\s*\.get\(model_name\)", source_text
    )
    if match is None:
        raise ValueError(f"{source}: CHGNet.load checkpoint map is missing")
    try:
        mapping = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"{source}: checkpoint map is not a literal dictionary") from error
    if not isinstance(mapping, dict) or frozenset(mapping) != _MODELS:
        raise ValueError(f"{source}: supported model-name set changed")
    rows = []
    for model_name, relative_path in mapping.items():
        if not isinstance(model_name, str) or not isinstance(relative_path, str):
            raise ValueError(f"{source}: checkpoint map entries must be strings")
        normalized = posixpath.normpath(f"chgnet/model/{relative_path}")
        if not normalized.startswith(_PRETRAINED_PREFIX) or ".." in normalized.split("/"):
            raise ValueError(f"{source}: invalid pretrained path for {model_name!r}")
        if not normalized.endswith(".pth.tar"):
            raise ValueError(f"{source}: unrecognised checkpoint file for {model_name!r}")
        rows.append((model_name, normalized))
    return tuple(rows)


def _parse_git_tree(
    payload: Any,
    model_paths: tuple[tuple[str, str], ...],
    source: str,
) -> dict[str, tuple[str, int]]:
    if not isinstance(payload, Mapping) or payload.get("truncated") is True:
        raise ValueError(f"{source}: Git tree is invalid or truncated")
    raw_tree = payload.get("tree")
    if not isinstance(raw_tree, list):
        raise ValueError(f"{source}: Git tree has no entries")
    tree: dict[str, Mapping[str, Any]] = {}
    for row in raw_tree:
        if isinstance(row, Mapping) and isinstance(row.get("path"), str):
            tree[row["path"]] = row
    result = {}
    for _, path in model_paths:
        row = tree.get(path)
        sha = row.get("sha") if row else None
        size = row.get("size") if row else None
        if (
            not row
            or row.get("type") != "blob"
            or not isinstance(sha, str)
            or not re.fullmatch(r"[0-9a-f]{40}", sha)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValueError(f"{source}: checkpoint {path!r} is absent or invalid in Git tree")
        result[path] = (sha, size)
    return result


__all__ = ["CHGNetPretrainedWeightsSourceAdapter"]
