"""Pinned first-party checkpoint inventory for the Geom2Vec GNN zoo."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

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

_REPOSITORY = "dinner-group/geom2vec"
_DIRECTORY = "checkpoints"
_CHECKPOINTS = (
    "et_l6_h64_rbf64_r75.pth",
    "et_l6_h128_rbf64_r75.pth",
    "et_l6_h256_rbf64_r75.pth",
    "tensornet_l2_h200_rbf32_r5.pth",
    "tensornet_l3_h64_rbf32_r5.pth",
    "tensornet_l3_h128_rbf32_r5.pth",
    "tensornet_l3_h256_rbf32_r5.pth",
    "visnet_l6_h64_rbf64_r75.pth",
    "visnet_l6_h128_rbf64_r75.pth",
    "visnet_l6_h256_rbf64_r75.pth",
    "visnet_l6_h256_rbf32_r5_pcqm.pth",
    "visnet_l9_h256_rbf32_r5_pcqm.pth",
)


class Geom2VecCheckpointFilesSourceAdapter(
    StaticJsonCheckpointRegistrySourceAdapter
):
    """Enumerate the twelve explicitly documented Geom2Vec checkpoint files."""

    coverage_limitation = (
        "Covers only the twelve `.pth` files named in Geom2Vec's pretrained-checkpoint "
        "README table and present in its first-party checkpoints directory. Model names "
        "are the published filenames; this adapter does not inspect payloads or fetch "
        "weight bytes."
    )

    def __init__(self, *, max_response_bytes: int = 4 * 1024 * 1024, **kwargs: Any) -> None:
        kwargs.setdefault("name", "dinner-group-geom2vec-checkpoints")
        kwargs.setdefault("repository", _REPOSITORY)
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", _DIRECTORY)
        kwargs.setdefault("provider_namespace", "geom2vec:checkpoint")
        kwargs["max_response_bytes"] = max_response_bytes
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "geom2vec-checkpoint-files-v1",
                "repository": self.repository,
                "branch": self.branch,
                "directory": _DIRECTORY,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": (
                    "twelve literal .pth files named in first-party README checkpoint table"
                ),
            }
        )

    def directory_url(self, revision: str) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/contents/{_DIRECTORY}"
            f"?ref={quote(revision, safe='')}"
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
            self.directory_url(revision),
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: checkpoint directory returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: checkpoint directory exceeds {self.max_response_bytes} bytes"
            )
        checkpoints = _parse_directory(
            response.json(), revision, self.name, maximum=self.max_entries
        )
        records = tuple(self._record(item, revision, response.body) for item in checkpoints)
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.directory_url(revision),
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
        return handle.removesuffix(".pth")


def _parse_directory(
    payload: Any, revision: str, source: str, *, maximum: int
) -> tuple[_Checkpoint, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{source}: GitHub directory response must be an array")
    if len(payload) > maximum:
        raise ValueError(f"{source}: checkpoint directory exceeds {maximum} entries")
    found: dict[str, _Checkpoint] = {}
    prefix = f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/{_DIRECTORY}/"
    for entry in payload:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        if name.endswith(".pth") and name not in _CHECKPOINTS:
            raise ValueError(f"{source}: unrecognized checkpoint file {name}")
        if name not in _CHECKPOINTS:
            continue
        if (
            entry.get("type") != "file"
            or entry.get("path") != f"{_DIRECTORY}/{name}"
            or not isinstance(entry.get("size"), int)
            or entry["size"] <= 0
        ):
            raise ValueError(f"{source}: invalid GitHub file entry for {name}")
        url = entry.get("download_url")
        if url != prefix + name:
            raise ValueError(f"{source}: unpinned or unexpected download URL for {name}")
        if name in found:
            raise ValueError(f"{source}: duplicate checkpoint file {name}")
        found[name] = _Checkpoint(
            handle=name,
            url=url,
            locator=f"{_DIRECTORY}/{name}",
        )
    if set(found) != set(_CHECKPOINTS):
        missing = sorted(set(_CHECKPOINTS) - set(found))
        raise ValueError(f"{source}: expected twelve checkpoint files; missing {missing}")
    return tuple(found[name] for name in _CHECKPOINTS)
