"""Pinned extraction of the official Transformer-M checkpoint table."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from modelome.http import HttpResponse
from modelome.models import SourcePage
from modelome.normalize import canonicalize_url, content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)

_ROW = re.compile(r"^\|\s*(?P<handle>L12_old|L12|L18)\s*\|(?P<cells>.*)\|\s*$")
_URL = re.compile(r"https://1drv\.ms/u/s![A-Za-z0-9_-]+\?e=[A-Za-z0-9_-]+$")
_HANDLES = ("L12", "L18", "L12_old")


class TransformerMCheckpointSourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Extract literal model names and OneDrive share links from README table."""

    coverage_limitation = (
        "Covers only the three L12/L18/L12_old checkpoint rows in the official "
        "Transformer-M README. It does not inspect OneDrive folders or infer "
        "checkpoint filenames beyond the source's model identities."
    )

    def __init__(self, *, document_path: str = "README.md", **kwargs: Any) -> None:
        self.document_path = document_path
        kwargs.setdefault("name", "transformer-m-official-checkpoints")
        kwargs.setdefault("repository", "lsj2408/Transformer-M")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", document_path)
        kwargs.setdefault("provider_namespace", "transformer-m:checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "transformer-m-checkpoints-v1",
                "repository": self.repository,
                "branch": self.branch,
                "document_path": self.document_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "three literal model/share-link rows in Checkpoints table",
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
            self.raw_url(revision), headers={"Accept": "text/markdown,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds {self.max_response_bytes} bytes")
        checkpoints = _readme_checkpoints(response.text(), self.name, self.document_path)
        records = tuple(self._record(item, revision, response.body) for item in checkpoints)
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


def _readme_checkpoints(document: str, source: str, path: str) -> tuple[_Checkpoint, ...]:
    in_section = False
    found: dict[str, _Checkpoint] = {}
    for line in document.splitlines():
        if line.startswith("## "):
            if in_section:
                break
            in_section = line.strip() == "## Checkpoints"
            continue
        if not in_section:
            continue
        match = _ROW.match(line)
        if not match:
            continue
        handle = match.group("handle")
        cells = match.group("cells").split("|")
        url_values = [cell.strip() for cell in cells if cell.strip().startswith("https://")]
        if len(url_values) != 1 or not _URL.fullmatch(url_values[0]):
            raise ValueError(f"{source}: unexpected checkpoint link for {handle}")
        if handle in found:
            raise ValueError(f"{source}: duplicate checkpoint row for {handle}")
        found[handle] = _Checkpoint(
            handle=handle,
            url=canonicalize_url(url_values[0]),
            locator=f"{path}:Checkpoints table",
        )
    if tuple(found) != _HANDLES:
        raise ValueError(f"{source}: expected exactly L12, L18, and L12_old checkpoint rows")
    return tuple(found[handle] for handle in _HANDLES)
