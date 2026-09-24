"""Read Google's first-party NeuralGCM checkpoint table.

The repository-maintained table enumerates exact checkpoint filenames under
``gs://neuralgcm/models/``.  This adapter turns those declared object paths
into HTTPS object locators; it does not list the bucket or fetch checkpoint
bytes.
"""

from __future__ import annotations

import re
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

_HEADER = "| Reference | Model Name | Path |"
_ROW = re.compile(r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*`([^`]+)`\s*\|\s*$")
_MODEL_LINK = re.compile(r"^\[([^\]]+)\]\((https://[^)]+)\)$")
_PATH = re.compile(r"^(?:v[0-9]+(?:_[a-z0-9]+)?/)+[A-Za-z0-9_.-]+\.pkl$")


class NeuralGCMCheckpointRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate exact checkpoint rows published by the NeuralGCM project."""

    coverage_limitation = (
        "Covers only model checkpoint rows in google-deepmind/neuralgcm's first-party "
        "docs/checkpoints.md table. It does not list the storage bucket, discover other "
        "objects, inspect checkpoint metadata, or download checkpoint bytes."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "google-neuralgcm-checkpoint-registry")
        kwargs.setdefault("repository", "google-deepmind/neuralgcm")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", "docs/checkpoints.md")
        kwargs.setdefault("provider_namespace", "neuralgcm:checkpoint")
        super().__init__(**kwargs)

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
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: registry exceeds {self.max_response_bytes} bytes")
        rows = _parse_table(response.text(), self.name, self.source_path, self.max_entries)
        records = tuple(self._record(row, revision, response.body) for row in rows)
        if not records:
            raise ValueError(f"{self.name}: checkpoint table contains no rows")
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
        return handle.rsplit("/", 1)[-1].removesuffix(".pkl").replace("_", " ")


def _parse_table(
    document: str, source: str, path: str, maximum: int
) -> tuple[_Checkpoint, ...]:
    if document.count(_HEADER) != 1:
        raise ValueError(f"{source}: expected exactly one checkpoint table")
    lines = document.splitlines()
    start = next(index for index, line in enumerate(lines) if _HEADER in line)
    rows: list[_Checkpoint] = []
    handles: set[str] = set()
    for line_number, line in enumerate(lines[start + 2 :], start + 3):
        if not line.lstrip().startswith("|"):
            if rows:
                break
            continue
        match = _ROW.fullmatch(line)
        if not match:
            raise ValueError(f"{source}: invalid checkpoint row at line {line_number}")
        reference, model_name, object_path = match.groups()
        if not reference:
            if not rows:
                raise ValueError(f"{source}: checkpoint table starts with an empty reference")
        else:
            ref_match = _MODEL_LINK.fullmatch(reference)
            if not ref_match:
                raise ValueError(f"{source}: invalid model reference at line {line_number}")
        if not model_name or not _PATH.fullmatch(object_path) or ".." in object_path.split("/"):
            raise ValueError(f"{source}: invalid checkpoint declaration at line {line_number}")
        if len(rows) >= maximum:
            raise ValueError(f"{source}: registry exceeds {maximum} entries")
        if object_path in handles:
            raise ValueError(f"{source}: duplicate checkpoint path {object_path!r}")
        handles.add(object_path)
        rows.append(
            _Checkpoint(
                handle=object_path,
                url=(
                    "https://storage.googleapis.com/neuralgcm/models/"
                    + quote(object_path, safe="/")
                ),
                locator=f"{path}:L{line_number}",
            )
        )
    if not rows:
        raise ValueError(f"{source}: checkpoint table is empty")
    return tuple(rows)
