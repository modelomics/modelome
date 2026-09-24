"""Pinned extraction of Graphcore's published OGB-LSC GPS++ checkpoints."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

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

_ROW = re.compile(
    r"^\|\s*(?P<label>GPS\+\+(?:\s+\d+M)?|GPS\+\+ trained on valid split)\s*"
    r"\|.*?\|\s*\[(?P<link>[^\]]+)\]\((?P<url>https://[^)]+)\)\s*\|$"
)
_REQUIRED = {
    "GPS++ 11M": "GPS_PCQ_4gps_11M.tar.gz",
    "GPS++ 22M": "GPS_PCQ_8gps_22M.tar.gz",
    "GPS++": "GPS_PCQ_16gps_44M.tar.gz",
    "GPS++ trained on valid split": "GPS_PCQ_16gps_44M_inc_valid.tar.gz",
}


class GraphcoreGPSPlusPlusCheckpointSourceAdapter(
    StaticJsonCheckpointRegistrySourceAdapter
):
    """Read the literal model/checkpoint rows in Graphcore's first-party README."""

    coverage_limitation = (
        "Covers only the four GPS++ model/checkpoint rows in Graphcore's OGB-LSC "
        "PCQM4Mv2 README table. The URLs point to tar.gz checkpoint archives; this "
        "adapter does not inspect or download their contents, and does not enumerate "
        "the 112-model challenge ensemble."
    )

    def __init__(self, *, document_path: str = "README.md", **kwargs: Any) -> None:
        self.document_path = document_path
        kwargs.setdefault("name", "graphcore-ogblsc-gpspp-checkpoints")
        kwargs.setdefault("repository", "graphcore/ogb-lsc-pcqm4mv2")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", document_path)
        kwargs.setdefault("provider_namespace", "graphcore-gpspp:checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "graphcore-gpspp-checkpoints-v1",
                "repository": self.repository,
                "branch": self.branch,
                "document_path": self.document_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "four literal GPS++ checkpoint rows in README table",
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
    in_table = False
    found: dict[str, _Checkpoint] = {}
    for line in document.splitlines():
        if line.strip() == "## Performance":
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table:
            continue
        match = _ROW.match(line)
        if not match:
            continue
        label = match.group("label")
        if label not in _REQUIRED:
            continue
        url = match.group("url")
        parsed = urlsplit(url)
        expected_file = _REQUIRED[label]
        if (
            parsed.scheme != "https"
            or parsed.hostname != "graphcore-ogblsc-pcqm4mv2.s3.us-west-1.amazonaws.com"
            or parsed.path.rsplit("/", 1)[-1] != expected_file
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"{source}: unexpected checkpoint URL for {label}")
        if label in found:
            raise ValueError(f"{source}: duplicate checkpoint row for {label}")
        found[label] = _Checkpoint(
            handle=label,
            url=canonicalize_url(url),
            locator=f"{path}:Performance table",
        )
    if set(found) != set(_REQUIRED):
        raise ValueError(f"{source}: expected exactly the four published GPS++ checkpoint rows")
    return tuple(found[label] for label in _REQUIRED)
