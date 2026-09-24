"""Read source-declared pretrained model rows from NASA-IMPACT/Prithvi-EO-2.0.

Only the first-party README's three-column Pre-trained Models table is admitted.
The weight destination is retained as the upstream-declared Hugging Face model
repository; this adapter does not query that service or download artifacts.
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

_ROW = re.compile(
    r"^\|\s*(Prithvi-EO-[A-Za-z0-9.-]+)\s*\|\s*([^|]+?)\s*\|\s*https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s*\|\s*$"
)
_TABLE_HEADER = "| Model  | Details  | Weights"


class GeospatialRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate Prithvi models listed with weight locations by NASA's project."""

    coverage_limitation = (
        "Covers only rows in NASA-IMPACT/Prithvi-EO-2.0's first-party pretrained-model "
        "README table. Weight links are source-declared model repository locations, "
        "not direct binary files; this adapter does not query Hugging Face or fetch weights."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "nasa-prithvi-geospatial-registry")
        kwargs.setdefault("repository", "NASA-IMPACT/Prithvi-EO-2.0")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", "README.md")
        kwargs.setdefault("provider_namespace", "nasa-prithvi:model")
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
        rows = _parse_table(response.text(), self.name, self.max_entries)
        records = tuple(self._record(row, revision, response.body) for row in rows)
        if not records:
            raise ValueError(f"{self.name}: pretrained model table contains no rows")
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

    def raw_url(self, revision: str) -> str:
        revision_path = quote(revision, safe="")
        source_path = quote(self.source_path, safe="/")
        return f"https://raw.githubusercontent.com/{self.repository}/{revision_path}/{source_path}"

    def _record(self, row: _Checkpoint, revision: str, source: bytes):
        record = super()._record(row, revision, source)
        # README locations are model repositories. Preserve them, but remove the
        # inherited binary claim and store a neutral upstream artifact locator.
        from dataclasses import replace

        links = tuple(
            replace(link, relation="model_repository")
            if link.relation == "weights"
            else link
            for link in record.links
        )
        raw = dict(record.raw)
        raw["model_repository_url"] = raw.pop("weight_url")
        releases = tuple(
            replace(
                release,
                metadata={
                    key: value for key, value in release.metadata.items() if key != "weight_url"
                }
                | {"model_repository_url": row.url},
            )
            for release in record.releases
        )
        return replace(record, links=links, raw=raw, releases=releases)


def _parse_table(document: str, source: str, maximum: int) -> tuple[_Checkpoint, ...]:
    if document.count(_TABLE_HEADER) != 1:
        raise ValueError(f"{source}: expected exactly one pretrained-model table")
    lines = document.splitlines()
    start = next(i for i, line in enumerate(lines) if _TABLE_HEADER in line)
    rows: list[_Checkpoint] = []
    for line_number, line in enumerate(lines[start + 2 :], start + 3):
        if not line.lstrip().startswith("|"):
            if rows:
                break
            continue
        match = _ROW.fullmatch(line)
        if not match:
            raise ValueError(f"{source}: invalid pretrained-model row at line {line_number}")
        handle, detail, owner_repo = match.groups()
        if len(rows) >= maximum:
            raise ValueError(f"{source}: registry exceeds {maximum} entries")
        rows.append(
            _Checkpoint(
                handle=handle,
                url=f"https://huggingface.co/{owner_repo}",
                locator=f"README.md:L{line_number}",
            )
        )
    if not rows:
        raise ValueError(f"{source}: pretrained model table is empty")
    if len({row.handle for row in rows}) != len(rows):
        raise ValueError(f"{source}: duplicate model handle")
    return tuple(rows)
