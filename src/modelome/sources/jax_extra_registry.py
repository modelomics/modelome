"""Pinned reader for Scenic's first-party JAX/Flax baseline model zoo."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urlsplit

from modelome.http import HttpResponse
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
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _header,
    _isoformat,
    _text,
)

_LINK = re.compile(r"^\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)]+)\)$")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._+-]+$")


class JaxExtraRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate Scenic baseline model-zoo rows with explicit GCS checkpoints.

    Rows must belong to the Model/Dataset/Pretraining/.../Checkpoint table in
    the configured first-party Scenic README. The parser admits only a single
    direct public Google Storage checkpoint link per row and never fetches it.
    """

    coverage_limitation = (
        "Covers explicit Model, Dataset, Pretraining, and Checkpoint rows in one "
        "configured Scenic baseline model-zoo Markdown document at a pinned "
        "revision. It does not enumerate other Scenic project README files, infer "
        "architectures, follow links, or download checkpoint contents."
    )

    def __init__(
        self,
        *,
        name: str = "scenic-baseline-model-zoo",
        repository: str = "google-research/scenic",
        branch: str = "main",
        source_path: str = "scenic/projects/baselines/README.md",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 10_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            repository=repository,
            branch=branch,
            source_path=source_path,
            provider_namespace="scenic:model",
            max_response_bytes=max_response_bytes,
            max_entries=max_entries,
            **kwargs,
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "jax-scenic-baseline-registry-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": (
                    "model-zoo table rows with public storage.googleapis.com checkpoint URLs"
                ),
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
                upstream_count=int(state.get("model_count", 0)),
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/markdown,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: registry exceeds {self.max_response_bytes} bytes")
        rows = _parse_model_zoo(response.text(), source=self.name, maximum=self.max_entries)
        records = tuple(self._record(row, revision, response.body) for row in rows)
        if not records:
            raise ValueError(f"{self.name}: registry contains no matching checkpoint rows")
        next_state = {
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

    def _record(
        self, row: tuple[str, str, str, str, str], revision: str, source: bytes
    ) -> SourceRecord:
        model_name, dataset, pretraining, accuracy, checkpoint_url = row
        identity = "|".join((model_name, dataset, pretraining, checkpoint_url))
        key = content_hash(identity)[:24]
        identifier = Identifier("scenic:model", identity)
        model = ModelHint(
            local_id=f"model:{key}",
            name=model_name,
            aliases=tuple(value for value in (dataset, pretraining) if value and value != "-"),
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=f"{self.source_path}:model-zoo:{model_name}:{dataset}:{pretraining}",
        )
        source_url = self.blob_url(revision)
        locator = model.locator
        return SourceRecord(
            source_record_id=f"scenic:{key}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(source_url),
            title=f"{model_name} ({dataset}; {pretraining})",
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "dataset": dataset,
                "pretraining": pretraining,
                "reported_accuracy": accuracy,
                "checkpoint_url": checkpoint_url,
            },
            text=(
                f"Scenic checkpoint: {model_name}; dataset: {dataset}; "
                f"pretraining: {pretraining}; accuracy: {accuracy}; {checkpoint_url}"
            ),
            identifiers=(identifier,),
            links=(
                Link(
                    source_url,
                    relation="model_card",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(model.local_id,),
                ),
                Link(
                    self.repository_url,
                    relation="source_implementation",
                    crawl=False,
                    model_local_ids=(model.local_id,),
                ),
                Link(
                    checkpoint_url,
                    relation="weights",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(model.local_id,),
                ),
            ),
            models=(model,),
            releases=(
                ReleaseHint(
                    local_id=f"release:{key}",
                    model_local_id=model.local_id,
                    revision=revision,
                    identifiers=(Identifier("scenic:release", identity),),
                    metadata={
                        "repository": self.repository,
                        "revision": revision,
                        "dataset": dataset,
                        "pretraining": pretraining,
                        "checkpoint_url": checkpoint_url,
                    },
                    locator=locator,
                ),
            ),
        )


def _parse_model_zoo(
    document: str, *, source: str, maximum: int
) -> tuple[tuple[str, str, str, str, str], ...]:
    rows: list[tuple[str, str, str, str, str]] = []
    seen: set[str] = set()
    active = False
    for line_no, line in enumerate(document.splitlines(), 1):
        cells = _cells(line)
        if cells is None:
            active = False
            continue
        normalized = tuple(cell.casefold() for cell in cells)
        if (
            normalized[0] == "model"
            and normalized[1] == "dataset"
            and normalized[2] == "pretraining"
            and normalized[-1] == "checkpoint"
        ):
            active = True
            continue
        if not active:
            continue
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        if len(cells) != 5:
            active = False
            continue
        model_name, dataset, pretraining, accuracy, checkpoint_cell = cells
        checkpoint = _LINK.fullmatch(checkpoint_cell)
        if checkpoint is None:
            active = False
            continue
        url = _safe_storage_url(checkpoint.group("url"), source, line_no)
        identity = "|".join((model_name, dataset, pretraining, url))
        if identity in seen:
            raise ValueError(f"{source}: duplicate model-zoo row at line {line_no}")
        seen.add(identity)
        rows.append((model_name, dataset, pretraining, accuracy, url))
        if len(rows) > maximum:
            raise ValueError(f"{source}: model zoo exceeds {maximum} entries")
    return tuple(rows)


def _cells(line: str) -> tuple[str, ...] | None:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    cells = tuple(cell.strip() for cell in stripped.strip("|").split("|"))
    return cells


def _safe_storage_url(url: str, source: str, line_no: int) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "storage.googleapis.com":
        raise ValueError(f"{source}: non-public Google Storage checkpoint URL at line {line_no}")
    parts = [unquote(part) for part in parsed.path.split("/")[1:]]
    if len(parts) < 2 or any(
        not part or part in {".", ".."} or not _SAFE_SEGMENT.fullmatch(part) for part in parts
    ):
        raise ValueError(f"{source}: unsafe checkpoint URL path at line {line_no}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{source}: checkpoint URL has query or fragment at line {line_no}")
    return canonicalize_url(url)
