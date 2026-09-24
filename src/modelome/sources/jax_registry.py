"""Pinned reader for the first-party T5X pretrained checkpoint table."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, unquote, urlsplit

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

_LINK = re.compile(r"^\[`?(?P<label>[^`\]]+?)`?\]\((?P<url>https?://[^)]+)\)$")
_GCS = re.compile(r"^gs://([a-z0-9._-]+)/([A-Za-z0-9._/-]+)$")
_OBJECT = re.compile(r"^[A-Za-z0-9._/-]+$")
_CHECKPOINT = re.compile(r"checkpoint_[0-9]+$")


class JaxRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate explicit T5X model/checkpoint rows from its official docs.

    The admitted scope is model tables whose second cell links a Gin config and
    third cell links a checkpoint under the documented T5X GCS prefix. The
    checkpoint cell may show a ``gs://`` URI or a relative checkpoint path, as
    the first-party document does. It reads the Markdown document only; model
    binaries and cloud objects are never requested.
    """

    coverage_limitation = (
        "Covers only explicit pretrained model table rows with Gin configs and "
        "Google Cloud Storage checkpoint directories in google-research/t5x "
        "docs/models.md at one pinned revision. It does not infer models, inspect "
        "checkpoint contents, or download cloud objects."
    )

    def __init__(
        self,
        *,
        name: str = "t5x-model-registry",
        repository: str = "google-research/t5x",
        branch: str = "main",
        source_path: str = "docs/models.md",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 10_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            repository=repository,
            branch=branch,
            source_path=source_path,
            provider_namespace="t5x:model",
            max_response_bytes=max_response_bytes,
            max_entries=max_entries,
            **kwargs,
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "jax-t5x-registry-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "markdown rows with Gin links and T5X GCS checkpoint paths",
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
        rows = []
        seen: set[str] = set()
        for line_no, line in enumerate(response.text().splitlines(), 1):
            cells = _table_cells(line)
            if cells is None:
                continue
            name, gin_cell, checkpoint_cell = cells
            gin_link = _LINK.fullmatch(gin_cell)
            checkpoint_link = _LINK.fullmatch(checkpoint_cell)
            if not gin_link or not checkpoint_link or name.casefold() == "model":
                continue
            gin_url = urlsplit(gin_link.group("url"))
            if gin_url.hostname not in {"github.com", "www.github.com"}:
                continue
            gs_path = _checkpoint_path(checkpoint_link.group("label"), checkpoint_link.group("url"))
            if gs_path is None:
                raise ValueError(f"{self.name}: invalid checkpoint table row at line {line_no}")
            gin = gin_link.group("label")
            if name in seen:
                raise ValueError(f"{self.name}: duplicate model row {name!r}")
            seen.add(name)
            rows.append((name, gin, gs_path, f"{self.source_path}:line:{line_no}"))
            if len(rows) > self.max_entries:
                raise ValueError(f"{self.name}: registry exceeds {self.max_entries} entries")
        if not rows:
            raise ValueError(f"{self.name}: registry contains no matching checkpoint rows")
        records = tuple(self._jax_record(row, revision, response.body) for row in rows)
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

    def _jax_record(
        self, row: tuple[str, str, str, str], revision: str, source: bytes
    ) -> SourceRecord:
        name, gin, gs_path, locator = row
        identifier = Identifier("t5x:model", name)
        model = ModelHint(
            local_id=f"model:{content_hash(name)[:24]}",
            name=name,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        bucket, obj = _GCS.fullmatch(gs_path).groups()  # type: ignore[union-attr]
        weight_url = f"https://storage.googleapis.com/{bucket}/{quote(obj, safe='/')}"
        source_url = self.blob_url(revision)
        return SourceRecord(
            source_record_id=f"t5x:{content_hash(name)[:24]}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(source_url),
            title=name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "gin_config": gin,
                "checkpoint_location": gs_path,
            },
            text=f"T5X pretrained model: {name}; Gin config: {gin}; checkpoint: {gs_path}",
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
                    weight_url,
                    relation="weights",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(model.local_id,),
                ),
            ),
            models=(model,),
            releases=(
                ReleaseHint(
                    local_id=f"release:{content_hash(name)[:24]}",
                    model_local_id=model.local_id,
                    revision=revision,
                    identifiers=(Identifier("t5x:release", name),),
                    metadata={
                        "repository": self.repository,
                        "revision": revision,
                        "gin_config": gin,
                        "checkpoint_location": gs_path,
                    },
                    locator=locator,
                ),
            ),
        )


def _table_cells(line: str) -> tuple[str, str, str] | None:
    """Split a simple Markdown row, allowing optional outer pipe delimiters."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = tuple(cell.strip() for cell in stripped.split("|"))
    return cells if len(cells) == 3 else None


def _checkpoint_path(label: str, target: str) -> str | None:
    """Validate a T5X checkpoint link and return its full source-native GCS URI."""
    parsed = urlsplit(target)
    if parsed.scheme != "https" or parsed.hostname != "console.cloud.google.com":
        return None
    prefix = "/storage/browser/"
    if not parsed.path.startswith(prefix):
        return None
    target_parts = [unquote(part) for part in parsed.path[len(prefix) :].split("/")]
    if len(target_parts) < 3:
        return None
    bucket, *object_parts = target_parts
    target_object = "/".join(object_parts)
    if bucket != "t5-data" or not target_object.startswith("pretrained_models/t5x/"):
        return None
    if not _OBJECT.fullmatch(target_object) or any(
        part in {"", ".", ".."} for part in object_parts
    ):
        return None
    if label.startswith("gs://"):
        match = _GCS.fullmatch(label)
        if not match or match.group(1) != bucket:
            return None
        object_path = match.group(2)
        if not object_path.startswith("pretrained_models/t5x/"):
            return None
        if target_object != object_path and not object_path.startswith(target_object + "/"):
            return None
    else:
        object_path = label.strip("/")
        if not _OBJECT.fullmatch(object_path) or any(
            part in {"", ".", ".."} for part in object_path.split("/")
        ):
            return None
        if not _CHECKPOINT.search(object_path):
            return None
        parent = object_path.rsplit("/", 1)[0]
        if target_object == object_path or target_object.endswith("/" + object_path):
            pass
        elif target_object.endswith("/" + parent):
            object_path = target_object + "/" + object_path.rsplit("/", 1)[-1]
        else:
            return None
        if not object_path.startswith("pretrained_models/t5x/"):
            object_path = "pretrained_models/t5x/" + object_path
    return f"gs://{bucket}/{object_path}"
