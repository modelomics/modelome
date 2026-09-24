"""Pinned ingestion of PaddleOCR's current first-party model list."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
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
from modelome.sources.html_catalog import _CatalogHtmlParser

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SAFE_MODEL_NAME = re.compile(r"^[^\x00-\x1f]{1,512}$")
_MODEL_HEADERS = frozenset({"模型", "模型名称"})
_WEIGHT_SUFFIXES = frozenset({".tar", ".tgz", ".zip", ".pdparams", ".pdmodel", ".pdiparams"})
_CONFIG_SUFFIXES = frozenset({".yaml", ".yml"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _ModelRow:
    name: str
    headers: tuple[str, ...]
    cells: tuple[str, ...]
    weights: tuple[str, ...]
    configs: tuple[str, ...]
    resources: tuple[str, ...]
    locator: str


class PaddleOcrCurrentModelListSourceAdapter:
    """Read direct artifact links in PaddleOCR's maintained HTML model tables.

    The document is Markdown containing HTML tables.  A row is eligible only
    when its first header is ``模型``/``模型名称``, a header explicitly names the
    model-download column, and that exact row declares a recognized artifact
    file URL.  This preserves table-row scope without executing PaddleOCR or
    retrieving any linked model bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only current PaddleOCR HTML-table rows that name a model, declare a "
        "model-download column, and directly link a recognized artifact file at one "
        "pinned commit. It does not infer models from prose, execute code, follow "
        "linked pages, or download model artifacts."
    )

    def __init__(
        self,
        *,
        name: str = "paddleocr-current-model-list",
        repository: str = "PaddlePaddle/PaddleOCR",
        branch: str = "main",
        source_path: str = "docs/version3.x/model_list.md",
        max_response_bytes: int = 8 * 1024 * 1024,
        max_entries: int = 10_000,
        max_anchors: int = 100_000,
        max_tables: int = 10_000,
        max_cells: int = 100_000,
        max_text_chars: int = 16 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.source_path = _safe_path(source_path)
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.max_anchors = _positive_int(max_anchors, "max_anchors")
        self.max_tables = _positive_int(max_tables, "max_tables")
        self.max_cells = _positive_int(max_cells, "max_cells")
        self.max_text_chars = _positive_int(max_text_chars, "max_text_chars")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paddleocr-current-model-list-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": (
                    "HTML model table row with model-download header and direct "
                    "recognized artifact"
                ),
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
        )

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(self.source_path, safe='/')}"
        )

    def blob_url(self, revision: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
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
            raise ValueError(f"{self.name}: source document returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: source document exceeds {self.max_response_bytes} bytes"
            )
        rows = _model_rows(
            response.text(),
            base_url=self.raw_url(revision),
            source=self.name,
            max_anchors=self.max_anchors,
            max_tables=self.max_tables,
            max_cells=self.max_cells,
            max_text_chars=self.max_text_chars,
        )
        if not rows:
            raise ValueError(f"{self.name}: source contains no direct model-artifact rows")
        models = _group_model_rows(rows)
        if len(models) > self.max_entries:
            raise ValueError(f"{self.name}: source exceeds {self.max_entries} named models")
        records = tuple(
            self._record(model_rows, revision, response.body)
            for model_rows in models.values()
        )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
            "artifact_row_count": len(rows),
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

    def _revision(self) -> tuple[str, HttpResponse]:
        response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _record(
        self,
        rows: tuple[_ModelRow, ...],
        revision: str,
        source: bytes,
    ) -> SourceRecord:
        name = rows[0].name
        model_key = f"paddleocr:current-model:{name}"
        local_id = f"model:{content_hash(model_key)[:24]}"
        model_identifier = Identifier("paddleocr:current-model", name)
        model = ModelHint(
            local_id=local_id,
            name=name,
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=rows[0].locator,
        )
        weights = _unique(url for row in rows for url in row.weights)
        configs = _unique(url for row in rows for url in row.configs)
        resources = _unique(url for row in rows for url in row.resources)
        links = [
            Link(
                self.repository_url,
                relation="source_repository",
                crawl=False,
                model_local_ids=(local_id,),
            )
        ]
        links.extend(
            Link(
                self.blob_url(revision),
                relation="model_card",
                locator=row.locator,
                crawl=False,
                model_local_ids=(local_id,),
            )
            for row in rows
        )
        links.extend(
            Link(
                url,
                relation="weights",
                locator=_locator_for(url, rows),
                crawl=False,
                model_local_ids=(local_id,),
            )
            for url in weights
        )
        links.extend(
            Link(
                url,
                relation="model_config",
                locator=_locator_for(url, rows),
                crawl=False,
                model_local_ids=(local_id,),
            )
            for url in configs
        )
        links.extend(
            Link(
                url,
                relation="related_resource",
                locator=_locator_for(url, rows),
                crawl=False,
                model_local_ids=(local_id,),
            )
            for url in resources
        )
        releases = tuple(
            ReleaseHint(
                local_id=f"release:{content_hash(_artifact_identity(name, url))[:24]}",
                model_local_id=local_id,
                revision=revision,
                identifiers=(
                    Identifier("paddleocr:current-artifact", _artifact_identity(name, url)),
                ),
                metadata={
                    "repository": self.repository,
                    "revision": revision,
                    "artifact_url": url,
                    "source_rows": [
                        _row_metadata(row) for row in rows if url in row.weights
                    ],
                },
                locator=_locator_for(url, rows),
            )
            for url in weights
        )
        source_url = self.blob_url(revision)
        return SourceRecord(
            source_record_id=f"model:{content_hash(model_key)[:24]}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(source_url),
            title=name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "model_name": name,
                "source_rows": [_row_metadata(row) for row in rows],
            },
            text="\n".join(
                _unique(
                    item
                    for row in rows
                    for item in (name, *row.headers, *row.cells)
                    if item
                )
            ),
            identifiers=(model_identifier,),
            links=tuple(dict.fromkeys(links)),
            models=(model,),
            releases=releases,
        )


def _model_rows(
    document: str,
    *,
    base_url: str,
    source: str,
    max_anchors: int,
    max_tables: int,
    max_cells: int,
    max_text_chars: int,
) -> tuple[_ModelRow, ...]:
    parser = _CatalogHtmlParser(
        max_anchors=max_anchors,
        max_tables=max_tables,
        max_cells=max_cells,
        max_text_chars=max_text_chars,
    )
    parser.feed(document)
    parser.close()
    parser.finish()
    result: list[_ModelRow] = []
    for table in parser.tables:
        header_index = _model_header_index(table.rows)
        if header_index is None:
            continue
        headers = tuple(_text(cell.text) for cell in table.rows[header_index].cells)
        for row in table.rows[header_index + 1 :]:
            if not row.cells or any(cell.is_header for cell in row.cells):
                continue
            name = _model_name(row.cells[0].text)
            if not name:
                continue
            urls = _unique(
                url
                for cell in row.cells
                for anchor in cell.anchors
                if (url := _web_link(base_url, anchor.href))
            )
            weights = tuple(url for url in urls if _url_suffix(url) in _WEIGHT_SUFFIXES)
            if not weights:
                continue
            configs = tuple(url for url in urls if _url_suffix(url) in _CONFIG_SUFFIXES)
            result.append(
                _ModelRow(
                    name=name,
                    headers=headers,
                    cells=tuple(_text(cell.text) for cell in row.cells),
                    weights=weights,
                    configs=configs,
                    resources=tuple(url for url in urls if url not in {*weights, *configs}),
                    locator=row.locator,
                )
            )
    return tuple(result)


def _group_model_rows(rows: tuple[_ModelRow, ...]) -> dict[str, tuple[_ModelRow, ...]]:
    result: dict[str, list[_ModelRow]] = {}
    for row in rows:
        result.setdefault(row.name, []).append(row)
    return {name: tuple(group) for name, group in result.items()}


def _model_header_index(rows: tuple[Any, ...]) -> int | None:
    for index, row in enumerate(rows):
        if not row.cells or not any(cell.is_header for cell in row.cells):
            continue
        first = _text(row.cells[0].text)
        headers = {_text(cell.text) for cell in row.cells}
        if first in _MODEL_HEADERS and "模型下载链接" in headers:
            return index
    return None


def _model_name(value: str) -> str:
    name = " ".join(_text(value).split())
    if not _SAFE_MODEL_NAME.fullmatch(name) or name.casefold() in {"n/a", "none", "-"}:
        return ""
    return name


def _web_link(base_url: str, href: str) -> str:
    value = _text(href)
    if not value or any(character.isspace() for character in value):
        return ""
    try:
        url = canonicalize_url(urljoin(base_url, value))
        parsed = urlsplit(url)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def _url_suffix(url: str) -> str:
    path = urlsplit(url).path.casefold()
    return next(
        (suffix for suffix in _WEIGHT_SUFFIXES | _CONFIG_SUFFIXES if path.endswith(suffix)),
        "",
    )


def _locator_for(url: str, rows: tuple[_ModelRow, ...]) -> str:
    return next(
        row.locator
        for row in rows
        if url in row.weights or url in row.configs or url in row.resources
    )


def _row_metadata(row: _ModelRow) -> dict[str, Any]:
    return {
        "headers": list(row.headers),
        "cells": list(row.cells),
        "weights": list(row.weights),
        "configs": list(row.configs),
        "resources": list(row.resources),
        "locator": row.locator,
    }


def _artifact_identity(name: str, url: str) -> str:
    return f"{name}\n{url}"


def _unique(values: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _safe_path(value: str) -> str:
    path = _required_text(value, "source path")
    invalid_part = any(part in {"", ".", ".."} for part in path.split("/"))
    if path.startswith("/") or "\\" in path or invalid_part:
        raise ValueError("source path is not a safe relative path")
    return path


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _header(headers: Mapping[str, Any], key: str) -> str:
    wanted = key.casefold()
    for header, value in headers.items():
        if isinstance(header, str) and header.casefold() == wanted:
            return _text(value)
    return ""


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["PaddleOcrCurrentModelListSourceAdapter"]
