"""Pinned first-party PaddleRec recommendation-algorithm catalog ingestion."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SAFE_TEXT = re.compile(r"^[^\x00-\x1f]{1,512}$")
_TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")
_MD_LINK = re.compile(r"\[(?P<label>[^\]\r\n]+)\]\((?P<target>[^)\r\n]+)\)")
_EXPECTED_HEADERS = (
    "type",
    "algorithm",
    "online environment",
    "parameter-server",
    "multi-gpu",
    "version",
    "paper",
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _AlgorithmRow:
    category: str
    name: str
    headers: tuple[str, ...]
    cells: tuple[str, ...]
    links: tuple[tuple[str, str], ...]
    locator: str


class PaddleRecCatalogSourceAdapter:
    """Read PaddleRec's source-declared support-table technique records.

    A row is admitted only from the table whose declared columns are ``Type``,
    ``Algorithm``, and ``Paper``. Its first Algorithm-cell link supplies the
    source implementation; model documentation, online environment, release,
    and paper links remain scoped to that same row. The table describes
    techniques and implementations, not a downloadable checkpoint inventory,
    so it produces documented modelome entries without synthetic releases.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only exact recommendation-algorithm rows in PaddleRec's first-party "
        "English support table at one pinned commit. It does not infer a trained "
        "release, resolve citations not linked by the row, fetch documentation pages, "
        "or download code or model files."
    )

    def __init__(
        self,
        *,
        name: str = "paddlerec-algorithm-catalog",
        repository: str = "PaddlePaddle/PaddleRec",
        branch: str = "master",
        source_path: str = "README_EN.md",
        max_response_bytes: int = 8 * 1024 * 1024,
        max_entries: int = 10_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.source_path = _safe_path(source_path)
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paddlerec-algorithm-catalog-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "PaddleRec Type/Algorithm/Paper support-table row",
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

    def blob_url(self, revision: str, path: str | None = None) -> str:
        target = self.source_path if path is None else path
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(target, safe='/')}"
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
        rows = _algorithm_rows(response.text(), self.name, self.source_path)
        if not rows:
            raise ValueError(f"{self.name}: source contains no supported-algorithm rows")
        if len(rows) > self.max_entries:
            raise ValueError(f"{self.name}: source exceeds {self.max_entries} algorithm rows")
        records = tuple(self._record(row, revision, response.body) for row in rows)
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

    def _record(self, row: _AlgorithmRow, revision: str, source: bytes) -> SourceRecord:
        identity = f"{row.category}\n{row.name}"
        local_id = f"model:{content_hash(identity)[:24]}"
        identifier = Identifier("paddlerec:recommendation-algorithm", identity)
        links = [
            Link(
                self.repository_url,
                relation="source_repository",
                crawl=False,
                model_local_ids=(local_id,),
            ),
            Link(
                self.blob_url(revision),
                relation="model_card",
                locator=row.locator,
                crawl=False,
                model_local_ids=(local_id,),
            ),
        ]
        links.extend(
            Link(
                self._resolved_link(url, revision),
                relation=relation,
                locator=row.locator,
                crawl=False,
                model_local_ids=(local_id,),
            )
            for url, relation in row.links
        )
        model = ModelHint(
            local_id=local_id,
            name=row.name,
            identifiers=(identifier,),
            status=ModelStatus.DOCUMENTED,
            locator=row.locator,
        )
        return SourceRecord(
            source_record_id=f"algorithm:{content_hash(identity)[:24]}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(self.blob_url(revision)),
            title=row.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "category": row.category,
                "algorithm": row.name,
                "headers": list(row.headers),
                "cells": list(row.cells),
                "row_links": [
                    {
                        "url": self._resolved_link(url, revision),
                        "relation": relation,
                    }
                    for url, relation in row.links
                ],
                "locator": row.locator,
            },
            text="\n".join((row.category, row.name, *row.cells)),
            identifiers=(identifier,),
            links=tuple(dict.fromkeys(links)),
            models=(model,),
        )

    def _resolved_link(self, url: str, revision: str) -> str:
        if _is_web_url(url):
            return url
        return self.blob_url(revision, _relative_target(self.source_path, url))


def _algorithm_rows(
    document: str, source: str, source_path: str
) -> tuple[_AlgorithmRow, ...]:
    result: list[_AlgorithmRow] = []
    lines = document.splitlines()
    index = 0
    while index + 1 < len(lines):
        headers = _table_cells(lines[index])
        separator = _table_cells(lines[index + 1])
        if _is_algorithm_header(headers) and _is_separator(separator):
            index += 2
            while index < len(lines):
                cells = _table_cells(lines[index])
                if not cells or len(cells) < 2:
                    break
                row = _algorithm_row(headers, cells, index + 1, source_path)
                if row is not None:
                    result.append(row)
                index += 1
            continue
        index += 1
    result = _deduplicate(result, source)
    return tuple(result)


def _algorithm_row(
    headers: tuple[str, ...],
    cells: tuple[str, ...],
    line_number: int,
    source_path: str,
) -> _AlgorithmRow | None:
    category = _plain(cells[0])
    algorithm_links = _links(cells[1])
    if not category or not algorithm_links:
        return None
    name = algorithm_links[0][0]
    if not _SAFE_TEXT.fullmatch(name) or name.casefold() in {"algorithm", "n/a", "none"}:
        return None
    links: list[tuple[str, str]] = []
    for cell_index, cell in enumerate(cells):
        for link_index, (_, url) in enumerate(_links(cell)):
            if cell_index == 1 and link_index == 0:
                relation = "source_implementation"
            elif _host(url).endswith("paddlerec.readthedocs.io"):
                relation = "model_card"
            elif _host(url).endswith("aistudio.baidu.com"):
                relation = "related_resource"
            elif cell_index == len(cells) - 1:
                relation = "paper_reference"
            elif _is_paddlerec_source(url):
                relation = "source_implementation"
            else:
                relation = "related_resource"
            links.append((url, relation))
    if not links:
        return None
    return _AlgorithmRow(
        category=category,
        name=name,
        headers=headers,
        cells=tuple(_plain(cell) for cell in cells),
        links=tuple(dict.fromkeys(links)),
        locator=f"{source_path}:line:{line_number}",
    )


def _table_cells(value: str) -> tuple[str, ...]:
    body = value.strip()
    if "|" not in body:
        return ()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return tuple(cell.strip() for cell in body.split("|"))


def _is_algorithm_header(cells: tuple[str, ...]) -> bool:
    return tuple(_plain(cell).casefold() for cell in cells) == _EXPECTED_HEADERS


def _is_separator(cells: tuple[str, ...]) -> bool:
    return bool(cells) and all(_TABLE_SEPARATOR.fullmatch(cell.replace(" ", "")) for cell in cells)


def _links(value: str) -> tuple[tuple[str, str], ...]:
    links = []
    for match in _MD_LINK.finditer(value):
        label = _text(match.group("label"))
        target = _text(match.group("target")).strip("<>")
        if label and _valid_target(target):
            links.append((label, target))
    return tuple(links)


def _valid_target(value: str) -> bool:
    if not value or value.startswith("#") or any(character.isspace() for character in value):
        return False
    if _is_web_url(value):
        try:
            canonicalize_url(value)
        except ValueError:
            return False
        return True
    try:
        _relative_target("README_EN.md", value)
    except ValueError:
        return False
    return True


def _relative_target(source_path: str, target: str) -> str:
    raw_path = target.split("#", 1)[0].split("?", 1)[0]
    path = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), raw_path))
    return _safe_path(path)


def _is_web_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _host(value: str) -> str:
    return (urlsplit(value).hostname or "").casefold()


def _is_paddlerec_source(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.hostname or ""
    ).casefold() == "github.com" and parsed.path.startswith("/PaddlePaddle/PaddleRec/")


def _plain(value: str) -> str:
    labels_only = _MD_LINK.sub(lambda match: _text(match.group("label")), value)
    return labels_only.replace("<br>", " ").strip()


def _deduplicate(rows: list[_AlgorithmRow], source: str) -> list[_AlgorithmRow]:
    result: dict[tuple[str, str], _AlgorithmRow] = {}
    for row in rows:
        key = (row.category, row.name)
        if key in result:
            raise ValueError(f"{source}: duplicate source-declared algorithm identity {key!r}")
        result[key] = row
    return list(result.values())


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


__all__ = ["PaddleRecCatalogSourceAdapter"]
