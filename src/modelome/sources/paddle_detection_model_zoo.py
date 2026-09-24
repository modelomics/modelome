"""Pinned source-archive ingestion for first-party Markdown checkpoint tables."""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, unquote, urljoin, urlsplit

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

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_NAMESPACE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_HEADING = re.compile(r"^#{1,6}[ \t]+(?P<title>.+?)\s*$")
_TABLE_SEPARATOR = re.compile(r"^:?-+:?$")
_MD_LINK = re.compile(r"\[(?P<label>[^\]\r\n]+)\]\((?P<url>[^)\s]+)\)")
_MARKDOWN = re.compile(r"[`*~]|<[^>]+>")
_DOCUMENT_PREFIX = "configs/"
_DOCUMENT_BASENAME = "README.md"
_WEIGHT_SUFFIX = re.compile(
    r"\.(?:pdparams|pdmodel|pdiparams|tar(?:\.gz)?|tgz|zip|pth|pt|ckpt|safetensors)"
    r"(?:$|[?#])",
    re.IGNORECASE,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Row:
    document_path: str
    heading: str
    cells: tuple[str, ...]
    aliases: tuple[str, ...]
    config_urls: tuple[str, ...]
    related_urls: tuple[str, ...]
    weight_url: str
    locator: str


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    url: str
    name: str
    rows: tuple[_Row, ...]


class PaddleDetectionModelZooSourceAdapter:
    """Enumerate direct checkpoint rows from one source archive.

    The configured first-party Markdown path family contains maintained model-zoo
    tables. This adapter accepts only a row that exposes an absolute direct
    artifact URL with a known checkpoint suffix. The source archive is parsed as
    text at one commit; project code is never executed and checkpoint URLs remain
    reference-only.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only direct checkpoint URLs in configured first-party Markdown model-zoo "
        "tables at one public Git commit. It does not infer weights from configurations, "
        "execute upstream code, enumerate other repository paths, or download artifacts."
    )

    def __init__(
        self,
        *,
        name: str = "paddledetection-model-zoo",
        repository: str = "PaddlePaddle/PaddleDetection",
        branch: str = "release/2.9",
        project_name: str = "PaddleDetection",
        provider_namespace: str = "paddledetection",
        document_prefix: str = _DOCUMENT_PREFIX,
        document_suffix: str = _DOCUMENT_BASENAME,
        document_paths: Sequence[str] = (),
        max_archive_bytes: int = 128 * 1024 * 1024,
        max_document_bytes: int = 2 * 1024 * 1024,
        max_total_document_bytes: int = 64 * 1024 * 1024,
        max_documents: int = 256,
        max_models: int = 100_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.project_name = _required_text(project_name, "project name")
        self.provider_namespace = _namespace(provider_namespace)
        self.document_prefix = _document_prefix(document_prefix)
        self.document_suffix = _document_suffix(document_suffix)
        self.document_paths = _document_paths(document_paths)
        self.max_archive_bytes = _positive_int(max_archive_bytes, "max_archive_bytes")
        self.max_document_bytes = _positive_int(max_document_bytes, "max_document_bytes")
        self.max_total_document_bytes = _positive_int(
            max_total_document_bytes, "max_total_document_bytes"
        )
        self.max_documents = _positive_int(max_documents, "max_documents")
        self.max_models = _positive_int(max_models, "max_models")
        self.client = client or HttpClient(max_response_bytes=self.max_archive_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "archive-markdown-checkpoint-zoo-v2",
                "repository": self.repository,
                "branch": self.branch,
                "project_name": self.project_name,
                "provider_namespace": self.provider_namespace,
                "document_prefix": self.document_prefix,
                "document_suffix": self.document_suffix,
                "document_paths": list(self.document_paths),
                "max_archive_bytes": self.max_archive_bytes,
                "max_document_bytes": self.max_document_bytes,
                "max_total_document_bytes": self.max_total_document_bytes,
                "max_documents": self.max_documents,
                "max_models": self.max_models,
                "admission": "configured Markdown table row with a direct checkpoint URL",
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

    def archive_url(self, revision: str) -> str:
        return f"{self.repository_url}/archive/{quote(revision, safe='')}.zip"

    def blob_url(self, revision: str, path: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(path, safe='/')}"
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
            self.archive_url(revision), headers={"Accept": "application/zip"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source archive returned HTTP {response.status}")
        if len(response.body) > self.max_archive_bytes:
            raise ValueError(
                f"{self.name}: source archive exceeds {self.max_archive_bytes} bytes"
            )
        documents = _archive_documents(
            response.body,
            source=self.name,
            document_prefix=self.document_prefix,
            document_suffix=self.document_suffix,
            document_paths=self.document_paths,
            max_document_bytes=self.max_document_bytes,
            max_total_document_bytes=self.max_total_document_bytes,
            max_documents=self.max_documents,
        )
        rows = tuple(
            row
            for path, document in documents.items()
            for row in _table_rows(
                document,
                path=path,
                source=self.name,
                revision=revision,
                blob_url=self.blob_url,
            )
        )
        checkpoints = _checkpoints(rows)
        if len(checkpoints) > self.max_models:
            raise ValueError(f"{self.name}: checkpoint catalog exceeds {self.max_models} models")
        if not checkpoints:
            raise ValueError(f"{self.name}: source archive contains no direct checkpoint rows")
        records = tuple(
            self._record(checkpoint, revision=revision, archive=response.body)
            for checkpoint in checkpoints
        )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "archive_url": self.archive_url(revision),
            "archive_sha256": content_hash(response.body),
            "document_count": len(documents),
            "checkpoint_row_count": len(rows),
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

    def _record(
        self, checkpoint: _Checkpoint, *, revision: str, archive: bytes
    ) -> SourceRecord:
        identity = checkpoint.url
        identifier = Identifier(f"{self.provider_namespace}:checkpoint-model", identity)
        local_id = f"model:{content_hash(identity)[:24]}"
        primary = checkpoint.rows[0]
        model = ModelHint(
            local_id=local_id,
            name=checkpoint.name,
            aliases=tuple(
                sorted({alias for row in checkpoint.rows for alias in row.aliases if alias})
            ),
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=primary.locator,
        )
        links: list[Link] = [
            Link(
                self.repository_url,
                relation="source_repository",
                crawl=False,
                model_local_ids=(local_id,),
            ),
            Link(
                checkpoint.url,
                relation="weights",
                locator=primary.locator,
                crawl=False,
                model_local_ids=(local_id,),
            ),
        ]
        for row in checkpoint.rows:
            links.append(
                Link(
                    self.blob_url(revision, row.document_path),
                    relation="model_card",
                    locator=row.locator,
                    crawl=False,
                    model_local_ids=(local_id,),
                )
            )
            links.extend(
                Link(
                    url,
                    relation="model_config",
                    locator=row.locator,
                    crawl=False,
                    model_local_ids=(local_id,),
                )
                for url in row.config_urls
            )
            links.extend(
                Link(
                    url,
                    relation="related_resource",
                    locator=row.locator,
                    crawl=False,
                    model_local_ids=(local_id,),
                )
                for url in row.related_urls
            )
        release = ReleaseHint(
            local_id=f"release:{content_hash(identity)[:24]}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier(f"{self.provider_namespace}:checkpoint", identity),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "checkpoint_url": checkpoint.url,
                "source_rows": [
                    {
                        "document_path": row.document_path,
                        "heading": row.heading,
                        "cells": list(row.cells),
                        "locator": row.locator,
                    }
                    for row in checkpoint.rows
                ],
            },
            locator=primary.locator,
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{content_hash(identity)[:24]}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(
                self.blob_url(revision, primary.document_path)
            ),
            title=checkpoint.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "archive_sha256": content_hash(archive),
                "checkpoint_url": checkpoint.url,
                "source_rows": [
                    {
                        "document_path": row.document_path,
                        "heading": row.heading,
                        "cells": list(row.cells),
                        "aliases": list(row.aliases),
                        "config_urls": list(row.config_urls),
                        "related_urls": list(row.related_urls),
                        "locator": row.locator,
                    }
                    for row in checkpoint.rows
                ],
            },
            text="\n".join(
                (
                    f"{self.project_name} checkpoint: {checkpoint.name}",
                    *(
                        f"{row.document_path}: {row.heading}"
                        for row in checkpoint.rows
                        if row.heading
                    ),
                )
            ),
            identifiers=(identifier,),
            links=tuple(dict.fromkeys(links)),
            models=(model,),
            releases=(release,),
        )


def _archive_documents(
    archive: bytes,
    *,
    source: str,
    document_prefix: str,
    document_suffix: str,
    document_paths: tuple[str, ...],
    max_document_bytes: int,
    max_total_document_bytes: int,
    max_documents: int,
) -> dict[str, str]:
    try:
        package = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as error:
        raise ValueError(f"{source}: source archive is not a ZIP file") from error
    with package:
        documents: dict[str, str] = {}
        total = 0
        for info in package.infolist():
            if info.is_dir():
                continue
            path = _archive_path(info.filename, source)
            if document_paths:
                selected = path in document_paths
            else:
                selected = path.startswith(document_prefix) and path.endswith(document_suffix)
            if not selected:
                continue
            if info.file_size > max_document_bytes:
                raise ValueError(
                    f"{source}: model-zoo document {path!r} exceeds {max_document_bytes} bytes"
                )
            total += info.file_size
            if total > max_total_document_bytes:
                raise ValueError(
                    f"{source}: model-zoo documents exceed {max_total_document_bytes} bytes"
                )
            if path in documents:
                raise ValueError(f"{source}: source archive has duplicate {path!r}")
            documents[path] = package.read(info).decode("utf-8", errors="strict")
        if len(documents) > max_documents:
            raise ValueError(f"{source}: source archive has more than {max_documents} documents")
    if not documents:
        raise ValueError(f"{source}: source archive has no matching model-zoo documents")
    return dict(sorted(documents.items()))


def _archive_path(value: str, source: str) -> str:
    parts = value.split("/")
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{source}: unsafe source archive member {value!r}")
    return "/".join(parts[1:])


def _table_rows(
    document: str,
    *,
    path: str,
    source: str,
    revision: str,
    blob_url: Callable[[str, str], str],
) -> tuple[_Row, ...]:
    heading = ""
    header: tuple[str, ...] | None = None
    rows: list[_Row] = []
    lines = document.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if match := _HEADING.match(line):
            heading = _plain(match.group("title"))
            header = None
            index += 1
            continue
        cells = _table_cells(line)
        if not cells:
            header = None
            index += 1
            continue
        if index + 1 < len(lines) and _table_separator(_table_cells(lines[index + 1])):
            header = cells
            index += 2
            continue
        if header is not None and _download_header(header):
            rows.extend(
                _row_entries(
                    cells,
                    document_path=path,
                    heading=heading,
                    line_number=index + 1,
                    source=source,
                    revision=revision,
                    blob_url=blob_url,
                )
            )
        index += 1
    return tuple(rows)


def _table_cells(line: str) -> tuple[str, ...] | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = tuple(cell.strip() for cell in stripped.split("|"))
    return cells if len(cells) >= 2 else None


def _table_separator(cells: tuple[str, ...] | None) -> bool:
    return bool(cells) and all(_TABLE_SEPARATOR.fullmatch(cell.strip()) for cell in cells)


def _download_header(header: tuple[str, ...]) -> bool:
    return any(
        re.search(
            r"download|weight|pretrained|checkpoint|model[ _-]?link|links?|模型下载|下载",
            _plain(cell),
            re.I,
        )
        for cell in header
    )


def _row_entries(
    cells: tuple[str, ...],
    *,
    document_path: str,
    heading: str,
    line_number: int,
    source: str,
    revision: str,
    blob_url: Callable[[str, str], str],
) -> tuple[_Row, ...]:
    links = [
        (label, url)
        for cell in cells
        for label, url in _markdown_links(cell)
    ]
    weight_urls = tuple(
        dict.fromkeys(
            canonicalize_url(url)
            for _, url in links
            if _is_direct_checkpoint(url)
        )
    )
    if not weight_urls:
        return ()
    config_urls = tuple(
        dict.fromkeys(
            _config_url(url, revision=revision, document_path=document_path, blob_url=blob_url)
            for _, url in links
            if _is_config_link(url)
        )
    )
    related_urls = tuple(
        dict.fromkeys(
            canonicalize_url(url)
            for _, url in links
            if _is_web_url(url) and not _is_direct_checkpoint(url) and not _is_config_link(url)
        )
    )
    aliases = (_plain(cells[0]),) if _plain(cells[0]) else ()
    locator = f"{document_path}:line:{line_number}"
    return tuple(
        _Row(
            document_path=document_path,
            heading=heading,
            cells=tuple(_plain(cell) for cell in cells),
            aliases=aliases,
            config_urls=config_urls,
            related_urls=related_urls,
            weight_url=url,
            locator=locator,
        )
        for url in weight_urls
    )


def _markdown_links(value: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (match.group("label").strip(), match.group("url").strip())
        for match in _MD_LINK.finditer(value)
    )


def _is_direct_checkpoint(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and bool(
        _WEIGHT_SUFFIX.search(parsed.path)
    )


def _is_web_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_config_link(url: str) -> bool:
    return urlsplit(url).path.casefold().endswith((".py", ".yml", ".yaml"))


def _config_url(
    url: str,
    *,
    revision: str,
    document_path: str,
    blob_url: Callable[[str, str], str],
) -> str:
    if _is_web_url(url):
        return canonicalize_url(url)
    base = blob_url(revision, document_path)
    resolved = canonicalize_url(urljoin(base, url))
    if urlsplit(resolved).netloc != "github.com":
        raise ValueError("PaddleDetection model config link resolved outside GitHub")
    return resolved


def _checkpoints(rows: tuple[_Row, ...]) -> tuple[_Checkpoint, ...]:
    grouped: dict[str, list[_Row]] = {}
    for row in rows:
        grouped.setdefault(row.weight_url, []).append(row)
    return tuple(
        _Checkpoint(
            url=url,
            name=_checkpoint_name(url),
            rows=tuple(grouped[url]),
        )
        for url in sorted(grouped)
    )


def _checkpoint_name(url: str) -> str:
    path = unquote(urlsplit(url).path)
    filename = path.rsplit("/", 1)[-1]
    suffixes = (
        ".safetensors",
        ".tar.gz",
        ".pdparams",
        ".pdmodel",
        ".pdiparams",
        ".ckpt",
        ".tar",
        ".tgz",
        ".zip",
        ".pth",
        ".pt",
    )
    for suffix in suffixes:
        if filename.casefold().endswith(suffix):
            name = filename[: -len(suffix)].strip()
            if name:
                if name.casefold() in {"model", "inference"}:
                    parent = path.rsplit("/", 2)[-2].strip()
                    if parent:
                        return parent
                return name
    raise ValueError(f"Paddle-project checkpoint URL has no usable filename: {url!r}")


def _plain(value: str) -> str:
    value = _MD_LINK.sub(lambda match: match.group("label"), value)
    value = _MARKDOWN.sub("", value)
    return " ".join(value.split()).strip()


def _document_prefix(value: str) -> str:
    prefix = _required_text(value, "document prefix").strip("/")
    if not prefix or "\\" in prefix or any(part in {".", ".."} for part in prefix.split("/")):
        raise ValueError("document prefix is not a safe relative path")
    return prefix + "/"


def _document_suffix(value: str) -> str:
    suffix = _required_text(value, "document suffix")
    if not suffix.startswith(".") and "/" not in suffix:
        suffix = "/" + suffix
    if "\\" in suffix or ".." in suffix.split("/"):
        raise ValueError("document suffix is not a safe filename suffix")
    return suffix


def _document_paths(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("document_paths must be an array of safe relative paths")
    paths = tuple(_safe_relative_path(item, "document path") for item in value)
    if len(set(paths)) != len(paths):
        raise ValueError("document_paths cannot contain duplicates")
    return paths


def _safe_relative_path(value: str, field: str) -> str:
    path = _required_text(value, field)
    invalid_part = any(part in {"", ".", ".."} for part in path.split("/"))
    if path.startswith("/") or "\\" in path or invalid_part:
        raise ValueError(f"{field} is not a safe relative path")
    return path


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _namespace(value: str) -> str:
    namespace = _required_text(value, "provider namespace").casefold()
    if not _NAMESPACE.fullmatch(namespace):
        raise ValueError("provider namespace must be a lowercase identifier")
    return namespace


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
    value = headers.get(key) or headers.get(key.casefold()) or headers.get(key.title())
    return _text(value)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["PaddleDetectionModelZooSourceAdapter"]
