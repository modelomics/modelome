"""Pinned enumeration of PaddlePaddle's public ModelCenter catalog."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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
from modelome.normalize import canonicalize_url, content_hash, extract_urls

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_INFO_PATH = re.compile(r"^modelcenter/(?P<family>[^/]+)/info\.ya?ml$", re.IGNORECASE)
_DOWNLOAD_PATH = re.compile(
    r"^modelcenter/(?P<family>[^/]+)/download_(?:en|cn)\.md$",
    re.IGNORECASE,
)
_MD_LINK = re.compile(r"\[(?P<label>[^\]\r\n]+)\]\((?P<url>https?://[^)\s]+)\)")
_MARKDOWN = re.compile(r"[`*~]|<[^>]+>")
_TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")
_SAFE_PROJECT = re.compile(r"^[A-Za-z0-9_.-]+$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _ModelInfo:
    family: str
    path: str
    name: str
    description: str | None
    tasks: tuple[str, ...]
    datasets: str | None
    publisher: str | None
    license: str | None
    source_project: str | None
    papers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DownloadRow:
    name: str
    paths: tuple[str, ...]
    locator: str
    links: tuple[tuple[str, str], ...]


class PaddleModelCenterSourceAdapter:
    """Read every exact ModelCenter family and declared downloadable variant.

    ModelCenter is a maintained, first-party subcatalog within PaddlePaddle's
    broader models repository. The adapter resolves a single commit, checks a
    complete recursive tree, then reads every small ``info.yaml`` and optional
    Chinese/English download table at that revision. It keeps family identities
    even if no download table is present, and turns only table rows with an
    explicit resource URL into variant release evidence.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the public ModelCenter family manifests and their source-declared "
        "download-table variants at one PaddlePaddle/models commit. It does not "
        "claim every model elsewhere in the broader repository, execute Paddle "
        "code, infer papers from names, or download model bytes."
    )

    def __init__(
        self,
        *,
        name: str = "paddle-model-center",
        repository: str = "PaddlePaddle/models",
        branch: str = "release/2.4",
        max_tree_bytes: int = 32 * 1024 * 1024,
        max_info_bytes: int = 1 * 1024 * 1024,
        max_download_bytes: int = 4 * 1024 * 1024,
        max_families: int = 1_000,
        max_download_files: int = 2_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.max_tree_bytes = _positive_int(max_tree_bytes, "max_tree_bytes")
        self.max_info_bytes = _positive_int(max_info_bytes, "max_info_bytes")
        self.max_download_bytes = _positive_int(max_download_bytes, "max_download_bytes")
        self.max_families = _positive_int(max_families, "max_families")
        self.max_download_files = _positive_int(max_download_files, "max_download_files")
        self.client = client or HttpClient(
            max_response_bytes=max(
                self.max_tree_bytes,
                self.max_info_bytes,
                self.max_download_bytes,
            )
        )
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paddle-model-center-v1",
                "repository": self.repository,
                "branch": self.branch,
                "max_tree_bytes": self.max_tree_bytes,
                "max_info_bytes": self.max_info_bytes,
                "max_download_bytes": self.max_download_bytes,
                "max_families": self.max_families,
                "max_download_files": self.max_download_files,
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

    def tree_url(self, revision: str) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/git/trees/"
            f"{quote(revision, safe='')}?recursive=1"
        )

    def raw_url(self, revision: str, path: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(path, safe='/')}"
        )

    def blob_url(self, revision: str, path: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(path, safe='/')}"
        )

    def tree_path_url(self, revision: str, path: str) -> str:
        return (
            f"{self.repository_url}/tree/{quote(revision, safe='')}/"
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

        tree_response = self.client.get(
            self.tree_url(revision),
            headers={"Accept": "application/vnd.github+json"},
        )
        if tree_response.status != 200:
            raise ValueError(f"{self.name}: repository tree returned HTTP {tree_response.status}")
        if len(tree_response.body) > self.max_tree_bytes:
            raise ValueError(f"{self.name}: repository tree exceeds {self.max_tree_bytes} bytes")
        info_paths, download_paths = self._catalog_paths(tree_response.json())
        records = []
        info_hashes: dict[str, str] = {}
        download_hashes: dict[str, str] = {}
        for family, info_path in info_paths:
            info_response = self._fetch_file(revision, info_path, self.max_info_bytes)
            info_hashes[info_path] = content_hash(info_response.body)
            info = _parse_info(family, info_path, info_response.text(), self.name)
            rows = []
            for download_path in download_paths.get(family, ()):
                response = self._fetch_file(revision, download_path, self.max_download_bytes)
                download_hashes[download_path] = content_hash(response.body)
                rows.extend(_download_rows(download_path, response.text(), self.name))
            records.extend(
                self._records(
                    info,
                    _merge_download_rows(rows),
                    revision=revision,
                    info_sha256=info_hashes[info_path],
                    download_hashes=download_hashes,
                )
            )
        if not records:
            raise ValueError(f"{self.name}: ModelCenter catalog produced no model records")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "tree_url": self.tree_url(revision),
            "tree_sha256": content_hash(tree_response.body),
            "family_count": len(info_paths),
            "download_file_count": sum(len(paths) for paths in download_paths.values()),
            "model_count": len(records),
            "info_sha256": info_hashes,
            "download_sha256": download_hashes,
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _revision(self) -> tuple[str, HttpResponse]:
        response: HttpResponse = self.client.get(
            self.commit_url,
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _fetch_file(self, revision: str, path: str, maximum: int) -> HttpResponse:
        response: HttpResponse = self.client.get(
            self.raw_url(revision, path),
            headers={"Accept": "text/plain,text/markdown,text/yaml"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog file {path} returned HTTP {response.status}")
        if len(response.body) > maximum:
            raise ValueError(f"{self.name}: catalog file {path} exceeds {maximum} bytes")
        return response

    def _catalog_paths(
        self,
        payload: Any,
    ) -> tuple[tuple[tuple[str, str], ...], Mapping[str, tuple[str, ...]]]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: repository tree is not a JSON object")
        if payload.get("truncated") is True:
            raise ValueError(f"{self.name}: recursive repository tree is truncated")
        tree = payload.get("tree")
        if not isinstance(tree, list):
            raise ValueError(f"{self.name}: repository tree lacks a tree list")
        infos: dict[str, str] = {}
        downloads: dict[str, list[str]] = {}
        for item in tree:
            if not isinstance(item, Mapping) or item.get("type") != "blob":
                continue
            path = _safe_path(item.get("path"), self.name)
            if match := _INFO_PATH.fullmatch(path):
                family = match.group("family")
                if family in infos:
                    raise ValueError(f"{self.name}: duplicate ModelCenter manifest for {family!r}")
                infos[family] = path
            elif match := _DOWNLOAD_PATH.fullmatch(path):
                downloads.setdefault(match.group("family"), []).append(path)
        if not infos:
            raise ValueError(f"{self.name}: repository tree contains no ModelCenter manifests")
        if len(infos) > self.max_families:
            raise ValueError(f"{self.name}: catalog exceeds {self.max_families} model families")
        count = sum(len(paths) for paths in downloads.values())
        if count > self.max_download_files:
            raise ValueError(
                f"{self.name}: catalog exceeds {self.max_download_files} download files"
            )
        return (
            tuple(sorted(infos.items())),
            {family: tuple(sorted(paths)) for family, paths in downloads.items()},
        )

    def _records(
        self,
        info: _ModelInfo,
        rows: tuple[_DownloadRow, ...],
        *,
        revision: str,
        info_sha256: str,
        download_hashes: Mapping[str, str],
    ) -> tuple[SourceRecord, ...]:
        declared = rows or (
            _DownloadRow(
                name=info.name,
                paths=(info.path,),
                locator=f"{info.path}:Model_Info.name",
                links=(),
            ),
        )
        records = []
        family_url = self.tree_path_url(revision, f"modelcenter/{info.family}")
        info_url = self.blob_url(revision, info.path)
        project_url = _project_url(info.source_project)
        for row in declared:
            identity = f"{info.family}/{row.name}"
            model_identifier = Identifier("paddle:model", identity)
            artifact_links = [
                url
                for url, relation in row.links
                if relation in {"weights", "inference_artifact", "model_artifact"}
            ]
            model = ModelHint(
                local_id=f"model:{identity}",
                name=row.name,
                aliases=(info.name,) if row.name != info.name and len(declared) == 1 else (),
                identifiers=(model_identifier,),
                status=ModelStatus.RELEASED if artifact_links else ModelStatus.DOCUMENTED,
                locator=row.locator,
            )
            links = [
                Link(family_url, relation="model_card", locator=row.locator, crawl=False),
                Link(info_url, relation="metadata", locator=f"{info.path}:Model_Info", crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ]
            if project_url:
                links.append(
                    Link(
                        project_url,
                        relation="source_implementation",
                        locator="from_repo",
                        crawl=False,
                    )
                )
            for source_path in row.paths:
                links.append(
                    Link(
                        self.blob_url(revision, source_path),
                        relation="documentation",
                        locator=row.locator,
                        crawl=False,
                    )
                )
            for paper in info.papers:
                links.append(
                    Link(
                        paper,
                        relation="paper",
                        locator=f"{info.path}:Paper",
                        crawl=True,
                    )
                )
            for url, relation in row.links:
                links.append(Link(url, relation=relation, locator=row.locator, crawl=False))
            releases = ()
            if artifact_links:
                releases = (
                    ReleaseHint(
                        local_id=f"release:{identity}",
                        model_local_id=model.local_id,
                        revision=revision,
                        identifiers=(Identifier("paddle:model-release", identity),),
                        metadata={
                            "repository": self.repository,
                            "revision": revision,
                            "family": info.family,
                            "source_project": info.source_project,
                            "artifacts": artifact_links,
                        },
                        locator=row.locator,
                    ),
                )
            records.append(
                SourceRecord(
                    source_record_id=f"model:{identity}",
                    kind=ArtifactKind.MODEL_CARD,
                    canonical_url=canonicalize_url(family_url),
                    title=row.name,
                    raw={
                        "repository": self.repository,
                        "revision": revision,
                        "family": info.family,
                        "info_path": info.path,
                        "info_sha256": info_sha256,
                        "download_paths": sorted(
                            {
                                source_path
                                for download in rows
                                for source_path in download.paths
                            }
                        ),
                        "download_sha256": {
                            path: download_hashes[path]
                            for path in sorted(
                                {
                                    source_path
                                    for download in rows
                                    for source_path in download.paths
                                }
                            )
                            if path in download_hashes
                        },
                        "model_info": {
                            "name": info.name,
                            "description": info.description,
                            "tasks": list(info.tasks),
                            "datasets": info.datasets,
                            "publisher": info.publisher,
                            "license": info.license,
                            "source_project": info.source_project,
                            "papers": list(info.papers),
                        },
                        "download_row": {
                            "name": row.name,
                            "links": [
                                {"url": url, "relation": relation}
                                for url, relation in row.links
                            ],
                        },
                    },
                    text="\n".join(
                        part
                        for part in (
                            row.name,
                            info.description,
                            "tasks: " + ", ".join(info.tasks) if info.tasks else "",
                            "datasets: " + info.datasets if info.datasets else "",
                            "publisher: " + info.publisher if info.publisher else "",
                        )
                        if part
                    ),
                    identifiers=(model_identifier,),
                    links=tuple(dict.fromkeys(links)),
                    models=(model,),
                    releases=releases,
                )
            )
        return tuple(records)


def _parse_info(family: str, path: str, document: str, source: str) -> _ModelInfo:
    fields = _yaml_sections(document)
    name = fields.get("Model_Info", {}).get("name") or family
    if not _text(name):
        raise ValueError(f"{source}: {path} does not declare Model_Info.name")
    info = fields.get("Model_Info", {})
    root = fields.get("", {})
    papers = tuple(
        dict.fromkeys(
            _section_urls(document, "Paper")
        )
    )
    tasks = tuple(
        dict.fromkeys(
            value
            for value in _section_values(
                document,
                "Task",
                {"tag_en", "sub_tag_en", "tag_cn", "sub_tag_cn"},
            )
            if value
        )
    )
    return _ModelInfo(
        family=family,
        path=path,
        name=_text(name),
        description=(
            _text(
                info.get("description_en")
                or info.get("description")
                or info.get("description_cn")
            )
            or None
        ),
        tasks=tasks,
        datasets=_text(root.get("Datasets")) or None,
        publisher=_text(root.get("Publisher")) or None,
        license=_text(root.get("License")) or None,
        source_project=_text(info.get("from_repo")) or None,
        papers=papers,
    )


def _yaml_sections(document: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {"": {}}
    current = ""
    for line in document.splitlines():
        top = re.match(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*):[ \t]*(?P<value>.*)$", line)
        if top:
            key = top.group("key")
            value = _yaml_scalar(top.group("value"))
            if value:
                sections[""][key] = value
                current = ""
            else:
                current = key
                sections.setdefault(current, {})
            continue
        nested = re.match(
            r"^[ \t]+(?:-[ \t]+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*):[ \t]*(?P<value>.*)$",
            line,
        )
        if nested and current:
            value = _yaml_scalar(nested.group("value"))
            if value and nested.group("key") not in sections[current]:
                sections[current][nested.group("key")] = value
    return sections


def _section_urls(document: str, section: str) -> tuple[str, ...]:
    active = False
    urls = []
    for line in document.splitlines():
        heading = re.match(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?:[ \t].*)?$", line)
        if heading:
            active = heading.group("key") == section
            continue
        if active and line and not line[:1].isspace():
            active = False
        if active:
            urls.extend(extract_urls(line))
    return tuple(dict.fromkeys(urls))


def _section_values(document: str, section: str, fields: set[str]) -> tuple[str, ...]:
    active = False
    values = []
    for line in document.splitlines():
        heading = re.match(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?:[ \t].*)?$", line)
        if heading:
            active = heading.group("key") == section
            continue
        if active and line and not line[:1].isspace():
            active = False
        if active:
            match = re.match(
                r"^[ \t]+(?:-[ \t]+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*):[ \t]*(?P<value>.*)$",
                line,
            )
            if match and match.group("key") in fields:
                value = _yaml_scalar(match.group("value"))
                if value:
                    values.append(value)
    return tuple(values)


def _download_rows(path: str, document: str, source: str) -> tuple[_DownloadRow, ...]:
    rows = []
    for line_number, line in enumerate(document.splitlines(), start=1):
        if not line.lstrip().startswith("|"):
            continue
        cells = _table_cells(line)
        if len(cells) < 2 or _table_separator(cells):
            continue
        name = _plain(cells[0])
        if not _model_cell(name):
            continue
        links = []
        for cell in cells[1:]:
            labelled = tuple(_MD_LINK.finditer(cell))
            for match in labelled:
                links.append((match.group("url"), _resource_relation(match.group("label"))))
            for url in extract_urls(_MD_LINK.sub("", cell)):
                links.append((url, "model_artifact"))
        if not links:
            continue
        rows.append(
            _DownloadRow(
                name=name,
                paths=(path,),
                locator=f"{path}:line:{line_number}",
                links=tuple(dict.fromkeys(links)),
            )
        )
    if len(rows) > 100_000:
        raise ValueError(f"{source}: download file {path} contains too many model rows")
    return tuple(rows)


def _merge_download_rows(rows: list[_DownloadRow]) -> tuple[_DownloadRow, ...]:
    merged: dict[str, _DownloadRow] = {}
    for row in rows:
        existing = merged.get(row.name)
        if existing is None:
            merged[row.name] = row
            continue
        merged[row.name] = _DownloadRow(
            name=row.name,
            paths=tuple(dict.fromkeys([*existing.paths, *row.paths])),
            locator=existing.locator,
            links=tuple(dict.fromkeys([*existing.links, *row.links])),
        )
    return tuple(merged.values())


def _table_cells(line: str) -> tuple[str, ...]:
    body = line.strip()
    if not body.startswith("|"):
        return ()
    body = body[1:-1] if body.endswith("|") else body[1:]
    return tuple(cell.strip() for cell in body.split("|"))


def _table_separator(cells: tuple[str, ...]) -> bool:
    return bool(cells) and all(_TABLE_SEPARATOR.fullmatch(cell.replace(" ", "")) for cell in cells)


def _model_cell(value: str) -> bool:
    compact = value.casefold()
    return bool(
        value
        and len(value) <= 200
        and compact not in {"model", "models", "model name", "模型", "模型名称", "name"}
        and not value.startswith("#")
    )


def _plain(value: str) -> str:
    value = _MD_LINK.sub(lambda match: match.group("label"), value)
    return _MARKDOWN.sub("", value).strip()


def _resource_relation(label: str) -> str:
    normalized = _plain(label).casefold()
    if any(token in normalized for token in ("pretrain", "weight", "parameter", "训练")):
        return "weights"
    if any(token in normalized for token in ("infer", "deploy", "推理")):
        return "inference_artifact"
    return "model_artifact"


def _project_url(value: str | None) -> str | None:
    if value and _SAFE_PROJECT.fullmatch(value):
        return f"https://github.com/PaddlePaddle/{quote(value, safe='')}"
    return None


def _safe_path(value: Any, source: str) -> str:
    path = _required_text(value, "repository tree path")
    invalid_part = any(part in {"", ".", ".."} for part in path.split("/"))
    if path.startswith("/") or "\\" in path or invalid_part:
        raise ValueError(f"{source}: repository tree path is not a safe relative path")
    return path


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _yaml_scalar(value: str) -> str:
    candidate = value.strip()
    if len(candidate) >= 2 and candidate[:1] in {"'", '"'} and candidate[-1:] == candidate[:1]:
        return candidate[1:-1]
    return candidate


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
