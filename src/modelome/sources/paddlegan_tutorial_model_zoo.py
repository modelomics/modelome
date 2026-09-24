"""Pinned first-party PaddleGAN tutorial-model catalog ingestion."""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, unquote, urlsplit

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
_HEADING = re.compile(r"^#{1,6}[ \t]+(?P<title>.+?)\s*$")
_TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")
_MD_LINK = re.compile(r"\[(?P<label>[^\]\r\n]+)\]\((?P<url>[^)\s]+)\)")
_URL = re.compile(r"https?://[^\s<>\[\]{}()\"']+", re.IGNORECASE)
_MARKDOWN = re.compile(r"[`*~]|<[^>]+>")
_ARTIFACT_SUFFIX = re.compile(
    r"\.(?:pdparams|pdmodel|pdiparams|tar(?:\.gz)?|tgz|zip|pth|pt)(?:$|[?#])",
    re.IGNORECASE,
)
_MODEL_COLUMN = re.compile(r"\b(?:model|method|architecture|network)\b|模型|方法")
_RESOURCE_COLUMN = re.compile(r"download|weight|artifact|checkpoint|model[ _-]?link|下载")
_PAPER_COLUMN = re.compile(r"paper|citation|reference|论文|引用")
_CONFIG_COLUMN = re.compile(r"config|yaml|yml|配置")
_CODE_COLUMN = re.compile(r"code|source|repo|implementation|实现|源码")
_PAPER_LINE = re.compile(r"\bpaper\b|论文|reference", re.IGNORECASE)
_CODE_LINE = re.compile(r"official\s+repo|source\s+code|official\s+code|代码|源码", re.IGNORECASE)
_RELATED_LINE = re.compile(r"ai\s*studio|aistudio|demo|online", re.IGNORECASE)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Observation:
    document_path: str
    title: str
    aliases: tuple[str, ...]
    locator: str
    resources: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _Artifact:
    url: str
    observations: tuple[_Observation, ...]


class PaddleGanTutorialModelZooSourceAdapter:
    """Read direct first-party model artifacts from PaddleGAN English tutorials.

    A model URL is admitted only if it is a direct file beneath PaddleGAN's
    first-party ``/models/`` host path. Table resources stay on their source
    row. A tutorial-level paper, code, or hosted-demo link is attached only
    when that tutorial declares exactly one model artifact, avoiding a shared
    VSR-family document's references being copied onto every checkpoint.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only direct first-party PaddleGAN model files declared in the "
        "configured English tutorial documents at one public commit. It does not "
        "infer checkpoints from code, ingest datasets, follow links, or transfer "
        "model, paper, or repository bytes."
    )

    def __init__(
        self,
        *,
        name: str = "paddlegan-tutorial-model-zoo",
        repository: str = "PaddlePaddle/PaddleGAN",
        branch: str = "develop",
        document_prefix: str = "docs/en_US/tutorials/",
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
        self.document_prefix = _document_prefix(document_prefix)
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
                "adapter": "paddlegan-tutorial-model-zoo-v1",
                "repository": self.repository,
                "branch": self.branch,
                "document_prefix": self.document_prefix,
                "max_archive_bytes": self.max_archive_bytes,
                "max_document_bytes": self.max_document_bytes,
                "max_total_document_bytes": self.max_total_document_bytes,
                "max_documents": self.max_documents,
                "max_models": self.max_models,
                "admission": "direct first-party PaddleGAN tutorial model URL",
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
            max_document_bytes=self.max_document_bytes,
            max_total_document_bytes=self.max_total_document_bytes,
            max_documents=self.max_documents,
        )
        artifacts = _artifacts(documents)
        if not artifacts:
            raise ValueError(
                f"{self.name}: source archive contains no direct first-party model artifacts"
            )
        if len(artifacts) > self.max_models:
            raise ValueError(f"{self.name}: model catalog exceeds {self.max_models} models")
        records = tuple(
            self._record(artifact, revision=revision, archive=response.body)
            for artifact in artifacts
        )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "archive_url": self.archive_url(revision),
            "archive_sha256": content_hash(response.body),
            "document_count": len(documents),
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
        self, artifact: _Artifact, *, revision: str, archive: bytes
    ) -> SourceRecord:
        identity = artifact.url
        local_id = f"model:{content_hash(identity)[:24]}"
        identifier = Identifier("paddlegan:checkpoint-model", identity)
        primary = artifact.observations[0]
        aliases = tuple(
            sorted(
                {
                    alias
                    for observation in artifact.observations
                    for alias in observation.aliases
                    if alias and alias != primary.title
                }
            )
        )
        links: list[Link] = [
            Link(
                self.repository_url,
                relation="source_repository",
                crawl=False,
                model_local_ids=(local_id,),
            ),
            Link(
                artifact.url,
                relation="weights",
                locator=primary.locator,
                crawl=False,
                model_local_ids=(local_id,),
            ),
        ]
        for observation in artifact.observations:
            links.append(
                Link(
                    self.blob_url(revision, observation.document_path),
                    relation="model_card",
                    locator=observation.locator,
                    crawl=False,
                    model_local_ids=(local_id,),
                )
            )
            links.extend(
                Link(
                    url,
                    relation=relation,
                    locator=observation.locator,
                    crawl=False,
                    model_local_ids=(local_id,),
                )
                for url, relation in observation.resources
            )
        model = ModelHint(
            local_id=local_id,
            name=primary.title,
            aliases=aliases,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=primary.locator,
        )
        release = ReleaseHint(
            local_id=f"release:{content_hash(identity)[:24]}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("paddlegan:checkpoint", identity),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "checkpoint_url": artifact.url,
                "source_rows": [
                    {
                        "document_path": observation.document_path,
                        "title": observation.title,
                        "aliases": list(observation.aliases),
                        "locator": observation.locator,
                        "resources": [
                            {"url": url, "relation": relation}
                            for url, relation in observation.resources
                        ],
                    }
                    for observation in artifact.observations
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
            title=primary.title,
            raw={
                "repository": self.repository,
                "revision": revision,
                "archive_sha256": content_hash(archive),
                "checkpoint_url": artifact.url,
                "source_rows": [
                    {
                        "document_path": observation.document_path,
                        "title": observation.title,
                        "aliases": list(observation.aliases),
                        "locator": observation.locator,
                        "resources": [
                            {"url": url, "relation": relation}
                            for url, relation in observation.resources
                        ],
                    }
                    for observation in artifact.observations
                ],
            },
            text="\n".join(
                (
                    f"PaddleGAN checkpoint: {primary.title}",
                    *(observation.document_path for observation in artifact.observations),
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
            if not path.startswith(document_prefix) or not path.endswith(".md"):
                continue
            if info.file_size > max_document_bytes:
                raise ValueError(
                    f"{source}: tutorial document {path!r} exceeds {max_document_bytes} bytes"
                )
            total += info.file_size
            if total > max_total_document_bytes:
                raise ValueError(
                    f"{source}: tutorial documents exceed {max_total_document_bytes} bytes"
                )
            if path in documents:
                raise ValueError(f"{source}: source archive has duplicate {path!r}")
            documents[path] = package.read(info).decode("utf-8", errors="strict")
        if len(documents) > max_documents:
            raise ValueError(f"{source}: source archive has more than {max_documents} documents")
    if not documents:
        raise ValueError(f"{source}: source archive has no matching tutorial documents")
    return dict(sorted(documents.items()))


def _archive_path(value: str, source: str) -> str:
    parts = value.split("/")
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{source}: unsafe source archive member {value!r}")
    return "/".join(parts[1:])


def _artifacts(documents: Mapping[str, str]) -> tuple[_Artifact, ...]:
    grouped: dict[str, list[_Observation]] = {}
    for path, document in documents.items():
        for url, observation in _document_artifacts(path, document):
            grouped.setdefault(url, []).append(observation)
    return tuple(
        _Artifact(url=url, observations=tuple(grouped[url])) for url in sorted(grouped)
    )


def _document_artifacts(path: str, document: str) -> tuple[tuple[str, _Observation], ...]:
    title = _document_title(document)
    if not title:
        return ()
    rows = _table_artifacts(path, document, title)
    document_urls = tuple(
        dict.fromkeys(
            canonicalize_url(url)
            for _, url in _links(document)
            if _is_direct_model_artifact(url)
        )
    )
    known = {url for url, _ in rows}
    result = list(rows)
    for url in document_urls:
        if url not in known:
            result.append(
                (
                    url,
                    _Observation(
                        document_path=path,
                        title=title,
                        aliases=(_artifact_filename(url),),
                        locator=f"{path}:line:{_first_url_line(document, url)}",
                        resources=(),
                    ),
                )
            )
    if len(document_urls) == 1:
        resources = _document_resources(document)
        result = [
            (
                url,
                _Observation(
                    document_path=observation.document_path,
                    title=observation.title,
                    aliases=observation.aliases,
                    locator=observation.locator,
                    resources=tuple(dict.fromkeys((*observation.resources, *resources))),
                ),
            )
            for url, observation in result
        ]
    return tuple(result)


def _table_artifacts(
    path: str, document: str, document_title: str
) -> tuple[tuple[str, _Observation], ...]:
    result: list[tuple[str, _Observation]] = []
    lines = document.splitlines()
    index = 0
    while index + 1 < len(lines):
        header = _table_cells(lines[index])
        separator = _table_cells(lines[index + 1])
        if not header or not _table_separator(separator):
            index += 1
            continue
        index += 2
        while index < len(lines):
            cells = _table_cells(lines[index])
            if not cells:
                break
            if len(cells) != len(header):
                index += 1
                continue
            for url in _direct_artifact_urls(cells):
                row_title = _table_title(header, cells, document_title)
                resources = _row_resources(header, cells, artifact_url=url)
                result.append(
                    (
                        url,
                        _Observation(
                            document_path=path,
                            title=row_title,
                            aliases=tuple(
                                dict.fromkeys(
                                    item
                                    for item in (_plain(cells[0]), _artifact_filename(url))
                                    if item and item != row_title
                                )
                            ),
                            locator=f"{path}:line:{index + 1}",
                            resources=resources,
                        ),
                    )
                )
            index += 1
    return tuple(result)


def _table_cells(line: str) -> tuple[str, ...] | None:
    body = line.strip()
    if "|" not in body:
        return None
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    cells = tuple(cell.strip() for cell in re.split(r"(?<!\\)\|", body))
    return cells if len(cells) >= 2 else None


def _table_separator(cells: tuple[str, ...] | None) -> bool:
    return bool(cells) and all(_TABLE_SEPARATOR.fullmatch(cell.replace(" ", "")) for cell in cells)


def _direct_artifact_urls(cells: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            canonicalize_url(url)
            for cell in cells
            for _, url in _links(cell)
            if _is_direct_model_artifact(url)
        )
    )


def _table_title(
    header: tuple[str, ...], cells: tuple[str, ...], fallback: str
) -> str:
    for index, value in enumerate(header):
        plain = _plain(value).casefold()
        if _MODEL_COLUMN.search(plain) and not _RESOURCE_COLUMN.search(plain):
            title = _plain(cells[index])
            if title:
                return title
    return fallback


def _row_resources(
    header: tuple[str, ...], cells: tuple[str, ...], *, artifact_url: str
) -> tuple[tuple[str, str], ...]:
    resources: list[tuple[str, str]] = []
    for index, cell in enumerate(cells):
        for _, url in _links(cell):
            if _is_direct_model_artifact(url):
                continue
            relation = _column_relation(header[index], url)
            if relation:
                resources.append((url, relation))
    return tuple(dict.fromkeys(resources))


def _column_relation(header: str, url: str) -> str | None:
    label = _plain(header)
    if _PAPER_COLUMN.search(label) or _is_paper_url(url):
        return "paper_reference"
    if _CONFIG_COLUMN.search(label) or urlsplit(url).path.casefold().endswith((".yaml", ".yml")):
        return "model_config"
    if _CODE_COLUMN.search(label):
        return "source_implementation"
    if _is_web_url(url):
        return "related_resource"
    return None


def _document_resources(document: str) -> tuple[tuple[str, str], ...]:
    resources: list[tuple[str, str]] = []
    for line in document.splitlines():
        if _PAPER_LINE.search(line):
            relation = "paper_reference"
        elif _CODE_LINE.search(line):
            relation = "source_implementation"
        elif _RELATED_LINE.search(line):
            relation = "related_resource"
        else:
            continue
        resources.extend(
            (url, relation)
            for _, url in _links(line)
            if _is_web_url(url) and not _is_direct_model_artifact(url)
        )
    return tuple(dict.fromkeys(resources))


def _document_title(document: str) -> str:
    for line in document.splitlines():
        if match := _HEADING.match(line):
            title = _plain(match.group("title"))
            if title:
                return title
    return ""


def _links(value: str) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for match in _MD_LINK.finditer(value):
        label = _text(match.group("label"))
        url = _clean_url(match.group("url"))
        if url and _is_web_url(url):
            found.append((label, canonicalize_url(url)))
    for match in _URL.finditer(value):
        url = _clean_url(match.group(0))
        if url and _is_web_url(url):
            found.append(("", canonicalize_url(url)))
    return tuple(dict.fromkeys(found))


def _clean_url(value: str) -> str:
    return _text(value).strip("<>,.;:!?")


def _is_direct_model_artifact(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme in {"http", "https"}
        and (parsed.hostname or "").casefold() == "paddlegan.bj.bcebos.com"
        and parsed.path.startswith("/models/")
        and bool(_ARTIFACT_SUFFIX.search(parsed.path))
    )


def _is_paper_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").casefold()
    return host in {"arxiv.org", "export.arxiv.org", "doi.org", "paperswithcode.com"}


def _is_web_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _artifact_filename(url: str) -> str:
    filename = unquote(urlsplit(url).path).rsplit("/", 1)[-1]
    suffixes = (
        ".tar.gz",
        ".pdparams",
        ".pdmodel",
        ".pdiparams",
        ".tar",
        ".tgz",
        ".zip",
        ".pth",
        ".pt",
    )
    for suffix in suffixes:
        if filename.casefold().endswith(suffix):
            return filename[: -len(suffix)].strip()
    return filename.strip()


def _first_url_line(document: str, url: str) -> int:
    for index, line in enumerate(document.splitlines(), start=1):
        if url in {candidate for _, candidate in _links(line)}:
            return index
    return 1


def _plain(value: str) -> str:
    labels = _MD_LINK.sub(lambda match: match.group("label"), value)
    return " ".join(_MARKDOWN.sub("", labels).split()).strip()


def _document_prefix(value: str) -> str:
    prefix = _required_text(value, "document prefix").strip("/")
    if not prefix or "\\" in prefix or any(part in {".", ".."} for part in prefix.split("/")):
        raise ValueError("document prefix is not a safe relative path")
    return prefix + "/"


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


__all__ = ["PaddleGanTutorialModelZooSourceAdapter"]
