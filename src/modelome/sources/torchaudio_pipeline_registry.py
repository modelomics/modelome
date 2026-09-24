"""Pinned static ingestion of Torchaudio's public pretrained pipeline declarations."""

from __future__ import annotations

import ast
import io
import re
import zipfile
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
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, extract_urls

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_DEFAULT_PIPELINE_ROOT = "src/torchaudio/pipelines"
_MAX_MODULES = 128
_MAX_PIPELINES = 10_000
_WEIGHT_FIELDS = frozenset(
    {
        "_path",
        "_model_path",
        "_rnnt_path",
        "_tacotron2_path",
        "_wavernn_path",
        "_global_stats_path",
        "_sp_model_path",
    }
)
_MODEL_BASE_MODULES = frozenset(
    {
        "_squim_pipeline.py",
        "_tts/impl.py",
        "_wav2vec2/impl.py",
    }
)
_DOWNLOAD_HELPER = "src/torchaudio/utils/download.py"
_WAV2VEC_HELPER = "src/torchaudio/pipelines/_wav2vec2/utils.py"
_TTS_MODULE = "src/torchaudio/pipelines/_tts/impl.py"


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Pipeline:
    name: str
    path: str
    source_sha256: str
    constructor: str
    weights: tuple[tuple[str, str], ...]
    docstring: str
    locator: str


@dataclass(frozen=True, slots=True)
class _DownloadBases:
    asset: str
    model: str


class TorchaudioPipelineRegistrySourceAdapter:
    """Enumerate public Torchaudio pipeline bundles at one immutable Git commit.

    The public pipeline initializer determines the eligible all-caps symbols. The
    adapter AST-parses their source declarations, preserving literal checkpoint
    paths, accompanying docstring URLs, and the pinned definition that declares
    them. It does not import Torchaudio, instantiate a bundle, or fetch a model.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public all-caps pretrained pipeline bundles statically declared in "
        "Torchaudio's public source tree at one Git commit. It preserves literal "
        "checkpoint paths and direct docstring URLs but does not execute package "
        "code, infer citation keys into paper URLs, enumerate deprecated releases, "
        "or download any artifact bytes."
    )

    def __init__(
        self,
        *,
        name: str = "torchaudio-pipeline-registry",
        repository: str = "pytorch/audio",
        branch: str = "main",
        pipeline_root: str = _DEFAULT_PIPELINE_ROOT,
        max_archive_bytes: int = 128 * 1024 * 1024,
        max_module_bytes: int = 2 * 1024 * 1024,
        max_modules: int = _MAX_MODULES,
        max_pipelines: int = _MAX_PIPELINES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.pipeline_root = _safe_path(pipeline_root, "pipeline_root")
        self.max_archive_bytes = _positive_int(max_archive_bytes, "max_archive_bytes")
        self.max_module_bytes = _positive_int(max_module_bytes, "max_module_bytes")
        self.max_modules = _positive_int(max_modules, "max_modules")
        self.max_pipelines = _positive_int(max_pipelines, "max_pipelines")
        self.client = client or HttpClient(max_response_bytes=self.max_archive_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "torchaudio-pipeline-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "pipeline_root": self.pipeline_root,
                "max_archive_bytes": self.max_archive_bytes,
                "max_module_bytes": self.max_module_bytes,
                "max_modules": self.max_modules,
                "max_pipelines": self.max_pipelines,
                "admission": "public initializer exports with literal bundle paths",
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
                upstream_count=_nonnegative_int(state.get("pipeline_count")),
            )

        archive_response = self.client.get(
            self.archive_url(revision), headers={"Accept": "application/zip"}
        )
        if archive_response.status != 200:
            raise ValueError(
                f"{self.name}: source archive returned HTTP {archive_response.status}"
            )
        if len(archive_response.body) > self.max_archive_bytes:
            raise ValueError(
                f"{self.name}: source archive exceeds {self.max_archive_bytes} bytes"
            )
        download_bases = _download_bases(
            archive_response.body,
            source=self.name,
            max_module_bytes=self.max_module_bytes,
        )
        pipelines, module_count = self._pipelines(archive_response.body)
        if len(pipelines) > self.max_pipelines:
            raise ValueError(
                f"{self.name}: source declares more than {self.max_pipelines} pipelines"
            )
        if not pipelines:
            raise ValueError(f"{self.name}: source archive contains no public pipelines")
        records = tuple(
            self._record(
                pipeline,
                revision,
                archive_response.body,
                download_bases,
            )
            for pipeline in pipelines
        )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "archive_url": self.archive_url(revision),
            "archive_sha256": content_hash(archive_response.body),
            "module_count": module_count,
            "pipeline_count": len(records),
            "weight_reference_count": sum(len(pipeline.weights) for pipeline in pipelines),
            "asset_download_base": download_bases.asset,
            "model_download_base": download_bases.model,
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
            raise ValueError(
                f"{self.name}: commit endpoint returned HTTP {response.status}"
            )
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _pipelines(self, archive: bytes) -> tuple[tuple[_Pipeline, ...], int]:
        modules = _pipeline_modules(
            archive,
            source=self.name,
            pipeline_root=self.pipeline_root,
            max_module_bytes=self.max_module_bytes,
        )
        if len(modules) > self.max_modules:
            raise ValueError(
                f"{self.name}: source has more than {self.max_modules} pipeline modules"
            )
        initializer = modules.get("__init__.py")
        if initializer is None:
            raise ValueError(f"{self.name}: source archive lacks pipeline initializer")
        public_names = _public_names(initializer, self.name, self.pipeline_root)
        result: list[_Pipeline] = []
        seen: set[str] = set()
        for relative_path, source in sorted(modules.items()):
            if relative_path == "__init__.py":
                continue
            for pipeline in _parse_module(
                source,
                path=f"{self.pipeline_root}/{relative_path}",
                relative_path=relative_path,
                public_names=public_names,
                name=self.name,
            ):
                if pipeline.name in seen:
                    raise ValueError(f"{self.name}: duplicate public pipeline {pipeline.name!r}")
                seen.add(pipeline.name)
                result.append(pipeline)
        missing = sorted(set(public_names) - seen)
        if missing:
            raise ValueError(
                f"{self.name}: public initializer exports without static declarations: "
                + ", ".join(missing[:20])
            )
        return tuple(sorted(result, key=lambda pipeline: pipeline.name)), len(modules)

    def _record(
        self,
        pipeline: _Pipeline,
        revision: str,
        archive: bytes,
        download_bases: _DownloadBases,
    ) -> SourceRecord:
        model_identifier = Identifier("torchaudio:pipeline", pipeline.name)
        model = ModelHint(
            local_id=f"model:{pipeline.name}",
            name=pipeline.name,
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=pipeline.locator,
        )
        definition_url = self.blob_url(revision, pipeline.path)
        links = [
            Link(self.repository_url, relation="source_repository", crawl=False),
            Link(
                definition_url,
                relation="pipeline_definition",
                locator=pipeline.locator,
                crawl=False,
                model_local_ids=(model.local_id,),
            ),
        ]
        releases = []
        for field, asset_path in pipeline.weights:
            weight_url = _weight_url(
                asset_path,
                pipeline.path,
                self.pipeline_root,
                download_bases,
                self.name,
                pipeline.locator,
            )
            links.append(
                Link(
                    weight_url,
                    relation="weights",
                    locator=f"{pipeline.locator}:{field}",
                    crawl=False,
                    model_local_ids=(model.local_id,),
                )
            )
            release_key = f"{pipeline.name}:{field}:{asset_path}"
            releases.append(
                ReleaseHint(
                    local_id=f"release:{content_hash(release_key)[:24]}",
                    model_local_id=model.local_id,
                    revision=revision,
                    identifiers=(
                        Identifier("torchaudio:pipeline-asset", release_key),
                    ),
                    metadata={
                        "repository": self.repository,
                        "revision": revision,
                        "pipeline": pipeline.name,
                        "constructor": pipeline.constructor,
                        "path_field": field,
                        "declared_path": asset_path,
                        "url": weight_url,
                    },
                    locator=f"{pipeline.locator}:{field}",
                )
            )
        for index, url in enumerate(extract_urls(pipeline.docstring)):
            links.append(
                Link(
                    url,
                    relation=_doc_relation(url),
                    locator=f"{pipeline.locator}:__doc__[{index}]",
                    crawl=False,
                    model_local_ids=(model.local_id,),
                )
            )
        text = "\n".join(part for part in (pipeline.name, pipeline.docstring) if part)
        return SourceRecord(
            source_record_id=f"pipeline:{pipeline.name}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(definition_url),
            title=pipeline.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "archive_sha256": content_hash(archive),
                "pipeline_path": pipeline.path,
                "pipeline_source_sha256": pipeline.source_sha256,
                "pipeline": pipeline.name,
                "constructor": pipeline.constructor,
                "asset_download_base": download_bases.asset,
                "model_download_base": download_bases.model,
                "weights": [
                    {"field": field, "declared_path": asset_path}
                    for field, asset_path in pipeline.weights
                ],
                "docstring": pipeline.docstring,
            },
            text=text,
            identifiers=(model_identifier,),
            links=tuple(dict.fromkeys(links)),
            models=(model,),
            releases=tuple(releases),
        )


def _pipeline_modules(
    archive: bytes,
    *,
    source: str,
    pipeline_root: str,
    max_module_bytes: int,
) -> dict[str, str]:
    try:
        package = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as error:
        raise ValueError(f"{source}: source archive is not a ZIP file") from error
    marker = f"/{pipeline_root}/"
    with package:
        modules: dict[str, str] = {}
        roots: set[str] = set()
        for info in package.infolist():
            if info.is_dir() or not info.filename.endswith(".py"):
                continue
            member = _archive_path(info.filename, source)
            if marker not in member:
                continue
            root, relative_path = member.split(marker, 1)
            if not root or not relative_path:
                raise ValueError(f"{source}: source archive has an invalid pipeline member")
            if info.file_size > max_module_bytes:
                raise ValueError(
                    f"{source}: pipeline module {member!r} exceeds {max_module_bytes} bytes"
                )
            if relative_path in modules:
                raise ValueError(f"{source}: source archive has duplicate module {relative_path!r}")
            roots.add(root)
            modules[relative_path] = package.read(info).decode("utf-8", errors="strict")
    if len(roots) != 1:
        raise ValueError(f"{source}: source archive must contain one pipeline root")
    if not modules:
        raise ValueError(f"{source}: source archive has no Python pipeline modules")
    return modules


def _download_bases(
    archive: bytes,
    *,
    source: str,
    max_module_bytes: int,
) -> _DownloadBases:
    """Trace the package's literal download-base helpers at this same commit."""

    sources = _archive_sources(
        archive,
        source=source,
        paths=(_DOWNLOAD_HELPER, _WAV2VEC_HELPER, _TTS_MODULE),
        max_module_bytes=max_module_bytes,
    )
    asset_base = _joined_url_prefix(
        sources[_DOWNLOAD_HELPER],
        function="_download",
        variable="url",
        source=source,
        path=_DOWNLOAD_HELPER,
    )
    model_base = _joined_url_prefix(
        sources[_WAV2VEC_HELPER],
        function="_get_state_dict",
        variable="url",
        source=source,
        path=_WAV2VEC_HELPER,
    )
    tts_base = _string_constant(
        sources[_TTS_MODULE],
        name="_BASE_URL",
        source=source,
        path=_TTS_MODULE,
    )
    expected_model_base = f"{asset_base.rstrip('/')}/models"
    if model_base.rstrip("/") != expected_model_base or tts_base.rstrip("/") != expected_model_base:
        raise ValueError(
            f"{source}: pipeline checkpoint download helpers disagree on their model base"
        )
    return _DownloadBases(asset=asset_base.rstrip("/"), model=expected_model_base)


def _archive_sources(
    archive: bytes,
    *,
    source: str,
    paths: tuple[str, ...],
    max_module_bytes: int,
) -> dict[str, str]:
    try:
        package = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as error:
        raise ValueError(f"{source}: source archive is not a ZIP file") from error
    wanted = set(paths)
    result: dict[str, str] = {}
    roots: set[str] = set()
    with package:
        for info in package.infolist():
            if info.is_dir():
                continue
            member = _archive_path(info.filename, source)
            for path in wanted:
                suffix = f"/{path}"
                if not member.endswith(suffix):
                    continue
                root = member.removesuffix(suffix)
                if not root:
                    raise ValueError(f"{source}: source archive has an invalid helper member")
                if info.file_size > max_module_bytes:
                    raise ValueError(
                        f"{source}: helper source {path!r} exceeds {max_module_bytes} bytes"
                    )
                if path in result:
                    raise ValueError(f"{source}: source archive has duplicate helper {path!r}")
                roots.add(root)
                result[path] = package.read(info).decode("utf-8", errors="strict")
    if len(roots) != 1:
        raise ValueError(f"{source}: source archive must contain one helper root")
    missing = sorted(wanted - set(result))
    if missing:
        raise ValueError(
            f"{source}: source archive is missing checkpoint helper(s): " + ", ".join(missing)
        )
    return result


def _joined_url_prefix(
    document: str,
    *,
    function: str,
    variable: str,
    source: str,
    path: str,
) -> str:
    try:
        tree = ast.parse(document, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{source}: cannot parse {path}: {error.msg}") from error
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != function:
            continue
        for statement in ast.walk(node):
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if not isinstance(target, ast.Name) or target.id != variable:
                continue
            prefix = _joined_prefix(statement.value)
            if prefix is not None:
                return prefix
    raise ValueError(f"{source}: {path} has no literal {function} download URL prefix")


def _string_constant(document: str, *, name: str, source: str, path: str) -> str:
    try:
        tree = ast.parse(document, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{source}: cannot parse {path}: {error.msg}") from error
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            value = _literal_string(statement.value)
            if value.startswith(("https://", "http://")):
                return canonicalize_url(value).rstrip("/")
    raise ValueError(f"{source}: {path} has no literal {name} download URL")


def _joined_prefix(node: ast.AST) -> str | None:
    if not isinstance(node, ast.JoinedStr):
        return None
    parts = []
    has_dynamic_part = False
    for value in node.values:
        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and not has_dynamic_part
        ):
            parts.append(value.value)
            continue
        if isinstance(value, ast.FormattedValue):
            has_dynamic_part = True
            continue
        return None
    prefix = "".join(parts)
    if has_dynamic_part and prefix.startswith(("https://", "http://")):
        return canonicalize_url(prefix).rstrip("/")
    return None


def _public_names(source: str, name: str, path: str) -> dict[str, str]:
    try:
        tree = ast.parse(source, filename=f"{path}/__init__.py")
    except SyntaxError as error:
        raise ValueError(f"{name}: cannot parse pipeline initializer: {error.msg}") from error
    result: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom):
            continue
        for imported in statement.names:
            if not imported.name.isupper():
                continue
            exported = imported.asname or imported.name
            if not exported.isupper():
                continue
            existing = result.get(imported.name)
            if existing is not None and existing != exported:
                raise ValueError(
                    f"{name}: public pipeline {imported.name!r} has conflicting aliases"
                )
            result[imported.name] = exported
    if not result:
        raise ValueError(f"{name}: pipeline initializer has no all-caps public imports")
    return result


def _parse_module(
    source: str,
    *,
    path: str,
    relative_path: str,
    public_names: Mapping[str, str],
    name: str,
) -> tuple[_Pipeline, ...]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{name}: cannot parse {path}: {error.msg}") from error
    docs = _docstrings(tree)
    result = []
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id not in public_names:
            continue
        if not isinstance(statement.value, ast.Call):
            raise ValueError(f"{name}: {path}:{target.id} is not a literal bundle call")
        constructor = _call_name(statement.value.func)
        if not constructor or not constructor.rsplit(".", maxsplit=1)[-1].endswith("Bundle"):
            raise ValueError(f"{name}: {path}:{target.id} has an unsupported constructor")
        weights = _bundle_paths(statement.value, name, path, target.id)
        if not weights:
            raise ValueError(f"{name}: {path}:{target.id} has no literal checkpoint path")
        result.append(
            _Pipeline(
                name=public_names[target.id],
                path=path,
                source_sha256=content_hash(source.encode()),
                constructor=constructor,
                weights=weights,
                docstring=docs.get(target.id, ""),
                locator=f"{path}:{target.id}",
            )
        )
    return tuple(result)


def _docstrings(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.attr == "__doc__"
            and target.value.id.isupper()
        ):
            value = _literal_string(statement.value)
            if value:
                result[target.value.id] = value
    return result


def _bundle_paths(
    call: ast.Call,
    source: str,
    path: str,
    symbol: str,
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for keyword in call.keywords:
        if keyword.arg in _WEIGHT_FIELDS:
            value = _literal_string(keyword.value)
            if not value:
                raise ValueError(
                    f"{source}: {path}:{symbol}:{keyword.arg} is not a literal path"
                )
            result.append((keyword.arg, value))
    if result:
        return tuple(result)
    if call.args:
        value = _literal_string(call.args[0])
        if value:
            return (("arg:0", value),)
    return ()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _weight_url(
    path: str,
    module_path: str,
    pipeline_root: str,
    download_bases: _DownloadBases,
    source: str,
    locator: str,
) -> str:
    value = path.strip()
    if value.startswith("https://") or value.startswith("http://"):
        return canonicalize_url(value)
    if not _safe_asset_path(value):
        raise ValueError(f"{source}: invalid checkpoint path at {locator}")
    relative_module = module_path.removeprefix(f"{pipeline_root}/")
    if value.startswith(("models/", "pipeline-assets/")):
        return canonicalize_url(f"{download_bases.asset}/{quote(value, safe='/')}")
    if relative_module in _MODEL_BASE_MODULES:
        return canonicalize_url(f"{download_bases.model}/{quote(value, safe='/')}")
    raise ValueError(
        f"{source}: cannot resolve relative checkpoint path at {locator} without "
        "a source-declared download base"
    )


def _doc_relation(url: str) -> str:
    normalized = canonicalize_url(url)
    parsed = urlsplit(normalized)
    host = parsed.netloc.casefold()
    path = parsed.path.casefold()
    if host in {"arxiv.org", "www.arxiv.org", "doi.org", "dx.doi.org"}:
        return "paper"
    if host == "github.com" and "/pytorch/audio/" in path:
        return "official_implementation"
    if "license" in path or "license" in host:
        return "license"
    if host == "huggingface.co":
        return "model_card"
    if host == "download.pytorch.org" and "/doc-assets/" in path:
        return "demo"
    return "related_resource"


def _safe_asset_path(value: str) -> bool:
    return bool(
        value
        and not value.startswith(("/", "\\"))
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )


def _literal_string(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    return ""


def _archive_path(value: str, source: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"{source}: source archive contains an unsafe path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{source}: source archive contains an unsafe path")
    return value


def _safe_path(value: str, field: str) -> str:
    path = _required_text(value, field)
    if path.startswith("/") or "\\" in path or any(
        part in {"", ".", ".."} for part in path.split("/")
    ):
        raise ValueError(f"{field} must be a safe relative path")
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
