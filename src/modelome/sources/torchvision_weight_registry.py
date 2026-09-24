"""Pinned static ingestion of TorchVision's source-declared weight enums.

TorchVision declares its official pretrained releases as ``WeightsEnum``
subclasses in the public repository.  This adapter resolves one Git commit,
reads a bounded source archive, and parses those declarations with ``ast``.
It never imports TorchVision or retrieves a checkpoint.
"""

from __future__ import annotations

import ast
import io
import re
import zipfile
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
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_DEFAULT_PACKAGE_PATH = "torchvision/models"
_MAX_MODULES = 1_000


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _WeightMember:
    name: str
    url: str
    locator: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _WeightEnum:
    name: str
    path: str
    source_sha256: str
    members: tuple[_WeightMember, ...]


class TorchvisionWeightRegistrySourceAdapter:
    """Enumerate official TorchVision ``WeightsEnum`` releases at one Git commit.

    A model identity is the exact source enum class (for example,
    ``ResNet50_Weights``).  Its literal ``Weights(url=...)`` members become
    versioned releases.  That keeps documented model families distinct from
    their individually named upstream weight releases without deriving either
    one from a filename or an external naming convention.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers literal WeightsEnum releases in TorchVision's public model-source "
        "tree at one observed Git commit. It does not execute package code, infer "
        "a paper or original architecture, enumerate third-party weights, or "
        "download checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "torchvision-weight-registry",
        repository: str = "pytorch/vision",
        branch: str = "main",
        package_path: str = _DEFAULT_PACKAGE_PATH,
        max_archive_bytes: int = 64 * 1024 * 1024,
        max_module_bytes: int = 2 * 1024 * 1024,
        max_modules: int = _MAX_MODULES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.package_path = _safe_package_path(package_path)
        self.max_archive_bytes = _positive_int(max_archive_bytes, "max_archive_bytes")
        self.max_module_bytes = _positive_int(max_module_bytes, "max_module_bytes")
        self.max_modules = _positive_int(max_modules, "max_modules")
        self.client = client or HttpClient(max_response_bytes=self.max_archive_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "torchvision-weight-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "package_path": self.package_path,
                "max_archive_bytes": self.max_archive_bytes,
                "max_module_bytes": self.max_module_bytes,
                "max_modules": self.max_modules,
                "admission": "literal WeightsEnum members with literal web URLs",
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

        archive_response = self.client.get(
            self.archive_url(revision),
            headers={"Accept": "application/zip"},
        )
        if archive_response.status != 200:
            raise ValueError(
                f"{self.name}: source archive returned HTTP {archive_response.status}"
            )
        if len(archive_response.body) > self.max_archive_bytes:
            raise ValueError(
                f"{self.name}: source archive exceeds {self.max_archive_bytes} bytes"
            )
        enums, module_count = self._weight_enums(archive_response.body)
        records = tuple(
            self._record(weight_enum, revision, archive_response.body)
            for weight_enum in enums
        )
        if not records:
            raise ValueError(f"{self.name}: source archive contains no weight enums")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "archive_url": self.archive_url(revision),
            "archive_sha256": content_hash(archive_response.body),
            "module_count": module_count,
            "model_count": len(records),
            "release_count": sum(len(weight_enum.members) for weight_enum in enums),
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
            self.commit_url,
            headers={"Accept": "application/vnd.github+json"},
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

    def _weight_enums(self, archive: bytes) -> tuple[tuple[_WeightEnum, ...], int]:
        modules = _archive_modules(
            archive,
            source=self.name,
            package_path=self.package_path,
            max_module_bytes=self.max_module_bytes,
        )
        if len(modules) > self.max_modules:
            raise ValueError(
                f"{self.name}: source archive has more than {self.max_modules} model modules"
            )
        result: list[_WeightEnum] = []
        seen: set[str] = set()
        for path, source in sorted(modules.items()):
            for weight_enum in _parse_weight_enums(source, path, self.name):
                if weight_enum.name in seen:
                    raise ValueError(
                        f"{self.name}: duplicate weight enum {weight_enum.name!r}"
                    )
                seen.add(weight_enum.name)
                result.append(weight_enum)
        return tuple(result), len(modules)

    def _record(
        self,
        weight_enum: _WeightEnum,
        revision: str,
        archive: bytes,
    ) -> SourceRecord:
        model_identifier = Identifier("torchvision:weight-enum", weight_enum.name)
        model_name = weight_enum.name.removesuffix("_Weights")
        model = ModelHint(
            local_id=f"model:{weight_enum.name}",
            name=model_name,
            aliases=(weight_enum.name,),
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=f"{weight_enum.path}:{weight_enum.name}",
        )
        code_url = self.blob_url(revision, weight_enum.path)
        links = [
            Link(self.repository_url, relation="source_repository", crawl=False),
            Link(
                code_url,
                relation="model_definition",
                locator=model.locator,
                crawl=False,
                model_local_ids=(model.local_id,),
            ),
        ]
        releases = []
        for member in weight_enum.members:
            links.append(
                Link(
                    member.url,
                    relation="weights",
                    locator=member.locator,
                    crawl=False,
                    model_local_ids=(model.local_id,),
                )
            )
            release_key = f"{weight_enum.name}.{member.name}"
            releases.append(
                ReleaseHint(
                    local_id=f"release:{release_key}",
                    model_local_id=model.local_id,
                    version=member.name,
                    revision=revision,
                    identifiers=(
                        Identifier("torchvision:weight-enum-member", release_key),
                    ),
                    metadata={
                        "repository": self.repository,
                        "revision": revision,
                        "module_path": weight_enum.path,
                        "weight_enum": weight_enum.name,
                        "member": member.name,
                        "aliases": member.aliases,
                        "weight_url": member.url,
                    },
                    locator=member.locator,
                )
            )
        return SourceRecord(
            source_record_id=f"weight-enum:{weight_enum.name}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(code_url),
            title=model_name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "archive_sha256": content_hash(archive),
                "module_path": weight_enum.path,
                "module_sha256": weight_enum.source_sha256,
                "weight_enum": weight_enum.name,
                "releases": [
                    {
                        "name": member.name,
                        "aliases": member.aliases,
                        "url": member.url,
                        "locator": member.locator,
                    }
                    for member in weight_enum.members
                ],
            },
            text=(
                f"TorchVision weight enum: {weight_enum.name}\n"
                f"Declared releases: {', '.join(member.name for member in weight_enum.members)}"
            ),
            identifiers=(model_identifier,),
            links=tuple(dict.fromkeys(links)),
            models=(model,),
            releases=tuple(releases),
        )


def _archive_modules(
    archive: bytes,
    *,
    source: str,
    package_path: str,
    max_module_bytes: int,
) -> dict[str, str]:
    try:
        package = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as error:
        raise ValueError(f"{source}: source archive is not a ZIP file") from error
    marker = f"/{package_path}/"
    with package:
        members: dict[str, str] = {}
        roots: set[str] = set()
        for info in package.infolist():
            if info.is_dir() or not info.filename.endswith(".py"):
                continue
            path = _archive_path(info.filename, source)
            if marker not in path:
                continue
            root, relative_path = path.split(marker, 1)
            if not root or not relative_path:
                raise ValueError(f"{source}: source archive has an invalid package member")
            if info.file_size > max_module_bytes:
                raise ValueError(
                    f"{source}: source file {path!r} exceeds {max_module_bytes} bytes"
                )
            model_path = f"{package_path}/{relative_path}"
            if model_path in members:
                raise ValueError(f"{source}: source archive has duplicate member {model_path!r}")
            roots.add(root)
            members[model_path] = package.read(info).decode("utf-8", errors="strict")
    if len(roots) != 1:
        raise ValueError(f"{source}: source archive must contain one package root")
    if not members:
        raise ValueError(f"{source}: source archive has no {package_path} Python files")
    return members


def _parse_weight_enums(source: str, path: str, name: str) -> tuple[_WeightEnum, ...]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{name}: cannot parse {path}: {error.msg}") from error
    enums = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not _is_weights_enum(node):
            continue
        members = _weight_members(node, path, name)
        if members:
            enums.append(
                _WeightEnum(
                    name=node.name,
                    path=path,
                    source_sha256=content_hash(source.encode()),
                    members=members,
                )
            )
    return tuple(enums)


def _is_weights_enum(node: ast.ClassDef) -> bool:
    return any(_node_name(base) == "WeightsEnum" for base in node.bases)


def _weight_members(
    weight_enum: ast.ClassDef,
    path: str,
    source: str,
) -> tuple[_WeightMember, ...]:
    members = []
    seen: set[str] = set()
    aliases: dict[str, str] = {}
    for statement in weight_enum.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(statement.value, ast.Name):
            aliases[target.id] = statement.value.id
            continue
        if not _is_weights_call(statement.value):
            continue
        if target.id in seen:
            raise ValueError(
                f"{source}: duplicate {weight_enum.name} member {target.id!r}"
            )
        url = _weight_url(statement.value)
        if url is None:
            raise ValueError(
                f"{source}: {weight_enum.name}.{target.id} has no literal web weight URL"
            )
        seen.add(target.id)
        members.append(
            _WeightMember(
                name=target.id,
                url=url,
                locator=f"{path}:{weight_enum.name}.{target.id}",
            )
        )
    members_by_name = {member.name: member for member in members}
    aliases_by_member: dict[str, list[str]] = {}
    for alias, target_name in aliases.items():
        if target_name in members_by_name and alias not in members_by_name:
            aliases_by_member.setdefault(target_name, []).append(alias)
    return tuple(
        _WeightMember(
            name=member.name,
            url=member.url,
            locator=member.locator,
            aliases=tuple(sorted(aliases_by_member.get(member.name, ()))),
        )
        for member in members
    )


def _is_weights_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _node_name(node.func) == "Weights"


def _weight_url(node: ast.AST) -> str | None:
    assert isinstance(node, ast.Call)
    for keyword in node.keywords:
        if keyword.arg != "url" or not isinstance(keyword.value, ast.Constant):
            continue
        value = keyword.value.value
        if isinstance(value, str) and _is_web_url(value):
            return value
    return None


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _archive_path(value: str, source: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"{source}: source archive contains an unsafe path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{source}: source archive contains an unsafe path")
    return value


def _safe_package_path(value: str) -> str:
    path = _required_text(value, "package_path")
    if path.startswith("/") or "\\" in path:
        raise ValueError("package_path must be a safe relative path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("package_path must be a safe relative path")
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


def _is_web_url(value: str) -> bool:
    return value.startswith("https://") or value.startswith("http://")
