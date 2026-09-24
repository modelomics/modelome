"""Pinned, static enumeration of timm's public model registry.

The ``timm.models`` package initializer is the upstream library's declared model
module list.  Each imported module registers architectures with
``@register_model`` and commonly declares pretrained configurations through
``generate_default_cfgs``.  This adapter reads that source only as data: it
never imports or executes upstream Python and never resolves model weights.
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
_MODULE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*$")
_HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MAX_MODULES = 1_000


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _PretrainedConfig:
    key: str
    fields: Mapping[str, Any]
    locator: str

    @property
    def architecture(self) -> str:
        return self.key.partition(".")[0]

    @property
    def tag(self) -> str | None:
        _, separator, tag = self.key.partition(".")
        return tag.strip("*") if separator else None

    @property
    def registry_key(self) -> str:
        """Return the architecture-and-tag spelling used by timm's registry.

        ``generate_default_cfgs`` accepts a trailing ``*`` as a source-level
        priority marker. Upstream strips it before registering model/tag names,
        so it must never escape into a Hub identity or release identifier.
        """

        return f"{self.architecture}.{self.tag}" if self.tag else self.architecture


@dataclass(frozen=True, slots=True)
class _ModuleCatalog:
    module: str
    path: str
    source_sha256: str
    models: tuple[str, ...]
    configs: tuple[_PretrainedConfig, ...]
    deprecated_aliases: Mapping[str, str]


class TimmModelRegistrySourceAdapter:
    """Enumerate timm's current model entrypoints at an immutable Git revision.

    The adapter resolves the public repository branch to a commit then performs
    one bounded GitHub source-archive fetch.  It validates every model module
    named by ``timm/models/__init__.py``, parses source with :mod:`ast`, and
    retains only explicit ``@register_model`` entrypoints plus literal
    ``generate_default_cfgs`` declarations.  This is deliberately a source
    registry plane, not a claim that every architecture has a paper or that any
    linked checkpoint remains available.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers architectures imported by the public timm.models package at one "
        "observed Git commit, with only statically declared pretrained-config "
        "resources. It does not execute timm, discover dynamically registered "
        "third-party models, establish a paper for an architecture, or download "
        "any checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "timm-model-registry",
        repository: str = "huggingface/pytorch-image-models",
        branch: str = "main",
        max_archive_bytes: int = 64 * 1024 * 1024,
        max_module_bytes: int = 2 * 1024 * 1024,
        max_modules: int = _MAX_MODULES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.max_archive_bytes = _positive_int(max_archive_bytes, "max_archive_bytes")
        self.max_module_bytes = _positive_int(max_module_bytes, "max_module_bytes")
        self.max_modules = _positive_int(max_modules, "max_modules")
        self.client = client or HttpClient(max_response_bytes=self.max_archive_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "timm-model-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "max_archive_bytes": self.max_archive_bytes,
                "max_module_bytes": self.max_module_bytes,
                "max_modules": self.max_modules,
                "admission": "timm.models initializer imports plus static register_model",
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
            raise ValueError(f"{self.name}: source archive returned HTTP {archive_response.status}")
        if len(archive_response.body) > self.max_archive_bytes:
            raise ValueError(
                f"{self.name}: source archive exceeds {self.max_archive_bytes} bytes"
            )
        catalogs = self._catalogs(archive_response.body)
        records = tuple(
            record
            for catalog in catalogs
            for record in self._records(catalog, revision, archive_response.body)
        )
        if not records:
            raise ValueError(
                f"{self.name}: registry archive contains no @register_model entrypoints"
            )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "archive_url": self.archive_url(revision),
            "archive_sha256": content_hash(archive_response.body),
            "module_count": len(catalogs),
            "model_count": len(records),
            "pretrained_config_count": sum(
                len(catalog.configs) for catalog in catalogs
            ),
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
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _catalogs(self, archive: bytes) -> tuple[_ModuleCatalog, ...]:
        members = _archive_members(archive, self.name, self.max_module_bytes)
        initializer_path = _initializer_path(members, self.name)
        root = initializer_path.removesuffix("/timm/models/__init__.py")
        modules = _model_modules(members[initializer_path], self.name)
        if len(modules) > self.max_modules:
            raise ValueError(
                f"{self.name}: initializer imports more than {self.max_modules} model modules"
            )
        catalogs = []
        for module in modules:
            path = f"{root}/timm/models/{module.replace('.', '/')}.py"
            source = members.get(path)
            if source is None:
                raise ValueError(
                    f"{self.name}: initializer imports {module!r}, but archive lacks {path!r}"
                )
            catalogs.append(_parse_module(module, path.removeprefix(f"{root}/"), source, self.name))
        return tuple(catalogs)

    def _records(
        self,
        catalog: _ModuleCatalog,
        revision: str,
        archive: bytes,
    ) -> tuple[SourceRecord, ...]:
        aliases_by_target: dict[str, list[str]] = {}
        registered = set(catalog.models)
        for alias, target in catalog.deprecated_aliases.items():
            target_architecture = target.partition(".")[0]
            if target_architecture in registered and alias not in registered:
                aliases_by_target.setdefault(target_architecture, []).append(alias)

        configs_by_model: dict[str, list[_PretrainedConfig]] = {}
        for config in catalog.configs:
            if config.architecture in registered:
                configs_by_model.setdefault(config.architecture, []).append(config)

        records = []
        for model_name in catalog.models:
            configs = tuple(configs_by_model.get(model_name, ()))
            code_url = self.blob_url(revision, catalog.path)
            model_identifier = Identifier("timm:model", model_name)
            has_weights = any(_has_declared_weights(config) for config in configs)
            model = ModelHint(
                local_id=f"model:{model_name}",
                name=model_name,
                identifiers=(model_identifier,),
                aliases=tuple(sorted(aliases_by_target.get(model_name, ()))),
                status=ModelStatus.RELEASED if has_weights else ModelStatus.DOCUMENTED,
                locator=f"{catalog.path}:@register_model({model_name})",
            )
            links = [
                Link(self.repository_url, relation="source_repository", crawl=False),
                Link(code_url, relation="model_definition", locator=model.locator, crawl=False),
            ]
            releases = []
            rendered_configs = []
            for config in configs:
                rendered = _render_config(model_name, config, self.name)
                rendered_configs.append(rendered)
                for url in _strings(rendered.get("url")):
                    if _is_web_url(url):
                        links.append(
                            Link(url, relation="weights", locator=config.locator, crawl=False)
                        )
                hub_id = _text(rendered.get("hf_hub_id"))
                if hub_id:
                    links.append(
                        Link(
                            f"https://huggingface.co/{hub_id}",
                            relation="linked_model_artifact",
                            locator=config.locator,
                            crawl=False,
                        )
                    )
                    hub_filename = _text(rendered.get("hf_hub_filename"))
                    if _safe_hub_filename(hub_filename):
                        links.append(
                            Link(
                                f"https://huggingface.co/{hub_id}/resolve/main/"
                                f"{quote(hub_filename, safe='/')}",
                                relation="weights",
                                locator=config.locator,
                                crawl=False,
                            )
                        )
                for license_url in _strings(rendered.get("license")):
                    if _is_web_url(license_url):
                        links.append(
                            Link(
                                license_url,
                                relation="license",
                                locator=config.locator,
                                crawl=False,
                            )
                        )
                for origin_url in _strings(rendered.get("origin_url")):
                    if _is_web_url(origin_url):
                        links.append(
                            Link(
                                origin_url,
                                relation="official_implementation",
                                locator=config.locator,
                                crawl=False,
                            )
                        )
                if _has_declared_weights(config):
                    releases.append(
                        ReleaseHint(
                            local_id=f"release:{config.registry_key}",
                            model_local_id=model.local_id,
                            version=config.tag,
                            revision=revision,
                            identifiers=(
                                Identifier("timm:pretrained-config", config.registry_key),
                            ),
                            metadata={
                                "repository": self.repository,
                                "revision": revision,
                                "module": catalog.module,
                                "source_config_key": config.key,
                                "config": rendered,
                            },
                            locator=config.locator,
                        )
                    )
            records.append(
                SourceRecord(
                    source_record_id=f"model:{model_name}",
                    kind=ArtifactKind.CATALOG_RECORD,
                    canonical_url=canonicalize_url(code_url),
                    title=model_name,
                    raw={
                        "repository": self.repository,
                        "revision": revision,
                        "archive_sha256": content_hash(archive),
                        "module": catalog.module,
                        "module_path": catalog.path,
                        "module_sha256": catalog.source_sha256,
                        "pretrained_configs": rendered_configs,
                        "deprecated_aliases": list(model.aliases),
                    },
                    text="\n".join(
                        item
                        for item in (
                            model_name,
                            f"timm module: {catalog.module}",
                            "pretrained configs: " + ", ".join(
                                config.key for config in configs
                            ) if configs else "",
                        )
                        if item
                    ),
                    identifiers=(model_identifier,),
                    links=tuple(dict.fromkeys(links)),
                    models=(model,),
                    releases=tuple(releases),
                )
            )
        return tuple(records)


def _archive_members(
    archive: bytes,
    source: str,
    max_module_bytes: int,
) -> dict[str, str]:
    try:
        package = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as error:
        raise ValueError(f"{source}: source archive is not a ZIP file") from error
    with package:
        members: dict[str, str] = {}
        for info in package.infolist():
            if info.is_dir():
                continue
            path = _archive_path(info.filename, source)
            if not path.endswith(".py"):
                continue
            if not (path.endswith("/timm/models/__init__.py") or "/timm/models/" in path):
                continue
            if info.file_size > max_module_bytes:
                raise ValueError(
                    f"{source}: source file {path!r} exceeds {max_module_bytes} bytes"
                )
            if path in members:
                raise ValueError(f"{source}: source archive has duplicate member {path!r}")
            members[path] = package.read(info).decode("utf-8", errors="strict")
    return members


def _initializer_path(members: Mapping[str, str], source: str) -> str:
    candidates = [
        path
        for path in members
        if path.endswith("/timm/models/__init__.py")
        and len(path.split("/")) == 4
    ]
    if len(candidates) != 1:
        raise ValueError(f"{source}: archive must contain exactly one timm models initializer")
    return candidates[0]


def _model_modules(source: str, name: str) -> tuple[str, ...]:
    tree = _parse_python(source, name, "timm/models/__init__.py")
    modules = []
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom) or statement.level != 1:
            continue
        module = statement.module or ""
        if module.startswith("_") or not _MODULE.fullmatch(module):
            continue
        if len(statement.names) != 1 or statement.names[0].name != "*":
            continue
        modules.append(module)
    if not modules:
        raise ValueError(f"{name}: models initializer declares no wildcard model modules")
    if len(set(modules)) != len(modules):
        raise ValueError(f"{name}: models initializer has duplicate model-module imports")
    return tuple(modules)


def _parse_module(module: str, path: str, source: str, name: str) -> _ModuleCatalog:
    tree = _parse_python(source, name, path)
    # timm occasionally factors a literal checkpoint URL or Hub ID into a
    # module-level constant. Resolve only simple literal assignments; this
    # keeps the source parser non-executing while preserving those references.
    constants: dict[str, Any] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value_node = statement.value
        else:
            continue
        value = _literal_value(value_node)
        if value is _UNSET:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
    helper_origin_urls = _helper_origin_urls(tree)
    models = []
    configs: dict[str, _PretrainedConfig] = {}
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _call_name(decorator) == "register_model" for decorator in node.decorator_list
        ):
            models.append(node.name)
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) == "generate_default_cfgs":
            if not node.args or not isinstance(node.args[0], ast.Dict):
                continue
            for key_node, value_node in zip(node.args[0].keys, node.args[0].values, strict=True):
                key = _literal_string(key_node)
                fields = _config_fields(value_node, constants, helper_origin_urls)
                if not key or fields is None:
                    continue
                if key in configs:
                    raise ValueError(f"{name}: duplicate pretrained config key {key!r} in {path}")
                configs[key] = _PretrainedConfig(
                    key=key,
                    fields=fields,
                    locator=f"{path}:generate_default_cfgs[{key!r}]",
                )
        elif _call_name(node.func) == "register_model_deprecations":
            mapping = node.args[1] if len(node.args) > 1 else None
            if not isinstance(mapping, ast.Dict):
                continue
            for alias_node, target_node in zip(mapping.keys, mapping.values, strict=True):
                alias = _literal_string(alias_node)
                target = _literal_string(target_node)
                if alias and target:
                    aliases[alias] = target
    if len(set(models)) != len(models):
        raise ValueError(f"{name}: duplicate @register_model entrypoint in {path}")
    return _ModuleCatalog(
        module=module,
        path=path,
        source_sha256=content_hash(source.encode()),
        models=tuple(models),
        configs=tuple(configs.values()),
        deprecated_aliases=aliases,
    )


def _config_fields(
    node: ast.AST,
    constants: Mapping[str, Any] | None = None,
    helper_origin_urls: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    if isinstance(node, ast.Call):
        fields = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            value = _literal_value(keyword.value, constants)
            if value is not _UNSET:
                fields[keyword.arg] = value
        helper_name = _call_name(node.func)
        if (
            "origin_url" not in fields
            and helper_origin_urls is not None
            and (origin_url := helper_origin_urls.get(helper_name))
        ):
            fields["origin_url"] = origin_url
        return fields
    if isinstance(node, ast.Dict):
        fields = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            key = _literal_string(key_node)
            value = _literal_value(value_node, constants)
            if key and value is not _UNSET:
                fields[key] = value
        return fields
    return None


def _helper_origin_urls(tree: ast.Module) -> dict[str, str]:
    """Collect literal ``origin_url`` defaults from config helper functions.

    timm commonly wraps ``_cfg`` in helpers such as ``_gcfg`` and places an
    exact upstream implementation URL in a literal dict merged into kwargs.
    Reading only that named string field captures the reference without
    evaluating arbitrary helper code or inheriting unrelated defaults.
    """

    helpers: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(statement):
            if not isinstance(node, ast.Dict):
                continue
            for key_node, value_node in zip(node.keys, node.values, strict=True):
                if _literal_string(key_node) == "origin_url":
                    value = _literal_string(value_node)
                    if _is_web_url(value):
                        helpers[statement.name] = value

    return helpers


_UNSET = object()


def _literal_value(node: ast.AST, constants: Mapping[str, Any] | None = None) -> Any:
    if isinstance(node, ast.Constant) and isinstance(
        node.value,
        str | int | float | bool | type(None),
    ):
        return node.value
    if isinstance(node, ast.Name) and constants is not None:
        return constants.get(node.id, _UNSET)
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_literal_value(value, constants) for value in node.elts]
        if any(value is _UNSET for value in values):
            return _UNSET
        return tuple(values)
    return _UNSET


def _literal_string(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    return ""


def _render_config(model_name: str, config: _PretrainedConfig, source: str) -> dict[str, Any]:
    rendered = dict(config.fields)
    hub_id = _text(rendered.get("hf_hub_id"))
    if hub_id == "timm/":
        hub_id = f"timm/{config.registry_key}"
        rendered["hf_hub_id"] = hub_id
    if hub_id and not _HUB_ID.fullmatch(hub_id):
        raise ValueError(
            f"{source}: {model_name} config {config.key!r} has invalid Hugging Face ID {hub_id!r}"
        )
    return rendered


def _has_declared_weights(config: _PretrainedConfig) -> bool:
    return any(
        bool(_strings(config.fields.get(field)))
        for field in ("url", "file", "hf_hub_id")
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def _parse_python(source: str, name: str, path: str) -> ast.Module:
    try:
        return ast.parse(source, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{name}: cannot parse {path}: {error.msg}") from error


def _archive_path(value: str, source: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"{source}: source archive contains an unsafe path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{source}: source archive contains an unsafe path")
    return value


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


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple | list):
        return tuple(item for item in value if isinstance(item, str) and item)
    return ()


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


def _safe_hub_filename(value: str) -> bool:
    """Accept relative Hub filenames without allowing path traversal."""

    return bool(value) and not value.startswith("/") and all(
        part not in {"", ".", ".."} for part in value.split("/")
    )
