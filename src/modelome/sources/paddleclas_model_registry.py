"""Pinned static ingestion of PaddleClas's public inference-model registry."""

from __future__ import annotations

import ast
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
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SAFE_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,255}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _ModelEntry:
    catalog: str
    family: str | None
    name: str
    url_template: str
    locator: str


class PaddleClasModelRegistrySourceAdapter:
    """Enumerate literal PaddleClas inference-model declarations at one commit.

    ``paddleclas.py`` is the upstream command-line registry that maps public
    model names to archive URL templates. This adapter parses its literal
    ImageNet-series and PULC lists paired with their templates, and resolves
    the ShiTu selector to the literal archive handles used by its runtime
    downloader. It neither imports PaddleClas nor follows/downloads an archive.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers ImageNet-series, PULC, and ShiTu inference models explicitly declared "
        "by PaddleClas's public static registry at one commit. It does not execute "
        "PaddleClas, infer undocumented models, verify archive availability, or "
        "download checkpoints."
    )

    def __init__(
        self,
        *,
        name: str = "paddleclas-model-registry",
        repository: str = "PaddlePaddle/PaddleClas",
        branch: str = "release/2.6",
        source_path: str = "paddleclas.py",
        max_response_bytes: int = 4 * 1024 * 1024,
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
                "adapter": "paddleclas-model-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": (
                    "literal ImageNet-series/PULC entries or SHITU runtime archive "
                    "handles paired with literal URL templates"
                ),
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
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
            self.raw_url(revision), headers={"Accept": "text/plain,text/x-python"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source file returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source file exceeds {self.max_response_bytes} bytes")
        entries = _parse_source(response.text(), self.name, self.source_path)
        if len(entries) > self.max_entries:
            raise ValueError(f"{self.name}: static registry exceeds {self.max_entries} models")
        if not entries:
            raise ValueError(f"{self.name}: static registry contains no admissible models")
        records = tuple(self._record(entry, revision, response.body) for entry in entries)
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
            "catalog_counts": _catalog_counts(entries),
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

    def _record(self, entry: _ModelEntry, revision: str, source: bytes) -> SourceRecord:
        identity = f"{entry.catalog}:{entry.name}"
        local_id = f"model:{identity}"
        model_identifier = Identifier("paddleclas:model", identity)
        source_url = self.blob_url(revision)
        weight_url = _render_url_template(entry.url_template, entry.name, self.name)
        model = ModelHint(
            local_id=local_id,
            name=entry.name,
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=entry.locator,
        )
        release = ReleaseHint(
            local_id=f"release:{identity}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("paddleclas:inference-model", identity),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "catalog": entry.catalog,
                "family": entry.family,
                "archive_url_template": entry.url_template,
                "archive_url": weight_url,
            },
            locator=entry.locator,
        )
        return SourceRecord(
            source_record_id=f"model:{identity}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(source_url),
            title=entry.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "catalog": entry.catalog,
                "family": entry.family,
                "model_name": entry.name,
                "archive_url_template": entry.url_template,
                "archive_url": weight_url,
                "locator": entry.locator,
            },
            text="\n".join(
                item
                for item in (
                    f"PaddleClas {entry.catalog} inference model: {entry.name}",
                    f"family: {entry.family}" if entry.family else "",
                )
                if item
            ),
            identifiers=(model_identifier,),
            links=(
                Link(
                    self.repository_url,
                    relation="source_repository",
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
                Link(
                    source_url,
                    relation="model_definition",
                    locator=entry.locator,
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
                Link(
                    weight_url,
                    relation="weights",
                    locator=entry.locator,
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_source(source: str, name: str, path: str) -> tuple[_ModelEntry, ...]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{name}: cannot parse source file: {error.msg}") from error

    assignments = _literal_assignments(tree)
    entries: list[_ModelEntry] = []
    imn_template = _url_template(assignments.get("IMN_MODEL_BASE_DOWNLOAD_URL"), name)
    imn_series = _literal_string_list_dict(assignments.get("IMN_MODEL_SERIES"), name)
    if imn_template is not None and imn_series is not None:
        for family, models in imn_series:
            for index, model in enumerate(models):
                entries.append(
                    _entry(
                        catalog="IMN",
                        family=family,
                        name=model,
                        url_template=imn_template,
                        locator=(f"{path}:IMN_MODEL_SERIES[{family!r}][{index}]"),
                        source=name,
                    )
                )

    pulc_template = _url_template(assignments.get("PULC_MODEL_BASE_DOWNLOAD_URL"), name)
    pulc_models = _literal_string_list(assignments.get("PULC_MODELS"), name)
    if pulc_template is not None and pulc_models is not None:
        for index, model in enumerate(pulc_models):
            entries.append(
                _entry(
                    catalog="PULC",
                    family=None,
                    name=model,
                    url_template=pulc_template,
                    locator=f"{path}:PULC_MODELS[{index}]",
                    source=name,
                )
            )

    shitu_template = _url_template(assignments.get("SHITU_MODEL_BASE_DOWNLOAD_URL"), name)
    shitu_models = _literal_string_list(assignments.get("SHITU_MODELS"), name)
    if shitu_template is not None and shitu_models is not None:
        if len(shitu_models) != 1:
            raise ValueError(f"{name}: expected one top-level SHITU selector")
        selector = shitu_models[0]
        for archive_name, line_number in _shitu_archive_handles(tree, name):
            entries.append(
                _entry(
                    catalog="SHITU",
                    family=selector,
                    name=archive_name,
                    url_template=shitu_template,
                    locator=f"{path}:line:{line_number} (SHITU selector {selector})",
                    source=name,
                )
            )

    if not entries:
        raise ValueError(f"{name}: source contains no supported literal inference model registry")
    seen = set()
    for entry in entries:
        key = (entry.catalog, entry.name)
        if key in seen:
            raise ValueError(f"{name}: duplicate static model declaration {key!r}")
        seen.add(key)
    return tuple(entries)


def _literal_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            result[target.id] = statement.value
    return result


def _shitu_archive_handles(tree: ast.Module, source: str) -> tuple[tuple[str, int], ...]:
    """Read the exact archive handles passed to runtime's SHITU downloader."""
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_check_input_model"
    ]
    if len(functions) != 1:
        raise ValueError(f"{source}: expected one _check_input_model function for SHITU handles")

    handles: list[tuple[str, int]] = []
    for node in ast.walk(functions[0]):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "check_model_file" or not node.args:
            continue
        model_type = node.args[0]
        if not isinstance(model_type, ast.Constant) or model_type.value != "shitu":
            continue
        if len(node.args) < 2:
            raise ValueError(f"{source}: SHITU download call lacks an archive handle")
        handle = node.args[1]
        if not isinstance(handle, ast.Constant) or not isinstance(handle.value, str):
            raise ValueError(f"{source}: SHITU archive handles must be literal strings")
        handles.append((handle.value, node.lineno))
    if not handles:
        raise ValueError(f"{source}: runtime declares no literal SHITU archive handles")
    return tuple(dict.fromkeys(handles))


def _literal_string_list_dict(
    node: ast.AST | None, source: str
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    if node is None:
        return None
    if not isinstance(node, ast.Dict):
        raise ValueError(f"{source}: IMN_MODEL_SERIES must be a literal dictionary")
    result = []
    for key, value in zip(node.keys, node.values, strict=True):
        family = _required_literal_string(key, source, "IMN series name")
        models = _literal_string_list(value, source)
        if not models:
            raise ValueError(f"{source}: IMN series {family!r} has no models")
        result.append((family, models))
    if not result:
        raise ValueError(f"{source}: IMN_MODEL_SERIES is empty")
    return tuple(result)


def _literal_string_list(node: ast.AST | None, source: str) -> tuple[str, ...] | None:
    if node is None:
        return None
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise ValueError(f"{source}: expected a literal model list")
    values = tuple(_required_literal_string(item, source, "model name") for item in node.elts)
    if not values:
        raise ValueError(f"{source}: static model list is empty")
    return values


def _url_template(node: ast.AST | None, source: str) -> str | None:
    if node is None:
        return None
    value = _required_literal_string(node, source, "model archive URL template")
    if value.count("{}") != 1 or "{" in value.replace("{}", "") or "}" in value.replace("{}", ""):
        raise ValueError(f"{source}: model archive URL template must contain one '{{}}' slot")
    parsed = urlsplit(value.replace("{}", "placeholder"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{source}: model archive URL template is not an HTTP URL")
    return value


def _entry(
    *,
    catalog: str,
    family: str | None,
    name: str,
    url_template: str,
    locator: str,
    source: str,
) -> _ModelEntry:
    if not _SAFE_MODEL_NAME.fullmatch(name) or any(
        part in {"", ".", ".."} for part in name.split("/")
    ):
        raise ValueError(f"{source}: invalid static model name {name!r}")
    return _ModelEntry(
        catalog=catalog,
        family=family,
        name=name,
        url_template=url_template,
        locator=locator,
    )


def _render_url_template(template: str, model_name: str, source: str) -> str:
    rendered = template.format(quote(model_name, safe="/"))
    parsed = urlsplit(rendered)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{source}: rendered model archive URL is not an HTTP URL")
    return canonicalize_url(rendered)


def _catalog_counts(entries: tuple[_ModelEntry, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.catalog] = counts.get(entry.catalog, 0) + 1
    return counts


def _required_literal_string(node: ast.AST | None, source: str, field: str) -> str:
    value = (
        node.value.strip() if isinstance(node, ast.Constant) and isinstance(node.value, str) else ""
    )
    if not value:
        raise ValueError(f"{source}: {field} must be a non-empty literal string")
    return value


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
    value = headers.get(key) or headers.get(key.casefold()) or headers.get(key.title())
    return _text(value)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["PaddleClasModelRegistrySourceAdapter"]
