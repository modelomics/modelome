from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
_KEY_VALUE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 _-]*):(?:[ ]*(?P<value>.*))?$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class _Collection:
    name: str = ""
    paper_url: str | None = None
    code_url: str | None = None
    readme: str | None = None
    nested_key: str | None = None


@dataclass(slots=True)
class _Model:
    name: str = ""
    collection: str | None = None
    config: str | None = None
    weights: str | None = None
    aliases: list[str] = field(default_factory=list)
    paper_url: str | None = None
    code_url: str | None = None
    nested_key: str | None = None
    alias_list_active: bool = False
    line: int = 0


class OpenMMLabModelIndexSourceAdapter:
    """Enumerate one OpenMMLab project's complete version-pinned model index.

    OpenMMLab project indexes import every model metafile. Each declared model
    becomes a separate artifact, linked directly to its version-pinned config,
    checkpoint, collection paper, and source code. The compact line parser
    intentionally reads only the stable public index fields instead of using a
    permissive YAML loader against untrusted repository content.
    """

    disable_derived_extraction = True

    def __init__(
        self,
        *,
        name: str,
        repository: str,
        branch: str = "main",
        max_manifests: int = 1_000,
        max_index_bytes: int = 2 * 1024 * 1024,
        max_manifest_bytes: int = 4 * 1024 * 1024,
        max_models: int = 100_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.max_manifests = _positive_int(max_manifests, "max_manifests")
        self.max_index_bytes = _positive_int(max_index_bytes, "max_index_bytes")
        self.max_manifest_bytes = _positive_int(max_manifest_bytes, "max_manifest_bytes")
        self.max_models = _positive_int(max_models, "max_models")
        self.client = client or HttpClient(
            max_response_bytes=max(self.max_index_bytes, self.max_manifest_bytes)
        )
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openmmlab-model-index-v1",
                "repository": self.repository,
                "branch": self.branch,
                "limits": {
                    "max_manifests": self.max_manifests,
                    "max_index_bytes": self.max_index_bytes,
                    "max_manifest_bytes": self.max_manifest_bytes,
                    "max_models": self.max_models,
                },
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        branch = quote(self.branch, safe="")
        return f"https://api.github.com/repos/{self.repository}/commits/{branch}"

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

        index_url = self._raw_url(revision, "model-index.yml")
        index_response: HttpResponse = self.client.get(
            index_url, headers={"Accept": "text/yaml,text/plain"}
        )
        _require_response(index_response, self.name, "model index", self.max_index_bytes)
        index_text = index_response.text()
        index_hash = content_hash(index_response.body)
        manifest_paths = _imports(index_text)
        if not manifest_paths:
            raise ValueError(f"{self.name}: model index contained no imported metafiles")
        if len(manifest_paths) > self.max_manifests:
            raise ValueError(
                f"{self.name}: model index exceeds {self.max_manifests} metafiles"
            )

        records: list[SourceRecord] = []
        manifest_hashes: dict[str, str] = {}
        for path in manifest_paths:
            response: HttpResponse = self.client.get(
                self._raw_url(revision, path),
                headers={"Accept": "text/yaml,text/plain"},
            )
            _require_response(response, self.name, f"metafile {path}", self.max_manifest_bytes)
            manifest_hash = content_hash(response.body)
            manifest_hashes[path] = manifest_hash
            collections, models = _parse_metafile(response.text())
            for model in models:
                if not model.name:
                    continue
                records.append(
                    self._record(
                        revision=revision,
                        index_hash=index_hash,
                        manifest_path=path,
                        manifest_hash=manifest_hash,
                        model=model,
                        collection=collections.get(model.collection or ""),
                    )
                )
                if len(records) > self.max_models:
                    raise ValueError(f"{self.name}: model index exceeds {self.max_models} models")
        if not records:
            raise ValueError(f"{self.name}: imported metafiles declared no named models")

        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "index_url": index_url,
            "index_sha256": index_hash,
            "index_import_count": len(manifest_paths),
            "manifest_sha256": manifest_hashes,
            "model_count": len(records),
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
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _raw_url(self, revision: str, path: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(path, safe='/')}"
        )

    def _blob_url(self, revision: str, path: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(path, safe='/')}"
        )

    def _record(
        self,
        *,
        revision: str,
        index_hash: str,
        manifest_path: str,
        manifest_hash: str,
        model: _Model,
        collection: _Collection | None,
    ) -> SourceRecord:
        config_path = _safe_relative_path(model.config) if model.config else None
        manifest_url = self._blob_url(revision, manifest_path)
        config_url = self._blob_url(revision, config_path) if config_path else manifest_url
        artifact_key = content_hash(
            {
                "repository": self.repository,
                "revision": revision,
                "manifest": manifest_path,
                "model": model.name,
                "config": config_path,
            }
        )
        model_key = f"{self.repository}:{config_path or model.name}"
        model_identifier = Identifier("openmmlab:model", model_key)
        artifact_identifier = Identifier("openmmlab:model-artifact", artifact_key)
        has_weights = bool(model.weights)
        model_hint = ModelHint(
            local_id=f"{artifact_key}#model",
            name=model.name,
            identifiers=(model_identifier,),
            aliases=tuple(_unique_text(model.aliases)),
            status=ModelStatus.RELEASED if has_weights else ModelStatus.DOCUMENTED,
            locator=f"{manifest_path}:line:{model.line}",
        )
        locator = f"{manifest_path}:line:{model.line}"
        links = [
            Link(config_url, relation="model_config", locator=locator),
            Link(manifest_url, relation="metadata", locator=locator),
            Link(self.repository_url, relation="source_repository", locator=locator),
        ]
        if model.weights:
            links.append(Link(model.weights, relation="weights", locator=locator))
        if collection is not None:
            self._collection_links(links, revision, collection, manifest_path, model.line)
        if model.paper_url:
            links.append(Link(model.paper_url, relation="paper_reference", locator=locator))
        if model.code_url:
            links.append(Link(model.code_url, relation="code_reference", locator=locator))
        releases = ()
        if model.weights:
            release = ReleaseHint(
                local_id=f"{artifact_key}#release",
                model_local_id=model_hint.local_id,
                identifiers=(Identifier("openmmlab:weights", model.weights),),
                revision=revision,
                metadata={
                    "repository": self.repository,
                    "config_path": config_path,
                    "weights_url": model.weights,
                    "manifest_path": manifest_path,
                },
                locator=f"{manifest_path}:line:{model.line}",
            )
            releases = (release,)
        collection_name = collection.name if collection is not None else model.collection
        text_parts = [model.name]
        if collection_name:
            text_parts.append(collection_name)
        return SourceRecord(
            source_record_id=f"model:{artifact_key[:32]}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=config_url,
            title=model.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "index_sha256": index_hash,
                "manifest_path": manifest_path,
                "manifest_sha256": manifest_hash,
                "model": {
                    "name": model.name,
                    "aliases": _unique_text(model.aliases),
                    "collection": model.collection,
                    "config": config_path,
                    "weights": model.weights,
                    "paper_url": model.paper_url,
                    "code_url": model.code_url,
                },
                "collection": None
                if collection is None
                else {
                    "name": collection.name,
                    "paper_url": collection.paper_url,
                    "code_url": collection.code_url,
                    "readme": collection.readme,
                },
            },
            text="\n".join(text_parts),
            identifiers=(artifact_identifier,),
            links=tuple(_unique_links(links)),
            models=(model_hint,),
            releases=releases,
        )

    def _collection_links(
        self,
        links: list[Link],
        revision: str,
        collection: _Collection,
        manifest_path: str,
        line: int,
    ) -> None:
        locator = f"{manifest_path}:line:{line}"
        if collection.paper_url:
            links.append(Link(collection.paper_url, relation="paper_reference", locator=locator))
        if collection.code_url:
            links.append(Link(collection.code_url, relation="code_reference", locator=locator))
        if collection.readme:
            links.append(
                Link(
                    self._blob_url(revision, collection.readme),
                    relation="documentation",
                    locator=locator,
                )
            )


def _imports(value: str) -> tuple[str, ...]:
    active = False
    paths: list[str] = []
    for line in value.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped == "Import:":
            active = True
            continue
        if not active:
            continue
        candidate = stripped
        if not line.startswith((" ", "\t")) and not candidate.startswith("- "):
            active = False
            continue
        if not candidate.startswith("- "):
            continue
        path = _safe_relative_path(_yaml_scalar(candidate[2:]))
        if path.casefold().endswith((".yml", ".yaml")):
            paths.append(path)
    return tuple(dict.fromkeys(paths))


def _parse_metafile(value: str) -> tuple[dict[str, _Collection], tuple[_Model, ...]]:
    section: str | None = None
    current_collection: _Collection | None = None
    current_model: _Model | None = None
    collections: dict[str, _Collection] = {}
    models: list[_Model] = []
    collection_list_indent: int | None = None
    model_list_indent: int | None = None

    def finish_collection() -> None:
        nonlocal current_collection
        if current_collection is not None and current_collection.name:
            collections[current_collection.name] = current_collection
        current_collection = None

    def finish_model() -> None:
        nonlocal current_model
        if current_model is not None and current_model.name:
            models.append(current_model)
        current_model = None

    for line_number, line in enumerate(value.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped in {"Collections:", "Models:"}:
            finish_collection()
            finish_model()
            section = stripped.removesuffix(":")
            collection_list_indent = None
            model_list_indent = None
            continue
        if section == "Collections":
            if stripped.startswith("- ") and (
                collection_list_indent is None or indent == collection_list_indent
            ):
                finish_collection()
                current_collection = _Collection()
                collection_list_indent = indent
                _collection_field(current_collection, stripped[2:])
            elif current_collection is not None:
                _collection_field(current_collection, stripped)
        elif section == "Models":
            if stripped.startswith("- ") and (
                model_list_indent is None or indent == model_list_indent
            ):
                finish_model()
                current_model = _Model(line=line_number)
                model_list_indent = indent
                _model_field(current_model, stripped[2:])
            elif current_model is not None:
                _model_field(current_model, stripped)
    finish_collection()
    finish_model()
    return collections, tuple(models)


def _collection_field(collection: _Collection, value: str) -> None:
    parsed = _key_value(value)
    if parsed is None:
        return
    key, raw = parsed
    if key in {"Paper", "Code"} and not raw:
        collection.nested_key = key
        return
    if key == "Name":
        collection.name = _yaml_scalar(raw)
    elif key == "README":
        collection.readme = _maybe_path(raw)
    elif key == "URL":
        url = _maybe_url(raw)
        if collection.nested_key == "Paper":
            collection.paper_url = url
        elif collection.nested_key == "Code":
            collection.code_url = url


def _model_field(model: _Model, value: str) -> None:
    if model.alias_list_active and value.startswith("- "):
        alias = _yaml_scalar(value[2:])
        if alias:
            model.aliases.append(alias)
        return
    parsed = _key_value(value)
    if parsed is None:
        return
    key, raw = parsed
    if key != "Alias":
        model.alias_list_active = False
    if key in {"Paper", "Code"} and not raw:
        model.nested_key = key
        return
    if key == "Name":
        model.name = _yaml_scalar(raw)
    elif key == "In Collection":
        model.collection = _yaml_scalar(raw) or None
    elif key == "Config":
        model.config = _maybe_path(raw)
    elif key == "Weights":
        model.weights = _maybe_url(raw)
    elif key == "Alias":
        if not raw:
            model.alias_list_active = True
            model.nested_key = "Alias"
        else:
            aliases = _yaml_sequence(raw)
            model.aliases.extend(aliases or [_yaml_scalar(raw)])
    elif key == "URL":
        url = _maybe_url(raw)
        if model.nested_key == "Paper":
            model.paper_url = url
        elif model.nested_key == "Code":
            model.code_url = url


def _key_value(value: str) -> tuple[str, str] | None:
    match = _KEY_VALUE.fullmatch(value)
    return None if match is None else (match.group("key"), match.group("value") or "")


def _yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


def _yaml_sequence(value: str) -> list[str]:
    """Read simple inline YAML sequences used by model alias declarations."""
    value = value.strip()
    if len(value) < 2 or value[0] != "[" or value[-1] != "]":
        return []
    result: list[str] = []
    token = ""
    quote_char: str | None = None
    for char in value[1:-1]:
        if quote_char:
            if char == quote_char:
                quote_char = None
            else:
                token += char
        elif char in {"'", '"'}:
            quote_char = char
        elif char == ",":
            if item := token.strip():
                result.append(item)
            token = ""
        else:
            token += char
    if item := token.strip():
        result.append(item)
    return result


def _maybe_url(value: str) -> str | None:
    result = _yaml_scalar(value)
    if not result.startswith(("https://", "http://")):
        return None
    try:
        return canonicalize_url(result)
    except ValueError:
        return None


def _maybe_path(value: str) -> str | None:
    try:
        return _safe_relative_path(_yaml_scalar(value))
    except ValueError:
        return None


def _safe_relative_path(value: str | None) -> str:
    if not isinstance(value, str):
        raise ValueError("repository path is required")
    path = value.strip().strip("/")
    if not path or "://" in path or "\\" in path:
        raise ValueError("repository path is invalid")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("repository path is unsafe")
    return path


def _repository(value: str) -> str:
    result = _required_text(value, "repository").strip("/")
    parts = result.split("/")
    if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts):
        raise ValueError("repository must be an owner/name GitHub repository")
    return result


def _require_response(
    response: HttpResponse, source: str, label: str, maximum: int
) -> None:
    if response.status != 200:
        raise ValueError(f"{source}: {label} returned HTTP {response.status}")
    if len(response.body) > maximum:
        raise ValueError(f"{source}: {label} exceeds {maximum} bytes")


def _unique_links(values: Sequence[Link]) -> tuple[Link, ...]:
    result: list[Link] = []
    seen: set[tuple[str, str, str | None]] = set()
    for value in values:
        key = (value.url, value.relation, value.locator)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _unique_text(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _required_text(value: Any, field: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == wanted), None)


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
