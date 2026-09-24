"""Pinned static ingestion of KerasHub's source-declared pretrained presets."""

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
_DEFAULT_PRESET_ROOT = "keras_hub/src/models"
_MAX_FILES = 5_000


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Preset:
    name: str
    collection: str
    path: str
    source_sha256: str
    handles: tuple[tuple[str, str], ...]
    metadata: Mapping[str, Any]
    locator: str


class KerasHubPresetRegistrySourceAdapter:
    """Enumerate literal KerasHub preset dictionaries at one public Git commit.

    KerasHub's model sources declare concrete pretrained presets in top-level
    ``*_presets`` dictionaries. A source-native preset identity includes its
    source file, dictionary, and exact key. The parser also accepts a
    ``**name`` expansion only when ``name`` was assigned an earlier literal
    dictionary in the same source file. ``kaggle://`` and ``hf://`` values are
    converted only to their corresponding public landing URLs; archive or
    weight bytes are never requested. For archived KerasCV only, nonliteral
    aggregate collections are skipped while literal per-model declarations are
    retained.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers literal pretrained-preset entries and their declared Kaggle or "
        "Hugging Face handles in KerasHub's public preset-source files at one Git "
        "commit. It does not execute KerasHub, infer a paper or origin model, "
        "enumerate dynamic presets, or download artifacts. KerasCV scans skip "
        "nonliteral aggregate collections while retaining literal per-model files."
    )

    def __init__(
        self,
        *,
        name: str = "keras-hub-preset-registry",
        repository: str = "keras-team/keras-hub",
        branch: str = "master",
        preset_root: str = _DEFAULT_PRESET_ROOT,
        max_archive_bytes: int = 64 * 1024 * 1024,
        max_file_bytes: int = 2 * 1024 * 1024,
        max_files: int = _MAX_FILES,
        max_presets: int = 100_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.preset_root = _safe_path(preset_root, "preset_root")
        self.max_archive_bytes = _positive_int(max_archive_bytes, "max_archive_bytes")
        self.max_file_bytes = _positive_int(max_file_bytes, "max_file_bytes")
        self.max_files = _positive_int(max_files, "max_files")
        self.max_presets = _positive_int(max_presets, "max_presets")
        self.client = client or HttpClient(max_response_bytes=self.max_archive_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "keras-hub-preset-registry-v2",
                "repository": self.repository,
                "branch": self.branch,
                "preset_root": self.preset_root,
                "max_archive_bytes": self.max_archive_bytes,
                "max_file_bytes": self.max_file_bytes,
                "max_files": self.max_files,
                "max_presets": self.max_presets,
                "admission": (
                    "literal *_presets dictionaries and source-local literal "
                    "dictionary expansions with provider handles"
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
                upstream_count=_nonnegative_int(state.get("preset_count")),
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
        presets, file_count = self._presets(archive_response.body)
        if len(presets) > self.max_presets:
            raise ValueError(f"{self.name}: source declares more than {self.max_presets} presets")
        if not presets:
            raise ValueError(f"{self.name}: source archive contains no provider-backed presets")
        records = tuple(
            self._record(preset, revision, archive_response.body) for preset in presets
        )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "archive_url": self.archive_url(revision),
            "archive_sha256": content_hash(archive_response.body),
            "preset_file_count": file_count,
            "preset_count": len(records),
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

    def _presets(self, archive: bytes) -> tuple[tuple[_Preset, ...], int]:
        files = _preset_files(
            archive,
            source=self.name,
            preset_root=self.preset_root,
            max_file_bytes=self.max_file_bytes,
        )
        if len(files) > self.max_files:
            raise ValueError(f"{self.name}: source has more than {self.max_files} preset files")
        result: list[_Preset] = []
        seen: set[tuple[str, str, str]] = set()
        for path, source in sorted(files.items()):
            for preset in _parse_presets(
                source,
                path,
                self.name,
                allow_dynamic_preset_collections=self.repository == "keras-team/keras-cv",
            ):
                key = (preset.path, preset.collection, preset.name)
                if key in seen:
                    raise ValueError(f"{self.name}: duplicate source preset {key!r}")
                seen.add(key)
                result.append(preset)
        return tuple(result), len(files)

    def _record(self, preset: _Preset, revision: str, archive: bytes) -> SourceRecord:
        identity = f"{preset.path}:{preset.collection}:{preset.name}"
        model_identifier = Identifier("keras-hub:preset", identity)
        model = ModelHint(
            local_id=f"model:{content_hash(identity)[:24]}",
            name=preset.name,
            aliases=(preset.collection,),
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=preset.locator,
        )
        code_url = self.blob_url(revision, preset.path)
        links = [
            Link(self.repository_url, relation="source_repository", crawl=False),
            Link(
                code_url,
                relation="preset_definition",
                locator=preset.locator,
                crawl=False,
                model_local_ids=(model.local_id,),
            ),
        ]
        releases = []
        for field, handle in preset.handles:
            url, relation = _handle_url(handle, self.name, preset.locator)
            handle_version = _handle_version(field, handle)
            links.append(
                Link(
                    url,
                    relation=relation,
                    locator=preset.locator,
                    crawl=False,
                    model_local_ids=(model.local_id,),
                )
            )
            release_id = f"{identity}:{field}:{handle}"
            releases.append(
                ReleaseHint(
                    local_id=f"release:{content_hash(release_id)[:24]}",
                    model_local_id=model.local_id,
                    version=handle_version,
                    revision=revision,
                    identifiers=(Identifier("keras-hub:preset-handle", release_id),),
                    metadata={
                        "repository": self.repository,
                        "revision": revision,
                        "preset_path": preset.path,
                        "preset_collection": preset.collection,
                        "preset_name": preset.name,
                        "handle_field": field,
                        "handle": handle,
                        "url": url,
                    },
                    locator=preset.locator,
                )
            )
        text = _preset_text(preset)
        return SourceRecord(
            source_record_id=f"preset:{content_hash(identity)[:24]}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(code_url),
            title=preset.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "archive_sha256": content_hash(archive),
                "preset_path": preset.path,
                "preset_source_sha256": preset.source_sha256,
                "preset_collection": preset.collection,
                "preset_name": preset.name,
                "metadata": dict(preset.metadata),
                "handles": [{"field": field, "handle": handle} for field, handle in preset.handles],
            },
            text=text,
            identifiers=(model_identifier,),
            links=tuple(dict.fromkeys(links)),
            models=(model,),
            releases=tuple(releases),
        )


def _preset_files(
    archive: bytes,
    *,
    source: str,
    preset_root: str,
    max_file_bytes: int,
) -> dict[str, str]:
    try:
        package = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as error:
        raise ValueError(f"{source}: source archive is not a ZIP file") from error
    marker = f"/{preset_root}/"
    with package:
        files: dict[str, str] = {}
        roots: set[str] = set()
        for info in package.infolist():
            if info.is_dir() or not info.filename.endswith("_presets.py"):
                continue
            path = _archive_path(info.filename, source)
            if marker not in path:
                continue
            root, relative_path = path.split(marker, 1)
            if not root or not relative_path:
                raise ValueError(f"{source}: source archive has an invalid preset member")
            if info.file_size > max_file_bytes:
                raise ValueError(
                    f"{source}: source file {path!r} exceeds {max_file_bytes} bytes"
                )
            source_path = f"{preset_root}/{relative_path}"
            if source_path in files:
                raise ValueError(f"{source}: source archive has duplicate member {source_path!r}")
            roots.add(root)
            files[source_path] = package.read(info).decode("utf-8", errors="strict")
    if len(roots) != 1:
        raise ValueError(f"{source}: source archive must contain one preset root")
    if not files:
        raise ValueError(f"{source}: source archive has no {preset_root} preset files")
    return files


def _parse_presets(
    source: str,
    path: str,
    name: str,
    *,
    allow_dynamic_preset_collections: bool = False,
) -> tuple[_Preset, ...]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{name}: cannot parse {path}: {error.msg}") from error
    result = []
    literal_mappings: dict[str, dict[str, ast.AST]] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        entries = _literal_mapping_entries(statement.value, literal_mappings)
        if entries is None:
            if target.id.endswith("_presets") and not allow_dynamic_preset_collections:
                raise ValueError(f"{name}: {path}:{target.id} is not a literal dictionary")
            continue
        literal_mappings[target.id] = entries
        if not target.id.endswith("_presets"):
            continue
        for preset_name, value_node in entries.items():
            fields = _dict_fields(value_node)
            if not preset_name or fields is None:
                if not allow_dynamic_preset_collections or not preset_name:
                    raise ValueError(
                        f"{name}: {path}:{target.id} has a nonliteral preset declaration"
                    )
                # KerasCV has literal row dictionaries with occasional computed
                # metadata/config fields. Keep only individually literal fields;
                # this cannot associate a handle from a sibling row or execute code.
                fields = _partial_dict_fields(value_node)
                if fields is None:
                    continue
            handles = tuple(
                (field, value)
                for field, value in fields.items()
                if field in {"kaggle_handle", "hf_handle"}
                and isinstance(value, str)
                and value
                and (
                    not allow_dynamic_preset_collections
                    or value.startswith(("kaggle://", "hf://"))
                )
            )
            if not handles:
                continue
            metadata = fields.get("metadata")
            result.append(
                _Preset(
                    name=preset_name,
                    collection=target.id,
                    path=path,
                    source_sha256=content_hash(source.encode()),
                    handles=handles,
                    metadata=metadata if isinstance(metadata, Mapping) else {},
                    locator=f"{path}:{target.id}[{preset_name!r}]",
                )
            )
    return tuple(result)


def _partial_dict_fields(node: ast.AST) -> dict[str, Any] | None:
    """Read literal top-level fields from a dictionary without evaluating expressions."""

    if not isinstance(node, ast.Dict):
        return None
    fields: dict[str, Any] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        key = _literal_string(key_node)
        if not key:
            continue
        value = _literal_value(value_node)
        if value is not _UNSET:
            fields[key] = value
    return fields


def _literal_mapping_entries(
    node: ast.AST,
    bindings: Mapping[str, Mapping[str, ast.AST]],
) -> dict[str, ast.AST] | None:
    """Resolve one source-local literal dict with earlier literal ``**name`` values."""

    if not isinstance(node, ast.Dict):
        return None
    result: dict[str, ast.AST] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        if key_node is None:
            if not isinstance(value_node, ast.Name) or value_node.id not in bindings:
                return None
            result.update(bindings[value_node.id])
            continue
        key = _literal_string(key_node)
        if not key:
            return None
        result[key] = value_node
    return result


def _dict_fields(node: ast.AST) -> dict[str, Any] | None:
    if not isinstance(node, ast.Dict):
        return None
    result = {}
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        key = _literal_string(key_node)
        value = _literal_value(value_node)
        if not key or value is _UNSET:
            return None
        result[key] = value
    return result


_UNSET = object()


def _literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(
        node.value, str | int | float | bool | type(None)
    ):
        return node.value
    if isinstance(node, ast.Dict):
        return _dict_fields(node) or _UNSET
    if isinstance(node, ast.List | ast.Tuple):
        values = [_literal_value(item) for item in node.elts]
        return values if all(value is not _UNSET for value in values) else _UNSET
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_value(node.left)
        right = _literal_value(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    return _UNSET


def _literal_string(node: ast.AST | None) -> str:
    value = _literal_value(node) if node is not None else _UNSET
    return value.strip() if isinstance(value, str) else ""


def _handle_url(handle: str, source: str, locator: str) -> tuple[str, str]:
    if handle.startswith("kaggle://"):
        path = handle.removeprefix("kaggle://").strip("/")
        parts = path.split("/")
        if len(parts) not in {4, 5} or any(not part for part in parts):
            raise ValueError(f"{source}: invalid Kaggle handle at {locator}")
        return (
            canonicalize_url(f"https://www.kaggle.com/models/{quote(path, safe='/')}"),
            "model_artifact",
        )
    if handle.startswith("hf://"):
        path = handle.removeprefix("hf://").strip("/")
        if len(path.split("/")) < 2 or any(not part for part in path.split("/")):
            raise ValueError(f"{source}: invalid Hugging Face handle at {locator}")
        return (
            canonicalize_url(f"https://huggingface.co/{quote(path, safe='/')}"),
            "model_card",
        )
    raise ValueError(f"{source}: unsupported preset handle at {locator}")


def _handle_version(field: str, handle: str) -> str | None:
    """Return a provider version only when the handle declares one explicitly."""
    if field != "kaggle_handle":
        return None
    parts = handle.removeprefix("kaggle://").split("/")
    # KerasHub's documented Kaggle form is
    # {owner}/{model}/keras/{variant}[/{version}].
    return parts[-1] if len(parts) == 5 else None


def _preset_text(preset: _Preset) -> str:
    parts = [preset.name, f"KerasHub preset collection: {preset.collection}"]
    for key in ("description", "path", "params"):
        value = preset.metadata.get(key)
        if value is not None:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


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
