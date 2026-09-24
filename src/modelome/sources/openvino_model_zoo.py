"""Version-pinned enumeration of Intel's OpenVINO Open Model Zoo."""

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
    ModelRelationHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, identifier_from_url

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MODEL_MANIFEST = re.compile(
    r"^models/(?P<tier>[^/]+)/(?P<model>.+)/model\.ya?ml$",
    re.IGNORECASE,
)
_WEIGHT_SUFFIXES = frozenset(
    {
        ".bin",
        ".blob",
        ".caffemodel",
        ".ckpt",
        ".h5",
        ".onnx",
        ".pb",
        ".pth",
        ".pt",
        ".safetensors",
        ".tflite",
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _ModelFile:
    name: str | None
    source: str | None
    original_source: str | None
    checksum: str | None
    size: int | None


@dataclass(frozen=True, slots=True)
class _Manifest:
    path: str
    tier: str
    model_id: str
    description: str
    task_type: str | None
    framework: str | None
    license: str | None
    files: tuple[_ModelFile, ...]


class OpenVinoModelZooSourceAdapter:
    """Enumerate every source-declared OpenVINO Model Zoo manifest.

    The provider's public repository is resolved at an immutable Git commit. A
    recursive tree enumerates every ``models/*/*/model.yml`` manifest, then each
    small manifest contributes one model/release observation with its declared
    file, checksum, source, license, and descriptive links. File URLs are kept
    as reference-only resources; this adapter never transfers model bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers model manifests present in the public OpenVINO Open Model Zoo "
        "repository at one observed commit. The maintenance-mode repository does "
        "not establish coverage of models published outside it; artifact bytes are "
        "referenced but never downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "openvino-model-zoo",
        repository: str = "openvinotoolkit/open_model_zoo",
        branch: str = "master",
        max_tree_bytes: int = 16 * 1024 * 1024,
        max_manifest_bytes: int = 1 * 1024 * 1024,
        max_manifests: int = 10_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.max_tree_bytes = _positive_int(max_tree_bytes, "max_tree_bytes", self.name)
        self.max_manifest_bytes = _positive_int(
            max_manifest_bytes,
            "max_manifest_bytes",
            self.name,
        )
        self.max_manifests = _positive_int(max_manifests, "max_manifests", self.name)
        self.client = client or HttpClient(
            max_response_bytes=max(self.max_tree_bytes, self.max_manifest_bytes)
        )
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openvino-model-zoo-v1",
                "repository": self.repository,
                "branch": self.branch,
                "max_tree_bytes": self.max_tree_bytes,
                "max_manifest_bytes": self.max_manifest_bytes,
                "max_manifests": self.max_manifests,
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
            self._tree_url(revision),
            headers={"Accept": "application/vnd.github+json"},
        )
        if tree_response.status != 200:
            raise ValueError(f"{self.name}: repository tree returned HTTP {tree_response.status}")
        if len(tree_response.body) > self.max_tree_bytes:
            raise ValueError(f"{self.name}: repository tree exceeds {self.max_tree_bytes} bytes")
        paths = self._manifest_paths(tree_response.json())

        records = []
        manifest_hashes: dict[str, str] = {}
        tree_paths = set(paths[1])
        for path in paths[0]:
            response = self.client.get(
                self._raw_url(revision, path),
                headers={"Accept": "text/yaml,text/plain"},
            )
            if response.status != 200:
                raise ValueError(f"{self.name}: manifest {path} returned HTTP {response.status}")
            if len(response.body) > self.max_manifest_bytes:
                raise ValueError(
                    f"{self.name}: manifest {path} exceeds {self.max_manifest_bytes} bytes"
                )
            manifest_hash = content_hash(response.body)
            manifest_hashes[path] = manifest_hash
            manifest = _parse_manifest(path, response.text())
            records.append(
                self._record(
                    manifest,
                    revision=revision,
                    manifest_hash=manifest_hash,
                    readme_exists=_readme_path(path) in tree_paths,
                )
            )

        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "tree_url": self._tree_url(revision),
            "tree_sha256": content_hash(tree_response.body),
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

    def _tree_url(self, revision: str) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/git/trees/"
            f"{quote(revision, safe='')}?recursive=1"
        )

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

    def _manifest_paths(self, payload: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: repository tree is not a JSON object")
        if payload.get("truncated") is True:
            raise ValueError(f"{self.name}: recursive repository tree is truncated")
        tree = payload.get("tree")
        if not isinstance(tree, list):
            raise ValueError(f"{self.name}: repository tree lacks a tree list")
        all_paths = []
        manifests = []
        for item in tree:
            if not isinstance(item, Mapping) or item.get("type") != "blob":
                continue
            path = _safe_path(item.get("path"), self.name, "repository tree path")
            all_paths.append(path)
            if _MODEL_MANIFEST.fullmatch(path):
                manifests.append(path)
        if not manifests:
            raise ValueError(f"{self.name}: repository tree contains no model manifests")
        if len(manifests) > self.max_manifests:
            raise ValueError(
                f"{self.name}: repository exceeds {self.max_manifests} model manifests"
            )
        return tuple(sorted(manifests)), tuple(sorted(all_paths))

    def _record(
        self,
        manifest: _Manifest,
        *,
        revision: str,
        manifest_hash: str,
        readme_exists: bool,
    ) -> SourceRecord:
        manifest_url = self._blob_url(revision, manifest.path)
        model_path = manifest.path.rsplit("/", 1)[0]
        model_url = (
            f"{self.repository_url}/tree/{quote(revision, safe='')}/"
            f"{quote(model_path, safe='/')}"
        )
        locator = f"{manifest.path}:model"
        model_identifier = Identifier(
            "openvino:model",
            f"{manifest.tier}/{manifest.model_id}",
        )
        model = ModelHint(
            local_id=f"{model_path}#model",
            name=manifest.model_id.rsplit("/", 1)[-1],
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        links = [
            Link(model_url, relation="model_card", locator=locator),
            Link(manifest_url, relation="metadata", locator=locator),
            Link(self.repository_url, relation="source_repository", locator=locator),
        ]
        if readme_exists:
            links.append(
                Link(
                    self._blob_url(revision, _readme_path(manifest.path)),
                    relation="documentation",
                    locator=locator,
                )
            )
        if manifest.license and _is_web_url(manifest.license):
            links.append(Link(manifest.license, relation="license", locator="license", crawl=False))

        relation_hints = []
        file_metadata = []
        for ordinal, item in enumerate(manifest.files):
            file_locator = f"files[{ordinal}]"
            payload = {
                "name": item.name,
                "source": item.source,
                "original_source": item.original_source,
                "checksum": item.checksum,
                "size": item.size,
            }
            file_metadata.append(payload)
            if item.source:
                relation = (
                    "weights"
                    if _is_weight_file(item.name, item.source)
                    else "model_artifact"
                )
                links.append(
                    Link(
                        item.source,
                        relation=relation,
                        locator=file_locator,
                        crawl=False,
                    )
                )
            if item.original_source:
                links.append(
                    Link(
                        item.original_source,
                        relation="original_model_source",
                        locator=f"{file_locator}.original_source",
                        crawl=False,
                    )
                )
                identifier = identifier_from_url(item.original_source)
                if identifier is not None:
                    relation_hints.append(
                        ModelRelationHint(
                            subject_local_id=model.local_id,
                            predicate="converted_from",
                            target=ModelHint(
                                local_id=f"original:{identifier.namespace}:{identifier.value}",
                                name=identifier.value,
                                identifiers=(identifier,),
                            ),
                            locator=f"{file_locator}.original_source",
                        )
                    )

        release = ReleaseHint(
            local_id=f"{model_path}#release",
            model_local_id=model.local_id,
            revision=revision,
            identifiers=(
                Identifier("openvino:model-manifest", model_path),
                Identifier("github:commit", f"{self.repository}@{revision}"),
            ),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "manifest_path": manifest.path,
                "manifest_sha256": manifest_hash,
                "tier": manifest.tier,
                "task_type": manifest.task_type,
                "framework": manifest.framework,
                "license": manifest.license,
                "files": file_metadata,
            },
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"manifest:{model_path}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(model_url),
            title=model.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "manifest_path": manifest.path,
                "manifest_sha256": manifest_hash,
                "tier": manifest.tier,
                "model_id": manifest.model_id,
                "task_type": manifest.task_type,
                "framework": manifest.framework,
                "license": manifest.license,
                "files": file_metadata,
            },
            text="\n".join(
                item
                for item in (
                    model.name,
                    manifest.task_type,
                    manifest.framework,
                    manifest.description,
                )
                if item
            ),
            identifiers=(Identifier("openvino:model-manifest", model_path),),
            links=tuple(_unique_links(links)),
            models=(model,),
            model_relations=tuple(_unique_relations(relation_hints)),
            releases=(release,),
        )


def _parse_manifest(path: str, text: str) -> _Manifest:
    match = _MODEL_MANIFEST.fullmatch(path)
    if match is None:  # pragma: no cover - caller owns this invariant
        raise ValueError(f"invalid OpenVINO manifest path: {path}")
    values: dict[str, str] = {}
    description_lines: list[str] = []
    files: list[dict[str, str]] = []
    description_block = False
    files_block = False
    current_file: dict[str, str] | None = None

    for line in text.splitlines():
        top = re.match(r"^(?P<key>[A-Za-z][A-Za-z0-9_]*):(?:[ \t]*(?P<value>.*))?$", line)
        if top is not None:
            key = top.group("key")
            value = _yaml_scalar(top.group("value") or "")
            description_block = key == "description" and value in {"|-", ">-", "|", ">"}
            files_block = key == "files" and not value
            current_file = None
            if not description_block and not files_block and value:
                values[key] = value
            continue
        if description_block and line[:1].isspace():
            description_lines.append(line.strip())
            continue
        if files_block:
            item = re.match(
                r"^[ \t]+-[ \t]+(?P<key>[A-Za-z][A-Za-z0-9_]*):[ \t]*(?P<value>.*)$",
                line,
            )
            if item is not None:
                current_file = {item.group("key"): _yaml_scalar(item.group("value"))}
                files.append(current_file)
                continue
            field = re.match(r"^[ \t]{4,}(?P<key>[A-Za-z][A-Za-z0-9_]*):[ \t]*(?P<value>.*)$", line)
            if current_file is not None and field is not None:
                current_file[field.group("key")] = _yaml_scalar(field.group("value"))

    description = " ".join(part for part in description_lines if part)
    if not description:
        description = values.get("description", "")
    parsed_files = tuple(
        _ModelFile(
            name=_text(row.get("name")) or None,
            source=_web_url_or_none(row.get("source")),
            original_source=_web_url_or_none(row.get("original_source")),
            checksum=_text(row.get("checksum")) or None,
            size=_nonnegative_int_or_none(row.get("size")),
        )
        for row in files
    )
    return _Manifest(
        path=path,
        tier=match.group("tier"),
        model_id=match.group("model"),
        description=description,
        task_type=_text(values.get("task_type")) or None,
        framework=_text(values.get("framework")) or None,
        license=_text(values.get("license")) or None,
        files=parsed_files,
    )


def _readme_path(manifest_path: str) -> str:
    return manifest_path.rsplit("/", 1)[0] + "/README.md"


def _is_weight_file(name: str | None, url: str) -> bool:
    candidate = (name or url.rsplit("/", 1)[-1]).casefold()
    return any(candidate.endswith(suffix) for suffix in _WEIGHT_SUFFIXES)


def _unique_links(links: list[Link]) -> tuple[Link, ...]:
    return tuple(dict.fromkeys(links))


def _unique_relations(relations: list[ModelRelationHint]) -> tuple[ModelRelationHint, ...]:
    return tuple(dict.fromkeys(relations))


def _yaml_scalar(value: str) -> str:
    candidate = value.strip()
    if len(candidate) >= 2 and candidate[:1] in {"'", '"'} and candidate[-1:] == candidate[:1]:
        return candidate[1:-1]
    return candidate


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _safe_path(value: Any, source: str, field: str) -> str:
    path = _required_text(value, field)
    invalid_parts = any(part in {"", ".", ".."} for part in path.split("/"))
    if path.startswith("/") or "\\" in path or invalid_parts:
        raise ValueError(f"{source}: {field} is not a safe relative path")
    return path


def _web_url_or_none(value: Any) -> str | None:
    candidate = _text(value)
    if not candidate:
        return None
    return canonicalize_url(candidate) if _is_web_url(candidate) else None


def _is_web_url(value: str) -> bool:
    return value.casefold().startswith(("https://", "http://"))


def _positive_int(value: Any, field: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {field} must be positive")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {field} must be positive") from error
    if result < 1:
        raise ValueError(f"{source}: {field} must be positive")
    return result


def _nonnegative_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("model_count must be non-negative")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("model_count must be non-negative") from error
    if result < 0:
        raise ValueError("model_count must be non-negative")
    return result


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not (result := value.strip()):
        raise ValueError(f"{field} must be non-empty text")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.casefold()
    for key, value in headers.items():
        if key.casefold() == lowered and value.strip():
            return value.strip()
    return None


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["OpenVinoModelZooSourceAdapter"]
