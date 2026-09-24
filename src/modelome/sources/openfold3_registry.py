"""First-party OpenFold3 parameter names and S3 object identities."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient
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
_REPOSITORY = "aqlaboratory/openfold-3"
_REGISTRY_PATH = "openfold3/entry_points/parameters.py"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_FILENAME = re.compile(r"^[A-Za-z0-9_.-]+\.pt$")
_MAX_ENTRIES = 100


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OpenFold3ParameterRegistryAdapter:
    """Parse the literal checkpoint registry in the OpenFold3 source repository.

    The adapter records object names in the public S3 bucket; it does not list
    the bucket or fetch parameter bytes. Deprecated entries are retained when
    the official registry retains them, with their exact compatibility text.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers parameter entries in OpenFold3's literal checkpoint registry. "
        "The source code currently marks legacy entries unsupported for current "
        "downloads; no weights are fetched and no bucket-wide inventory is inferred."
    )

    def __init__(
        self,
        *,
        name: str = "openfold3-parameter-registry",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_source_bytes: int = 512 * 1024,
        max_entries: int = _MAX_ENTRIES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or max_source_bytes <= 0 or max_entries <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_source_bytes = max_source_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_source_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openfold3-parameter-registry-v1",
                "repository": repository,
                "branch": branch,
                "registry_path": _REGISTRY_PATH,
                "max_source_bytes": max_source_bytes,
                "max_entries": max_entries,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                (), {**state, "checked_at": checked}, True,
                upstream_count=state.get("parameter_count"),
            )
        source_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{revision}/{_REGISTRY_PATH}"
        )
        response = self.client.get(source_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_source_bytes:
            raise ValueError(f"{self.name}: registry exceeds response limit")
        entries, bucket, key_prefix = _parse_registry(response.text(), self.max_entries)
        record = self._record(entries, bucket, key_prefix, source_url, revision)
        return SourcePage(
            (record,),
            {"completed_revision": revision, "checked_at": checked,
             "parameter_count": len(entries)},
            True,
            upstream_count=len(entries),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        entries: list[dict[str, Any]],
        bucket: str,
        key_prefix: str,
        source_url: str,
        revision: str,
    ) -> SourceRecord:
        namespace = "openfold3:parameter"
        models: list[ModelHint] = []
        releases: list[ReleaseHint] = []
        links: list[Link] = []
        raw_entries: list[dict[str, str]] = []
        for item in entries:
            name, filename = item["name"], item["file_name"]
            local_id = f"model:{name}"
            s3_uri = f"s3://{bucket}/{key_prefix}{filename}"
            models.append(
                ModelHint(
                    local_id,
                    f"OpenFold3 {name}",
                    identifiers=(Identifier(namespace, name),),
                    aliases=(filename,),
                    status=ModelStatus.RELEASED,
                )
            )
            releases.append(
                ReleaseHint(
                    f"release:{name}", local_id,
                    identifiers=(Identifier(f"{namespace}:release", name),),
                    metadata={
                        "filename": filename,
                        "s3_uri": s3_uri,
                        "version_compatibility": item["version_compatibility"],
                        "download_supported_by_current_registry": not item["legacy"],
                    },
                    locator=f"OPENFOLD_MODEL_CHECKPOINT_REGISTRY[{name!r}]",
                )
            )
            links.append(
                Link(s3_uri, "weights", crawl=False, model_local_ids=(local_id,))
            )
            raw_entries.append({**item, "s3_uri": s3_uri})
        page_url = f"{self.repository_url}/blob/{revision}/{_REGISTRY_PATH}"
        return SourceRecord(
            source_record_id=f"{self.name}:{revision}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(page_url),
            title="OpenFold3 parameter registry",
            raw={"repository": self.repository, "revision": revision,
                 "source_url": source_url, "parameters": raw_entries},
            text="\n".join(
                f"{item['name']} {item['file_name']} {item['version_compatibility']}"
                for item in entries
            ),
            identifiers=(Identifier("openfold3:registry-revision", revision),),
            links=(Link(page_url, "model_card", crawl=False), *links),
            models=tuple(models),
            releases=tuple(releases),
        )


def _parse_registry(source: str, limit: int) -> tuple[list[dict[str, Any]], str, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("OpenFold3 parameter registry is invalid Python") from exc
    registry: ast.Dict | None = None
    legacy_names: set[str] | None = None
    bucket: str | None = None
    key_prefix: str | None = None
    saw_registry = False
    saw_bucket = False
    saw_key_prefix = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "OPENFOLD_MODEL_CHECKPOINT_REGISTRY"
            for target in node.targets
        ):
            if saw_registry:
                raise ValueError("OpenFold3 parameter registry is declared more than once")
            saw_registry = True
            if not isinstance(node.value, ast.Dict):
                raise ValueError("OpenFold3 parameter registry must be a literal dictionary")
            registry = node.value
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "LEGACY_CHECKPOINTS"
            for target in node.targets
        ):
            if saw_bucket:
                raise ValueError("OpenFold3 bucket is declared more than once")
            saw_bucket = True
            try:
                raw_legacy = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise ValueError("OpenFold3 legacy names must be a literal list") from exc
            if not isinstance(raw_legacy, list) or not all(
                isinstance(item, str) and _NAME.fullmatch(item) for item in raw_legacy
            ):
                raise ValueError("OpenFold3 legacy names must be valid literal names")
            if legacy_names is not None:
                raise ValueError("OpenFold3 legacy list is declared more than once")
            legacy_names = set(raw_legacy)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "OPENFOLD_BUCKET"
            for target in node.targets
        ):
            try:
                bucket = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise ValueError("OpenFold3 bucket must be a literal string") from exc
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "checkpoint_s3_key"
            for target in node.targets
        ):
            if saw_key_prefix:
                raise ValueError("OpenFold3 S3 key is declared more than once")
            saw_key_prefix = True
            key_prefix = _s3_key_prefix(node.value)
    if registry is None:
        raise ValueError("OpenFold3 parameter registry literal was not found")
    if legacy_names is None:
        raise ValueError("OpenFold3 legacy checkpoint list was not found")
    if not isinstance(bucket, str) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", bucket):
        raise ValueError("OpenFold3 bucket is missing or invalid")
    if key_prefix is None:
        raise ValueError("OpenFold3 S3 key prefix was not found")
    if len(registry.keys) > limit:
        raise ValueError("OpenFold3 parameter registry exceeds entry limit")
    entries: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for key_node, value_node in zip(registry.keys, registry.values, strict=True):
        if key_node is None:
            raise ValueError("OpenFold3 registry unpacking is not supported")
        try:
            name = ast.literal_eval(key_node)
        except (ValueError, TypeError) as exc:
            raise ValueError("OpenFold3 registry keys must be literals") from exc
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("OpenFold3 registry contains an invalid parameter name")
        if name in seen_names:
            raise ValueError("OpenFold3 registry contains a duplicate parameter name")
        seen_names.add(name)
        if (
            not isinstance(value_node, ast.Call)
            or not isinstance(value_node.func, ast.Name)
            or value_node.func.id != "CheckpointEntry"
        ):
            raise ValueError("OpenFold3 registry entry must be a CheckpointEntry call")
        fields: dict[str, Any] = {}
        if len(value_node.args) > 2:
            raise ValueError("OpenFold3 registry entry has unexpected positional fields")
        if value_node.args:
            try:
                fields["file_name"] = ast.literal_eval(value_node.args[0])
            except (ValueError, TypeError) as exc:
                raise ValueError("OpenFold3 registry entry lacks a literal filename") from exc
        if len(value_node.args) == 2:
            try:
                fields["version_compatibility"] = ast.literal_eval(value_node.args[1])
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "OpenFold3 registry entry has a nonliteral compatibility"
                ) from exc
        for keyword in value_node.keywords:
            if keyword.arg not in {"file_name", "version_compatibility"}:
                raise ValueError("OpenFold3 registry entry has an unsupported field")
            if keyword.arg in fields:
                raise ValueError("OpenFold3 registry entry repeats a field")
            try:
                fields[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError) as exc:
                field = "literal filename" if keyword.arg == "file_name" else "literal field"
                raise ValueError(f"OpenFold3 registry entry lacks a {field}") from exc
        if "file_name" not in fields:
            raise ValueError("OpenFold3 registry entry lacks a literal filename")
        filename = fields["file_name"]
        compatibility = fields.get("version_compatibility")
        if compatibility is None:
            compatibility = "unspecified"
        if not isinstance(filename, str) or not _FILENAME.fullmatch(filename):
            raise ValueError("OpenFold3 registry contains an invalid checkpoint filename")
        if not isinstance(compatibility, str):
            raise ValueError("OpenFold3 registry compatibility must be literal text")
        entries.append({"name": name, "file_name": filename,
                        "version_compatibility": compatibility,
                        "legacy": name in legacy_names})
    if not entries:
        raise ValueError("OpenFold3 parameter registry is empty")
    if legacy_names.difference(seen_names):
        raise ValueError("OpenFold3 legacy list refers to an unknown parameter name")
    return entries, bucket, key_prefix


def _s3_key_prefix(value: ast.expr) -> str | None:
    """Extract only the official filename interpolation pattern."""
    if not isinstance(value, ast.JoinedStr) or len(value.values) != 2:
        return None
    prefix, filename = value.values
    if not isinstance(prefix, ast.Constant) or not isinstance(prefix.value, str):
        return None
    if not isinstance(filename, ast.FormattedValue) or not isinstance(filename.value, ast.Name):
        return None
    if filename.value.id != "checkpoint_file_name":
        return None
    if filename.conversion != -1 or filename.format_spec is not None:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/", prefix.value):
        return None
    return prefix.value


__all__ = ["OpenFold3ParameterRegistryAdapter"]
