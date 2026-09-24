"""Pinned static ingestion of Detectron2's official pretrained-model map."""

from __future__ import annotations

import ast
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
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SAFE_CONFIG = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+$")
_WEB_PREFIX = re.compile(r"^https?://[^/]+(?:/.*)?$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _ModelEntry:
    config_path: str
    weight_suffix: str
    locator: str


class Detectron2ModelZooSourceAdapter:
    """Enumerate each official Detectron2 config/checkpoint declaration.

    The adapter reads only the upstream source file containing
    ``_ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX``. Python is parsed with ``ast``
    and never imported; a drift from the static literal map fails closed.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers config/checkpoint pairs declared in Detectron2's public static "
        "model-zoo source at one commit. It does not enumerate external projects, "
        "verify current weight availability, infer papers, or download checkpoints."
    )

    def __init__(
        self,
        *,
        name: str = "detectron2-model-zoo",
        repository: str = "facebookresearch/detectron2",
        branch: str = "main",
        source_path: str = "detectron2/model_zoo/model_zoo.py",
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
                "adapter": "detectron2-model-zoo-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
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

    def config_url(self, revision: str, config_path: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/configs/"
            f"{quote(config_path, safe='/')}.yaml"
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
            self.raw_url(revision),
            headers={"Accept": "text/plain,text/x-python"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source file returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source file exceeds {self.max_response_bytes} bytes")
        prefix, entries = _parse_source(response.text(), self.name, self.source_path)
        if len(entries) > self.max_entries:
            raise ValueError(f"{self.name}: model map exceeds {self.max_entries} entries")
        records = tuple(self._record(entry, prefix, revision, response.body) for entry in entries)
        if not records:
            raise ValueError(f"{self.name}: model map contains no entries")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
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

    def _record(
        self,
        entry: _ModelEntry,
        prefix: str,
        revision: str,
        source: bytes,
    ) -> SourceRecord:
        model_identifier = Identifier("detectron2:model", entry.config_path)
        model = ModelHint(
            local_id=f"model:{entry.config_path}",
            name=entry.config_path.rsplit("/", 1)[-1],
            aliases=(entry.config_path,),
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=entry.locator,
        )
        source_url = self.blob_url(revision)
        config_url = self.config_url(revision, entry.config_path)
        weight_url = f"{prefix.rstrip('/')}/{entry.config_path}/{entry.weight_suffix.lstrip('/')}"
        release = ReleaseHint(
            local_id=f"release:{entry.config_path}",
            model_local_id=model.local_id,
            revision=revision,
            identifiers=(Identifier("detectron2:model-zoo-config", entry.config_path),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "config_path": entry.config_path,
                "weight_suffix": entry.weight_suffix,
            },
            locator=entry.locator,
        )
        return SourceRecord(
            source_record_id=f"model:{entry.config_path}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(source_url),
            title=model.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "config_path": entry.config_path,
                "config_url": config_url,
                "weight_suffix": entry.weight_suffix,
                "weight_url": weight_url,
            },
            text=f"Detectron2 model-zoo config: {entry.config_path}",
            identifiers=(model_identifier,),
            links=(
                Link(source_url, relation="model_card", locator=entry.locator, crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
                Link(config_url, relation="model_config", locator=entry.locator, crawl=False),
                Link(weight_url, relation="weights", locator=entry.locator, crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_source(source: str, name: str, path: str) -> tuple[str, tuple[_ModelEntry, ...]]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{name}: cannot parse source file: {error.msg}") from error
    prefix = ""
    mapping: ast.Dict | None = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "_ModelZooUrls":
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id == "S3_PREFIX":
                prefix = _literal_string(statement.value)
            elif target.id == "CONFIG_PATH_TO_URL_SUFFIX" and isinstance(statement.value, ast.Dict):
                mapping = statement.value
    if not _WEB_PREFIX.fullmatch(prefix):
        raise ValueError(f"{name}: static model map has no valid S3_PREFIX")
    if mapping is None:
        raise ValueError(f"{name}: static model map is absent or not a literal dictionary")
    entries = []
    seen = set()
    for key, value in zip(mapping.keys, mapping.values, strict=True):
        config_path = _literal_string(key)
        suffix = _literal_string(value)
        if not _SAFE_CONFIG.fullmatch(config_path):
            raise ValueError(f"{name}: invalid static config path {config_path!r}")
        if (
            not suffix
            or suffix.startswith("/")
            or "\\" in suffix
            or ".." in suffix.split("/")
        ):
            raise ValueError(f"{name}: invalid weight suffix for {config_path!r}")
        if config_path in seen:
            raise ValueError(f"{name}: duplicate static config path {config_path!r}")
        seen.add(config_path)
        entries.append(
            _ModelEntry(
                config_path=config_path,
                weight_suffix=suffix,
                locator=f"{path}:CONFIG_PATH_TO_URL_SUFFIX[{config_path!r}]",
            )
        )
    return prefix, tuple(entries)


def _literal_string(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    return ""


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
