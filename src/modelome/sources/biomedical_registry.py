"""First-party StarDist pretrained model registry source."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY = "stardist/stardist"
_SOURCE_PATH = "stardist/models/__init__.py"
_ARCHIVE_PREFIX = "/stardist/stardist-models/releases/download/"
_MAX_MODELS = 1_000


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StarDistPretrainedRegistrySourceAdapter:
    """Enumerate StarDist's own code-registered pretrained model archives.

    StarDist registers its built-in 2D and 3D pretrained models through literal
    ``register_model`` calls in ``stardist/models/__init__.py``. The source is
    resolved at a Git commit and the declared first-party release archive URL
    and SHA-256 are retained without downloading model weights.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the built-in pretrained model registrations in StarDist's public "
        "Python package. User-trained models and models shared through BioImage.IO "
        "are outside this registry. Archive bytes are not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "stardist-pretrained-models",
        repository: str = _REPOSITORY,
        branch: str = "main",
        source_path: str = _SOURCE_PATH,
        max_source_bytes: int = 2 * 1024 * 1024,
        max_models: int = _MAX_MODELS,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        if repository != _REPOSITORY:
            raise ValueError(f"{self.name}: repository must be {_REPOSITORY!r}")
        if source_path != _SOURCE_PATH:
            raise ValueError(f"{self.name}: source_path must be {_SOURCE_PATH!r}")
        self.repository = repository
        self.branch = _required_text(branch, "branch")
        self.source_path = source_path
        self.max_source_bytes = _positive_int(max_source_bytes, "max_source_bytes")
        self.max_models = _positive_int(max_models, "max_models")
        self.client = client or HttpClient(max_response_bytes=self.max_source_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "stardist-pretrained-registry-v1",
                "repository": repository,
                "branch": self.branch,
                "source_path": source_path,
                "max_source_bytes": self.max_source_bytes,
                "max_models": self.max_models,
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
        commit_response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(
                f"{self.name}: commit endpoint returned HTTP {commit_response.status}"
            )
        commit = commit_response.json()
        revision = _text(commit.get("sha")) if isinstance(commit, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            return SourcePage(
                records=(),
                next_state={**state, "checked_at": checked_at},
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        source_url = self._raw_url(revision)
        response: HttpResponse = self.client.get(
            source_url, headers={"Accept": "text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source file returned HTTP {response.status}")
        if len(response.body) > self.max_source_bytes:
            raise ValueError(
                f"{self.name}: source file exceeds {self.max_source_bytes} bytes"
            )
        source = response.text()
        entries = self._registrations(source)
        if not entries:
            raise ValueError(f"{self.name}: no built-in pretrained registrations found")
        if len(entries) > self.max_models:
            raise ValueError(f"{self.name}: source exceeds {self.max_models} models")

        models = []
        releases = []
        links = []
        lines = []
        for entry in entries:
            local_id = f"stardist:{entry['name']}#model"
            models.append(
                ModelHint(
                    local_id=local_id,
                    name=entry["name"],
                    identifiers=(Identifier("stardist:model", entry["name"]),),
                    status=ModelStatus.RELEASED,
                    locator=f"{self.source_path}:{entry['line']}",
                )
            )
            releases.append(
                ReleaseHint(
                    local_id=f"{local_id}:release:{entry['checksum']}",
                    model_local_id=local_id,
                    version=entry["version"],
                    revision=entry["checksum"],
                    identifiers=(
                        Identifier(
                            "stardist:model-release",
                            f"{entry['name']}@{entry['checksum']}",
                        ),
                    ),
                    metadata={
                        "archive_url": entry["url"],
                        "archive_sha256": entry["checksum"],
                        "model_class": entry["model_class"],
                    },
                    locator=f"{self.source_path}:{entry['line']}",
                )
            )
            links.append(
                Link(
                    entry["url"],
                    relation="weights",
                    locator=f"{self.source_path}:{entry['line']}",
                    crawl=False,
                    model_local_ids=(local_id,),
                )
            )
            lines.append(f"{entry['name']} ({entry['model_class']})")

        next_state = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "model_count": len(entries),
            "source_sha256": content_hash(response.body),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        record = SourceRecord(
            source_record_id=f"{self.name}:registry",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self._blob_url(revision),
            title="StarDist built-in pretrained model registry",
            raw={
                "repository": self.repository,
                "repository_revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(response.body),
                "response_url": response.url or source_url,
                "registrations": entries,
            },
            text="\n".join(lines),
            identifiers=(Identifier("stardist:model-registry", revision),),
            links=tuple(links),
            models=tuple(models),
            releases=tuple(releases),
        )
        return SourcePage(
            records=(record,),
            next_state=next_state,
            complete=True,
            upstream_count=len(entries),
            authoritative_snapshot=True,
        )

    def _registrations(self, source: str) -> list[dict[str, Any]]:
        try:
            tree = ast.parse(source, filename=self.source_path)
        except SyntaxError as error:
            raise ValueError(f"{self.name}: source file is not valid Python") from error
        entries: dict[str, dict[str, Any]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "register_model" or len(node.args) < 4:
                continue
            if not isinstance(node.args[0], ast.Name) or node.args[0].id not in {
                "StarDist2D",
                "StarDist3D",
            }:
                continue
            values = [_literal_string(item) for item in node.args[1:4]]
            name, url, checksum = values
            if not name or not url or not checksum:
                raise ValueError(
                    f"{self.name}: registration at line {node.lineno} must use literal strings"
                )
            archive = self._archive(url)
            checksum = checksum.casefold()
            if not _SHA256.fullmatch(checksum):
                raise ValueError(
                    f"{self.name}: registration {name!r} has an invalid SHA-256"
                )
            if name in entries:
                raise ValueError(f"{self.name}: duplicate model registration {name!r}")
            entries[name] = {
                "name": name,
                "model_class": node.args[0].id,
                "url": archive,
                "version": self._archive_version(archive),
                "checksum": checksum,
                "line": node.lineno,
            }
        return [entries[name] for name in sorted(entries)]

    def _archive(self, value: str) -> str:
        url = canonicalize_url(value)
        parts = urlsplit(url)
        archive_parts = parts.path[len(_ARCHIVE_PREFIX) :].split("/")
        if (
            parts.scheme != "https"
            or parts.hostname != "github.com"
            or not parts.path.startswith(_ARCHIVE_PREFIX)
            or len(archive_parts) != 2
            or any(part in {"", ".", ".."} for part in archive_parts)
            or not parts.path.endswith(".zip")
            or parts.query
            or parts.fragment
        ):
            raise ValueError(f"{self.name}: model archive is outside the StarDist release tree")
        return url

    def _archive_version(self, url: str) -> str:
        return urlsplit(url).path[len(_ARCHIVE_PREFIX) :].split("/", 1)[0]

    def _raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(self.source_path, safe='/')}"
        )

    def _blob_url(self, revision: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )


def _literal_string(node: ast.AST) -> str:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return ""
    return value.strip() if isinstance(value, str) else ""


def _required_text(value: Any, label: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if not result:
        raise ValueError(f"{label} must be a non-empty string")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _header(headers: Mapping[str, Any], name: str) -> str:
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return _text(value)
    return ""


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = ["StarDistPretrainedRegistrySourceAdapter"]
