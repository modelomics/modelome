from __future__ import annotations

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
_BUNDLE_KEY = re.compile(
    r"^(?P<model>[A-Za-z0-9][A-Za-z0-9._-]*?)_v"
    r"(?P<version>[0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9._+-]*)?)$"
)
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MAX_ENTRIES = 10_000


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MonaiModelZooSourceAdapter:
    """Enumerate every released bundle in MONAI's public model registry.

    The Project MONAI model-zoo repository publishes ``models/model_info.json``
    as the authoritative, machine-readable mapping from versioned bundle names
    to immutable archive URLs and checksums.  The registry is resolved at one
    public Git commit, so the adapter can retain a release identity and archive
    checksum without downloading bundle or weight bytes.

    BioImage.IO and MONAI have overlapping application areas but distinct
    publishing ecosystems.  A MONAI bundle is emitted as release evidence only
    when it appears in this registry; model-family, task, or name filtering is
    deliberately absent.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers versioned bundles present in Project MONAI's public "
        "models/model_info.json registry. Draft, private, removed, or "
        "independently distributed MONAI bundles are outside this source. "
        "Archive bytes and their embedded licenses are not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "monai-model-zoo",
        repository: str = "Project-MONAI/model-zoo",
        branch: str = "dev",
        registry_path: str = "models/model_info.json",
        max_registry_bytes: int = 4 * 1024 * 1024,
        max_entries: int = _MAX_ENTRIES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.registry_path = _registry_path(registry_path)
        self.max_registry_bytes = _positive_int(
            max_registry_bytes, "max_registry_bytes", self.name
        )
        self.max_entries = _bounded_positive_int(max_entries, "max_entries", self.name)
        self.client = client or HttpClient(max_response_bytes=self.max_registry_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "monai-model-zoo-v1",
                "repository": self.repository,
                "branch": self.branch,
                "registry_path": self.registry_path,
                "max_registry_bytes": self.max_registry_bytes,
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
                upstream_count=_nonnegative_int(state.get("record_count")),
            )

        registry_url = self._raw_url(revision, self.registry_path)
        response: HttpResponse = self.client.get(
            registry_url, headers={"Accept": "application/json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_registry_bytes:
            raise ValueError(
                f"{self.name}: registry exceeds {self.max_registry_bytes} bytes"
            )
        controls = self._controls(response.json())
        current_controls = {control.release_key: control for control in controls}
        known_controls = self._checkpoint_controls(state.get("known_records"))

        records = [
            self._record(control, revision=revision, registry_url=registry_url)
            for control in controls
        ]
        records.extend(
            self._tombstone(control, revision=revision, registry_url=registry_url)
            for release_key, control in sorted(known_controls.items())
            if release_key not in current_controls
        )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "registry_url": registry_url,
            "registry_sha256": content_hash(response.body),
            "record_count": len(controls),
            "known_records": {
                release_key: control.checkpoint_value()
                for release_key, control in current_controls.items()
            },
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=True,
            upstream_count=len(controls),
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

    def _model_page_url(self, revision: str, model_id: str) -> str:
        return canonicalize_url(
            f"{self.repository_url}/tree/{quote(revision, safe='')}/models/"
            f"{quote(model_id, safe='')}"
        )

    def _controls(self, payload: Any) -> tuple[_BundleControl, ...]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: registry is not a JSON object")
        if not payload:
            raise ValueError(f"{self.name}: registry contains no bundles")
        if len(payload) > self.max_entries:
            raise ValueError(f"{self.name}: registry exceeds {self.max_entries} bundles")
        controls = []
        for release_key, raw in sorted(payload.items()):
            controls.append(self._control(release_key, raw))
        return tuple(controls)

    def _control(self, release_key: Any, raw: Any) -> _BundleControl:
        key = _required_text(release_key, "registry bundle key")
        match = _BUNDLE_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"{self.name}: bundle key {key!r} lacks a version suffix")
        if not isinstance(raw, Mapping):
            raise ValueError(f"{self.name}: bundle {key!r} is not a JSON object")
        source_url = _web_url(raw.get("source"), self.name, f"bundle {key!r} source")
        checksum = _text(raw.get("checksum")).casefold() or None
        if checksum is not None and not _SHA1.fullmatch(checksum):
            raise ValueError(f"{self.name}: bundle {key!r} checksum is not a SHA-1")
        return _BundleControl(
            release_key=key,
            model_id=match.group("model"),
            version=match.group("version"),
            source_url=source_url,
            checksum=checksum,
        )

    def _checkpoint_controls(self, value: Any) -> dict[str, _BundleControl]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{self.name}: checkpoint known_records is not a mapping")
        result = {}
        for release_key, raw in value.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"{self.name}: checkpoint bundle {release_key!r} is invalid")
            result[_required_text(release_key, "checkpoint bundle key")] = self._control(
                release_key,
                {
                    "source": raw.get("source_url"),
                    "checksum": raw.get("checksum"),
                },
            )
        return result

    def _record(
        self, control: _BundleControl, *, revision: str, registry_url: str
    ) -> SourceRecord:
        model_page_url = self._model_page_url(revision, control.model_id)
        model_identifier = Identifier("monai:model", control.model_id)
        release_identifier = Identifier("monai:bundle", control.release_key)
        archive_identifier = Identifier("monai:bundle-archive", control.source_url)
        model = ModelHint(
            local_id=f"{control.release_key}#model",
            name=control.model_id,
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=f"$.{control.release_key}",
        )
        release = ReleaseHint(
            local_id=f"{control.release_key}#release",
            model_local_id=model.local_id,
            version=control.version,
            revision=revision,
            identifiers=(release_identifier, archive_identifier),
            metadata={
                "archive_sha1": control.checksum,
                "archive_url": control.source_url,
                "registry_path": self.registry_path,
                "registry_revision": revision,
            },
            locator=f"$.{control.release_key}",
        )
        return SourceRecord(
            source_record_id=f"bundle:{control.release_key}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=model_page_url,
            title=control.model_id,
            raw={
                "registry_url": registry_url,
                "registry_path": self.registry_path,
                "registry_revision": revision,
                "release_key": control.release_key,
                "model_id": control.model_id,
                "version": control.version,
                "source": control.source_url,
                "checksum": control.checksum,
                "locator": f"$.{control.release_key}",
            },
            text=f"{control.model_id}\n{control.release_key}",
            identifiers=(Identifier("monai:bundle-record", control.release_key),),
            links=(
                Link(model_page_url, relation="model_page", locator=f"$.{control.release_key}"),
                Link(registry_url, relation="metadata", locator=f"$.{control.release_key}"),
                Link(
                    self.repository_url,
                    relation="source_repository",
                    locator=f"$.{control.release_key}",
                ),
                Link(
                    control.source_url,
                    relation="weights",
                    locator=f"$.{control.release_key}.source",
                    crawl=False,
                ),
            ),
            models=(model,),
            releases=(release,),
        )

    def _tombstone(
        self, control: _BundleControl, *, revision: str, registry_url: str
    ) -> SourceRecord:
        model_page_url = self._model_page_url(revision, control.model_id)
        return SourceRecord(
            source_record_id=f"bundle:{control.release_key}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=model_page_url,
            title=control.model_id,
            raw={
                "registry_url": registry_url,
                "registry_path": self.registry_path,
                "registry_revision": revision,
                "release_key": control.release_key,
                "model_id": control.model_id,
                "version": control.version,
                "source": control.source_url,
                "checksum": control.checksum,
                "removal_observed": True,
            },
            identifiers=(Identifier("monai:bundle-record", control.release_key),),
            deleted=True,
        )


class _BundleControl:
    __slots__ = ("release_key", "model_id", "version", "source_url", "checksum")

    def __init__(
        self,
        *,
        release_key: str,
        model_id: str,
        version: str,
        source_url: str,
        checksum: str | None,
    ) -> None:
        self.release_key = release_key
        self.model_id = model_id
        self.version = version
        self.source_url = source_url
        self.checksum = checksum

    def checkpoint_value(self) -> dict[str, str | None]:
        return {"source_url": self.source_url, "checksum": self.checksum}


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _registry_path(value: str) -> str:
    path = _required_text(value, "registry_path").strip("/")
    if any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("registry_path must be a safe relative path")
    return path


def _web_url(value: Any, source: str, field: str) -> str:
    url = canonicalize_url(_required_text(value, field))
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"{source}: {field} is not an HTTP(S) URL")
    return url


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str, source: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {field} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{source}: {field} must be a positive integer")
    return result


def _bounded_positive_int(value: Any, field: str, source: str) -> int:
    result = _positive_int(value, field, source)
    if result > _MAX_ENTRIES:
        raise ValueError(f"{source}: {field} exceeds {_MAX_ENTRIES}")
    return result


def _nonnegative_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _header(headers: Mapping[str, Any], name: str) -> str:
    needle = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == needle:
            return _text(value)
    return ""


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["MonaiModelZooSourceAdapter"]
