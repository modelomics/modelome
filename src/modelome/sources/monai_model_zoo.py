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
_ARCHIVE_TAG = "hosting_storage_v1"
_ASSET_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ASSET_PAGE_SIZE = 100


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MonaiModelZooSourceAdapter:
    """Enumerate released bundles and versions in MONAI's public model zoo.

    The repository's pinned ``models/model_info.json`` provides archive URLs
    and checksums. Its ``hosting_storage_v1`` GitHub release retains versioned
    bundle assets, including historical assets absent from the current index.
    The adapter joins both inventories without downloading bundle bytes.

    BioImage.IO and MONAI have overlapping application areas but distinct
    publishing ecosystems.  A MONAI bundle is emitted as release evidence only
    when it appears in this registry; model-family, task, or name filtering is
    deliberately absent.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers versioned bundles present in Project MONAI's public "
        "models/model_info.json registry or hosting_storage_v1 release. "
        "Draft, private, removed, or "
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
                "adapter": "monai-model-zoo-v2",
                "repository": self.repository,
                "branch": self.branch,
                "registry_path": self.registry_path,
                "max_registry_bytes": self.max_registry_bytes,
                "max_entries": self.max_entries,
                "archive_release_tag": _ARCHIVE_TAG,
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

    @property
    def archive_release_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/releases/tags/"
            f"{_ARCHIVE_TAG}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        archive_response: HttpResponse = self.client.get(
            self.archive_release_url,
            headers={"Accept": "application/vnd.github+json"},
        )
        if archive_response.status != 200:
            raise ValueError(
                f"{self.name}: archived release returned HTTP {archive_response.status}"
            )
        if len(archive_response.body) > self.max_registry_bytes:
            raise ValueError(f"{self.name}: archived release exceeds response byte limit")
        archive_payload = archive_response.json()
        archive_assets = self._release_assets(archive_payload)
        archive_controls = self._archive_controls(archive_payload, archive_assets)
        archive_digest = content_hash(
            {"release": archive_payload, "assets": list(archive_assets)}
        )
        if (
            revision == _text(state.get("completed_revision"))
            and archive_digest == _text(state.get("archive_assets_digest"))
        ):
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
        current_controls = self._controls(response.json())
        controls_by_key = {control.release_key: control for control in current_controls}
        for archive_control in archive_controls:
            current = controls_by_key.get(archive_control.release_key)
            if current is None:
                controls_by_key[archive_control.release_key] = archive_control
            else:
                current.archive_asset_id = archive_control.archive_asset_id
                current.archive_asset_digest = archive_control.archive_asset_digest
                current.archive_asset_updated_at = archive_control.archive_asset_updated_at
        controls = tuple(controls_by_key[key] for key in sorted(controls_by_key))
        if len(controls) > self.max_entries:
            raise ValueError(
                f"{self.name}: combined registry and archive exceed "
                f"{self.max_entries} bundles"
            )
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
            "archive_assets_digest": archive_digest,
            "archive_asset_count": len(archive_controls),
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
        if len(payload) > self.max_entries:
            raise ValueError(f"{self.name}: registry exceeds {self.max_entries} bundles")
        controls = []
        for release_key, raw in sorted(payload.items()):
            controls.append(self._control(release_key, raw))
        return tuple(controls)

    def _release_assets(self, payload: Any) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(payload, Mapping) or _text(payload.get("tag_name")) != _ARCHIVE_TAG:
            raise ValueError(f"{self.name}: archive endpoint returned the wrong release tag")
        release_id = _optional_asset_id(payload.get("id"), _ARCHIVE_TAG)
        if release_id is None:
            raise ValueError(f"{self.name}: archive release is missing its ID")
        assets_count = _nonnegative_int(payload.get("assets_count"))
        if assets_count is None:
            raise ValueError(f"{self.name}: archive release has an invalid assets_count")
        if assets_count > self.max_entries:
            raise ValueError(f"{self.name}: archived release exceeds {self.max_entries} assets")
        assets_url = _archive_assets_url(
            payload.get("assets_url"), self.repository, release_id
        )
        collected: list[Mapping[str, Any]] = []
        page_number = 1
        while len(collected) < assets_count:
            response: HttpResponse = self.client.get(
                assets_url,
                params={"per_page": _ASSET_PAGE_SIZE, "page": page_number},
                headers={"Accept": "application/vnd.github+json"},
            )
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: archive asset page {page_number} returned HTTP "
                    f"{response.status}"
                )
            if len(response.body) > self.max_registry_bytes:
                raise ValueError(
                    f"{self.name}: archive asset page {page_number} exceeds response byte limit"
                )
            page_assets = response.json()
            if not isinstance(page_assets, list) or len(page_assets) > _ASSET_PAGE_SIZE:
                raise ValueError(f"{self.name}: archive asset page is not a bounded array")
            if not page_assets and len(collected) < assets_count:
                raise ValueError(f"{self.name}: archive asset listing ended before assets_count")
            if len(collected) + len(page_assets) > assets_count:
                raise ValueError(f"{self.name}: archive asset listing exceeds assets_count")
            for asset in page_assets:
                if not isinstance(asset, Mapping):
                    raise ValueError(f"{self.name}: archived release asset is not an object")
                collected.append(asset)
            page_number += 1
        if len(collected) != assets_count:
            raise ValueError(f"{self.name}: incomplete archive asset listing")
        return tuple(collected)

    def _archive_controls(
        self, payload: Any, assets: tuple[Mapping[str, Any], ...]
    ) -> tuple[_BundleControl, ...]:
        if not isinstance(payload, Mapping) or _text(payload.get("tag_name")) != _ARCHIVE_TAG:
            raise ValueError(f"{self.name}: archive endpoint returned the wrong release tag")
        controls = []
        seen: set[str] = set()
        seen_ids: set[int] = set()
        for asset in assets:
            name = _text(asset.get("name"))
            asset_id = _optional_asset_id(asset.get("id"), name)
            if asset_id is None or asset_id in seen_ids:
                raise ValueError(f"{self.name}: missing or duplicate GitHub asset ID {name!r}")
            seen_ids.add(asset_id)
            if not name.endswith(".zip"):
                continue
            release_key = name[:-4]
            match = _BUNDLE_KEY.fullmatch(release_key)
            if match is None:
                continue
            if release_key in seen:
                raise ValueError(f"{self.name}: duplicate archived bundle asset {name!r}")
            seen.add(release_key)
            source_url = _archive_asset_url(
                asset.get("browser_download_url"), self.repository, _ARCHIVE_TAG, name
            )
            digest = _optional_asset_digest(asset.get("digest"), name)
            controls.append(
                _BundleControl(
                    release_key=release_key,
                    model_id=match.group("model"),
                    version=match.group("version"),
                    source_url=source_url,
                    checksum=None,
                    archive_asset_id=asset_id,
                    archive_asset_digest=digest,
                    archive_asset_updated_at=_optional_datetime_text(
                        asset.get("updated_at"), name
                    ),
                )
            )
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
            archive_asset_id=_optional_asset_id(raw.get("archive_asset_id"), key),
            archive_asset_digest=_optional_asset_digest(
                raw.get("archive_asset_digest"), key
            ),
            archive_asset_updated_at=_optional_datetime_text(
                raw.get("archive_asset_updated_at"), key
            ),
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
                    "archive_asset_id": raw.get("archive_asset_id"),
                    "archive_asset_digest": raw.get("archive_asset_digest"),
                    "archive_asset_updated_at": raw.get("archive_asset_updated_at"),
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
                "github_archive_asset_id": control.archive_asset_id,
                "github_archive_asset_digest": control.archive_asset_digest,
                "github_archive_asset_updated_at": control.archive_asset_updated_at,
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
                "archive_asset_id": control.archive_asset_id,
                "archive_asset_digest": control.archive_asset_digest,
                "archive_asset_updated_at": control.archive_asset_updated_at,
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
                *(
                    (
                        Link(
                            _archive_url(self.repository, _ARCHIVE_TAG, control.release_key),
                            relation="github_archive",
                            locator=f"$.{control.release_key}.archive_asset_id",
                            crawl=False,
                        ),
                    )
                    if control.archive_asset_id is not None
                    and control.source_url
                    != _archive_url(self.repository, _ARCHIVE_TAG, control.release_key)
                    else ()
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
                "archive_asset_id": control.archive_asset_id,
                "archive_asset_digest": control.archive_asset_digest,
                "archive_asset_updated_at": control.archive_asset_updated_at,
                "removal_observed": True,
            },
            identifiers=(Identifier("monai:bundle-record", control.release_key),),
            deleted=True,
        )


class _BundleControl:
    __slots__ = (
        "release_key",
        "model_id",
        "version",
        "source_url",
        "checksum",
        "archive_asset_id",
        "archive_asset_digest",
        "archive_asset_updated_at",
    )

    def __init__(
        self,
        *,
        release_key: str,
        model_id: str,
        version: str,
        source_url: str,
        checksum: str | None,
        archive_asset_id: int | None = None,
        archive_asset_digest: str | None = None,
        archive_asset_updated_at: str | None = None,
    ) -> None:
        self.release_key = release_key
        self.model_id = model_id
        self.version = version
        self.source_url = source_url
        self.checksum = checksum
        self.archive_asset_id = archive_asset_id
        self.archive_asset_digest = archive_asset_digest
        self.archive_asset_updated_at = archive_asset_updated_at

    def checkpoint_value(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "checksum": self.checksum,
            "archive_asset_id": self.archive_asset_id,
            "archive_asset_digest": self.archive_asset_digest,
            "archive_asset_updated_at": self.archive_asset_updated_at,
        }


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


def _archive_url(repository: str, tag: str, release_key: str) -> str:
    filename = f"{release_key}.zip"
    return canonicalize_url(
        f"https://github.com/{repository}/releases/download/{quote(tag, safe='')}/"
        f"{quote(filename, safe='')}"
    )


def _archive_assets_url(value: Any, repository: str, release_id: int) -> str:
    url = _web_url(value, "monai-model-zoo", "archive assets URL")
    expected = f"https://api.github.com/repos/{repository}/releases/{release_id}/assets"
    if url != expected:
        raise ValueError("monai-model-zoo: unexpected archive assets API URL")
    return url


def _archive_asset_url(value: Any, repository: str, tag: str, filename: str) -> str:
    url = _web_url(value, "monai-model-zoo", f"archive asset {filename!r}")
    expected = _archive_url(repository, tag, filename[:-4])
    if url != expected:
        raise ValueError(f"monai-model-zoo: unexpected archive asset URL for {filename!r}")
    return url


def _optional_asset_id(value: Any, filename: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"monai-model-zoo: invalid GitHub asset ID for {filename!r}")
    return value


def _optional_asset_digest(value: Any, filename: str) -> str | None:
    digest = _text(value).casefold()
    if not digest:
        return None
    if not _ASSET_DIGEST.fullmatch(digest):
        raise ValueError(f"monai-model-zoo: invalid GitHub asset digest for {filename!r}")
    return digest


def _optional_datetime_text(value: Any, filename: str) -> str | None:
    timestamp = _text(value)
    if not timestamp:
        return None
    if len(timestamp) > 128 or any(char in timestamp for char in "\r\n\x00"):
        raise ValueError(f"monai-model-zoo: invalid GitHub asset timestamp for {filename!r}")
    return timestamp


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
