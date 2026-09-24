from __future__ import annotations

import gzip
import ipaddress
import json
import re
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import content_hash

Clock = Callable[[], datetime]

_CATALOG_URL = "https://index.commoncrawl.org/collinfo.json"
_DATA_URL = "https://data.commoncrawl.org/"
_SCAN_STAGE = "wet_shards"
_COLLECTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_PAIR_RE = re.compile(r"%([0-9A-Fa-f]{2})")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Collection:
    collection_id: str
    name: str
    collection_from: str
    collection_to: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _Catalog:
    collections: tuple[_Collection, ...]
    digest: str

    @property
    def by_id(self) -> dict[str, _Collection]:
        return {collection.collection_id: collection for collection in self.collections}


@dataclass(frozen=True, slots=True)
class _WetManifest:
    paths: tuple[str, ...]
    digest: str
    etag: str | None
    compressed_bytes: int


class _ManifestError(ValueError):
    pass


class CommonCrawlWetSourceAdapter:
    """Enumerate every Common Crawl WET object without downloading WET data.

    Common Crawl's collection catalog and each collection's ``wet.paths.gz``
    file are a small control plane over a much larger payload plane. This
    adapter downloads only those control files. It freezes a canonical catalog
    digest for every in-progress scan and emits stable records that a separate
    bounded bulk loader can consume later.
    """

    def __init__(
        self,
        *,
        name: str = "commoncrawl-wet",
        catalog_url: str = _CATALOG_URL,
        data_url: str = _DATA_URL,
        page_size: int = 1_000,
        manifests_per_page: int = 16,
        max_catalog_bytes: int = 8 * 1024 * 1024,
        max_collections: int = 10_000,
        max_manifest_compressed_bytes: int = 64 * 1024 * 1024,
        max_manifest_bytes: int = 512 * 1024 * 1024,
        max_manifest_shards: int = 2_000_000,
        max_path_bytes: int = 16 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.catalog_url = _https_url(catalog_url, self.name, "catalog URL")
        self.data_url = _https_directory_url(data_url, self.name, "data URL")
        self.page_size = _positive_int(page_size, self.name, "page size")
        self.manifests_per_page = _positive_int(
            manifests_per_page, self.name, "manifests per page"
        )
        self.max_catalog_bytes = _positive_int(
            max_catalog_bytes, self.name, "maximum catalog bytes"
        )
        self.max_collections = _positive_int(
            max_collections, self.name, "maximum collections"
        )
        self.max_manifest_compressed_bytes = _positive_int(
            max_manifest_compressed_bytes,
            self.name,
            "maximum compressed manifest bytes",
        )
        self.max_manifest_bytes = _positive_int(
            max_manifest_bytes, self.name, "maximum manifest bytes"
        )
        self.max_manifest_shards = _positive_int(
            max_manifest_shards, self.name, "maximum manifest shards"
        )
        self.max_path_bytes = _positive_int(
            max_path_bytes, self.name, "maximum WET path bytes"
        )
        self.client = client or HttpClient(
            max_response_bytes=max(
                self.max_catalog_bytes,
                self.max_manifest_compressed_bytes,
            )
        )
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "commoncrawl-wet-control-v1",
                "catalog_url": self.catalog_url,
                "data_url": self.data_url,
                "page_size": self.page_size,
                "manifests_per_page": self.manifests_per_page,
                "max_catalog_bytes": self.max_catalog_bytes,
                "max_collections": self.max_collections,
                "max_manifest_compressed_bytes": self.max_manifest_compressed_bytes,
                "max_manifest_bytes": self.max_manifest_bytes,
                "max_manifest_shards": self.max_manifest_shards,
                "max_path_bytes": self.max_path_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if not isinstance(state, Mapping):
            raise TypeError(f"{self.name}: checkpoint state must be a mapping")
        try:
            catalog = self._read_catalog()
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            return self._failed_page(
                state,
                source_record_id=f"{self.name}:collection-catalog",
                error=error,
                retry_state=_restart_boundary(state),
                summary={"catalog_url": self.catalog_url},
            )

        stage = _optional_text(state.get("stage"))
        if stage:
            if stage != _SCAN_STAGE:
                raise ValueError(f"{self.name}: unknown checkpoint stage {stage!r}")
            return self._resume_scan(state, catalog)
        return self._begin_scan(state, catalog)

    def _begin_scan(self, state: Mapping[str, Any], catalog: _Catalog) -> SourcePage:
        completed = _completed_collections(state, self.name)
        prior_catalog_digest = _optional_digest(
            state.get("catalog_digest"), self.name, "catalog digest"
        )
        current = catalog.by_id

        missing = sorted(set(completed) - set(current))
        if missing:
            return self._failed_page(
                state,
                source_record_id=f"{self.name}:collection-catalog:regression",
                error=_ManifestError(
                    "collection catalog dropped previously observed collection IDs"
                ),
                retry_state=dict(state),
                summary={
                    "catalog_url": self.catalog_url,
                    "missing_count": len(missing),
                    "missing_ids": missing[:100],
                },
            )

        if prior_catalog_digest == catalog.digest:
            _validate_complete_coverage(completed, catalog, self.name)
            return SourcePage(
                records=(),
                next_state=dict(state),
                complete=True,
                upstream_count=0,
            )

        plan = [
            collection.collection_id
            for collection in catalog.collections
            if completed.get(collection.collection_id, {}).get("collection_fingerprint")
            != collection.fingerprint
        ]
        if not plan:
            return SourcePage(
                records=(),
                next_state=_complete_state(
                    state,
                    catalog=catalog,
                    completed=completed,
                    completed_at=_isoformat(self.clock()),
                ),
                complete=True,
                upstream_count=0,
            )

        scan_state = _checkpoint_metadata(state)
        if prior_catalog_digest is not None:
            scan_state["base_catalog_digest"] = prior_catalog_digest
        scan_state.update(
            {
                "stage": _SCAN_STAGE,
                "scan_catalog_digest": catalog.digest,
                "scan_collection_ids": plan,
                "scan_collection_fingerprints": {
                    collection_id: current[collection_id].fingerprint
                    for collection_id in plan
                },
                "collection_index": 0,
                "manifest_offset": 0,
                "completed_collections": completed,
                "scan_shards_emitted": 0,
                "started_at": _isoformat(self.clock()),
            }
        )
        return self._emit_page(scan_state, catalog)

    def _resume_scan(self, state: Mapping[str, Any], catalog: _Catalog) -> SourcePage:
        expected_catalog = _required_digest(
            state.get("scan_catalog_digest"), self.name, "scan catalog digest"
        )
        if expected_catalog != catalog.digest:
            return self._failed_page(
                state,
                source_record_id=f"{self.name}:collection-catalog:drift",
                error=_ManifestError("collection catalog changed during the frozen scan"),
                retry_state=_restart_boundary(state),
                summary={
                    "catalog_url": self.catalog_url,
                    "expected_catalog_digest": expected_catalog,
                    "observed_catalog_digest": catalog.digest,
                },
            )
        return self._emit_page(dict(state), catalog)

    def _emit_page(
        self,
        state: Mapping[str, Any],
        catalog: _Catalog,
    ) -> SourcePage:
        scan_state = dict(state)
        collection_ids = _scan_collection_ids(scan_state, self.name)
        collection_fingerprints = _scan_collection_fingerprints(
            scan_state, collection_ids, self.name
        )
        catalog_by_id = catalog.by_id
        for collection_id in collection_ids:
            collection = catalog_by_id.get(collection_id)
            if collection is None:
                raise ValueError(
                    f"{self.name}: scan collection is absent from the frozen catalog"
                )
            if collection.fingerprint != collection_fingerprints[collection_id]:
                raise ValueError(
                    f"{self.name}: scan collection fingerprint is inconsistent"
                )

        collection_index = _state_index(
            scan_state.get("collection_index"),
            len(collection_ids),
            self.name,
            "collection index",
        )
        offset = _nonnegative_int(
            scan_state.get("manifest_offset"), self.name, "manifest offset"
        )
        completed = _completed_collections(scan_state, self.name)
        emitted_total = _nonnegative_int(
            scan_state.get("scan_shards_emitted"),
            self.name,
            "scan shards emitted",
        )
        records: list[SourceRecord] = []
        manifests_read = 0

        while (
            collection_index < len(collection_ids)
            and len(records) < self.page_size
            and manifests_read < self.manifests_per_page
        ):
            collection_id = collection_ids[collection_index]
            collection = catalog_by_id[collection_id]
            manifest_url = self._manifest_url(collection_id)
            try:
                manifest = self._read_manifest(manifest_url, collection_id)
            except (
                TypeError,
                ValueError,
                UnicodeError,
                gzip.BadGzipFile,
                EOFError,
                zlib.error,
            ) as error:
                return self._failed_page(
                    scan_state,
                    source_record_id=f"{self.name}:{collection_id}:wet-manifest",
                    error=error,
                    retry_state=_restart_boundary(scan_state),
                    summary={
                        "collection_id": collection_id,
                        "manifest_url": manifest_url,
                    },
                )
            manifests_read += 1

            active_id = _optional_text(scan_state.get("active_collection_id"))
            if active_id:
                if active_id != collection_id:
                    raise ValueError(
                        f"{self.name}: active manifest does not match collection index"
                    )
                expected_digest = _required_digest(
                    scan_state.get("active_manifest_digest"),
                    self.name,
                    "active manifest digest",
                )
                expected_total = _nonnegative_int(
                    scan_state.get("active_manifest_total"),
                    self.name,
                    "active manifest total",
                )
                if expected_digest != manifest.digest or expected_total != len(
                    manifest.paths
                ):
                    return self._failed_page(
                        scan_state,
                        source_record_id=(
                            f"{self.name}:{collection_id}:wet-manifest:drift"
                        ),
                        error=_ManifestError(
                            "WET manifest changed during the frozen collection scan"
                        ),
                        retry_state=_restart_boundary(scan_state),
                        summary={
                            "collection_id": collection_id,
                            "expected_manifest_digest": expected_digest,
                            "observed_manifest_digest": manifest.digest,
                            "expected_total_shards": expected_total,
                            "observed_total_shards": len(manifest.paths),
                        },
                    )
            else:
                if offset != 0:
                    raise ValueError(
                        f"{self.name}: manifest offset has no active collection"
                    )
                scan_state["active_collection_id"] = collection_id
                scan_state["active_manifest_digest"] = manifest.digest
                scan_state["active_manifest_total"] = len(manifest.paths)

            if offset > len(manifest.paths):
                raise ValueError(
                    f"{self.name}: manifest offset is beyond the frozen manifest"
                )

            prior = completed.get(collection_id)
            if offset == 0 and prior is not None and (
                prior.get("manifest_digest") == manifest.digest
                and prior.get("total_shards") == len(manifest.paths)
            ):
                completed[collection_id] = _completed_collection(
                    collection, manifest
                )
                collection_index += 1
                scan_state["collection_index"] = collection_index
                scan_state["manifest_offset"] = 0
                _clear_active_manifest(scan_state)
                continue

            capacity = self.page_size - len(records)
            selected = manifest.paths[offset : offset + capacity]
            records.extend(
                self._shard_record(
                    collection=collection,
                    manifest=manifest,
                    manifest_url=manifest_url,
                    path=path,
                    manifest_index=offset + index,
                )
                for index, path in enumerate(selected)
            )
            offset += len(selected)
            emitted_total += len(selected)
            scan_state["manifest_offset"] = offset
            scan_state["scan_shards_emitted"] = emitted_total

            if offset == len(manifest.paths):
                completed[collection_id] = _completed_collection(
                    collection, manifest
                )
                collection_index += 1
                offset = 0
                scan_state["collection_index"] = collection_index
                scan_state["manifest_offset"] = 0
                _clear_active_manifest(scan_state)

        scan_state["completed_collections"] = completed
        if collection_index >= len(collection_ids):
            return SourcePage(
                records=tuple(records),
                next_state=_complete_state(
                    scan_state,
                    catalog=catalog,
                    completed=completed,
                    completed_at=_isoformat(self.clock()),
                ),
                complete=True,
                upstream_count=emitted_total,
            )

        if not records and manifests_read == 0:
            return self._failed_page(
                scan_state,
                source_record_id=f"{self.name}:pagination",
                error=_ManifestError("scan page made no progress"),
                retry_state=_restart_boundary(scan_state),
                summary={
                    "collection_index": collection_index,
                    "manifest_offset": offset,
                },
            )
        return SourcePage(
            records=tuple(records),
            next_state=scan_state,
            complete=False,
        )

    def _read_catalog(self) -> _Catalog:
        response: HttpResponse = self.client.get(
            self.catalog_url,
            headers={"Accept": "application/json"},
        )
        _validate_response(
            response,
            expected_url=self.catalog_url,
            maximum_bytes=self.max_catalog_bytes,
            source=self.name,
            label="collection catalog",
        )
        try:
            text = response.body.decode("utf-8", errors="strict")
            payload = json.loads(text, parse_constant=_reject_json_constant)
        except UnicodeDecodeError as error:
            raise _ManifestError("collection catalog is not valid UTF-8") from error
        except json.JSONDecodeError as error:
            raise _ManifestError("collection catalog is not valid JSON") from error
        if not isinstance(payload, list):
            raise _ManifestError("collection catalog must be a JSON array")
        if not payload:
            raise _ManifestError("collection catalog is empty or truncated")
        if len(payload) > self.max_collections:
            raise _ManifestError(
                f"collection catalog exceeds {self.max_collections} entries"
            )

        collections: list[_Collection] = []
        seen: set[str] = set()
        for index, raw in enumerate(payload):
            if not isinstance(raw, Mapping):
                raise _ManifestError(
                    f"collection catalog entry {index} must be an object"
                )
            collection = _collection(raw, index=index)
            if collection.collection_id in seen:
                raise _ManifestError(
                    f"duplicate collection ID {collection.collection_id!r}"
                )
            seen.add(collection.collection_id)
            collections.append(collection)
        return _Catalog(tuple(collections), content_hash(payload))

    def _read_manifest(self, url: str, collection_id: str) -> _WetManifest:
        response: HttpResponse = self.client.get(
            url,
            headers={"Accept": "application/gzip,application/octet-stream"},
        )
        _validate_response(
            response,
            expected_url=url,
            maximum_bytes=self.max_manifest_compressed_bytes,
            source=self.name,
            label=f"{collection_id} WET manifest",
        )
        paths, digest = _parse_wet_manifest(
            response.body,
            collection_id=collection_id,
            max_uncompressed_bytes=self.max_manifest_bytes,
            max_shards=self.max_manifest_shards,
            max_path_bytes=self.max_path_bytes,
        )
        headers = {str(key).casefold(): str(value) for key, value in response.headers.items()}
        etag = _optional_text(headers.get("etag")) or None
        return _WetManifest(paths, digest, etag, len(response.body))

    def _manifest_url(self, collection_id: str) -> str:
        encoded = quote(collection_id, safe="-._~")
        return f"{self.data_url}crawl-data/{encoded}/wet.paths.gz"

    def _shard_record(
        self,
        *,
        collection: _Collection,
        manifest: _WetManifest,
        manifest_url: str,
        path: str,
        manifest_index: int,
    ) -> SourceRecord:
        stable_url = f"{self.data_url}{path}"
        shard_key = content_hash(
            {
                "collection_id": collection.collection_id,
                "stable_object_url": stable_url,
            }
        )
        raw: dict[str, Any] = {
            "record_type": "commoncrawl_wet_shard",
            "collection_id": collection.collection_id,
            "collection_name": collection.name,
            "collection_from": collection.collection_from,
            "collection_to": collection.collection_to,
            "manifest_url": manifest_url,
            "manifest_digest": manifest.digest,
            "manifest_digest_algorithm": "sha256",
            "manifest_compressed_bytes": manifest.compressed_bytes,
            "stable_object_url": stable_url,
            "manifest_index": manifest_index,
            "total_shards": len(manifest.paths),
            "path": path,
        }
        if manifest.etag is not None:
            raw["manifest_etag"] = manifest.etag
        return SourceRecord(
            source_record_id=f"commoncrawl:wet-shard:{shard_key}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=stable_url,
            title=(
                f"Common Crawl {collection.collection_id} WET shard "
                f"{manifest_index + 1} of {len(manifest.paths)}"
            ),
            raw=raw,
            published_at=collection.collection_to,
            identifiers=(Identifier("commoncrawl:wet-shard", shard_key),),
            links=(
                Link(
                    stable_url,
                    relation="bulk_payload",
                    locator="$.stable_object_url",
                    crawl=False,
                ),
                Link(
                    manifest_url,
                    relation="control_manifest",
                    locator="$.manifest_url",
                    crawl=False,
                ),
            ),
        )

    def _failed_page(
        self,
        state: Mapping[str, Any],
        *,
        source_record_id: str,
        error: Exception,
        retry_state: Mapping[str, Any],
        summary: Mapping[str, Any],
    ) -> SourcePage:
        retry = dict(retry_state)
        return SourcePage(
            records=(),
            next_state=retry,
            complete=False,
            issues=(
                SourceIssue(
                    source_record_id=source_record_id,
                    stage="source_manifest",
                    error=f"{type(error).__name__}: {error}",
                    summary=dict(summary),
                ),
            ),
            retry_state=retry,
        )


def _collection(raw: Mapping[str, Any], *, index: int) -> _Collection:
    collection_id = _required_exact_text(raw.get("id"), f"collection {index} ID")
    if not _COLLECTION_ID_RE.fullmatch(collection_id) or collection_id in {".", ".."}:
        raise _ManifestError(f"malformed collection ID {collection_id!r}")
    name = _optional_exact_text(raw.get("name"), f"collection {collection_id} name")
    collection_from = _required_exact_text(
        raw.get("from"), f"collection {collection_id} from"
    )
    collection_to = _required_exact_text(
        raw.get("to"), f"collection {collection_id} to"
    )
    start = _collection_time(collection_from, collection_id, "from")
    end = _collection_time(collection_to, collection_id, "to")
    if end < start:
        raise _ManifestError(f"collection {collection_id} ends before it starts")
    identity = {
        "id": collection_id,
        "name": name,
        "from": collection_from,
        "to": collection_to,
    }
    return _Collection(
        collection_id,
        name,
        collection_from,
        collection_to,
        content_hash(identity),
    )


def _parse_wet_manifest(
    body: bytes,
    *,
    collection_id: str,
    max_uncompressed_bytes: int,
    max_shards: int,
    max_path_bytes: int,
) -> tuple[tuple[str, ...], str]:
    paths: list[str] = []
    seen: set[str] = set()
    uncompressed_bytes = 0
    digest_lines: list[str] = []
    stream = BytesIO(body)
    try:
        with gzip.GzipFile(fileobj=stream, mode="rb") as manifest:
            while True:
                line = manifest.readline(max_path_bytes + 2)
                if not line:
                    break
                uncompressed_bytes += len(line)
                if uncompressed_bytes > max_uncompressed_bytes:
                    raise _ManifestError(
                        f"WET manifest exceeds {max_uncompressed_bytes} uncompressed bytes"
                    )
                if len(line) > max_path_bytes + 1 or not line.endswith(b"\n"):
                    raise _ManifestError(
                        "WET manifest contains an overlong or truncated path line"
                    )
                raw_path = line[:-1]
                if raw_path.endswith(b"\r"):
                    raw_path = raw_path[:-1]
                try:
                    path = raw_path.decode("utf-8", errors="strict")
                except UnicodeDecodeError as error:
                    raise _ManifestError(
                        "WET manifest contains a path that is not valid UTF-8"
                    ) from error
                _validate_wet_path(path, collection_id)
                if path in seen:
                    raise _ManifestError(f"duplicate WET path {path!r}")
                if len(paths) >= max_shards:
                    raise _ManifestError(
                        f"WET manifest exceeds {max_shards} shard paths"
                    )
                seen.add(path)
                paths.append(path)
                digest_lines.append(path)
    except (gzip.BadGzipFile, EOFError, zlib.error) as error:
        raise _ManifestError("WET manifest is invalid or truncated gzip data") from error

    if stream.tell() != len(body):
        raise _ManifestError("WET manifest contains unread trailing compressed data")
    if not paths:
        raise _ManifestError("WET manifest is empty or truncated")
    # The normalized LF-delimited list is independent of gzip headers and transport.
    digest = content_hash("".join(f"{path}\n" for path in digest_lines))
    return tuple(paths), digest


def _validate_wet_path(path: str, collection_id: str) -> None:
    if not path or path != path.strip() or "\\" in path or "//" in path:
        raise _ManifestError(f"unsafe WET path {path!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        raise _ManifestError("WET path contains control characters")
    parts = urlsplit(path)
    if (
        parts.scheme
        or parts.netloc
        or parts.query
        or parts.fragment
        or path.startswith("/")
    ):
        raise _ManifestError(f"WET path must be a relative object path: {path!r}")
    segments = path.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise _ManifestError(f"WET path contains path traversal: {path!r}")
    dangerous_encoding = any(
        chr(int(match.group(1), 16)) in {"/", "\\", "."}
        or int(match.group(1), 16) < 32
        or int(match.group(1), 16) == 127
        for match in _HEX_PAIR_RE.finditer(path)
    )
    if dangerous_encoding or any(
        segment in {".", ".."} for segment in unquote(path).split("/")
    ):
        raise _ManifestError(f"WET path contains encoded path traversal: {path!r}")
    if len(segments) < 5 or segments[0] != "crawl-data" or segments[1] != collection_id:
        raise _ManifestError(
            f"WET path is outside collection {collection_id!r}: {path!r}"
        )
    if "wet" not in segments[2:-1] or not segments[-1].endswith(".warc.wet.gz"):
        raise _ManifestError(f"manifest entry is not a WET shard path: {path!r}")


def _validate_response(
    response: HttpResponse,
    *,
    expected_url: str,
    maximum_bytes: int,
    source: str,
    label: str,
) -> None:
    if not isinstance(response, HttpResponse):
        raise TypeError(f"{source}: {label} client returned an invalid response")
    if response.status != 200:
        raise _ManifestError(f"{label} returned HTTP {response.status}")
    if len(response.body) > maximum_bytes:
        raise _ManifestError(f"{label} exceeds {maximum_bytes} bytes")
    headers = {str(key).casefold(): str(value) for key, value in response.headers.items()}
    content_length = _optional_text(headers.get("content-length"))
    if content_length:
        try:
            declared = int(content_length)
        except ValueError as error:
            raise _ManifestError(f"{label} has an invalid Content-Length") from error
        if declared < 0 or declared != len(response.body):
            raise _ManifestError(f"{label} is truncated or has an invalid Content-Length")
    # Keep the expected URL in the signature and error surface even when an injected
    # test transport does not expose a meaningful final response URL.
    _required_text(expected_url, "expected response URL")


def _complete_state(
    state: Mapping[str, Any],
    *,
    catalog: _Catalog,
    completed: Mapping[str, Mapping[str, Any]],
    completed_at: str,
) -> dict[str, Any]:
    result = _checkpoint_metadata(state)
    result.update(
        {
            "catalog_digest": catalog.digest,
            "collection_count": len(catalog.collections),
            "completed_collections": {
                collection.collection_id: dict(completed[collection.collection_id])
                for collection in catalog.collections
            },
            "completed_at": completed_at,
        }
    )
    return result


def _completed_collection(
    collection: _Collection,
    manifest: _WetManifest,
) -> dict[str, Any]:
    return {
        "collection_fingerprint": collection.fingerprint,
        "manifest_digest": manifest.digest,
        "total_shards": len(manifest.paths),
    }


def _restart_boundary(state: Mapping[str, Any]) -> dict[str, Any]:
    completed = state.get("completed_collections")
    result = _checkpoint_metadata(state)
    if isinstance(completed, Mapping):
        normalized = {
            str(key): dict(value)
            for key, value in completed.items()
            if isinstance(key, str) and isinstance(value, Mapping)
        }
        if normalized:
            result["completed_collections"] = normalized
    base_catalog = _optional_text(state.get("base_catalog_digest"))
    if base_catalog:
        result["catalog_digest"] = base_catalog
    return result


def _checkpoint_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("_modelome_")
    }


def _completed_collections(
    state: Mapping[str, Any], source: str
) -> dict[str, dict[str, Any]]:
    raw = state.get("completed_collections", {})
    if not isinstance(raw, Mapping):
        raise TypeError(f"{source}: completed collections must be a mapping")
    result: dict[str, dict[str, Any]] = {}
    for collection_id, value in raw.items():
        if not isinstance(collection_id, str) or not _COLLECTION_ID_RE.fullmatch(
            collection_id
        ):
            raise ValueError(f"{source}: completed collection ID is malformed")
        if not isinstance(value, Mapping):
            raise TypeError(f"{source}: completed collection entry must be a mapping")
        fingerprint = _required_digest(
            value.get("collection_fingerprint"),
            source,
            "completed collection fingerprint",
        )
        manifest_digest = _required_digest(
            value.get("manifest_digest"), source, "completed manifest digest"
        )
        total_shards = _nonnegative_int(
            value.get("total_shards"), source, "completed total shards"
        )
        result[collection_id] = {
            "collection_fingerprint": fingerprint,
            "manifest_digest": manifest_digest,
            "total_shards": total_shards,
        }
    return result


def _validate_complete_coverage(
    completed: Mapping[str, Mapping[str, Any]],
    catalog: _Catalog,
    source: str,
) -> None:
    expected = {collection.collection_id for collection in catalog.collections}
    if set(completed) != expected:
        raise ValueError(f"{source}: completed checkpoint does not cover its catalog")
    for collection in catalog.collections:
        if completed[collection.collection_id].get(
            "collection_fingerprint"
        ) != collection.fingerprint:
            raise ValueError(
                f"{source}: completed collection fingerprint does not match catalog"
            )


def _scan_collection_ids(state: Mapping[str, Any], source: str) -> tuple[str, ...]:
    raw = state.get("scan_collection_ids")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise TypeError(f"{source}: scan collection IDs must be an array")
    result = tuple(_required_exact_text(value, "scan collection ID") for value in raw)
    if not result:
        raise ValueError(f"{source}: scan collection IDs cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{source}: scan collection IDs contain a pagination cycle")
    if any(not _COLLECTION_ID_RE.fullmatch(value) for value in result):
        raise ValueError(f"{source}: scan collection ID is malformed")
    return result


def _scan_collection_fingerprints(
    state: Mapping[str, Any],
    collection_ids: Sequence[str],
    source: str,
) -> dict[str, str]:
    raw = state.get("scan_collection_fingerprints")
    if not isinstance(raw, Mapping) or set(raw) != set(collection_ids):
        raise ValueError(f"{source}: scan collection fingerprints are incomplete")
    return {
        collection_id: _required_digest(
            raw.get(collection_id), source, "scan collection fingerprint"
        )
        for collection_id in collection_ids
    }


def _clear_active_manifest(state: dict[str, Any]) -> None:
    state.pop("active_collection_id", None)
    state.pop("active_manifest_digest", None)
    state.pop("active_manifest_total", None)


def _collection_time(value: str, collection_id: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _ManifestError(
            f"collection {collection_id} {field} is not an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _https_url(value: Any, source: str, field: str) -> str:
    text = _required_text(value, field)
    parts = urlsplit(text)
    try:
        port = parts.port
    except ValueError:
        raise ValueError(f"{source}: {field} has an invalid port") from None
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or port not in {None, 443}
    ):
        raise ValueError(f"{source}: {field} must be a credential-free HTTPS URL")
    host = parts.hostname.casefold().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"{source}: {field} host must be public")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError(f"{source}: {field} host must be public")
    netloc = f"[{host}]" if ":" in host else host
    return urlunsplit(("https", netloc, parts.path, "", ""))


def _https_directory_url(value: Any, source: str, field: str) -> str:
    return _https_url(value, source, field).rstrip("/") + "/"


def _required_text(value: Any, field: str) -> str:
    result = _optional_text(value)
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _required_exact_text(value: Any, field: str) -> str:
    result = _required_text(value, field)
    if not isinstance(value, str) or value != result:
        raise _ManifestError(f"{field} contains surrounding whitespace")
    return result


def _optional_exact_text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or value != value.strip():
        raise _ManifestError(f"{field} must be an exact string")
    return value


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _required_digest(value: Any, source: str, field: str) -> str:
    result = _optional_digest(value, source, field)
    if result is None:
        raise ValueError(f"{source}: {field} is required")
    return result


def _optional_digest(value: Any, source: str, field: str) -> str | None:
    if value is None:
        return None
    result = _optional_text(value)
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise ValueError(f"{source}: {field} must be a lowercase SHA-256 digest")
    return result


def _positive_int(value: Any, source: str, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{source}: {field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise TypeError(f"{source}: {field} must be an integer") from None
    if result < 1 or result != value:
        raise ValueError(f"{source}: {field} must be a positive integer")
    return result


def _nonnegative_int(value: Any, source: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{source}: {field} must be an integer")
    if value < 0:
        raise ValueError(f"{source}: {field} must be nonnegative")
    return value


def _state_index(
    value: Any,
    length: int,
    source: str,
    field: str,
) -> int:
    result = _nonnegative_int(value, source, field)
    if result > length:
        raise ValueError(f"{source}: {field} is beyond the frozen scan")
    return result


def _reject_json_constant(value: str) -> None:
    raise _ManifestError(f"collection catalog contains invalid JSON constant {value}")


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["CommonCrawlWetSourceAdapter"]
