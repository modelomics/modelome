from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import content_hash

Clock = Callable[[], datetime]

_DEFAULT_API_URL = "https://zenodo.org/api/records"
_DEFAULT_CONCEPT_RECORD_ID = "3516917"
_FILES_STAGE = "release_files"
_CHECKSUM_RE = re.compile(r"^(?P<algorithm>md5|sha256|sha512):(?P<digest>[0-9a-fA-F]+)$")
_CHECKSUM_LENGTHS = {"md5": 32, "sha256": 64, "sha512": 128}
_PARTITION_RE = re.compile(r"^(?P<family>.+?)(?:_(?P<index>[1-9][0-9]*))?$")
_DOI_RE = re.compile(r"^10\.[0-9]{4,9}/\S+$", re.IGNORECASE)
_SCAN_KEYS = frozenset(
    {
        "stage",
        "target_release",
        "target_version",
        "target_manifest_signature",
        "target_file_count",
        "file_cursor",
        "control_records_seen",
        "started_at",
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _ReleaseFile:
    key: str
    file_id: str | None
    size: int
    checksum_algorithm: str
    checksum_digest: str
    download_url: str
    partition: str
    partition_index: int | None

    @property
    def checksum(self) -> str:
        return f"{self.checksum_algorithm}:{self.checksum_digest}"


@dataclass(frozen=True, slots=True)
class _Release:
    record_id: str
    concept_record_id: str
    version: str | None
    title: str
    publication_date: str
    created_at: str | None
    updated_at: str | None
    doi: str | None
    concept_doi: str | None
    canonical_url: str
    api_url: str
    versions_url: str | None
    metadata: Mapping[str, Any]
    files: tuple[_ReleaseFile, ...]
    signature: str


@dataclass(frozen=True, slots=True)
class _ScanBoundary:
    target_release: str
    target_version: str | None
    manifest_signature: str
    file_count: int
    cursor: int
    control_records_seen: int


class _ManifestError(ValueError):
    pass


class OpenAireGraphSourceAdapter:
    """Enumerate immutable OpenAIRE Graph full-download release controls.

    OpenAIRE publishes the complete graph as a versioned Zenodo dataset whose
    public API record is a machine-readable release and file catalog. This
    adapter emits only small ``CATALOG_RECORD`` controls: one release record and
    one record for every upstream-named tar partition. It deliberately selects
    no entity, research-product type, topic, venue, or journal. A downstream
    loader must stream every tar/gzip JSONL payload into Parquet and resolve an
    object's type from that object's own upstream metadata.

    Full downloads do not expose an incremental tombstone stream. A successfully
    loaded release is therefore an authoritative replacement snapshot; removals
    are computed by comparing sealed releases, never from an incomplete scan.
    """

    def __init__(
        self,
        *,
        name: str = "openaire-graph",
        url: str = _DEFAULT_API_URL,
        concept_record_id: str = _DEFAULT_CONCEPT_RECORD_ID,
        page_size: int = 25,
        max_manifest_bytes: int = 8 * 1024 * 1024,
        max_files: int = 10_000,
        max_file_bytes: int = 1 << 40,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _api_base_url(url, self.name)
        self.concept_record_id = _record_id(concept_record_id, "concept record ID", self.name)
        self.page_size = _bounded_positive_integer(page_size, "page size", self.name, maximum=1_000)
        self.max_manifest_bytes = _bounded_positive_integer(
            max_manifest_bytes,
            "maximum manifest bytes",
            self.name,
            maximum=64 * 1024 * 1024,
        )
        self.max_files = _bounded_positive_integer(
            max_files, "maximum file count", self.name, maximum=100_000
        )
        self.max_file_bytes = _bounded_positive_integer(
            max_file_bytes,
            "maximum file size",
            self.name,
            maximum=1 << 50,
        )
        self.client = client or HttpClient(max_response_bytes=self.max_manifest_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openaire-graph-full-download-v1",
                "api_url": self.url,
                "concept_record_id": self.concept_record_id,
                "page_size": self.page_size,
                "max_manifest_bytes": self.max_manifest_bytes,
                "max_files": self.max_files,
                "max_file_bytes": self.max_file_bytes,
            }
        )

    @property
    def repository_identity(self) -> Mapping[str, str]:
        """Return the durable identity of the upstream release catalog."""

        return {
            "api_url": self.url,
            "concept_record_id": self.concept_record_id,
            "checkpoint_signature": self.checkpoint_signature,
        }

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        self._validate_state_identity(state)
        stage = _optional_text(state.get("stage"), "checkpoint stage", self.name)
        if stage is None:
            return self._begin_scan(state)
        if stage == _FILES_STAGE:
            return self._file_page(state)
        raise ValueError(f"{self.name}: unknown checkpoint stage {stage!r}")

    def _begin_scan(self, state: Mapping[str, Any]) -> SourcePage:
        retry_state = _stable_state(state)
        try:
            release = self._load_release(self.concept_record_id)
            watermark = _optional_record_id(state.get("watermark"), "watermark", self.name)
            if watermark is not None and int(release.record_id) < int(watermark):
                raise _ManifestError(
                    f"release catalog moved backward from {watermark} to {release.record_id}"
                )
            if watermark == release.record_id:
                expected = _required_text(
                    state.get("release_signature"),
                    "completed release signature",
                )
                expected_count = _state_integer(
                    state, "file_count", self.name, maximum=self.max_files
                )
                if expected != release.signature or expected_count != len(release.files):
                    raise _ManifestError("completed immutable release manifest changed upstream")
                next_state = dict(retry_state)
                next_state.update(
                    {
                        "repository_signature": self.checkpoint_signature,
                        "watermark": release.record_id,
                        "release_signature": release.signature,
                        "release_version": release.version,
                        "file_count": len(release.files),
                        "completed_at": _isoformat(self.clock()),
                    }
                )
                return SourcePage(
                    records=(),
                    next_state=next_state,
                    complete=True,
                    upstream_count=0,
                    retry_state=retry_state,
                )
        except _ManifestError as error:
            return self._failure_page(
                retry_state,
                error,
                source_record_id=f"{self.name}:release-catalog",
            )

        now = _isoformat(self.clock())
        next_state = dict(retry_state)
        next_state.update(
            {
                "repository_signature": self.checkpoint_signature,
                "stage": _FILES_STAGE,
                "target_release": release.record_id,
                "target_version": release.version,
                "target_manifest_signature": release.signature,
                "target_file_count": len(release.files),
                "file_cursor": 0,
                "control_records_seen": 1,
                "started_at": now,
            }
        )
        return SourcePage(
            records=(self._release_record(release),),
            next_state=next_state,
            complete=False,
            upstream_count=1 + len(release.files),
            retry_state=retry_state,
        )

    def _file_page(self, state: Mapping[str, Any]) -> SourcePage:
        scan = self._scan_boundary(state)
        retry_state = dict(state)
        try:
            release = self._load_release(scan.target_release)
            self._validate_continuity(release, scan)
        except _ManifestError as error:
            stable = _stable_state(state)
            return self._failure_page(
                stable,
                error,
                source_record_id=(f"{self.name}:release:{scan.target_release}:manifest-continuity"),
                summary={
                    "target_release": scan.target_release,
                    "file_cursor": scan.cursor,
                    "expected_file_count": scan.file_count,
                },
            )

        stop = min(scan.cursor + self.page_size, scan.file_count)
        selected = release.files[scan.cursor : stop]
        if len(selected) != stop - scan.cursor:
            stable = _stable_state(state)
            return self._failure_page(
                stable,
                _ManifestError("pinned manifest was truncated while paging"),
                source_record_id=(f"{self.name}:release:{scan.target_release}:manifest-continuity"),
            )
        records = tuple(
            self._file_record(
                release,
                item,
                manifest_index=scan.cursor + offset,
            )
            for offset, item in enumerate(selected)
        )
        next_cursor = stop
        records_seen = scan.control_records_seen + len(records)
        total = 1 + scan.file_count

        if next_cursor < scan.file_count:
            next_state = dict(state)
            next_state.update(
                {
                    "file_cursor": next_cursor,
                    "control_records_seen": records_seen,
                }
            )
            complete = False
        else:
            if records_seen != total:
                stable = _stable_state(state)
                return self._failure_page(
                    stable,
                    _ManifestError(
                        f"control count {records_seen} does not match frozen total {total}"
                    ),
                    source_record_id=(f"{self.name}:release:{scan.target_release}:pagination"),
                )
            next_state = _stable_state(state)
            next_state.update(
                {
                    "repository_signature": self.checkpoint_signature,
                    "watermark": release.record_id,
                    "release_signature": release.signature,
                    "release_version": release.version,
                    "file_count": len(release.files),
                    "completed_at": _isoformat(self.clock()),
                }
            )
            complete = True

        return SourcePage(
            records=records,
            next_state=next_state,
            complete=complete,
            upstream_count=total,
            retry_state=retry_state,
        )

    def _load_release(self, selector: str) -> _Release:
        request_url = self._record_url(selector)
        payload = self._get_json(request_url)
        if not isinstance(payload, Mapping):
            raise _ManifestError("Zenodo release manifest must be a JSON object")

        record_id = _manifest_record_id(payload.get("id"), "release ID")
        concept_record_id = _manifest_record_id(payload.get("conceptrecid"), "concept record ID")
        if concept_record_id != self.concept_record_id:
            raise _ManifestError(
                f"release belongs to concept {concept_record_id}, expected {self.concept_record_id}"
            )
        if selector != self.concept_record_id and record_id != selector:
            raise _ManifestError(f"requested immutable release {selector}, received {record_id}")

        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise _ManifestError("release metadata must be a JSON object")
        title = _manifest_text(metadata.get("title"), "release title", maximum=4_096)
        version = _manifest_optional_text(metadata.get("version"), "release version", maximum=256)
        publication_date = _manifest_date(metadata.get("publication_date"), "publication date")
        created_at = _manifest_optional_datetime(payload.get("created"), "created date")
        updated_at = _manifest_optional_datetime(payload.get("updated"), "updated date")
        doi = _manifest_optional_doi(payload.get("doi"), "release DOI")
        concept_doi = _manifest_optional_doi(payload.get("conceptdoi"), "concept DOI")

        links = payload.get("links")
        if not isinstance(links, Mapping):
            raise _ManifestError("release links must be a JSON object")
        canonical_url = self._release_html_url(record_id, links.get("self_html"))
        api_url = self._release_api_url(record_id, links.get("self"))
        versions_url = self._optional_catalog_url(links.get("versions"), "versions URL")

        raw_files = payload.get("files")
        if not _is_sequence(raw_files):
            raise _ManifestError("release files must be a JSON array")
        if not raw_files:
            raise _ManifestError("release files must not be empty")
        if len(raw_files) > self.max_files:
            raise _ManifestError(
                f"release has {len(raw_files)} files, exceeding limit {self.max_files}"
            )

        files = self._release_files(raw_files, record_id)
        release_metadata = _json_value(metadata, "release metadata")
        signature = content_hash(
            {
                "record_id": record_id,
                "concept_record_id": concept_record_id,
                "doi": doi,
                "concept_doi": concept_doi,
                "created_at": created_at,
                "updated_at": updated_at,
                # Zenodo's concept and immutable-record endpoints can return
                # set-like metadata arrays (notably communities) in different
                # orders. Sign semantic JSON contents, while preserving the
                # upstream representation verbatim on the control record.
                "metadata": _semantic_json(release_metadata),
                "files": [
                    {
                        "key": item.key,
                        "file_id": item.file_id,
                        "size": item.size,
                        "checksum": item.checksum,
                        "download_url": item.download_url,
                    }
                    for item in files
                ],
            }
        )
        return _Release(
            record_id=record_id,
            concept_record_id=concept_record_id,
            version=version,
            title=title,
            publication_date=publication_date,
            created_at=created_at,
            updated_at=updated_at,
            doi=doi,
            concept_doi=concept_doi,
            canonical_url=canonical_url,
            api_url=api_url,
            versions_url=versions_url,
            metadata=release_metadata,
            files=files,
            signature=signature,
        )

    def _release_files(
        self,
        values: Sequence[Any],
        record_id: str,
    ) -> tuple[_ReleaseFile, ...]:
        files: list[_ReleaseFile] = []
        seen_keys: set[str] = set()
        seen_urls: set[str] = set()
        for index, raw in enumerate(values):
            if not isinstance(raw, Mapping):
                raise _ManifestError(f"release file[{index}] must be a JSON object")
            key = _manifest_file_key(raw.get("key"), index)
            folded = key.casefold()
            if folded in seen_keys:
                raise _ManifestError(f"duplicate release file key {key!r}")
            seen_keys.add(folded)

            file_id = _manifest_optional_text(
                raw.get("id"), f"release file[{index}] ID", maximum=512
            )
            size = _manifest_integer(raw.get("size"), f"release file[{index}] size")
            if size > self.max_file_bytes:
                raise _ManifestError(
                    f"release file {key!r} size {size} exceeds limit {self.max_file_bytes}"
                )
            algorithm, digest = _manifest_checksum(
                raw.get("checksum"), f"release file[{index}] checksum"
            )
            file_links = raw.get("links")
            if not isinstance(file_links, Mapping):
                raise _ManifestError(f"release file[{index}] links must be a JSON object")
            download_url = self._download_url(file_links.get("self"), record_id=record_id, key=key)
            if download_url in seen_urls:
                raise _ManifestError(f"duplicate release file URL for {key!r}")
            seen_urls.add(download_url)

            stem = key[:-4]
            match = _PARTITION_RE.fullmatch(stem)
            if match is None:
                raise _ManifestError(f"release file {key!r} has no partition name")
            partition_index = match.group("index")
            files.append(
                _ReleaseFile(
                    key=key,
                    file_id=file_id,
                    size=size,
                    checksum_algorithm=algorithm,
                    checksum_digest=digest,
                    download_url=download_url,
                    partition=match.group("family"),
                    partition_index=(int(partition_index) if partition_index is not None else None),
                )
            )
        return tuple(sorted(files, key=lambda item: _natural_key(item.key)))

    def _scan_boundary(self, state: Mapping[str, Any]) -> _ScanBoundary:
        if state.get("stage") != _FILES_STAGE:
            raise ValueError(f"{self.name}: checkpoint stage changed during scan")
        target_release = _required_record_id(
            state.get("target_release"), "target release", self.name
        )
        target_version = _optional_text(state.get("target_version"), "target version", self.name)
        manifest_signature = _required_text(
            state.get("target_manifest_signature"), "target manifest signature"
        )
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_signature):
            raise ValueError(f"{self.name}: target manifest signature is malformed")
        file_count = _state_integer(state, "target_file_count", self.name, maximum=self.max_files)
        if file_count < 1:
            raise ValueError(f"{self.name}: target_file_count must be positive")
        cursor = _state_integer(state, "file_cursor", self.name, maximum=file_count)
        if cursor >= file_count:
            raise ValueError(f"{self.name}: file_cursor is outside the frozen manifest")
        records_seen = _state_integer(
            state,
            "control_records_seen",
            self.name,
            maximum=1 + file_count,
        )
        if records_seen != 1 + cursor:
            raise ValueError(f"{self.name}: control_records_seen does not match file_cursor")
        _required_datetime(state.get("started_at"), "scan start", self.name)
        return _ScanBoundary(
            target_release=target_release,
            target_version=target_version,
            manifest_signature=manifest_signature,
            file_count=file_count,
            cursor=cursor,
            control_records_seen=records_seen,
        )

    def _validate_continuity(self, release: _Release, scan: _ScanBoundary) -> None:
        if release.record_id != scan.target_release:
            raise _ManifestError(
                f"pinned release changed from {scan.target_release} to {release.record_id}"
            )
        if release.version != scan.target_version:
            raise _ManifestError("pinned release version changed while resuming")
        if len(release.files) != scan.file_count:
            raise _ManifestError(
                f"pinned manifest count changed from {scan.file_count} to {len(release.files)}"
            )
        if release.signature != scan.manifest_signature:
            raise _ManifestError("pinned manifest contents changed while resuming")

    def _release_record(self, release: _Release) -> SourceRecord:
        identifiers = [
            Identifier("openaire:graph-release", release.record_id),
            Identifier("zenodo", release.record_id),
        ]
        links = [Link(release.api_url, relation="release_metadata", crawl=False)]
        if release.doi is not None:
            identifiers.append(Identifier("doi", release.doi))
            links.append(Link(_doi_url(release.doi), relation="doi", crawl=False))
        if release.concept_doi is not None:
            identifiers.append(Identifier("doi", release.concept_doi))
            links.append(Link(_doi_url(release.concept_doi), relation="concept_doi", crawl=False))
        if release.versions_url is not None:
            links.append(Link(release.versions_url, relation="release_catalog", crawl=False))
        raw = {
            "record_type": "release_manifest",
            "release_id": release.record_id,
            "concept_record_id": release.concept_record_id,
            "version": release.version,
            "publication_date": release.publication_date,
            "created_at": release.created_at,
            "updated_at": release.updated_at,
            "release_doi": release.doi,
            "concept_doi": release.concept_doi,
            "metadata": release.metadata,
            "manifest_signature": release.signature,
            "file_count": len(release.files),
            "total_bytes": sum(item.size for item in release.files),
            "upstream_partitions": [item.partition for item in release.files],
            "selection": {
                "scope": "complete_graph_release",
                "partition_filter": None,
                "scientific_filter": None,
                "research_product_type_resolution": "each_payload_object.type",
            },
            "payload_contract": _payload_contract(),
            "snapshot_semantics": _snapshot_semantics(),
        }
        return SourceRecord(
            source_record_id=f"openaire-graph:release:{release.record_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=release.canonical_url,
            title=release.title,
            raw=raw,
            published_at=release.publication_date,
            modified_at=release.updated_at,
            identifiers=tuple(identifiers),
            links=tuple(links),
        )

    def _file_record(
        self,
        release: _Release,
        item: _ReleaseFile,
        *,
        manifest_index: int,
    ) -> SourceRecord:
        identity = f"{release.record_id}/{item.key}"
        raw = {
            "record_type": "dataset_shard",
            "operation": "snapshot",
            "release_id": release.record_id,
            "release_version": release.version,
            "manifest_signature": release.signature,
            "manifest_index": manifest_index,
            "manifest_count": len(release.files),
            "file_key": item.key,
            "file_id": item.file_id,
            "size": item.size,
            "checksum": {
                "algorithm": item.checksum_algorithm,
                "value": item.checksum_digest,
            },
            "entity_partition": item.partition,
            "partition_index": item.partition_index,
            "partition_classification": "upstream_filename_metadata",
            "download_url": item.download_url,
            "license_metadata_record_id": (f"openaire-graph:release:{release.record_id}"),
            "payload_contract": _payload_contract(),
            "snapshot_semantics": _snapshot_semantics(),
        }
        return SourceRecord(
            source_record_id=(
                f"openaire-graph:file:{release.record_id}:{quote(item.key, safe='')}"
            ),
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=item.download_url,
            title=f"{release.title} — {item.key}",
            raw=raw,
            published_at=release.publication_date,
            modified_at=release.updated_at,
            identifiers=(
                Identifier("openaire:graph-file", identity),
                Identifier(item.checksum_algorithm, item.checksum_digest),
            ),
            links=(
                Link(item.download_url, relation="bulk_payload", crawl=False),
                Link(release.api_url, relation="release_metadata", crawl=False),
            ),
        )

    def _failure_page(
        self,
        retry_state: Mapping[str, Any],
        error: Exception,
        *,
        source_record_id: str,
        summary: Mapping[str, Any] | None = None,
    ) -> SourcePage:
        issue = SourceIssue(
            source_record_id=source_record_id,
            stage="source_manifest",
            error=f"{type(error).__name__}: {error}",
            summary=dict(summary or {}),
        )
        return SourcePage(
            records=(),
            next_state=dict(retry_state),
            complete=False,
            issues=(issue,),
            retry_state=dict(retry_state),
        )

    def _get_json(self, url: str) -> Any:
        try:
            response: HttpResponse = self.client.get(
                url,
                headers={"Accept": "application/json"},
                redirect_validator=self._validate_redirect,
            )
        except _ManifestError:
            raise
        if response.status != 200:
            raise _ManifestError(f"release catalog returned HTTP {response.status}")
        self._validate_redirect(response.url)
        content_type = response.headers.get("content-type")
        if content_type is not None and "json" not in content_type.casefold():
            raise _ManifestError("release catalog did not return JSON content")
        if len(response.body) > self.max_manifest_bytes:
            raise _ManifestError(f"release catalog exceeded {self.max_manifest_bytes} bytes")
        try:
            return json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _ManifestError("release catalog is not valid JSON") from error

    def _validate_redirect(self, target: str) -> None:
        parts = _safe_url(target, "release catalog redirect")
        base = urlsplit(self.url)
        if _url_origin(parts) != _url_origin(base):
            raise _ManifestError("release catalog redirect changed origin")
        prefix = base.path.rstrip("/") + "/"
        if not parts.path.startswith(prefix):
            raise _ManifestError("release catalog redirect left the records API")

    def _record_url(self, record_id: str) -> str:
        return f"{self.url}/{quote(record_id, safe='')}"

    def _release_api_url(self, record_id: str, value: Any) -> str:
        expected = self._record_url(record_id)
        if value is None:
            return expected
        actual = self._catalog_url(value, "release API URL")
        if actual != expected:
            raise _ManifestError("release API URL is not canonical")
        return actual

    def _release_html_url(self, record_id: str, value: Any) -> str:
        base = urlsplit(self.url)
        expected = urlunsplit(
            ("https", base.netloc, f"/records/{quote(record_id, safe='')}", "", "")
        )
        if value is None:
            return expected
        actual = self._catalog_url(value, "release HTML URL")
        if actual != expected:
            raise _ManifestError("release HTML URL is not canonical")
        return actual

    def _optional_catalog_url(self, value: Any, field: str) -> str | None:
        if value is None:
            return None
        return self._catalog_url(value, field)

    def _catalog_url(self, value: Any, field: str) -> str:
        url = _manifest_text(value, field, maximum=4_096)
        parts = _safe_url(url, field)
        if _url_origin(parts) != _url_origin(urlsplit(self.url)):
            raise _ManifestError(f"{field} changed origin")
        return urlunsplit(("https", parts.netloc.casefold(), parts.path, "", ""))

    def _download_url(self, value: Any, *, record_id: str, key: str) -> str:
        url = self._catalog_url(value, f"download URL for {key!r}")
        parts = urlsplit(url)
        base_path = urlsplit(self.url).path.rstrip("/")
        expected_prefix = f"{base_path}/{record_id}/files/"
        if not parts.path.startswith(expected_prefix) or not parts.path.endswith("/content"):
            raise _ManifestError(f"download URL for {key!r} is not a file content URL")
        encoded_key = parts.path[len(expected_prefix) : -len("/content")]
        if "/" in encoded_key or unquote(encoded_key) != key:
            raise _ManifestError(f"download URL does not match file key {key!r}")
        return url

    def _validate_state_identity(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        actual = _required_text(
            state.get("repository_signature"), "checkpoint repository signature"
        )
        if actual != self.checkpoint_signature:
            raise ValueError(f"{self.name}: checkpoint belongs to another repository")


def _payload_contract() -> Mapping[str, Any]:
    return {
        "archive": "tar",
        "archive_members": "gzip",
        "member_records": "one_json_object_per_line",
        "normalization": "lossless_before_projection",
        "enumerate_every_partition": True,
        "entity_type_resolution": "each_json_object.type",
        "preserve_all_pids_relations_urls_creators_hosts_dates_rights": True,
        "storage": "parquet",
    }


def _snapshot_semantics() -> Mapping[str, Any]:
    return {
        "authoritative_after_complete_release_is_sealed": True,
        "replace_prior_release": True,
        "explicit_tombstone_stream": False,
        "deletions": "derive_from_absence_between_complete_sealed_releases",
        "partial_release_must_not_be_committed": True,
        "incremental_update_stream": False,
    }


def _stable_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key not in _SCAN_KEYS}


def _api_base_url(value: Any, source: str) -> str:
    url = _required_text(value, "OpenAIRE release catalog URL").rstrip("/")
    parts = _safe_url(url, "OpenAIRE release catalog URL", error_type=ValueError)
    if parts.query or parts.fragment:
        raise ValueError(f"{source}: release catalog URL cannot have query or fragment")
    if not parts.path or parts.path == "/":
        raise ValueError(f"{source}: release catalog URL must include an API path")
    return urlunsplit(("https", parts.netloc.casefold(), parts.path.rstrip("/"), "", ""))


def _safe_url(
    value: Any,
    field: str,
    *,
    error_type: type[ValueError] = _ManifestError,
) -> Any:
    if not isinstance(value, str) or not value or value != value.strip():
        raise error_type(f"{field} must be a nonempty URL")
    parts = urlsplit(value)
    try:
        port = parts.port
    except ValueError:
        raise error_type(f"{field} has an invalid port") from None
    host = (parts.hostname or "").casefold().rstrip(".")
    if (
        parts.scheme.casefold() != "https"
        or not host
        or parts.username is not None
        or parts.password is not None
        or port not in {None, 443}
        or parts.query
        or parts.fragment
        or not parts.path
    ):
        raise error_type(f"{field} must be a credential-free query-free HTTPS URL")
    if host == "localhost" or host.endswith(".localhost"):
        raise error_type(f"{field} host must be public")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise error_type(f"{field} host must be public")
    return parts


def _url_origin(parts: Any) -> tuple[str, str, int]:
    return (
        parts.scheme.casefold(),
        (parts.hostname or "").casefold().rstrip("."),
        parts.port or 443,
    )


def _record_id(value: Any, field: str, source: str) -> str:
    try:
        return _manifest_record_id(value, field)
    except _ManifestError as error:
        raise ValueError(f"{source}: {error}") from None


def _required_record_id(value: Any, field: str, source: str) -> str:
    return _record_id(value, field, source)


def _optional_record_id(value: Any, field: str, source: str) -> str | None:
    if value is None:
        return None
    return _record_id(value, field, source)


def _manifest_record_id(value: Any, field: str) -> str:
    if isinstance(value, bool):
        raise _ManifestError(f"{field} must be a positive decimal integer")
    text = str(value) if isinstance(value, int) else value
    if not isinstance(text, str) or not re.fullmatch(r"[1-9][0-9]*", text):
        raise _ManifestError(f"{field} must be a positive decimal integer")
    return text


def _manifest_file_key(value: Any, index: int) -> str:
    key = _manifest_text(value, f"release file[{index}] key", maximum=512)
    if (
        not key.endswith(".tar")
        or key in {".", ".."}
        or "/" in key
        or "\\" in key
        or any(ord(character) < 32 for character in key)
    ):
        raise _ManifestError(f"release file[{index}] key must be a safe .tar filename")
    return key


def _manifest_checksum(value: Any, field: str) -> tuple[str, str]:
    text = _manifest_text(value, field, maximum=256)
    match = _CHECKSUM_RE.fullmatch(text)
    if match is None:
        raise _ManifestError(f"{field} must be an md5, sha256, or sha512 checksum")
    algorithm = match.group("algorithm").casefold()
    digest = match.group("digest").casefold()
    if len(digest) != _CHECKSUM_LENGTHS[algorithm]:
        raise _ManifestError(f"{field} has the wrong digest length")
    return algorithm, digest


def _manifest_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _ManifestError(f"{field} must be a positive integer")
    return value


def _bounded_positive_integer(
    value: Any,
    field: str,
    source: str,
    *,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: {field} must be a positive integer") from None
    if parsed < 1 or parsed > maximum:
        raise ValueError(f"{source}: {field} must be between 1 and {maximum}")
    return parsed


def _state_integer(
    state: Mapping[str, Any],
    key: str,
    source: str,
    *,
    maximum: int,
) -> int:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{source}: checkpoint {key} must be an integer between 0 and {maximum}")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _optional_text(value: Any, field: str, source: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{source}: {field} must be a nonempty string")
    return value


def _manifest_text(value: Any, field: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
    ):
        raise _ManifestError(f"{field} must be a nonempty string of at most {maximum} characters")
    return value


def _manifest_optional_text(value: Any, field: str, *, maximum: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    return _manifest_text(value, field, maximum=maximum)


def _manifest_date(value: Any, field: str) -> str:
    text = _manifest_text(value, field, maximum=32)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise _ManifestError(f"{field} must be an ISO calendar date") from None
    if parsed.isoformat() != text:
        raise _ManifestError(f"{field} must use YYYY-MM-DD format")
    return text


def _manifest_optional_datetime(value: Any, field: str) -> str | None:
    if value is None:
        return None
    text = _manifest_text(value, field, maximum=64)
    return _normalized_datetime(text, field, _ManifestError)


def _required_datetime(value: Any, field: str, source: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source}: {field} must be an ISO UTC timestamp")
    return _normalized_datetime(value, field, ValueError, prefix=f"{source}: ")


def _normalized_datetime(
    value: str,
    field: str,
    error_type: type[ValueError],
    *,
    prefix: str = "",
) -> str:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        raise error_type(f"{prefix}{field} must be an ISO timestamp") from None
    if parsed.tzinfo is None:
        raise error_type(f"{prefix}{field} must include a UTC offset")
    return _isoformat(parsed)


def _manifest_optional_doi(value: Any, field: str) -> str | None:
    if value is None:
        return None
    text = _manifest_text(value, field, maximum=512)
    normalized = text.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    if not _DOI_RE.fullmatch(normalized):
        raise _ManifestError(f"{field} is malformed")
    return normalized.casefold()


def _doi_url(doi: str) -> str:
    return f"https://doi.org/{quote(doi, safe='/()')}"


def _json_value(value: Any, field: str) -> Any:
    try:
        serialized = json.dumps(value, allow_nan=False, sort_keys=True)
        return json.loads(serialized)
    except (TypeError, ValueError):
        raise _ManifestError(f"{field} must contain only finite JSON values") from None


def _semantic_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if _is_sequence(value):
        normalized = [_semantic_json(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (1, int(component)) if component.isdigit() else (0, component.casefold())
        for component in re.split(r"([0-9]+)", value)
        if component
    )


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
