from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit
from xml.etree import ElementTree

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourcePage, SourceRecord
from modelome.normalize import content_hash

Clock = Callable[[], datetime]

_BUCKET_URL = "https://softwareheritage.s3.amazonaws.com/"
_GRAPH_PREFIX = "graph/"
_RELEASE_LIST_URL = (
    "https://docs.softwareheritage.org/_sources/devel/swh-export/graph/dataset.rst.txt"
)
_SCAN_STAGE = "origin_orc_shards"
_ORIGIN_FILE_RE = re.compile(r"^origin-[A-Za-z0-9][A-Za-z0-9._-]*\.orc$")
_SCAN_KEYS = frozenset(
    {
        "stage",
        "target_release",
        "target_manifest_signature",
        "target_approval_document_sha256",
        "target_shard_count",
        "shard_cursor",
        "started_at",
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Object:
    key: str
    size: int
    etag: str
    last_modified: str
    checksum_algorithms: tuple[str, ...]
    checksum_type: str | None

    def descriptor(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "size": self.size,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "checksum_algorithms": list(self.checksum_algorithms),
            "checksum_type": self.checksum_type,
        }


@dataclass(frozen=True, slots=True)
class _Manifest:
    release: str
    objects: tuple[_Object, ...]
    metadata: _ExportMetadata
    approval_document_sha256: str
    signature: str


@dataclass(frozen=True, slots=True)
class _ExportMetadata:
    url: str
    sha256: str
    flavor: str
    formats: tuple[str, ...]
    object_types: tuple[str, ...]
    export_start: str
    export_end: str
    tool_name: str
    tool_version: str


@dataclass(frozen=True, slots=True)
class _ListPage:
    common_prefixes: tuple[str, ...]
    objects: tuple[_Object, ...]
    next_token: str | None


class SoftwareHeritageOriginSourceAdapter:
    """Enumerate the complete origin table of an immutable SWH graph export.

    The public S3 ``ListObjectsV2`` API is used as a small control plane. The
    latest dated full graph export is discovered dynamically, then every ORC
    object under ``orc/origin/`` is frozen into a manifest before controls are
    emitted. No forge, repository name, topic, language, or model term is used.

    This is a census of origins held by Software Heritage at one export instant,
    not a census of every repository that has ever existed. Acquisition controls
    contain no neural-model assertion; a later projection may select exact GitHub
    repository URLs while the lake retains all origin rows.

    Contracts:
    https://docs.softwareheritage.org/devel/swh-export/graph/dataset.html
    https://docs.softwareheritage.org/devel/swh-export/graph/schema.html
    """

    def __init__(
        self,
        *,
        name: str = "software-heritage-origins",
        bucket_url: str = _BUCKET_URL,
        graph_prefix: str = _GRAPH_PREFIX,
        release_list_url: str = _RELEASE_LIST_URL,
        page_size: int = 32,
        list_page_size: int = 1_000,
        max_list_bytes: int = 16 * 1024 * 1024,
        max_release_list_bytes: int = 4 * 1024 * 1024,
        max_metadata_bytes: int = 8 * 1024 * 1024,
        max_list_pages: int = 1_000,
        max_manifest_objects: int = 100_000,
        settlement_age_hours: int = 7 * 24,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.bucket_url = _https_directory_url(bucket_url, "bucket URL")
        self.graph_prefix = _safe_prefix(graph_prefix)
        self.release_list_url = _https_url(release_list_url, "release list URL")
        self.page_size = _positive_integer(page_size, "page size", maximum=10_000)
        self.list_page_size = _positive_integer(list_page_size, "list page size", maximum=1_000)
        self.max_list_bytes = _positive_integer(
            max_list_bytes, "maximum list bytes", maximum=64 * 1024 * 1024
        )
        self.max_release_list_bytes = _positive_integer(
            max_release_list_bytes,
            "maximum release list bytes",
            maximum=64 * 1024 * 1024,
        )
        self.max_metadata_bytes = _positive_integer(
            max_metadata_bytes,
            "maximum metadata bytes",
            maximum=64 * 1024 * 1024,
        )
        self.max_list_pages = _positive_integer(
            max_list_pages, "maximum list pages", maximum=100_000
        )
        self.max_manifest_objects = _positive_integer(
            max_manifest_objects,
            "maximum manifest objects",
            maximum=10_000_000,
        )
        self.settlement_age_hours = _positive_integer(
            settlement_age_hours,
            "settlement age hours",
            maximum=24 * 365,
        )
        self.client = client or HttpClient(
            max_response_bytes=max(
                self.max_list_bytes,
                self.max_release_list_bytes,
                self.max_metadata_bytes,
            )
        )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "software-heritage-origin-orc-control-v1",
                "bucket_url": self.bucket_url,
                "graph_prefix": self.graph_prefix,
                "release_list_url": self.release_list_url,
                "page_size": self.page_size,
                "list_page_size": self.list_page_size,
                "max_manifest_objects": self.max_manifest_objects,
                "settlement_age_hours": self.settlement_age_hours,
            }
        )

    @property
    def coverage_semantics(self) -> Mapping[str, Any]:
        return {
            "enumerates": "every origin row in the selected immutable full graph export",
            "discovery_basis": "Software Heritage archive holdings",
            "github_name_or_keyword_filter": False,
            "historical_github_census": False,
            "contains_repository_content": False,
            "snapshot_semantics": True,
            "settlement_age_hours": self.settlement_age_hours,
            "release_approval": "official full-graph dataset list plus export metadata",
        }

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if not isinstance(state, Mapping):
            raise TypeError(f"{self.name}: checkpoint state must be a mapping")
        self._validate_signature(state)
        stage = _optional_text(state.get("stage"))
        if not stage:
            return self._begin_scan(state)
        if stage != _SCAN_STAGE:
            raise ValueError(f"{self.name}: unknown checkpoint stage {stage!r}")
        return self._resume_scan(state)

    def _begin_scan(self, state: Mapping[str, Any]) -> SourcePage:
        release, approval_sha256 = self._latest_release()
        manifest = self._manifest(release, approval_document_sha256=approval_sha256)
        self._validate_settlement(manifest)
        stable = _stable_state(state)
        watermark = _optional_release(state.get("watermark"), self.name)
        if watermark is not None and release < watermark:
            raise ValueError(
                f"{self.name}: release listing moved backward from {watermark} to {release}"
            )
        if release == watermark:
            prior_signature = _required_digest(
                state.get("manifest_signature"), "completed manifest signature"
            )
            prior_count = _nonnegative_integer(state.get("shard_count"), "completed shard count")
            if prior_signature != manifest.signature or prior_count != len(manifest.objects):
                raise ValueError(
                    f"{self.name}: completed immutable origin manifest changed upstream"
                )
            complete_state = dict(stable)
            complete_state.update(
                {
                    "checkpoint_signature": self.checkpoint_signature,
                    "watermark": release,
                    "manifest_signature": manifest.signature,
                    "shard_count": len(manifest.objects),
                    "checked_at": _isoformat(self.clock()),
                }
            )
            return SourcePage(
                records=(),
                next_state=complete_state,
                complete=True,
                upstream_count=0,
                retry_state=stable,
            )

        scan = dict(stable)
        scan.update(
            {
                "checkpoint_signature": self.checkpoint_signature,
                "stage": _SCAN_STAGE,
                "target_release": release,
                "target_manifest_signature": manifest.signature,
                "target_approval_document_sha256": approval_sha256,
                "target_shard_count": len(manifest.objects),
                "shard_cursor": 0,
                "started_at": _isoformat(self.clock()),
            }
        )
        return self._emit_page(scan, manifest)

    def _resume_scan(self, state: Mapping[str, Any]) -> SourcePage:
        release = _required_release(state.get("target_release"), self.name)
        approved, approval_sha256 = self._approved_releases()
        if release not in approved:
            raise ValueError(f"{self.name}: frozen release is no longer officially approved")
        expected_approval = _required_digest(
            state.get("target_approval_document_sha256"),
            "target approval document digest",
        )
        if approval_sha256 != expected_approval:
            raise ValueError(f"{self.name}: official release list changed during frozen scan")
        manifest = self._manifest(
            release,
            approval_document_sha256=approval_sha256,
        )
        self._validate_settlement(manifest)
        signature = _required_digest(
            state.get("target_manifest_signature"), "target manifest signature"
        )
        count = _nonnegative_integer(state.get("target_shard_count"), "target shard count")
        if signature != manifest.signature or count != len(manifest.objects):
            raise ValueError(f"{self.name}: frozen origin manifest changed during scan")
        return self._emit_page(state, manifest)

    def _emit_page(self, state: Mapping[str, Any], manifest: _Manifest) -> SourcePage:
        cursor = _nonnegative_integer(state.get("shard_cursor"), "shard cursor")
        if cursor >= len(manifest.objects):
            raise ValueError(f"{self.name}: shard cursor is outside the manifest")
        stop = min(cursor + self.page_size, len(manifest.objects))
        records = tuple(
            self._control_record(manifest, index, item)
            for index, item in enumerate(
                manifest.objects[cursor:stop],
                start=cursor,
            )
        )
        retry_state = dict(state)
        if stop < len(manifest.objects):
            next_state = dict(state)
            next_state["shard_cursor"] = stop
            return SourcePage(
                records=records,
                next_state=next_state,
                complete=False,
                upstream_count=len(manifest.objects),
                retry_state=retry_state,
            )

        next_state = _stable_state(state)
        next_state.update(
            {
                "checkpoint_signature": self.checkpoint_signature,
                "watermark": manifest.release,
                "manifest_signature": manifest.signature,
                "shard_count": len(manifest.objects),
                "completed_at": _isoformat(self.clock()),
            }
        )
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(manifest.objects),
            retry_state=retry_state,
        )

    def _latest_release(self) -> tuple[str, str]:
        prefixes = self._list_all(prefix=self.graph_prefix, delimiter="/")[0]
        releases: list[str] = []
        expected_prefix = re.escape(self.graph_prefix)
        release_re = re.compile(
            rf"^{expected_prefix}(?P<release>[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})/$"
        )
        for prefix in prefixes:
            match = release_re.fullmatch(prefix)
            if match is None:
                continue
            release = match.group("release")
            _required_release(release, self.name)
            releases.append(release)
        if not releases:
            raise ValueError(f"{self.name}: no dated full graph exports were listed")
        if len(releases) != len(set(releases)):
            raise ValueError(f"{self.name}: release listing contains duplicates")
        approved, approval_sha256 = self._approved_releases()
        candidates = sorted(set(releases).intersection(approved))
        if not candidates:
            raise ValueError(
                f"{self.name}: S3 has no date approved by the official full-graph list"
            )
        return candidates[-1], approval_sha256

    def _approved_releases(self) -> tuple[frozenset[str], str]:
        response = self.client.get(
            self.release_list_url,
            headers={"Accept": "text/plain", "Accept-Encoding": "identity"},
            redirect_validator=lambda url: _validate_exact_document_url(
                url, expected=self.release_list_url, label="release list"
            ),
        )
        body = _response_body(
            response,
            expected_url=self.release_list_url,
            maximum=self.max_release_list_bytes,
            label=f"{self.name}: release list",
        )
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{self.name}: release list is not UTF-8") from error
        start_match = re.search(r"(?m)^Full graph datasets\r?\n-+\r?$", text)
        if start_match is None:
            raise ValueError(f"{self.name}: official full-graph section is absent")
        remainder = text[start_match.end() :]
        stop_match = re.search(r"(?m)^Teaser datasets\r?\n-+\r?$", remainder)
        if stop_match is None:
            raise ValueError(f"{self.name}: official teaser boundary is absent")
        full_section = remainder[: stop_match.start()]
        releases = tuple(
            re.findall(
                r"(?m)^\.\. _graph-dataset-([0-9]{4}-[0-9]{2}-[0-9]{2}):\s*$",
                full_section,
            )
        )
        if not releases or len(releases) != len(set(releases)):
            raise ValueError(f"{self.name}: official full-graph release list is invalid")
        for release in releases:
            _required_release(release, self.name)
        return frozenset(releases), hashlib.sha256(body).hexdigest()

    def _manifest(
        self,
        release: str,
        *,
        approval_document_sha256: str,
    ) -> _Manifest:
        release = _required_release(release, self.name)
        approval_document_sha256 = _required_digest(
            approval_document_sha256, "approval document digest"
        )
        metadata = self._export_metadata(release)
        prefix = f"{self.graph_prefix}{release}/orc/origin/"
        _, listed = self._list_all(prefix=prefix, delimiter=None)
        objects: list[_Object] = []
        for item in listed:
            if item.key == prefix and item.size == 0:
                continue
            filename = item.key.removeprefix(prefix)
            if (
                not item.key.startswith(prefix)
                or "/" in filename
                or _ORIGIN_FILE_RE.fullmatch(filename) is None
            ):
                raise ValueError(
                    f"{self.name}: unexpected object in origin ORC prefix: {item.key!r}"
                )
            if item.size < 1:
                raise ValueError(f"{self.name}: origin ORC object is empty")
            objects.append(item)
        objects.sort(key=lambda item: item.key)
        if not objects:
            raise ValueError(f"{self.name}: origin ORC manifest is empty for {release}")
        if len(objects) > self.max_manifest_objects:
            raise ValueError(f"{self.name}: origin ORC manifest exceeds configured bound")
        keys = [item.key for item in objects]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.name}: origin ORC manifest contains duplicate keys")
        signature = content_hash(
            {
                "release": release,
                "table": "origin",
                "export_metadata_sha256": metadata.sha256,
                "objects": [item.descriptor() for item in objects],
            }
        )
        return _Manifest(
            release=release,
            objects=tuple(objects),
            metadata=metadata,
            approval_document_sha256=approval_document_sha256,
            signature=signature,
        )

    def _export_metadata(self, release: str) -> _ExportMetadata:
        url = f"{self.bucket_url}{self.graph_prefix}{release}/meta/export.json"
        response = self.client.get(
            url,
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
            redirect_validator=lambda target: _validate_exact_document_url(
                target, expected=url, label="export metadata"
            ),
        )
        body = _response_body(
            response,
            expected_url=url,
            maximum=self.max_metadata_bytes,
            label=f"{self.name}: export metadata",
        )
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{self.name}: export metadata is invalid JSON") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"{self.name}: export metadata must be an object")
        if value.get("flavor") != "full":
            raise ValueError(f"{self.name}: export metadata is not a full graph")
        formats = _string_sequence(value.get("formats"), "export formats")
        if "orc" not in {item.casefold() for item in formats}:
            raise ValueError(f"{self.name}: full graph does not publish ORC")
        object_types = _string_sequence(value.get("object_types"), "object types")
        if "origin" not in {item.casefold() for item in object_types}:
            raise ValueError(f"{self.name}: full graph does not include origins")
        export_start = _required_exact_text(value.get("export_start"), "export_start")
        export_end = _required_exact_text(value.get("export_end"), "export_end")
        start = _utc_instant(export_start, f"{self.name}: export_start")
        end = _utc_instant(export_end, f"{self.name}: export_end")
        if end < start or start.date().isoformat() != release:
            raise ValueError(f"{self.name}: export time window is inconsistent")
        tool = value.get("tool")
        if not isinstance(tool, Mapping):
            raise ValueError(f"{self.name}: export metadata lacks tool identity")
        tool_name = _required_exact_text(tool.get("name"), "export tool name")
        tool_version = _required_exact_text(tool.get("version"), "export tool version")
        if tool_name != "swh.export":
            raise ValueError(f"{self.name}: export metadata has an unexpected tool")
        return _ExportMetadata(
            url=url,
            sha256=hashlib.sha256(body).hexdigest(),
            flavor="full",
            formats=formats,
            object_types=object_types,
            export_start=export_start,
            export_end=export_end,
            tool_name=tool_name,
            tool_version=tool_version,
        )

    def _validate_settlement(self, manifest: _Manifest) -> None:
        now = self.clock()
        if not isinstance(now, datetime):
            raise TypeError("clock must return a datetime")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        boundary = now.astimezone(UTC) - timedelta(hours=self.settlement_age_hours)
        newest = max(
            _utc_instant(item.last_modified, f"{self.name}: object LastModified")
            for item in manifest.objects
        )
        export_end = _utc_instant(
            manifest.metadata.export_end,
            f"{self.name}: export_end",
        )
        newest = max(newest, export_end)
        if newest > boundary:
            raise ValueError(
                f"{self.name}: latest origin export has not reached its "
                f"{self.settlement_age_hours}-hour settlement age"
            )

    def _list_all(
        self,
        *,
        prefix: str,
        delimiter: str | None,
    ) -> tuple[tuple[str, ...], tuple[_Object, ...]]:
        token: str | None = None
        pages = 0
        prefixes: list[str] = []
        objects: list[_Object] = []
        seen_tokens: set[str] = set()
        while True:
            pages += 1
            if pages > self.max_list_pages:
                raise ValueError(f"{self.name}: S3 listing exceeds page bound")
            params: dict[str, str | int] = {
                "list-type": "2",
                "prefix": prefix,
                "max-keys": self.list_page_size,
            }
            if delimiter is not None:
                params["delimiter"] = delimiter
            if token is not None:
                params["continuation-token"] = token
            response = self.client.get(
                self.bucket_url,
                params=params,
                headers={"Accept": "application/xml", "Accept-Encoding": "identity"},
                redirect_validator=self._validate_redirect,
            )
            page = self._parse_list_response(response, requested_prefix=prefix)
            prefixes.extend(page.common_prefixes)
            objects.extend(page.objects)
            if page.next_token is None:
                return tuple(prefixes), tuple(objects)
            if page.next_token in seen_tokens:
                raise ValueError(f"{self.name}: S3 continuation token repeated")
            seen_tokens.add(page.next_token)
            token = page.next_token

    def _parse_list_response(
        self,
        response: HttpResponse | Any,
        *,
        requested_prefix: str,
    ) -> _ListPage:
        if int(getattr(response, "status", 0)) != 200:
            raise ValueError(f"{self.name}: S3 list returned HTTP {response.status}")
        body = getattr(response, "body", None)
        if not isinstance(body, bytes):
            raise TypeError(f"{self.name}: S3 list body must be bytes")
        if len(body) > self.max_list_bytes:
            raise ValueError(f"{self.name}: S3 list response exceeds byte bound")
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as error:
            raise ValueError(f"{self.name}: invalid S3 ListObjectsV2 XML") from error
        if _local(root.tag) != "ListBucketResult":
            raise ValueError(f"{self.name}: unexpected S3 listing document")
        prefix = _one_text(root, "Prefix", required=True)
        if prefix != requested_prefix:
            raise ValueError(f"{self.name}: S3 listing prefix changed")
        truncated_text = _one_text(root, "IsTruncated", required=True)
        if truncated_text not in {"true", "false"}:
            raise ValueError(f"{self.name}: invalid S3 truncation flag")
        truncated = truncated_text == "true"
        next_token = _one_text(root, "NextContinuationToken")
        if truncated != (next_token is not None):
            raise ValueError(f"{self.name}: inconsistent S3 continuation metadata")

        common_prefixes = tuple(
            _one_text(element, "Prefix", required=True)
            for element in _children(root, "CommonPrefixes")
        )
        objects = tuple(_parse_object(item, self.name) for item in _children(root, "Contents"))
        return _ListPage(
            common_prefixes=common_prefixes,
            objects=objects,
            next_token=next_token,
        )

    def _control_record(
        self,
        manifest: _Manifest,
        index: int,
        item: _Object,
    ) -> SourceRecord:
        url = f"{self.bucket_url}{quote(item.key, safe='/-._~')}"
        filename = item.key.rsplit("/", 1)[-1]
        source_record_id = f"software-heritage:origin-orc:{manifest.release}:{filename}"
        return SourceRecord(
            source_record_id=source_record_id,
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=url,
            title=f"Software Heritage origin table {manifest.release} shard {index + 1}",
            published_at=f"{manifest.release}T00:00:00Z",
            identifiers=(
                Identifier(
                    "software-heritage:origin-orc-shard",
                    f"{manifest.release}/{filename}",
                ),
            ),
            links=(Link(url, relation="bulk_payload", crawl=False),),
            raw={
                "record_type": "software_heritage_origin_orc_shard",
                "release": manifest.release,
                "table": "origin",
                "object_key": item.key,
                "stable_object_url": url,
                "expected_bytes": item.size,
                "object_etag": item.etag,
                "object_last_modified": item.last_modified,
                "object_checksum_algorithms": list(item.checksum_algorithms),
                "object_checksum_type": item.checksum_type,
                "manifest_index": index,
                "manifest_count": len(manifest.objects),
                "manifest_signature": manifest.signature,
                "release_approval_url": self.release_list_url,
                "release_approval_document_sha256": (manifest.approval_document_sha256),
                "export_metadata_url": manifest.metadata.url,
                "export_metadata_sha256": manifest.metadata.sha256,
                "export_flavor": manifest.metadata.flavor,
                "export_formats": list(manifest.metadata.formats),
                "export_object_types": list(manifest.metadata.object_types),
                "export_start": manifest.metadata.export_start,
                "export_end": manifest.metadata.export_end,
                "export_tool": {
                    "name": manifest.metadata.tool_name,
                    "version": manifest.metadata.tool_version,
                },
                "archive_format": "orc",
                "coverage_scope": "all_software_heritage_origins_in_export",
                "github_name_or_keyword_filter": False,
                "historical_github_census": False,
                "contains_repository_content": False,
                "settlement_age_hours": self.settlement_age_hours,
            },
        )

    def _validate_signature(self, state: Mapping[str, Any]) -> None:
        signature = _optional_text(state.get("checkpoint_signature"))
        if signature and signature != self.checkpoint_signature:
            raise ValueError(
                f"{self.name}: checkpoint belongs to a different adapter configuration"
            )

    def _validate_redirect(self, url: str) -> None:
        parts = urlsplit(url)
        expected = urlsplit(self.bucket_url)
        if (
            parts.scheme.casefold() != "https"
            or (parts.hostname or "").casefold() != (expected.hostname or "").casefold()
            or parts.port not in {None, 443}
            or parts.username is not None
            or parts.password is not None
        ):
            raise ValueError(f"{self.name}: S3 listing redirected outside its bucket")


def _parse_object(element: ElementTree.Element, source: str) -> _Object:
    key = _required_exact_text(_one_text(element, "Key", required=True), "object key")
    size = _nonnegative_integer(_one_text(element, "Size", required=True), "object size")
    etag = _required_exact_text(_one_text(element, "ETag", required=True), "object ETag")
    last_modified = _required_exact_text(
        _one_text(element, "LastModified", required=True), "object LastModified"
    )
    _utc_instant(last_modified, f"{source}: object LastModified")
    algorithms = tuple(
        _required_exact_text(value, "checksum algorithm")
        for value in _all_text(element, "ChecksumAlgorithm")
    )
    checksum_type = _one_text(element, "ChecksumType")
    if checksum_type is not None:
        checksum_type = _required_exact_text(checksum_type, "checksum type")
    return _Object(
        key=key,
        size=size,
        etag=etag,
        last_modified=last_modified,
        checksum_algorithms=algorithms,
        checksum_type=checksum_type,
    )


def _children(element: ElementTree.Element, local_name: str) -> list[ElementTree.Element]:
    return [child for child in element if _local(child.tag) == local_name]


def _one_text(
    element: ElementTree.Element,
    local_name: str,
    *,
    required: bool = False,
) -> str | None:
    values = _all_text(element, local_name)
    if len(values) > 1:
        raise ValueError(f"S3 XML repeats singleton {local_name}")
    if not values:
        if required:
            raise ValueError(f"S3 XML is missing {local_name}")
        return None
    return values[0]


def _all_text(element: ElementTree.Element, local_name: str) -> list[str]:
    values: list[str] = []
    for child in element:
        if _local(child.tag) != local_name:
            continue
        if child.text is None:
            raise ValueError(f"S3 XML {local_name} is empty")
        values.append(child.text)
    return values


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _stable_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in state.items() if key not in _SCAN_KEYS}


def _required_release(value: Any, source: str) -> str:
    text = _required_exact_text(value, "release")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{source}: release must be an ISO calendar date") from error
    if parsed.isoformat() != text:
        raise ValueError(f"{source}: release must be canonical")
    return text


def _optional_release(value: Any, source: str) -> str | None:
    return None if value is None else _required_release(value, source)


def _required_digest(value: Any, label: str) -> str:
    text = _required_exact_text(value, label).casefold()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _safe_prefix(value: Any) -> str:
    prefix = _required_exact_text(value, "graph prefix")
    if (
        not prefix.endswith("/")
        or prefix.startswith("/")
        or "//" in prefix
        or any(segment in {"", ".", ".."} for segment in prefix[:-1].split("/"))
        or not prefix.isascii()
    ):
        raise ValueError("graph prefix must be a safe relative directory")
    return prefix


def _response_body(
    response: HttpResponse | Any,
    *,
    expected_url: str,
    maximum: int,
    label: str,
) -> bytes:
    if int(getattr(response, "status", 0)) != 200:
        raise ValueError(f"{label} returned HTTP {response.status}")
    _validate_exact_document_url(
        str(getattr(response, "url", "")), expected=expected_url, label=label
    )
    body = getattr(response, "body", None)
    if not isinstance(body, bytes):
        raise TypeError(f"{label} body must be bytes")
    if len(body) > maximum:
        raise ValueError(f"{label} exceeds byte bound")
    return body


def _validate_exact_document_url(url: str, *, expected: str, label: str) -> None:
    if url != expected:
        raise ValueError(f"{label} URL changed")
    parts = urlsplit(url)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"{label} URL is unsafe")


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    result = tuple(_required_exact_text(item, label) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicates")
    return result


def _https_url(value: Any, label: str) -> str:
    text = _required_exact_text(value, label)
    parts = urlsplit(text)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"{label} must be a credential-free HTTPS URL")
    return text


def _https_directory_url(value: Any, label: str) -> str:
    text = _required_exact_text(value, label)
    parts = urlsplit(text)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"{label} must be an HTTPS origin or directory URL")
    path = parts.path or "/"
    if not path.endswith("/"):
        path += "/"
    return urlunsplit(("https", parts.netloc.casefold(), path, "", ""))


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _required_exact_text(value: Any, label: str) -> str:
    text = _required_text(value, label)
    if text != value:
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return text


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_integer(value: Any, label: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1 or value > maximum:
        raise ValueError(f"{label} must be between 1 and {maximum}")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _utc_instant(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{label} must be a UTC instant")
    return parsed


def _isoformat(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("clock must return a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
