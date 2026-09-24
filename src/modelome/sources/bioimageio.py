from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelRelationHint,
    ModelStatus,
    ReleaseHint,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, extract_urls, identifier_from_url

Clock = Callable[[], datetime]

_DEFAULT_INDEX_URL = "https://bioimage-io.github.io/collection/index.json"
_DEFAULT_ARTIFACT_BASE_URL = "https://hypha.aicell.io/bioimage-io/artifacts"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOI_RE = re.compile(r"(?i)^10\.\d{4,9}/\S+$")
_MAX_ID_CHARS = 1_024
_MAX_VERSION_CHARS = 256
_MAX_PATH_CHARS = 16_384


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _CatalogRecord:
    source_record_id: str
    model_id: str
    alias: str
    version: str | None
    source: str | None
    sha256: str | None
    created_at: str | None
    signature: str
    raw: Mapping[str, Any]

    def checkpoint_value(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "alias": self.alias,
            "version": self.version,
            "source": self.source,
            "sha256": self.sha256,
            "created_at": self.created_at,
            "signature": self.signature,
        }


class BioImageIoSourceAdapter:
    """Enumerate every version in the official public BioImage.IO model index.

    BioImage.IO publishes a static index containing every public resource and
    the exact SHA-256 of each version's resource-description file.  This adapter
    selects only records whose upstream resource type is ``model``; it does not
    use model names, tags, tasks, architectures, or scientific-domain filters.

    Each changed version is resolved through the public artifact endpoint and
    its exact ``bioimageio.yaml``/``rdf.yaml`` bytes are downloaded and verified
    against the index checksum.  Thus the retained evidence includes both the
    structured artifact manifest and the byte-exact upstream model card.  Page
    size bounds the number of model versions (and therefore detail/RDF download
    pairs) attempted in one call.

    Completed checkpoints keep only compact catalog controls.  A later scan
    fetches details for new or changed versions and emits explicit tombstones
    for removed versions.  An index that changes during a paginated scan causes
    a safe restart from the previous completed checkpoint.
    """

    coverage_limitation = (
        "Covers model resources present in BioImage.IO's official public collection "
        "index. Draft, private, rejected, or otherwise unpublished resources are not "
        "visible through that index. Referenced weight files are indexed and checksum "
        "evidence is retained, but binary weights are not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "bioimageio",
        artifact_source: str | None = None,
        index_url: str = _DEFAULT_INDEX_URL,
        artifact_base_url: str = _DEFAULT_ARTIFACT_BASE_URL,
        workspace: str = "bioimage-io",
        page_size: int = 20,
        max_index_bytes: int = 16 * 1024 * 1024,
        max_artifact_bytes: int = 16 * 1024 * 1024,
        max_rdf_bytes: int = 16 * 1024 * 1024,
        max_items: int = 100_000,
        max_versions_per_model: int = 10_000,
        max_links: int = 4_096,
        max_text_chars: int = 8 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name", _MAX_ID_CHARS)
        self.artifact_source = _required_text(
            artifact_source or self.name, "artifact source", _MAX_ID_CHARS
        )
        self.index_url = _web_url(index_url, self.name, "index URL")
        self.artifact_base_url = _web_url(
            artifact_base_url, self.name, "artifact base URL"
        ).rstrip("/")
        self.workspace = _required_text(workspace, "workspace", _MAX_ID_CHARS)
        if "/" in self.workspace or any(character.isspace() for character in self.workspace):
            raise ValueError(f"{self.name}: workspace must be one URL path segment")
        self.page_size = _bounded_positive_int(page_size, "page_size", self.name, 100)
        self.max_index_bytes = _positive_int(max_index_bytes, "max_index_bytes", self.name)
        self.max_artifact_bytes = _positive_int(
            max_artifact_bytes, "max_artifact_bytes", self.name
        )
        self.max_rdf_bytes = _positive_int(max_rdf_bytes, "max_rdf_bytes", self.name)
        self.max_items = _positive_int(max_items, "max_items", self.name)
        self.max_versions_per_model = _positive_int(
            max_versions_per_model, "max_versions_per_model", self.name
        )
        self.max_links = _positive_int(max_links, "max_links", self.name)
        self.max_text_chars = _positive_int(max_text_chars, "max_text_chars", self.name)
        self.client = client or HttpClient(
            max_response_bytes=max(
                self.max_index_bytes,
                self.max_artifact_bytes,
                self.max_rdf_bytes,
            )
        )
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "bioimageio-public-collection-v1",
                "artifact_source": self.artifact_source,
                "index_url": self.index_url,
                "artifact_base_url": self.artifact_base_url,
                "workspace": self.workspace,
                "page_size": self.page_size,
                "max_index_bytes": self.max_index_bytes,
                "max_artifact_bytes": self.max_artifact_bytes,
                "max_rdf_bytes": self.max_rdf_bytes,
                "max_items": self.max_items,
                "max_versions_per_model": self.max_versions_per_model,
                "max_links": self.max_links,
                "max_text_chars": self.max_text_chars,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        known_records = self._checkpoint_records(state.get("known_records"))
        in_progress = "index_digest" in state
        operation_offset = _state_count(state, "operation_offset", self.name) or 0
        if operation_offset and not in_progress:
            raise ValueError(
                f"{self.name}: operation_offset checkpoint is missing index_digest"
            )

        headers = {"Accept": "application/json"}
        if not in_progress:
            if etag := _text(state.get("etag")):
                headers["If-None-Match"] = etag
            if modified := _text(state.get("http_last_modified")):
                headers["If-Modified-Since"] = modified

        response: HttpResponse = self.client.get(self.index_url, headers=headers)
        if response.status == 304:
            if in_progress:
                raise ValueError(
                    f"{self.name}: index returned 304 during an unfinished scan"
                )
            if not known_records:
                raise ValueError(
                    f"{self.name}: index returned 304 without a completed catalog checkpoint"
                )
            next_state = self._completed_state(
                known_records,
                watermark=_text(state.get("watermark")) or None,
                etag=_header(response.headers, "etag") or _text(state.get("etag")) or None,
                http_last_modified=(
                    _header(response.headers, "last-modified")
                    or _text(state.get("http_last_modified"))
                    or None
                ),
            )
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=len(known_records),
            )

        _require_status(response, self.name, "collection index")
        if len(response.body) > self.max_index_bytes:
            raise ValueError(
                f"{self.name}: collection index exceeds {self.max_index_bytes} bytes"
            )
        digest = hashlib.sha256(response.body).hexdigest()
        payload = response.json()
        controls, index_timestamp = self._parse_index(payload)
        current_records = {
            control.source_record_id: control.checkpoint_value() for control in controls
        }

        if in_progress and _required_text(
            state.get("index_digest"), "checkpoint index_digest", 64
        ) != digest:
            retry_state = self._completed_state_from_checkpoint(state, known_records)
            issue = SourceIssue(
                source_record_id=f"{self.name}:collection-index",
                stage="source_consistency",
                error="collection index changed during a paginated scan; restart required",
                summary={
                    "previous_digest": state.get("index_digest"),
                    "current_digest": digest,
                },
            )
            return SourcePage(
                records=(),
                next_state=retry_state,
                retry_state=retry_state,
                complete=False,
                upstream_count=len(current_records),
                issues=(issue,),
            )

        operations: list[tuple[str, _CatalogRecord | Mapping[str, Any]]] = [
            ("upsert", control)
            for control in controls
            if known_records.get(control.source_record_id, {}).get("signature")
            != control.signature
        ]
        operations.extend(
            ("delete", known_records[record_id])
            for record_id in sorted(known_records.keys() - current_records.keys())
        )
        if operation_offset > len(operations):
            raise ValueError(
                f"{self.name}: operation_offset {operation_offset} exceeds "
                f"the frozen operation count {len(operations)}"
            )

        page_operations = operations[operation_offset : operation_offset + self.page_size]
        page_retry_state = self._scan_state(
            known_records,
            digest=digest,
            index_timestamp=index_timestamp,
            operation_offset=operation_offset,
            scan_total=len(current_records),
            etag=_header(response.headers, "etag"),
            http_last_modified=_header(response.headers, "last-modified"),
        )
        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for absolute_index, (operation, value) in enumerate(
            page_operations, start=operation_offset
        ):
            try:
                if operation == "delete":
                    records.append(self._tombstone(value))
                else:
                    if not isinstance(value, _CatalogRecord):
                        raise TypeError("upsert operation has invalid catalog control")
                    records.append(self._resolve_record(value))
            except (KeyError, TypeError, UnicodeError, ValueError) as error:
                record_id = (
                    value.source_record_id
                    if isinstance(value, _CatalogRecord)
                    else _text(value.get("source_record_id"))
                    or f"{self.name}:operation:{absolute_index}"
                )
                issues.append(
                    SourceIssue(
                        source_record_id=record_id,
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"operation": operation, "operation_index": absolute_index},
                    )
                )

        next_offset = operation_offset + len(page_operations)
        complete = next_offset >= len(operations)
        if complete:
            next_state = self._completed_state(
                current_records,
                watermark=index_timestamp,
                etag=(
                    _text(state.get("scan_etag"))
                    if in_progress
                    else _header(response.headers, "etag")
                )
                or None,
                http_last_modified=(
                    _text(state.get("scan_http_last_modified"))
                    if in_progress
                    else _header(response.headers, "last-modified")
                )
                or None,
            )
        else:
            next_state = self._scan_state(
                known_records,
                digest=digest,
                index_timestamp=index_timestamp,
                operation_offset=next_offset,
                scan_total=len(current_records),
                etag=(
                    _text(state.get("scan_etag"))
                    if in_progress
                    else _header(response.headers, "etag")
                )
                or None,
                http_last_modified=(
                    _text(state.get("scan_http_last_modified"))
                    if in_progress
                    else _header(response.headers, "last-modified")
                )
                or None,
            )

        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            retry_state=page_retry_state,
            complete=complete,
            upstream_count=len(current_records),
            issues=tuple(issues),
        )

    def _parse_index(self, payload: Any) -> tuple[tuple[_CatalogRecord, ...], str | None]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: collection index must be a JSON object")
        raw_items = payload.get("items")
        if not _is_sequence(raw_items):
            raise ValueError(f"{self.name}: collection index items must be an array")
        if len(raw_items) > self.max_items:
            raise ValueError(
                f"{self.name}: collection index contains more than {self.max_items} items"
            )
        declared_total = _required_count(payload.get("total"), "index.total", self.name)
        if declared_total != len(raw_items):
            raise ValueError(
                f"{self.name}: index.total {declared_total} does not match "
                f"the {len(raw_items)} returned items"
            )

        count_per_type = payload.get("count_per_type")
        if not isinstance(count_per_type, Mapping):
            raise ValueError(f"{self.name}: index.count_per_type must be an object")
        declared_models = _required_count(
            count_per_type.get("model", 0), "index.count_per_type.model", self.name
        )
        actual_models = sum(
            isinstance(item, Mapping) and _text(item.get("type")) == "model"
            for item in raw_items
        )
        if declared_models != actual_models:
            raise ValueError(
                f"{self.name}: declared model count {declared_models} does not match "
                f"the {actual_models} model items"
            )

        index_timestamp = _optional_text(payload.get("timestamp"), _MAX_VERSION_CHARS)
        controls: list[_CatalogRecord] = []
        seen_record_ids: set[str] = set()
        seen_model_ids: set[str] = set()
        for item_index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, Mapping):
                raise ValueError(f"{self.name}: index item {item_index} is not an object")
            if _text(raw_item.get("type")) != "model":
                continue
            model_id = _required_text(
                raw_item.get("id"), f"index.items[{item_index}].id", _MAX_ID_CHARS
            )
            if model_id in seen_model_ids:
                raise ValueError(
                    f"{self.name}: duplicate model resource id {model_id!r} "
                    "in collection index"
                )
            seen_model_ids.add(model_id)
            alias = self._model_alias(model_id)
            raw_versions = raw_item.get("versions")
            if not _is_sequence(raw_versions):
                raise ValueError(
                    f"{self.name}: index model {model_id!r} versions must be an array"
                )
            if len(raw_versions) > self.max_versions_per_model:
                raise ValueError(
                    f"{self.name}: index model {model_id!r} has more than "
                    f"{self.max_versions_per_model} versions"
                )
            versions: Sequence[Any] = raw_versions or (None,)
            for version_index, raw_version in enumerate(versions):
                if raw_version is None:
                    version = None
                    source = None
                    sha256 = None
                    created_at = None
                    raw_control: dict[str, Any] = {
                        "id": model_id,
                        "type": "model",
                        "version": None,
                    }
                else:
                    if not isinstance(raw_version, Mapping):
                        raise ValueError(
                            f"{self.name}: version {version_index} of {model_id!r} "
                            "is not an object"
                        )
                    version = _required_text(
                        raw_version.get("version"),
                        f"version of {model_id!r}",
                        _MAX_VERSION_CHARS,
                    )
                    sha256 = _required_text(
                        raw_version.get("sha256"),
                        f"SHA-256 of {model_id!r}@{version}",
                        64,
                    ).casefold()
                    if not _SHA256_RE.fullmatch(sha256):
                        raise ValueError(
                            f"{self.name}: invalid SHA-256 for {model_id!r}@{version}"
                        )
                    source = self._safe_rdf_url(
                        _required_text(
                            raw_version.get("source"),
                            f"source of {model_id!r}@{version}",
                            _MAX_PATH_CHARS,
                        ),
                        alias=alias,
                        version=version,
                    )
                    created_at = _optional_text(
                        raw_version.get("created_at"), _MAX_VERSION_CHARS
                    )
                    raw_control = {
                        "id": model_id,
                        "type": "model",
                        "version": dict(raw_version),
                    }

                suffix = version if version is not None else "unversioned"
                record_id = f"{model_id}@{suffix}"
                if record_id in seen_record_ids:
                    raise ValueError(
                        f"{self.name}: duplicate catalog version record {record_id!r}"
                    )
                seen_record_ids.add(record_id)
                controls.append(
                    _CatalogRecord(
                        source_record_id=record_id,
                        model_id=model_id,
                        alias=alias,
                        version=version,
                        source=source,
                        sha256=sha256,
                        created_at=created_at,
                        signature=content_hash(raw_control),
                        raw=raw_control,
                    )
                )
        return tuple(controls), index_timestamp

    def _resolve_record(self, control: _CatalogRecord) -> SourceRecord:
        detail_url = canonicalize_url(
            f"{self.artifact_base_url}/{quote(control.alias, safe='')}"
        )
        detail_response: HttpResponse = self.client.get(
            detail_url,
            params={"version": control.version} if control.version is not None else None,
            headers={"Accept": "application/json"},
        )
        _require_status(detail_response, self.name, f"artifact {control.model_id!r}")
        if len(detail_response.body) > self.max_artifact_bytes:
            raise ValueError(
                f"artifact {control.model_id!r} exceeds {self.max_artifact_bytes} bytes"
            )
        detail = detail_response.json()
        if not isinstance(detail, Mapping):
            raise ValueError(f"artifact {control.model_id!r} must be a JSON object")
        returned_id = _text(detail.get("id"))
        if returned_id and returned_id != control.model_id:
            raise ValueError(
                f"artifact identity mismatch: expected {control.model_id!r}, "
                f"received {returned_id!r}"
            )
        manifest = detail.get("manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError(f"artifact {control.model_id!r} has no manifest object")
        if _text(manifest.get("type")) != "model":
            raise ValueError(f"artifact {control.model_id!r} manifest is not a model")
        name = _required_text(
            manifest.get("name"), f"manifest name of {control.model_id!r}", _MAX_ID_CHARS
        )

        rdf_text: str | None = None
        if control.source is not None:
            rdf_response: HttpResponse = self.client.get(
                control.source,
                headers={
                    "Accept": "application/yaml, text/yaml, text/plain, application/octet-stream"
                },
            )
            _require_status(rdf_response, self.name, f"RDF {control.source!r}")
            if len(rdf_response.body) > self.max_rdf_bytes:
                raise ValueError(
                    f"RDF for {control.source_record_id!r} exceeds "
                    f"{self.max_rdf_bytes} bytes"
                )
            actual_sha256 = hashlib.sha256(rdf_response.body).hexdigest()
            if actual_sha256 != control.sha256:
                raise ValueError(
                    f"RDF SHA-256 mismatch for {control.source_record_id!r}: "
                    f"expected {control.sha256}, received {actual_sha256}"
                )
            rdf_text = rdf_response.body.decode("utf-8")

        model_identifier = Identifier("bioimageio:model", control.model_id)
        links = [
            Link(detail_url, relation="metadata", locator="$.artifact.id", crawl=False),
        ]
        if control.source is not None:
            links.append(
                Link(
                    control.source,
                    relation="model_card_source",
                    locator="$.catalog.version.source",
                    crawl=False,
                )
            )
        archive_url = canonicalize_url(
            f"{self.artifact_base_url}/{quote(control.alias, safe='')}/create-zip-file"
            + (
                f"?version={quote(control.version, safe='')}"
                if control.version is not None
                else ""
            )
        )
        links.append(
            Link(archive_url, relation="model_package", locator="$.artifact.id", crawl=False)
        )

        manifest_links = self._manifest_evidence(
            manifest,
            alias=control.alias,
            version=control.version,
        )
        links.extend(manifest_links)

        manifest_id = _optional_text(manifest.get("id"), _MAX_ID_CHARS)
        aliases = [control.alias]
        model_identifiers = [model_identifier]
        if manifest_id and manifest_id not in {control.model_id, control.alias, name}:
            aliases.append(manifest_id)
            if identifier := _identifier_from_scalar(manifest_id):
                model_identifiers.append(identifier)

        local_model_id = f"{control.source_record_id}#model"
        model = ModelHint(
            local_id=local_model_id,
            name=name,
            identifiers=_unique_identifiers(model_identifiers),
            aliases=tuple(dict.fromkeys(alias for alias in aliases if alias != name)),
            status=ModelStatus.RELEASED,
            locator="$.artifact.manifest.name",
        )
        model_relations: tuple[ModelRelationHint, ...] = ()
        parent = manifest.get("parent")
        if isinstance(parent, Mapping):
            parent_id = _optional_text(parent.get("id"), _MAX_ID_CHARS)
            if parent_id:
                parent_locator = "$.artifact.manifest.parent.id"
                model_relations = (
                    ModelRelationHint(
                        subject_local_id=local_model_id,
                        predicate="derived_from",
                        target=ModelHint(
                            local_id=f"{control.source_record_id}#parent:{parent_id}",
                            name=parent_id,
                            identifiers=(Identifier("bioimageio:model", parent_id),),
                            status=ModelStatus.DOCUMENTED,
                            locator=parent_locator,
                        ),
                        locator=parent_locator,
                    ),
                )
        release_identifiers = (
            (
                Identifier(
                    "bioimageio:version",
                    f"{control.model_id}@{control.version}",
                ),
            )
            if control.version is not None
            else ()
        )
        weight_metadata = _weight_metadata(manifest.get("weights"))
        releases = ()
        if control.version is not None or control.sha256 is not None:
            releases = (
                ReleaseHint(
                    local_id=f"{control.source_record_id}#release",
                    model_local_id=local_model_id,
                    version=control.version,
                    revision=control.sha256,
                    identifiers=release_identifiers,
                    released_at=control.created_at,
                    metadata={
                        "rdf_sha256": control.sha256,
                        "rdf_source": control.source,
                        "format_version": _optional_text(
                            manifest.get("format_version"), _MAX_VERSION_CHARS
                        ),
                        "weight_formats": weight_metadata[0],
                        "weight_sources": weight_metadata[1],
                        "artifact_current_version": _optional_text(
                            detail.get("current_version"), _MAX_VERSION_CHARS
                        ),
                        "artifact_file_count": _optional_count(detail.get("file_count")),
                        "catalog_comment": (
                            _bounded_free_text(
                                control.raw["version"].get("comment"),
                                self.max_text_chars,
                            )
                            if isinstance(control.raw.get("version"), Mapping)
                            else None
                        ),
                    },
                    locator="$.catalog.version",
                ),
            )

        raw = {
            "catalog": dict(control.raw),
            "artifact": dict(detail),
            "rdf": {
                "source": control.source,
                "sha256": control.sha256,
                "text": rdf_text,
            },
        }
        text = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        if len(text) > self.max_text_chars:
            text = text[: self.max_text_chars]
        canonical_url = control.source or detail_url
        artifact_identifiers = (
            (
                Identifier(
                    "bioimageio:resource-version",
                    f"{control.model_id}@{control.version}",
                ),
            )
            if control.version is not None
            else (Identifier("bioimageio:resource", control.model_id),)
        )
        return SourceRecord(
            source_record_id=control.source_record_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonical_url,
            title=name,
            raw=raw,
            text=text,
            published_at=control.created_at,
            modified_at=control.created_at,
            identifiers=artifact_identifiers,
            links=_unique_links(links, self.max_links, self.name),
            models=(model,),
            model_relations=model_relations,
            releases=releases,
        )

    def _manifest_evidence(
        self,
        manifest: Mapping[str, Any],
        *,
        alias: str,
        version: str | None,
    ) -> list[Link]:
        links: list[Link] = []

        weights = manifest.get("weights")
        if isinstance(weights, Mapping):
            for weight_format, descriptor in sorted(
                weights.items(), key=lambda item: str(item[0])
            ):
                if not isinstance(descriptor, Mapping):
                    continue
                locator = f"$.artifact.manifest.weights.{weight_format}.source"
                if url := self._reference_url(descriptor.get("source"), alias, version):
                    links.append(Link(url, relation="weights", locator=locator, crawl=False))
                architecture = descriptor.get("architecture")
                if isinstance(architecture, Mapping):
                    architecture_locator = (
                        f"$.artifact.manifest.weights.{weight_format}.architecture.source"
                    )
                    if url := self._reference_url(
                        architecture.get("source"), alias, version
                    ):
                        links.append(
                            Link(
                                url,
                                relation="implementation",
                                locator=architecture_locator,
                            )
                        )

        if documentation_url := self._reference_url(
            manifest.get("documentation"), alias, version
        ):
            links.append(
                Link(
                    documentation_url,
                    relation="documentation",
                    locator="$.artifact.manifest.documentation",
                )
            )
        if (
            (git_repo := _optional_text(manifest.get("git_repo"), _MAX_PATH_CHARS))
            and _is_web_url(git_repo)
        ):
            links.append(
                Link(
                    canonicalize_url(git_repo),
                    relation="implementation",
                    locator="$.artifact.manifest.git_repo",
                )
            )

        raw_links = manifest.get("links")
        if _is_sequence(raw_links):
            for index, value in enumerate(raw_links):
                if isinstance(value, Mapping):
                    url = _optional_text(value.get("url"), _MAX_PATH_CHARS)
                else:
                    url = _optional_text(value, _MAX_PATH_CHARS)
                if url and _is_web_url(url):
                    links.append(
                        Link(
                            canonicalize_url(url),
                            relation="references",
                            locator=f"$.artifact.manifest.links[{index}]",
                        )
                    )

        raw_citations = manifest.get("cite")
        if _is_sequence(raw_citations):
            for index, value in enumerate(raw_citations):
                if not isinstance(value, Mapping):
                    continue
                doi = _normalize_doi(value.get("doi"))
                if doi:
                    links.append(
                        Link(
                            canonicalize_url(f"https://doi.org/{doi}"),
                            relation="publication",
                            locator=f"$.artifact.manifest.cite[{index}].doi",
                        )
                    )
                url = _optional_text(value.get("url"), _MAX_PATH_CHARS)
                if url and _is_web_url(url):
                    links.append(
                        Link(
                            canonicalize_url(url),
                            relation="publication",
                            locator=f"$.artifact.manifest.cite[{index}].url",
                        )
                    )

        for url in extract_urls(manifest):
            links.append(
                Link(url, relation="metadata_reference", locator="$.artifact.manifest")
            )
        return links

    def _reference_url(self, value: Any, alias: str, version: str | None) -> str | None:
        if isinstance(value, Mapping):
            value = value.get("source")
        reference = _optional_text(value, _MAX_PATH_CHARS)
        if not reference:
            return None
        if _is_web_url(reference):
            return canonicalize_url(reference)
        parts = urlsplit(reference)
        if (
            parts.scheme
            or parts.netloc
            or parts.query
            or parts.fragment
            or reference.startswith("/")
        ):
            return None
        segments = reference.replace("\\", "/").split("/")
        if any(not segment or segment in {".", ".."} for segment in segments):
            return None
        encoded_path = "/".join(quote(segment, safe="") for segment in segments)
        url = f"{self.artifact_base_url}/{quote(alias, safe='')}/files/{encoded_path}"
        if version is not None:
            url += f"?version={quote(version, safe='')}"
        return canonicalize_url(url)

    def _safe_rdf_url(self, value: str, *, alias: str, version: str) -> str:
        parts = urlsplit(value)
        base = urlsplit(self.artifact_base_url)
        if (
            parts.scheme.casefold() != "https"
            or parts.username is not None
            or parts.password is not None
            or _origin(value) != _origin(self.artifact_base_url)
        ):
            raise ValueError(f"{self.name}: RDF source is outside artifact origin")
        base_path = base.path.rstrip("/")
        expected_prefix = f"{base_path}/{quote(alias, safe='')}/files/"
        if not parts.path.startswith(expected_prefix) or parts.path == expected_prefix:
            raise ValueError(f"{self.name}: RDF source is outside its artifact file tree")
        query = parse_qsl(parts.query, keep_blank_values=True)
        if query != [("version", version)]:
            raise ValueError(f"{self.name}: RDF source does not pin the catalog version")
        return canonicalize_url(value)

    def _model_alias(self, model_id: str) -> str:
        prefix = f"{self.workspace}/"
        if not model_id.startswith(prefix):
            raise ValueError(
                f"{self.name}: model id {model_id!r} is outside workspace "
                f"{self.workspace!r}"
            )
        alias = model_id[len(prefix) :]
        if (
            not alias
            or "/" in alias
            or any(ord(character) < 33 for character in alias)
            or len(alias) > _MAX_ID_CHARS
        ):
            raise ValueError(f"{self.name}: invalid model artifact alias in {model_id!r}")
        return alias

    def _tombstone(self, value: Mapping[str, Any]) -> SourceRecord:
        record_id = _required_text(
            value.get("source_record_id"), "deleted source_record_id", _MAX_ID_CHARS * 2
        )
        model_id = _required_text(value.get("model_id"), "deleted model_id", _MAX_ID_CHARS)
        alias = _required_text(value.get("alias"), "deleted alias", _MAX_ID_CHARS)
        source = _optional_text(value.get("source"), _MAX_PATH_CHARS)
        detail_url = canonicalize_url(
            f"{self.artifact_base_url}/{quote(alias, safe='')}"
        )
        return SourceRecord(
            source_record_id=record_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=source or detail_url,
            title=model_id,
            raw={"catalog_removed": True, "previous_control": dict(value)},
            deleted=True,
        )

    def _checkpoint_records(self, value: Any) -> dict[str, dict[str, Any]]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{self.name}: checkpoint known_records must be an object")
        maximum = self.max_items * min(self.max_versions_per_model, 100)
        if len(value) > maximum:
            raise ValueError(
                f"{self.name}: checkpoint known_records exceeds safe bound {maximum}"
            )
        result: dict[str, dict[str, Any]] = {}
        for raw_record_id, raw_control in value.items():
            record_id = _required_text(
                raw_record_id, "checkpoint source_record_id", _MAX_ID_CHARS * 2
            )
            if not isinstance(raw_control, Mapping):
                raise ValueError(
                    f"{self.name}: checkpoint control for {record_id!r} must be an object"
                )
            model_id = _required_text(
                raw_control.get("model_id"), "checkpoint model_id", _MAX_ID_CHARS
            )
            alias = _required_text(
                raw_control.get("alias"), "checkpoint alias", _MAX_ID_CHARS
            )
            self._model_alias(model_id)
            if model_id != f"{self.workspace}/{alias}":
                raise ValueError(
                    f"{self.name}: checkpoint model id and alias do not match"
                )
            version = _optional_text(raw_control.get("version"), _MAX_VERSION_CHARS)
            expected_record_id = f"{model_id}@{version or 'unversioned'}"
            if record_id != expected_record_id:
                raise ValueError(
                    f"{self.name}: checkpoint source record identity mismatch"
                )
            sha256 = _optional_text(raw_control.get("sha256"), 64)
            if sha256 and not _SHA256_RE.fullmatch(sha256):
                raise ValueError(f"{self.name}: invalid checkpoint SHA-256")
            signature = _required_text(
                raw_control.get("signature"), "checkpoint signature", 64
            )
            if not _SHA256_RE.fullmatch(signature):
                raise ValueError(f"{self.name}: invalid checkpoint signature")
            source = _optional_text(raw_control.get("source"), _MAX_PATH_CHARS)
            if source and version:
                source = self._safe_rdf_url(source, alias=alias, version=version)
            result[record_id] = {
                "source_record_id": record_id,
                "model_id": model_id,
                "alias": alias,
                "version": version,
                "source": source,
                "sha256": sha256,
                "created_at": _optional_text(
                    raw_control.get("created_at"), _MAX_VERSION_CHARS
                ),
                "signature": signature,
            }
        return result

    def _scan_state(
        self,
        known_records: Mapping[str, Mapping[str, Any]],
        *,
        digest: str,
        index_timestamp: str | None,
        operation_offset: int,
        scan_total: int,
        etag: str | None,
        http_last_modified: str | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "known_records": {
                key: _checkpoint_copy(value) for key, value in sorted(known_records.items())
            },
            "index_digest": digest,
            "operation_offset": operation_offset,
            "scan_total": scan_total,
            "started_at": _isoformat(self.clock()),
        }
        if index_timestamp:
            result["index_timestamp"] = index_timestamp
        if etag:
            result["scan_etag"] = etag
        if http_last_modified:
            result["scan_http_last_modified"] = http_last_modified
        return result

    def _completed_state(
        self,
        known_records: Mapping[str, Mapping[str, Any]],
        *,
        watermark: str | None,
        etag: str | None,
        http_last_modified: str | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "known_records": {
                key: _checkpoint_copy(value) for key, value in sorted(known_records.items())
            },
            "completed_at": _isoformat(self.clock()),
        }
        if watermark:
            result["watermark"] = watermark
        if etag:
            result["etag"] = etag
        if http_last_modified:
            result["http_last_modified"] = http_last_modified
        return result

    def _completed_state_from_checkpoint(
        self,
        state: Mapping[str, Any],
        known_records: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self._completed_state(
            known_records,
            watermark=_text(state.get("watermark")) or None,
            etag=_text(state.get("etag")) or None,
            http_last_modified=_text(state.get("http_last_modified")) or None,
        )


def _weight_metadata(value: Any) -> tuple[list[str], list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        return [], []
    formats: list[str] = []
    sources: list[dict[str, Any]] = []
    for raw_format, descriptor in sorted(value.items(), key=lambda item: str(item[0])):
        weight_format = str(raw_format)
        formats.append(weight_format)
        if not isinstance(descriptor, Mapping):
            continue
        source = descriptor.get("source")
        if isinstance(source, Mapping):
            source = source.get("source")
        source_text = _text(source)
        sources.append(
            {
                "format": weight_format,
                "source": source_text or None,
                "sha256": _optional_text(descriptor.get("sha256"), 64),
            }
        )
    return formats, sources


def _checkpoint_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_record_id": _text(value.get("source_record_id")) or None,
        "model_id": value.get("model_id"),
        "alias": value.get("alias"),
        "version": value.get("version"),
        "source": value.get("source"),
        "sha256": value.get("sha256"),
        "created_at": value.get("created_at"),
        "signature": value.get("signature"),
    }


def _identifier_from_scalar(value: str) -> Identifier | None:
    if _is_web_url(value):
        return identifier_from_url(value)
    doi = _normalize_doi(value)
    return Identifier("doi", doi) if doi else None


def _normalize_doi(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    lowered = text.casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    text = text.rstrip(".,;:!?)\"]}").casefold()
    return text if _DOI_RE.fullmatch(text) else None


def _required_count(value: Any, field: str, source: str) -> int:
    result = _optional_count(value)
    if result is None:
        raise ValueError(f"{source}: {field} must be a non-negative integer")
    return result


def _optional_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _state_count(state: Mapping[str, Any], key: str, source: str) -> int | None:
    if key not in state:
        return None
    value = state.get(key)
    result = _optional_count(value)
    if result is None:
        raise ValueError(f"{source}: invalid checkpoint {key}: {value!r}")
    return result


def _unique_identifiers(values: Iterable[Identifier]) -> tuple[Identifier, ...]:
    return tuple(dict.fromkeys(values))


def _unique_links(
    values: Iterable[Link], maximum: int, source: str
) -> tuple[Link, ...]:
    result: list[Link] = []
    seen: set[tuple[str, str, bool]] = set()
    for value in values:
        key = (value.url, value.relation, value.crawl)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) > maximum:
            raise ValueError(f"{source}: model manifest exceeds {maximum} retained links")
    return tuple(result)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == wanted), None)


def _require_status(response: HttpResponse, source: str, resource: str) -> None:
    if response.status < 200 or response.status >= 300:
        raise ValueError(f"{source}: {resource} returned HTTP {response.status}")


def _required_text(value: Any, field: str, maximum: int) -> str:
    result = _optional_text(value, maximum)
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _optional_text(value: Any, maximum: int) -> str:
    result = _text(value)
    if len(result) > maximum:
        raise ValueError(f"text value exceeds {maximum} characters")
    if any(character in result for character in "\r\n\x00"):
        raise ValueError("text value contains control characters")
    return result


def _bounded_free_text(value: Any, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    if len(value) > maximum:
        raise ValueError(f"text value exceeds {maximum} characters")
    if "\x00" in value:
        raise ValueError("text value contains a NUL character")
    return value


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {field} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{source}: {field} must be a positive integer")
    return result


def _bounded_positive_int(value: Any, field: str, source: str, maximum: int) -> int:
    result = _positive_int(value, field, source)
    if result > maximum:
        raise ValueError(f"{source}: {field} must not exceed {maximum}")
    return result


def _web_url(value: str, source: str, field: str) -> str:
    result = canonicalize_url(_required_text(value, field, _MAX_PATH_CHARS))
    parts = urlsplit(result)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
    ):
        raise ValueError(f"{source}: {field} must be a public HTTPS URL")
    return result


def _is_web_url(value: str) -> bool:
    parts = urlsplit(value)
    return (
        parts.scheme.casefold() in {"http", "https"}
        and bool(parts.hostname)
        and parts.username is None
        and parts.password is None
    )


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    scheme = parts.scheme.casefold()
    port = parts.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (parts.hostname or "").casefold(), port


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["BioImageIoSourceAdapter"]
