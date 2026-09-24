"""Public CivitAI model-catalog ingestion.

The public ``/api/v1/models`` endpoint exposes models, their published versions,
file identifiers and checksums, declared base-model families, and download
references.  This adapter retains that evidence without downloading model bytes
or treating a provider label as a cross-provider identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit, urlunsplit

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
from modelome.normalize import canonicalize_url, content_hash, extract_urls


class CivitaiModelsSourceAdapter:
    """Page every public model exposed by CivitAI's model-list endpoint."""

    coverage_limitation = (
        "The public models API exposes only Published and requested early-access versions "
        "to non-moderator callers; "
        "archived models omit file links, and NSFW results can be silently restricted "
        "by region even when nsfw=true. This source cannot enumerate unpublished or "
        "moderator-only historical versions."
    )

    def __init__(
        self,
        *,
        name: str = "civitai-models",
        url: str = "https://civitai.com/api/v1/models",
        page_size: int = 100,
        sort_by: str = "Newest",
        period: str = "AllTime",
        include_nsfw: bool = True,
        include_early_access: bool = True,
        query: str | None = None,
        tag: str | None = None,
        username: str | None = None,
        model_types: Sequence[str] = (),
        base_models: Sequence[str] = (),
        artifact_kind: str | ArtifactKind = ArtifactKind.MODEL_CARD,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, "catalog URL")
        self.page_size = int(page_size)
        if not 1 <= self.page_size <= 100:
            raise ValueError(f"{self.name}: page_size must be from 1 through 100")
        self.sort_by = _required_text(sort_by, "sort_by")
        if self.sort_by not in {"Highest Rated", "Most Downloaded", "Newest"}:
            raise ValueError(f"{self.name}: unsupported CivitAI model sort {self.sort_by!r}")
        self.period = _required_text(period, "period")
        if self.period not in {"AllTime", "Year", "Month", "Week", "Day"}:
            raise ValueError(f"{self.name}: unsupported CivitAI sort period {self.period!r}")
        self.include_nsfw = bool(include_nsfw)
        self.include_early_access = bool(include_early_access)
        self.query = _optional_text(query)
        self.tag = _optional_text(tag)
        self.username = _optional_text(username)
        self.model_types = _text_options(model_types, "model_types")
        self.base_models = _text_options(base_models, "base_models")
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.client = client or HttpClient()
        self.checkpoint_signature = content_hash(
            {
                "adapter": "civitai-models-v1",
                "url": self.url,
                "page_size": self.page_size,
                "sort_by": self.sort_by,
                "period": self.period,
                "include_nsfw": self.include_nsfw,
                "include_early_access": self.include_early_access,
                "query": self.query,
                "tag": self.tag,
                "username": self.username,
                "model_types": self.model_types,
                "base_models": self.base_models,
                "artifact_kind": self.artifact_kind.value,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        next_url = _optional_text(state.get("next_url"))
        resuming = next_url is not None
        raw_items_seen = _state_count(state, "raw_items_seen") if resuming else 0
        scan_total = _state_count(state, "scan_total") if resuming else None
        if next_url is not None:
            request_url = _pagination_url(next_url, self.url)
            response: HttpResponse = self.client.get(
                request_url,
                headers={"Accept": "application/json"},
            )
        else:
            request_url = self.url
            response = self.client.get(
                self.url,
                params=self._request_params(),
                headers={"Accept": "application/json"},
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: expected a JSON object from {response.url}")
        items = payload.get("items")
        if not _is_sequence(items):
            raise ValueError(f"{self.name}: response items must be a list")
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(f"{self.name}: response metadata must be an object")
        metadata = metadata or {}
        raw_items_seen = (raw_items_seen or 0) + len(items)
        response_total = _optional_nonnegative_int(metadata.get("totalItems"))
        if response_total is not None:
            scan_total = max(scan_total or 0, response_total)

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, item in enumerate(items):
            try:
                records.append(self._record(item, index))
            except (TypeError, ValueError) as error:
                raw = dict(item) if isinstance(item, Mapping) else {"value": repr(item)[:1000]}
                record_id = _optional_text(item.get("id")) if isinstance(item, Mapping) else None
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            record_id
                            or f"{self.name}:malformed:{content_hash(raw)[:32]}"
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index, "raw": raw},
                    )
                )

        page_url = _optional_web_url(metadata.get("nextPage"), response.url)
        if page_url is not None:
            page_url = _merge_pagination_params(page_url, self._request_params())
        if page_url is None:
            cursor = _optional_text(metadata.get("nextCursor"))
            if cursor is not None:
                page_url = _cursor_url(
                    self.url,
                    page_size=self.page_size,
                    sort_by=self.sort_by,
                    period=self.period,
                    include_nsfw=self.include_nsfw,
                    cursor=cursor,
                    extra_params=self._filter_params(),
                )
        if page_url is not None:
            page_url = _pagination_url(page_url, self.url)
            if page_url == request_url:
                raise ValueError(f"{self.name}: pagination URL did not advance")
        if page_url is None and scan_total is not None and raw_items_seen < scan_total:
            raise ValueError(
                f"{self.name}: pagination ended after {raw_items_seen} model(s), "
                f"before the provider-reported total of {scan_total}"
            )

        next_state: dict[str, Any] = {}
        if page_url is not None:
            next_state = {"next_url": page_url, "raw_items_seen": raw_items_seen}
            if scan_total is not None:
                next_state["scan_total"] = scan_total
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=page_url is None,
            upstream_count=response_total if response_total is not None else scan_total,
            issues=tuple(issues),
        )

    def _filter_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in ("query", "tag", "username"):
            value = getattr(self, key)
            if value is not None:
                params[key] = value
        if self.model_types:
            params["types"] = list(self.model_types)
        if self.base_models:
            params["baseModels"] = list(self.base_models)
        return params

    def _request_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "limit": self.page_size,
            "sort": self.sort_by,
            "nsfw": str(self.include_nsfw).lower(),
            "earlyAccess": str(self.include_early_access).lower(),
        }
        if self.period != "AllTime":
            params["period"] = self.period
        params.update(self._filter_params())
        return params

    def _record(self, item: Any, index: int) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError(f"catalog item {index} is not an object")
        model_id = _required_text(item.get("id"), f"catalog item {index} id")
        model_url = _model_url(model_id, item.get("url"))
        title = _optional_text(item.get("name")) or model_id
        model_local_id = f"{model_id}#model"
        model_identifier = Identifier("civitai:model", model_id)
        model_mode = _optional_text(item.get("mode"))
        model = ModelHint(
            local_id=model_local_id,
            name=title,
            identifiers=(model_identifier,),
            status=(
                ModelStatus.DOCUMENTED
                if model_mode in {"Archived", "TakenDown"}
                else ModelStatus.RELEASED
            ),
            locator="$.id",
        )

        description = _optional_text(item.get("description"))
        tags = tuple(
            tag for tag in (_optional_text(value) for value in _sequence(item.get("tags"))) if tag
        )
        text_parts = [part for part in (description,) if part]
        if tags:
            text_parts.append("tags: " + ", ".join(sorted(set(tags))))
        links: list[Link] = [Link(model_url, relation="model_page", locator="$.id")]
        if description:
            links.extend(
                Link(url, relation="documentation_reference", locator="$.description")
                for url in extract_urls(description)
            )

        relations = list(
            _base_model_relations(
                model_id,
                model_local_id,
                _sequence(item.get("baseModels")),
                "$.baseModels",
            )
        )
        releases: list[ReleaseHint] = []
        for version_index, version in enumerate(_sequence(item.get("modelVersions"))):
            if not isinstance(version, Mapping):
                continue
            release, version_links, relation = self._version(
                model_id,
                model_local_id,
                version,
                version_index,
                model_mode,
            )
            if release is not None:
                releases.append(release)
            links.extend(version_links)
            if relation is not None:
                relations.append(relation)

        return SourceRecord(
            source_record_id=model_id,
            kind=self.artifact_kind,
            canonical_url=model_url,
            title=title,
            raw=dict(item),
            text="\n".join(text_parts),
            published_at=_published_at(item),
            modified_at=_optional_text(item.get("updatedAt")),
            identifiers=(model_identifier,),
            links=_unique_links(links),
            models=(model,),
            model_relations=tuple(relations),
            releases=tuple(releases),
        )

    def _version(
        self,
        model_id: str,
        model_local_id: str,
        version: Mapping[str, Any],
        index: int,
        model_mode: str | None,
    ) -> tuple[ReleaseHint | None, tuple[Link, ...], ModelRelationHint | None]:
        version_id = _optional_text(version.get("id"))
        version_name = _optional_text(version.get("name"))
        version_url = _model_version_url(model_id, version_id)
        links: list[Link] = []
        if version_url:
            links.append(
                Link(
                    version_url,
                    relation="model_variant",
                    locator=f"$.modelVersions[{index}].id",
                )
            )
        version_download = _optional_web_url(version.get("downloadUrl"), self.url)
        if version_download:
            links.append(
                Link(
                    version_download,
                    relation="weights",
                    locator=f"$.modelVersions[{index}].downloadUrl",
                    crawl=False,
                )
            )

        identifiers: list[Identifier] = []
        if version_id:
            identifiers.append(Identifier("civitai:model-version", version_id))
        for file_index, file in enumerate(_sequence(version.get("files"))):
            if not isinstance(file, Mapping):
                continue
            file_id = _optional_text(file.get("id"))
            if file_id:
                identifiers.append(Identifier("civitai:model-file", file_id))
            sha256 = _sha256(file.get("hashes"))
            if sha256:
                identifiers.append(Identifier("sha256", sha256))
            download_url = _optional_web_url(file.get("downloadUrl"), self.url)
            if download_url:
                links.append(
                    Link(
                        download_url,
                        relation="weights",
                        locator=(
                            f"$.modelVersions[{index}].files[{file_index}].downloadUrl"
                        ),
                        crawl=False,
                    )
                )
        base_model = _optional_text(version.get("baseModel"))
        relation = None
        if base_model:
            relation = _base_model_relation(
                model_id,
                model_local_id,
                base_model,
                f"$.modelVersions[{index}].baseModel",
            )

        if version_id is None:
            return None, _unique_links(links), relation
        release = ReleaseHint(
            local_id=f"{model_id}#release:{version_id}",
            model_local_id=model_local_id,
            version=version_name or version_id,
            revision=version_id,
            identifiers=tuple(dict.fromkeys(identifiers)),
            released_at=_optional_text(version.get("publishedAt")),
            metadata={
                "version_index": _optional_nonnegative_int(version.get("index")),
                "base_model": base_model,
                "base_model_type": _optional_text(version.get("baseModelType")),
                "availability": _optional_text(version.get("availability")),
                "nsfw_level": _optional_nonnegative_int(version.get("nsfwLevel")),
                "model_type": _optional_text(version.get("type")),
                "trained_words": tuple(
                    word
                    for word in (
                        _optional_text(value)
                        for value in _sequence(version.get("trainedWords"))
                    )
                    if word
                ),
                "air": _optional_text(version.get("air")),
                "status": _optional_text(version.get("status")),
                "model_mode": model_mode,
                "upload_type": _optional_text(version.get("uploadType")),
                "usage_control": _optional_text(version.get("usageControl")),
                "created_at": _optional_text(version.get("createdAt")),
                "updated_at": _optional_text(version.get("updatedAt")),
                "supports_generation": (
                    version.get("supportsGeneration")
                    if isinstance(version.get("supportsGeneration"), bool)
                    else None
                ),
                "stats": _safe_stats(version.get("stats")),
                "files": tuple(
                    _file_metadata(file)
                    for file in _sequence(version.get("files"))
                    if isinstance(file, Mapping)
                ),
            },
            locator=f"$.modelVersions[{index}]",
        )
        return release, _unique_links(links), relation


def _model_url(model_id: str, value: Any) -> str:
    supplied = _optional_web_url(value, "https://civitai.com")
    if supplied:
        return supplied
    return canonicalize_url(f"https://civitai.com/models/{quote(model_id, safe='')}")


def _model_version_url(model_id: str, version_id: str | None) -> str | None:
    if version_id is None:
        return None
    return canonicalize_url(
        f"https://civitai.com/models/{quote(model_id, safe='')}?"
        f"modelVersionId={quote(version_id, safe='')}"
    )


def _base_model_relations(
    model_id: str,
    model_local_id: str,
    values: Sequence[Any],
    locator: str,
) -> tuple[ModelRelationHint, ...]:
    return tuple(
        _base_model_relation(model_id, model_local_id, value, f"{locator}[{index}]")
        for index, value in enumerate(values)
        if _optional_text(value)
    )


def _base_model_relation(
    model_id: str,
    model_local_id: str,
    value: str,
    locator: str,
) -> ModelRelationHint:
    """Keep CivitAI's provider-scoped base-model taxonomy as declared lineage."""

    return ModelRelationHint(
        subject_local_id=model_local_id,
        predicate="base_model",
        target=ModelHint(
            local_id=f"{model_id}#base-model:{value}",
            name=value,
            identifiers=(Identifier("civitai:base-model", value),),
            status=ModelStatus.DOCUMENTED,
            locator=locator,
        ),
        locator=locator,
    )


def _published_at(item: Mapping[str, Any]) -> str | None:
    for key in ("publishedAt", "createdAt"):
        if value := _optional_text(item.get(key)):
            return value
    for version in _sequence(item.get("modelVersions")):
        if isinstance(version, Mapping) and (value := _optional_text(version.get("publishedAt"))):
            return value
    return None


def _sha256(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    candidate = _optional_text(value.get("SHA256"))
    if candidate is None:
        return None
    normalized = candidate.casefold()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        return None
    return normalized


def _cursor_url(
    base_url: str,
    *,
    page_size: int,
    sort_by: str,
    period: str,
    include_nsfw: bool,
    cursor: str,
    extra_params: Mapping[str, Any] | None = None,
) -> str:
    params: list[tuple[str, str]] = [
        ("limit", str(page_size)),
        ("sort", sort_by),
        ("nsfw", str(include_nsfw).lower()),
        ("cursor", cursor),
    ]
    if period != "AllTime":
        params.append(("period", period))
    for key, value in (extra_params or {}).items():
        if isinstance(value, (list, tuple)):
            params.extend((key, str(item)) for item in value)
        else:
            params.append((key, str(value)))
    return canonicalize_url(
        f"{base_url}?{urlencode(params)}"
    )


def _merge_pagination_params(value: str, params: Mapping[str, Any]) -> str:
    """Retain the configured scan scope if upstream nextPage omits query keys."""

    parts = urlsplit(value)
    query = parse_qsl(parts.query, keep_blank_values=True)
    configured_keys = set(params)
    query = [(key, item) for key, item in query if key not in configured_keys]
    for key, item in params.items():
        values = item if isinstance(item, (list, tuple)) else (item,)
        query.extend((key, str(value)) for value in values)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def _text_options(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence of strings")
    return tuple(dict.fromkeys(_required_text(value, label) for value in values))


def _safe_stats(value: Any) -> Mapping[str, int | float] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def _file_metadata(file: Mapping[str, Any]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key in ("id", "name", "type", "sizeKB", "primary"):
        value = file.get(key)
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
    hashes = file.get("hashes")
    if isinstance(hashes, Mapping):
        result["hashes"] = {
            str(key): raw.strip()
            for key, value in hashes.items()
            if isinstance(value, str) and (raw := value.strip())
        }
    file_metadata = file.get("metadata")
    if isinstance(file_metadata, Mapping):
        result["metadata"] = dict(file_metadata)
    return result


def _pagination_url(value: str, base_url: str) -> str:
    result = _web_url(value, "pagination URL")
    expected = urlsplit(base_url)
    actual = urlsplit(result)
    expected_port = expected.port or (443 if expected.scheme == "https" else 80)
    actual_port = actual.port or (443 if actual.scheme == "https" else 80)
    if (
        actual.scheme != expected.scheme
        or actual.hostname != expected.hostname
        or actual_port != expected_port
        or actual.path != expected.path
    ):
        raise ValueError(f"{base_url}: pagination URL changed endpoint")
    return result


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda link: (link.url, link.relation, link.locator or "", link.crawl),
        )
    )


def _web_url(value: str, label: str) -> str:
    result = canonicalize_url(value)
    parsed = urlsplit(result)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    return result


def _optional_web_url(value: Any, base_url: str) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    result = canonicalize_url(urljoin(base_url, text))
    parsed = urlsplit(result)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return result


def _required_text(value: Any, label: str) -> str:
    result = _optional_text(value)
    if result is None:
        raise ValueError(f"{label} must be non-empty text")
    return result


def _optional_text(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _state_count(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key)
    if value is None:
        return 0
    parsed = _optional_nonnegative_int(value)
    if parsed is None:
        raise ValueError(f"CivitAI model sync state {key} must be a nonnegative integer")
    return parsed


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


__all__ = ["CivitaiModelsSourceAdapter"]
