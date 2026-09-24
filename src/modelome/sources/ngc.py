"""Public NVIDIA NGC model-catalog ingestion.

NGC exposes a guest-readable search endpoint for the currently public ``MODEL``
resource group. The endpoint paginates a moving catalog, so this adapter checks
the provider's total during a run but deliberately does not invoke authoritative
snapshot deletion semantics. By default, it records each public model-card page
and the declared latest-version handle. Optional one-page version expansion
uses NGC's guest-readable model-version metadata endpoint without fetching
individual cards, archives, or weight bytes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from math import ceil
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote

from modelome.http import HttpClient, HttpFailure, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, extract_urls

Clock = Callable[[], datetime]

_QUERY_FIELDS = (
    "all",
    "description",
    "displayName",
    "name",
    "resourceId",
)
_FIELDS = (
    "isPublic",
    "attributes",
    "guestAccess",
    "name",
    "orgName",
    "teamName",
    "displayName",
    "dateModified",
    "labels",
    "description",
    "resourceId",
)
_MAX_PAGE_SIZE = 200


def _utcnow() -> datetime:
    return datetime.now(UTC)


class NgcModelsSourceAdapter:
    """Enumerate guest-visible current NGC ``MODEL`` resource rows.

    This is provider-page and latest-release evidence only.  NGC's search
    endpoint is paginated rather than a versioned provider export: a fixed
    total protects one attempted sweep from silent drift, while
    ``authoritative_snapshot`` remains false so a final page can never
    tombstone records observed on earlier pages.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the guest-visible, currently listed NVIDIA NGC MODEL resource "
        "group reachable through its public search endpoint. It does not fetch "
        "individual cards, archives, or weights. Historical versions can be "
        "requested with include_all_versions when the guest endpoint reports "
        "them in one page; multi-page version continuation is currently "
        "rejected because its contract is unverified. It does not cover private, "
        "gated, removed, or unlisted resources. Because the "
        "paginated catalog is mutable rather than a versioned export, completed "
        "sweeps do not use authoritative-snapshot deletion semantics."
    )

    def __init__(
        self,
        *,
        name: str = "ngc-models",
        url: str = "https://api.ngc.nvidia.com/v2/search/catalog/resources/MODEL",
        model_page_base_url: str = "https://catalog.ngc.nvidia.com",
        metadata_base_url: str = "https://api.ngc.nvidia.com",
        page_size: int = _MAX_PAGE_SIZE,
        include_all_versions: bool = False,
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, "catalog URL")
        self.model_page_base_url = _web_url(
            model_page_base_url,
            "model-page base URL",
        ).rstrip("/")
        self.metadata_base_url = _web_url(
            metadata_base_url,
            "model-metadata base URL",
        ).rstrip("/")
        self.page_size = _positive_int(page_size, "page_size")
        if self.page_size > _MAX_PAGE_SIZE:
            raise ValueError(
                f"{self.name}: page_size must be from 1 through {_MAX_PAGE_SIZE}"
            )
        self.max_response_bytes = _positive_int(
            max_response_bytes,
            "max_response_bytes",
        )
        if not isinstance(include_all_versions, bool):
            raise ValueError(f"{self.name}: include_all_versions must be a boolean")
        self.include_all_versions = include_all_versions
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "ngc-models-v1",
                "url": self.url,
                "model_page_base_url": self.model_page_base_url,
                "metadata_base_url": self.metadata_base_url,
                "page_size": self.page_size,
                "include_all_versions": self.include_all_versions,
                "max_response_bytes": self.max_response_bytes,
                "query_fields": _QUERY_FIELDS,
                "fields": _FIELDS,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        page_number = _state_nonnegative_int(state.get("page"), "page", 0)
        response: HttpResponse = self.client.get(
            self.url,
            params={"q": self._query(page_number)},
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog response exceeds {self.max_response_bytes} bytes"
            )
        items, provider_total, page_total = self._items(response, page_number)
        prior_total = state.get("provider_total")
        if prior_total is not None:
            prior_total = _state_nonnegative_int(prior_total, "provider_total")
            if prior_total != provider_total:
                raise ValueError(
                    f"{self.name}: provider total changed from {prior_total} to "
                    f"{provider_total}; restart the catalog sweep"
                )

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, item in enumerate(items):
            try:
                records.append(self._record(item, page_number, provider_total))
            except (TypeError, ValueError) as error:
                raw = dict(item) if isinstance(item, Mapping) else {"value": repr(item)[:1000]}
                resource_id = _text(item.get("resourceId")) if isinstance(item, Mapping) else ""
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            resource_id
                            or f"{self.name}:malformed:{content_hash(raw)[:32]}"
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index, "raw": raw},
                    )
                )

        next_page = page_number + 1
        complete = next_page >= page_total
        checked_at = _isoformat(self.clock())
        if complete:
            next_state: dict[str, Any] = {
                "completed_at": checked_at,
                "provider_total": provider_total,
                "page_total": page_total,
            }
        else:
            next_state = {
                "page": next_page,
                "provider_total": provider_total,
                "page_total": page_total,
                "started_at": _text(state.get("started_at")) or checked_at,
            }
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=provider_total if complete else None,
            issues=tuple(issues),
        )

    def _query(self, page_number: int) -> str:
        return json.dumps(
            {
                "query": "",
                "orderBy": [
                    {"field": "nameSort", "value": "ASC"},
                    {"field": "resourceId", "value": "ASC"},
                ],
                "queryFields": list(_QUERY_FIELDS),
                "fields": list(_FIELDS),
                "page": page_number,
                "pageSize": self.page_size,
                "filters": [],
            },
            separators=(",", ":"),
        )

    def _items(
        self,
        response: HttpResponse,
        page_number: int,
    ) -> tuple[tuple[Any, ...], int, int]:
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: catalog response is not a JSON object")
        provider_total = _state_nonnegative_int(
            payload.get("resultTotal"),
            "provider resultTotal",
        )
        page_total = _state_nonnegative_int(
            payload.get("resultPageTotal"),
            "provider resultPageTotal",
        )
        expected_page_total = ceil(provider_total / self.page_size)
        if page_total != expected_page_total:
            raise ValueError(
                f"{self.name}: resultPageTotal {page_total} does not match "
                f"resultTotal {provider_total} and page size {self.page_size}"
            )
        if page_number >= page_total and provider_total:
            raise ValueError(f"{self.name}: checkpoint page is outside provider range")
        params = payload.get("params")
        if not isinstance(params, Mapping):
            raise ValueError(f"{self.name}: catalog response has no params object")
        returned_page = _state_nonnegative_int(params.get("page"), "provider page")
        returned_size = _positive_int(params.get("pageSize"), "provider pageSize")
        if returned_page != page_number or returned_size != self.page_size:
            raise ValueError(
                f"{self.name}: catalog pagination response does not match request"
            )
        groups = payload.get("results")
        if not _is_sequence(groups):
            raise ValueError(f"{self.name}: catalog results must be a list")
        model_groups = [
            group
            for group in groups
            if isinstance(group, Mapping) and _text(group.get("groupValue")).upper() == "MODEL"
        ]
        if len(model_groups) != 1:
            raise ValueError(f"{self.name}: expected exactly one MODEL result group")
        model_group = model_groups[0]
        group_total = _state_nonnegative_int(
            model_group.get("totalCount"),
            "MODEL group totalCount",
        )
        if group_total != provider_total:
            raise ValueError(
                f"{self.name}: MODEL group total {group_total} does not match "
                f"resultTotal {provider_total}"
            )
        items = model_group.get("resources")
        if not _is_sequence(items):
            raise ValueError(f"{self.name}: MODEL group resources must be a list")
        expected_rows = min(
            self.page_size,
            max(provider_total - page_number * self.page_size, 0),
        )
        if len(items) != expected_rows:
            raise ValueError(
                f"{self.name}: catalog returned {len(items)} row(s) for page "
                f"{page_number}, expected {expected_rows} from provider total"
            )
        return tuple(items), provider_total, page_total

    def _record(
        self,
        item: Any,
        page_number: int,
        provider_total: int,
    ) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError("catalog model is not an object")
        resource_id = _resource_id(item.get("resourceId"))
        org_name = _required_text(item.get("orgName"), "catalog model orgName")
        team_name = _text(item.get("teamName"))
        model_name = _required_text(item.get("name"), "catalog model name")
        _validate_resource_id(resource_id, org_name, team_name, model_name)
        title = _text(item.get("displayName")) or resource_id
        local_id = f"{resource_id}#model"
        identifier = Identifier("ngc:model", resource_id)
        model = ModelHint(
            local_id=local_id,
            name=title,
            aliases=_aliases(resource_id, model_name, title),
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator="$.resourceId",
        )
        model_page = self._model_page_url(org_name, team_name, model_name)
        description = _text(item.get("description"))
        # The catalog declares this one first-party card for this exact model.
        # The adjacent exact-version metadata endpoint is the crawlable form:
        # it returns the card's Markdown without making a catalog sweep fan out
        # into individual cards.
        links = [
            Link(
                model_page,
                relation="model_card",
                locator="$.resourceId",
                crawl=False,
            ),
        ]
        links.extend(
            Link(url, relation="documentation_reference", locator="$.description")
            for url in extract_urls(description)
        )
        attributes = _attributes(item.get("attributes"))
        latest_releases = _latest_release(
            resource_id,
            local_id,
            attributes,
            _text(item.get("dateModified")) or None,
        )
        if self.include_all_versions:
            releases, version_inventory_status, version_inventory_error = (
                self._model_versions(resource_id, local_id, latest_releases)
            )
        else:
            releases = latest_releases
            version_inventory_status = "not_requested"
            version_inventory_error = None
        latest_version = _text(attributes.get("latestVersionIdStr"))
        metadata_url = (
            self._model_metadata_url(org_name, team_name, model_name, latest_version)
            if latest_version
            else None
        )
        if metadata_url:
            # This exact-version endpoint returns the model-card Markdown. It
            # is queued for a future one-entry resolver, never fetched while
            # enumerating the catalog itself.
            links.append(
                Link(
                    metadata_url,
                    relation="model_card_metadata",
                    locator="$.attributes.latestVersionIdStr",
                )
            )
        text = _record_text(description, item, attributes)
        raw = dict(item)
        raw["catalog_page"] = page_number
        raw["provider_total"] = provider_total
        raw["version_inventory_status"] = version_inventory_status
        if version_inventory_error is not None:
            raw["version_inventory_error"] = version_inventory_error
        return SourceRecord(
            source_record_id=resource_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=model_page,
            title=title,
            raw=raw,
            text=text,
            modified_at=_text(item.get("dateModified")) or None,
            identifiers=(identifier,),
            links=_unique_links(links),
            models=(model,),
            releases=releases,
        )

    def _model_versions(
        self,
        resource_id: str,
        model_local_id: str,
        latest_releases: tuple[ReleaseHint, ...],
    ) -> tuple[tuple[ReleaseHint, ...], str, str | None]:
        """Expand guest-visible exact versions from NGC's model metadata route.

        The guest endpoint currently reports its version list in one page. We
        validate that contract and retain the base catalog row with an
        unavailable inventory status if one model cannot be expanded. This
        keeps a single inaccessible model from dropping its catalog row or
        interrupting later catalog pages.
        """
        try:
            return self._fetch_model_versions(resource_id, model_local_id, latest_releases)
        except Exception as error:
            status, message = _version_failure_status(error)
            fallback = tuple(
                _with_version_inventory_status(release, status, message)
                for release in latest_releases
            )
            return fallback, status, message

    def _fetch_model_versions(
        self,
        resource_id: str,
        model_local_id: str,
        latest_releases: tuple[ReleaseHint, ...],
    ) -> tuple[tuple[ReleaseHint, ...], str, str | None]:
        url = f"{self.metadata_base_url}/v2/models/{_quoted_resource_id(resource_id)}/versions"
        response: HttpResponse = self.client.get(
            url,
            headers={"Accept": "application/json"},
        )
        if response.status in {401, 403}:
            message = f"HTTP {response.status}"
            fallback = tuple(
                _with_version_inventory_status(
                    release,
                    "unavailable_unauthorized",
                    message,
                )
                for release in latest_releases
            )
            return fallback, "unavailable_unauthorized", message
        if response.status != 200:
            if response.status in {408, 425, 429, 500, 502, 503, 504}:
                message = f"HTTP {response.status}"
                fallback = tuple(
                    _with_version_inventory_status(
                        release,
                        "unavailable_transient",
                        message,
                    )
                    for release in latest_releases
                )
                return fallback, "unavailable_transient", message
            raise ValueError(
                f"{self.name}: versions for {resource_id} returned HTTP {response.status}"
            )
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: versions for {resource_id} exceed "
                f"{self.max_response_bytes} bytes"
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: versions response for {resource_id} is not an object")
        rows = payload.get("modelVersions")
        if not _is_sequence(rows) or any(
            not isinstance(row, Mapping) for row in rows
        ):
            raise ValueError(
                f"{self.name}: modelVersions for {resource_id} is not a list of objects"
            )
        pagination = payload.get("paginationInfo")
        if not isinstance(pagination, Mapping):
            raise ValueError(
                f"{self.name}: versions response for {resource_id} has no paginationInfo"
            )
        total = _state_nonnegative_int(
            pagination.get("totalResults"),
            "version totalResults",
        )
        total_pages = _state_nonnegative_int(
            pagination.get("totalPages"),
            "version totalPages",
        )
        page_index = _state_nonnegative_int(pagination.get("index"), "version page index")
        page_size = _positive_int(pagination.get("size"), "version page size")
        valid_total_pages = (
            total_pages in {0, 1}
            if total == 0
            else total_pages == ceil(total / page_size)
        )
        if page_index != 0 or not valid_total_pages:
            raise ValueError(f"{self.name}: invalid version pagination metadata for {resource_id}")
        if total == 0 and not rows:
            return (), "complete", None
        if total_pages != 1 or len(rows) != total:
            raise ValueError(
                f"{self.name}: versions for {resource_id} span {total_pages} page(s); "
                "the guest endpoint's continuation contract is not verified"
            )
        releases: list[ReleaseHint] = []
        seen: set[str] = set()
        for index, row in enumerate(rows):
            version = _required_text(row.get("versionId"), f"version row {index} versionId")
            if version in seen:
                raise ValueError(f"{self.name}: duplicate version {version!r} for {resource_id}")
            seen.add(version)
            released_at = _text(row.get("createdDate")) or None
            identity = Identifier("ngc:model-version", f"{resource_id}:{version}")
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"versionId", "createdDate"}
            }
            metadata["version_inventory_status"] = "complete"
            releases.append(
                ReleaseHint(
                    local_id=f"{resource_id}#version:{version}",
                    model_local_id=model_local_id,
                    version=version,
                    identifiers=(identity,),
                    released_at=released_at,
                    metadata=metadata,
                    locator=f"$.modelVersions[{index}]",
                )
            )
        return tuple(releases), "complete", None

    def _model_page_url(self, org_name: str, team_name: str, model_name: str) -> str:
        segments = ["orgs", quote(org_name, safe="")]
        if team_name:
            segments.append(quote(team_name, safe=""))
        segments.extend(("models", quote(model_name, safe="")))
        return canonicalize_url(f"{self.model_page_base_url}/{'/'.join(segments)}")

    def _model_metadata_url(
        self,
        org_name: str,
        team_name: str,
        model_name: str,
        version: str,
    ) -> str:
        segments = ["v2", "models", quote(org_name, safe="")]
        if team_name:
            segments.append(quote(team_name, safe=""))
        segments.extend(
            (quote(model_name, safe=""), "versions", quote(version, safe=""))
        )
        return canonicalize_url(f"{self.metadata_base_url}/{'/'.join(segments)}")


def _latest_release(
    resource_id: str,
    model_local_id: str,
    attributes: Mapping[str, Any],
    released_at: str | None,
) -> tuple[ReleaseHint, ...]:
    version_id = _text(attributes.get("latestVersionIdStr"))
    if not version_id:
        return ()
    metadata = {
        key: value
        for key in ("application", "format", "latestVersionSizeInBytes")
        if (value := attributes.get(key)) is not None
        and isinstance(value, str | int | float | bool)
    }
    return (
        ReleaseHint(
            local_id=f"{resource_id}#latest:{version_id}",
            model_local_id=model_local_id,
            version=version_id,
            identifiers=(Identifier("ngc:model-version", f"{resource_id}:{version_id}"),),
            released_at=released_at,
            metadata=metadata,
            locator="$.attributes.latestVersionIdStr",
        ),
    )


def _record_text(
    description: str,
    item: Mapping[str, Any],
    attributes: Mapping[str, Any],
) -> str:
    parts = [description] if description else []
    labels = _label_values(item.get("labels"))
    if labels:
        parts.append("labels: " + ", ".join(labels))
    for key in ("application", "format"):
        if value := _text(attributes.get(key)):
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


def _attributes(value: Any) -> Mapping[str, Any]:
    """Normalize NGC's ``[{key, value}]`` attributes without losing raw data."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if not _is_sequence(value):
        raise ValueError("catalog model attributes is not an object or list")
    result: dict[str, Any] = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"catalog model attribute {index} is not an object")
        key = _required_text(item.get("key"), f"catalog model attribute {index} key")
        if key in result:
            raise ValueError(f"catalog model has duplicate attribute {key!r}")
        result[key] = item.get("value")
    return result


def _label_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return _text_values(value)
    if not _is_sequence(value):
        return ()
    result = []
    for item in value:
        if isinstance(item, str):
            result.append(item.strip())
            continue
        if not isinstance(item, Mapping):
            continue
        key = _text(item.get("key"))
        values = _text_values(item.get("values")) or _text_values(
            item.get("unresolvedValues")
        )
        result.extend(f"{key}: {label}" if key else label for label in values)
    return tuple(label for label in result if label)


def _resource_id(value: Any) -> str:
    result = _required_text(value, "catalog model resourceId")
    has_invalid_part = any(
        not part or any(character.isspace() for character in part)
        for part in result.split("/")
    )
    if has_invalid_part:
        raise ValueError("catalog model resourceId has an invalid path segment")
    return result


def _quoted_resource_id(value: str) -> str:
    return "/".join(quote(part, safe="") for part in value.split("/"))


def _version_failure_status(error: Exception) -> tuple[str, str]:
    """Classify per-model HTTP failures without copying request URLs or tokens."""
    cause = error.__cause__ if isinstance(error, HttpFailure) else error
    if isinstance(cause, HTTPError):
        code = cause.code
        if code in {401, 403}:
            return "unavailable_unauthorized", f"HTTP {code}"
        if code in {408, 425, 429, 500, 502, 503, 504}:
            return "unavailable_transient", f"HTTP {code}"
        return "unavailable_error", f"HTTP {code}"
    if isinstance(cause, TimeoutError):
        return "unavailable_transient", "network timeout"
    if isinstance(cause, URLError):
        return "unavailable_transient", "network request failed"
    if isinstance(error, HttpFailure):
        return "unavailable_error", "HTTP request failed"
    return "unavailable_error", f"{type(error).__name__}: {error}"


def _with_version_inventory_status(
    release: ReleaseHint,
    status: str,
    error: str | None = None,
) -> ReleaseHint:
    metadata = {**release.metadata, "version_inventory_status": status}
    if error is not None:
        metadata["version_inventory_error"] = error
    return replace(
        release,
        metadata=metadata,
    )


def _validate_resource_id(
    resource_id: str,
    org_name: str,
    team_name: str,
    model_name: str,
) -> None:
    expected = "/".join((org_name, *([team_name] if team_name else []), model_name))
    if resource_id != expected:
        raise ValueError(
            "catalog model resourceId does not match orgName/teamName/name fields"
        )


def _aliases(resource_id: str, model_name: str, title: str) -> tuple[str, ...]:
    return tuple(
        value
        for value in dict.fromkeys((resource_id, model_name))
        if value and value != title
    )


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda link: (
                link.url,
                link.relation,
                link.locator or "",
                link.crawl,
                link.model_local_ids,
            ),
        )
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _text_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        values = value.values()
    elif _is_sequence(value):
        values = value
    else:
        return ()
    return tuple(item for item in (_text(value) for value in values) if item)


def _web_url(value: str, label: str) -> str:
    result = canonicalize_url(value)
    if not result.startswith(("https://", "http://")):
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    return result


def _required_text(value: Any, field: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _state_nonnegative_int(value: Any, field: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
