"""Bounded public ModelScope model-catalog ingestion.

ModelScope's public OpenAPI model list intentionally limits deep pagination to
3,000 rows per sort.  This adapter therefore takes a bounded union of the
provider's documented catalog orders instead of pretending that the API exposes
an exhaustive snapshot.  Every retained row has a source-native model ID and a
first-party model-card URL; no repository, card, or artifact bytes are fetched.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
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
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, extract_urls

Clock = Callable[[], datetime]

_SORTS = frozenset({"default", "downloads", "likes", "last_modified"})
_MAX_OFFSET = 3_000


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ModelScopeModelsSourceAdapter:
    """Enumerate a transparent bounded union of ModelScope catalog windows.

    The source's official OpenAPI contract caps ``page_number * page_size`` at
    3,000, which makes an exhaustive anonymous scan impossible.  By default we
    capture the first 3,000 public rows in each documented ordering.  A run is
    complete when those configured windows, not ModelScope's whole catalog, have
    been observed.  This distinction prevents snapshot deletion semantics and
    coverage reporting from claiming the provider's larger total.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the configured bounded union of ModelScope's public model-list "
        "sorts. ModelScope caps anonymous deep pagination at 3,000 rows per "
        "sort, so this is not an exhaustive ModelScope catalog snapshot. It "
        "does not fetch model cards, repository files, papers, or weights."
    )

    def __init__(
        self,
        *,
        name: str = "modelscope-models",
        url: str = "https://modelscope.cn/openapi/v1/models",
        page_size: int = 50,
        max_pages_per_sort: int = 60,
        sorts: Sequence[str] = ("default", "downloads", "likes", "last_modified"),
        search_queries: Sequence[str] = (),
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, "catalog URL")
        self.page_size = _positive_int(page_size, "page_size")
        if self.page_size > 50:
            raise ValueError(f"{self.name}: page_size must be from 1 through 50")
        self.max_pages_per_sort = _positive_int(
            max_pages_per_sort,
            "max_pages_per_sort",
        )
        if self.page_size * self.max_pages_per_sort > _MAX_OFFSET:
            raise ValueError(
                f"{self.name}: page_size * max_pages_per_sort must not exceed "
                f"ModelScope's {_MAX_OFFSET:,}-row per-sort ceiling"
            )
        self.sorts = _sorts(sorts, self.name)
        self.search_queries = _search_queries(search_queries, self.name)
        self.max_response_bytes = _positive_int(
            max_response_bytes,
            "max_response_bytes",
        )
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "modelscope-models-v1",
                "url": self.url,
                "page_size": self.page_size,
                "max_pages_per_sort": self.max_pages_per_sort,
                "sorts": self.sorts,
                "search_queries": self.search_queries,
                "max_response_bytes": self.max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        sort_index = _state_nonnegative_int(state.get("sort_index"), "sort_index", 0)
        page_number = _state_positive_int(state.get("page_number"), "page_number", 1)
        if sort_index >= len(self.sorts):
            raise ValueError(f"{self.name}: checkpoint sort_index is out of range")
        if page_number > self.max_pages_per_sort:
            raise ValueError(f"{self.name}: checkpoint page_number exceeds configured window")
        sort = self.sorts[sort_index]
        query_index = _state_nonnegative_int(state.get("query_index"), "query_index", 0)
        if query_index > len(self.search_queries):
            raise ValueError(f"{self.name}: checkpoint query_index is out of range")
        query = self.search_queries[query_index - 1] if query_index else None
        params = {"sort": sort, "page_number": page_number, "page_size": self.page_size}
        if query is not None:
            params["search"] = query
        response: HttpResponse = self.client.get(
            self.url,
            params=params,
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog response exceeds {self.max_response_bytes} bytes"
            )
        items, total = self._items(response, sort, page_number)
        provider_totals = _provider_totals(state.get("provider_totals"), self.name)
        total_key = sort if query is None else f"search:{query}:{sort}"
        prior_total = provider_totals.get(total_key)
        if prior_total is not None and prior_total != total:
            raise ValueError(
                f"{self.name}: provider total for {sort!r} changed from "
                f"{prior_total} to {total}; restart the bounded scan"
            )
        provider_totals[total_key] = total

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, item in enumerate(items):
            if isinstance(item, Mapping) and item.get("private") is True:
                continue
            try:
                records.append(self._record(item, sort, page_number, total))
            except (TypeError, ValueError) as error:
                raw = dict(item) if isinstance(item, Mapping) else {"value": repr(item)[:1000]}
                source_id = _text(item.get("id")) if isinstance(item, Mapping) else ""
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            source_id
                            or f"{self.name}:malformed:{content_hash(raw)[:32]}"
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index, "raw": raw},
                    )
                )

        prior_rows = _state_nonnegative_int(
            state.get("observed_window_rows"),
            "observed_window_rows",
            0,
        )
        observed_rows = prior_rows + len(items)
        window_ends = (
            page_number == self.max_pages_per_sort
            or page_number * self.page_size >= total
        )
        next_sort_index = sort_index + 1 if window_ends else sort_index
        next_query_index = query_index
        if window_ends and next_sort_index >= len(self.sorts):
            next_query_index += 1
            next_sort_index = 0
        next_page_number = 1 if window_ends else page_number + 1
        complete = next_query_index > len(self.search_queries)
        checked_at = _isoformat(self.clock())
        if complete:
            next_state: dict[str, Any] = {
                "completed_at": checked_at,
                "observed_window_rows": observed_rows,
                "provider_totals": provider_totals,
            }
        else:
            next_state = {
                "sort_index": next_sort_index,
                "query_index": next_query_index,
                "page_number": next_page_number,
                "observed_window_rows": observed_rows,
                "provider_totals": provider_totals,
                "started_at": _text(state.get("started_at")) or checked_at,
            }
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=observed_rows if complete else None,
            issues=tuple(issues),
        )

    def _items(
        self,
        response: HttpResponse,
        sort: str,
        page_number: int,
    ) -> tuple[tuple[Any, ...], int]:
        payload = response.json()
        if not isinstance(payload, Mapping) or payload.get("success") is not True:
            raise ValueError(f"{self.name}: expected a successful catalog envelope")
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise ValueError(f"{self.name}: successful catalog response has no data object")
        items = data.get("models")
        if not _is_sequence(items):
            raise ValueError(f"{self.name}: catalog data.models must be a list")
        returned_page = _state_positive_int(
            data.get("page_number"),
            "provider page_number",
            page_number,
        )
        returned_size = _state_positive_int(
            data.get("page_size"),
            "provider page_size",
            self.page_size,
        )
        if returned_page != page_number or returned_size != self.page_size:
            raise ValueError(
                f"{self.name}: catalog pagination response does not match request "
                f"for sort {sort!r}"
            )
        total = _state_nonnegative_int(data.get("total_count"), "provider total_count")
        expected_rows = min(
            self.page_size,
            max(total - (page_number - 1) * self.page_size, 0),
        )
        if len(items) != expected_rows:
            raise ValueError(
                f"{self.name}: catalog returned {len(items)} row(s) for page "
                f"{page_number}, expected {expected_rows} from provider total"
            )
        return tuple(items), total

    def _record(
        self,
        item: Any,
        sort: str,
        page_number: int,
        provider_total: int,
    ) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError("catalog model is not an object")
        model_id = _model_id(item.get("id"))
        encoded_id = quote(model_id, safe="/")
        model_url = canonicalize_url(f"https://modelscope.cn/models/{encoded_id}")
        metadata_url = canonicalize_url(
            f"https://modelscope.cn/openapi/v1/models/{encoded_id}"
        )
        title = _text(item.get("display_name")) or model_id
        identifier = Identifier("modelscope:model", model_id)
        model = ModelHint(
            local_id=f"{model_id}#model",
            name=title,
            aliases=(model_id,) if title != model_id else (),
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator="$.id",
        )
        description = _text(item.get("description"))
        tasks = _text_values(item.get("tasks"))
        tags = _text_values(item.get("tags"))
        text_parts = [description] if description else []
        if tasks:
            text_parts.append("tasks: " + ", ".join(tasks))
        if tags:
            text_parts.append("tags: " + ", ".join(tags))
        links = [
            Link(model_url, relation="model_page", locator="$.id"),
            Link(metadata_url, relation="metadata", locator="$.id"),
        ]
        links.extend(
            Link(url, relation="documentation_reference", locator="$.description")
            for url in extract_urls(description)
        )
        raw = dict(item)
        raw["catalog_sort"] = sort
        raw["catalog_page_number"] = page_number
        raw["provider_total_count"] = provider_total
        return SourceRecord(
            source_record_id=model_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=model_url,
            title=title,
            raw=raw,
            text="\n".join(text_parts),
            published_at=_text(item.get("created_at")) or None,
            modified_at=_text(item.get("last_modified")) or None,
            identifiers=(identifier,),
            links=_unique_links(links),
            models=(model,),
        )


def _sorts(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError(f"{name}: sorts must be a non-empty array")
    result = tuple(_required_text(value, "sort") for value in values)
    is_valid = (
        bool(result)
        and len(set(result)) == len(result)
        and all(value in _SORTS for value in result)
    )
    if not is_valid:
        raise ValueError(f"{name}: sorts must be unique documented ModelScope sorts")
    return result


def _search_queries(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError(f"{name}: search_queries must be an array")
    result = tuple(_required_text(value, "search query") for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name}: search_queries must be unique")
    return result


def _provider_totals(value: Any, name: str) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}: checkpoint provider_totals must be an object")
    result: dict[str, int] = {}
    for key, total in value.items():
        sort = _required_text(key, "checkpoint provider total sort")
        if sort not in _SORTS and not (
            sort.startswith("search:") and sort.rsplit(":", 1)[-1] in _SORTS
        ):
            raise ValueError(f"{name}: checkpoint has an unknown sort {sort!r}")
        result[sort] = _state_nonnegative_int(total, "checkpoint provider total")
    return result


def _model_id(value: Any) -> str:
    model_id = _required_text(value, "model id")
    parts = model_id.split("/")
    has_invalid_part = any(
        not part or any(character.isspace() for character in part) for part in parts
    )
    if len(parts) != 2 or has_invalid_part:
        raise ValueError("model id must be an owner/name pair")
    return model_id


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda link: (link.url, link.relation, link.locator or "", link.crawl),
        )
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _text_values(value: Any) -> tuple[str, ...]:
    if not _is_sequence(value):
        return ()
    return tuple(item for item in (_text(item) for item in value) if item)


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


def _state_positive_int(value: Any, field: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
