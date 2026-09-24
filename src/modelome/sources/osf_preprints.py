from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]
_NO_VALUE = frozenset({"", "na", "n/a", "none", "not available", "null"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OsfPreprintSourceAdapter:
    """Enumerate every public OSF-hosted community preprint in closed UTC windows.

    OSF's top-level preprint endpoint deliberately spans every provider. That keeps
    this source independent of a hand-maintained list of current rXiv communities
    and retains their legacy records when a community migrates to another host.
    """

    def __init__(
        self,
        *,
        name: str = "osf-preprints",
        url: str = "https://api.osf.io/v2/preprints/",
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        page_size: int = 100,
        initial_lookback_days: int = 7,
        overlap_days: int = 2,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name)
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.page_size = int(page_size)
        self.initial_lookback_days = int(initial_lookback_days)
        self.overlap_days = int(overlap_days)
        if not 1 <= self.page_size <= 100:
            raise ValueError(f"{self.name}: page_size must be between 1 and 100")
        if self.initial_lookback_days < 0:
            raise ValueError(f"{self.name}: initial lookback days must be nonnegative")
        if self.overlap_days < 0:
            raise ValueError(f"{self.name}: overlap days must be nonnegative")
        self.client = client or HttpClient()
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "osf-preprints-v1",
                "url": self.url,
                "artifact_kind": self.artifact_kind.value,
                "page_size": self.page_size,
                "initial_lookback_days": self.initial_lookback_days,
                "overlap_days": self.overlap_days,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        now = _as_utc(self.clock())
        closed_through = now.date() - timedelta(days=1)
        prior_watermark = _optional_date(state.get("watermark"), "watermark", self.name)
        window_start, window_end, frozen = self._window(
            state,
            prior_watermark=prior_watermark,
            closed_through=closed_through,
        )
        page_number = _state_page(state, frozen=frozen, source=self.name)

        if window_start > window_end:
            next_state: dict[str, Any] = {"completed_at": _isoformat(now)}
            if prior_watermark is not None:
                next_state["watermark"] = prior_watermark.isoformat()
            return SourcePage(records=(), next_state=next_state, complete=True, upstream_count=0)

        retry_state = self._scan_state(
            state,
            page_number=page_number,
            window_start=window_start,
            window_end=window_end,
            now=now,
            prior_watermark=prior_watermark,
        )
        response: HttpResponse = self.client.get(
            self.url,
            params={
                "page": page_number,
                "page[size]": self.page_size,
                "sort": "date_modified",
                "filter[date_modified][gte]": _start_of_day(window_start),
                "filter[date_modified][lte]": _end_of_day(window_end),
            },
            headers={"Accept": "application/vnd.api+json, application/json"},
        )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: expected a JSON object from {response.url}")
        raw_items = payload.get("data")
        if not _is_sequence(raw_items):
            raise ValueError(f"{self.name}: response.data must be an array")
        links = payload.get("links")
        if not isinstance(links, Mapping):
            raise ValueError(f"{self.name}: response.links must be an object")
        total = _response_total(links, self.name)
        next_page = _next_page(links.get("next"), self.name)

        issues: list[SourceIssue] = []
        prior_total = _state_count(state, "scan_total", self.name) if frozen else None
        replay_required = False
        if prior_total is not None and prior_total != total:
            replay_required = True
            issues.append(
                self._pagination_issue(
                    page_number,
                    window_start,
                    window_end,
                    f"window total changed from {prior_total} to {total}",
                    total,
                    len(raw_items),
                )
            )
            # A changed total can shift number-based pages. Replaying the frozen
            # window prevents an insertion or deletion from skipping a record.
            retry_state = self._scan_state(
                {},
                page_number=1,
                window_start=window_start,
                window_end=window_end,
                now=now,
                prior_watermark=prior_watermark,
            )

        if next_page is not None and next_page != page_number + 1:
            issues.append(
                self._pagination_issue(
                    page_number,
                    window_start,
                    window_end,
                    f"response next page {next_page}, expected {page_number + 1}",
                    total,
                    len(raw_items),
                )
            )
            next_page = None

        records: list[SourceRecord] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, Mapping):
                issues.append(
                    SourceIssue(
                        source_record_id=f"{self.name}:item:{page_number}:{index}",
                        stage="source_normalize",
                        error="TypeError: OSF preprint result is not a JSON object",
                        summary={"page": page_number, "index": index, "value": repr(item)[:1000]},
                    )
                )
                continue
            try:
                records.append(self._record(item))
            except (KeyError, TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=self._malformed_id(item, page_number, index),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"page": page_number, "index": index, "raw": dict(item)},
                    )
                )

        raw_items_seen = (
            _state_count(state, "raw_items_seen", self.name) or 0
        ) + len(raw_items)
        if raw_items_seen > total:
            issues.append(
                self._pagination_issue(
                    page_number,
                    window_start,
                    window_end,
                    f"page ended at item {raw_items_seen}, beyond declared total {total}",
                    total,
                    len(raw_items),
                )
            )
            next_page = None
        if next_page is not None and not raw_items:
            issues.append(
                self._pagination_issue(
                    page_number,
                    window_start,
                    window_end,
                    "pagination returned no records before its next page",
                    total,
                    0,
                )
            )
            next_page = None

        complete = not replay_required and next_page is None and raw_items_seen == total
        if next_page is None and raw_items_seen != total:
            issues.append(
                self._pagination_issue(
                    page_number,
                    window_start,
                    window_end,
                    f"pagination ended after {raw_items_seen} records, expected {total}",
                    total,
                    len(raw_items),
                )
            )

        if complete:
            next_state = {
                "watermark": window_end.isoformat(),
                "completed_at": _isoformat(now),
            }
        elif next_page is not None and not replay_required:
            next_state = self._scan_state(
                state,
                page_number=next_page,
                window_start=window_start,
                window_end=window_end,
                now=now,
                prior_watermark=prior_watermark,
                raw_items_seen=raw_items_seen,
                scan_total=total,
            )
        else:
            next_state = retry_state

        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=total,
            issues=tuple(issues),
            retry_state=retry_state,
        )

    def _window(
        self,
        state: Mapping[str, Any],
        *,
        prior_watermark: date | None,
        closed_through: date,
    ) -> tuple[date, date, bool]:
        raw_start = state.get("window_start")
        raw_end = state.get("window_end")
        if raw_start is not None or raw_end is not None:
            if raw_start is None or raw_end is None:
                raise ValueError(f"{self.name}: frozen window requires both boundaries")
            window_start = _required_date(raw_start, "window_start", self.name)
            window_end = _required_date(raw_end, "window_end", self.name)
            if window_start > window_end:
                raise ValueError(f"{self.name}: window_start must not follow window_end")
            if window_end > closed_through:
                raise ValueError(f"{self.name}: window_end must be a closed UTC day")
            return window_start, window_end, True

        if "page" in state:
            raise ValueError(f"{self.name}: page checkpoint is missing its frozen window")
        if prior_watermark is None:
            lookback = max(self.initial_lookback_days, 1)
            window_start = closed_through - timedelta(days=lookback - 1)
        else:
            if prior_watermark > closed_through:
                raise ValueError(f"{self.name}: watermark is later than the last closed UTC day")
            window_start = prior_watermark + timedelta(days=1 - self.overlap_days)
        return window_start, closed_through, False

    def _scan_state(
        self,
        state: Mapping[str, Any],
        *,
        page_number: int,
        window_start: date,
        window_end: date,
        now: datetime,
        prior_watermark: date | None,
        raw_items_seen: int | None = None,
        scan_total: int | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "page": page_number,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "started_at": _text(state.get("started_at")) or _isoformat(now),
            "raw_items_seen": (
                raw_items_seen
                if raw_items_seen is not None
                else _state_count(state, "raw_items_seen", self.name) or 0
            ),
        }
        known_total = scan_total
        if known_total is None:
            known_total = _state_count(state, "scan_total", self.name)
        if known_total is not None:
            result["scan_total"] = known_total
        if prior_watermark is not None:
            result["watermark"] = prior_watermark.isoformat()
        return result

    def _record(self, item: Mapping[str, Any]) -> SourceRecord:
        identifier = _required_text(item.get("id"), "OSF preprint id")
        item_type = _required_text(item.get("type"), "OSF preprint type")
        if item_type != "preprints":
            raise ValueError(f"OSF item type must be 'preprints', got {item_type!r}")
        attributes = item.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError("OSF preprint attributes must be an object")
        links = item.get("links")
        if not isinstance(links, Mapping):
            raise ValueError("OSF preprint links must be an object")
        preprint_url = _required_web_url(links.get("html"), "OSF preprint html URL")
        title = _required_text(attributes.get("title"), "OSF preprint title")
        provider = _provider_id(item)

        identifiers = [Identifier("osf:preprint", identifier)]
        record_links = [Link(preprint_url, relation="preprint", locator="$.links.html")]
        for locator, value in (
            ("$.attributes.doi", attributes.get("doi")),
            ("$.links.preprint_doi", links.get("preprint_doi")),
        ):
            if doi := _optional_doi(value):
                identifiers.append(Identifier("doi", doi))
                record_links.append(Link(_doi_url(doi), relation="doi", locator=locator))
        if project_url := _optional_web_url(links.get("iri")):
            record_links.append(
                Link(project_url, relation="related_project", locator="$.links.iri")
            )

        raw = dict(item)
        raw["provider"] = provider
        return SourceRecord(
            source_record_id=f"osf-preprints:{identifier}",
            kind=self.artifact_kind,
            canonical_url=preprint_url,
            title=title,
            raw=raw,
            text=_text(attributes.get("description")),
            published_at=_optional_timestamp(attributes.get("date_published")),
            modified_at=_optional_timestamp(attributes.get("date_modified")),
            identifiers=_unique_identifiers(identifiers),
            links=_unique_links(record_links),
        )

    def _malformed_id(self, item: Mapping[str, Any], page_number: int, index: int) -> str:
        identifier = _text(item.get("id"))
        if identifier:
            return f"osf-preprints:{identifier}"
        return f"{self.name}:malformed:{content_hash(dict(item))[:32]}:{page_number}:{index}"

    def _pagination_issue(
        self,
        page_number: int,
        window_start: date,
        window_end: date,
        message: str,
        total: int,
        received: int,
    ) -> SourceIssue:
        summary = {
            "page": page_number,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "declared_total": total,
            "received_count": received,
        }
        return SourceIssue(
            source_record_id=(
                f"{self.name}:pagination:{content_hash({**summary, 'message': message})[:32]}"
            ),
            stage="source_pagination",
            error=f"{self.name}: {message}",
            summary=summary,
        )


def _response_total(links: Mapping[str, Any], source: str) -> int:
    meta = links.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError(f"{source}: response.links.meta must be an object")
    total = _optional_count(meta.get("total"))
    if total is None:
        raise ValueError(f"{source}: response.links.meta.total must be a nonnegative integer")
    return total


def _next_page(value: Any, source: str) -> int | None:
    if value is None:
        return None
    url = _required_web_url(value, f"{source} response next URL")
    parsed = urlsplit(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    values = query.get("page")
    if values is None or len(values) != 1:
        raise ValueError(f"{source}: response next URL must have exactly one page parameter")
    page = _optional_count(values[0])
    if page is None or page < 1:
        raise ValueError(f"{source}: response next URL has an invalid page parameter")
    return page


def _provider_id(item: Mapping[str, Any]) -> str | None:
    relationships = item.get("relationships")
    if not isinstance(relationships, Mapping):
        return None
    provider = relationships.get("provider")
    if not isinstance(provider, Mapping):
        return None
    data = provider.get("data")
    if not isinstance(data, Mapping):
        return None
    value = _text(data.get("id"))
    return value or None


def _state_page(state: Mapping[str, Any], *, frozen: bool, source: str) -> int:
    if "page" not in state:
        return 1
    if not frozen:
        raise ValueError(f"{source}: page checkpoint is missing its frozen window")
    page = _optional_count(state.get("page"))
    if page is None or page < 1:
        raise ValueError(f"{source}: invalid page checkpoint")
    return page


def _state_count(state: Mapping[str, Any], key: str, source: str) -> int | None:
    if key not in state:
        return None
    count = _optional_count(state.get(key))
    if count is None:
        raise ValueError(f"{source}: invalid {key} checkpoint")
    return count


def _optional_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _optional_date(value: Any, field: str, source: str) -> date | None:
    if value is None or _text(value) == "":
        return None
    return _required_date(value, field, source)


def _required_date(value: Any, field: str, source: str) -> date:
    if isinstance(value, datetime):
        return _as_utc(value).date()
    if isinstance(value, date):
        return value
    text = _text(value)
    try:
        if len(text) == 10:
            return date.fromisoformat(text)
        return _as_utc(datetime.fromisoformat(text.replace("Z", "+00:00"))).date()
    except ValueError:
        raise ValueError(f"{source}: invalid {field}: {value!r}") from None


def _optional_timestamp(value: Any) -> str | None:
    timestamp = _text(value)
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"OSF preprint timestamp is invalid: {value!r}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return _isoformat(parsed)


def _start_of_day(value: date) -> str:
    return _isoformat(datetime.combine(value, time.min, tzinfo=UTC))


def _end_of_day(value: date) -> str:
    return _isoformat(datetime.combine(value, time.max, tzinfo=UTC))


def _doi_url(doi: str) -> str:
    return canonicalize_url(f"https://doi.org/{doi}")


def _optional_doi(value: Any) -> str:
    text = unquote(_text(value)).strip()
    if text.casefold() in _NO_VALUE:
        return ""
    lowered = text.casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.strip().casefold()
    if not text.startswith("10.") or "/" not in text or any(char.isspace() for char in text):
        return ""
    return text


def _required_web_url(value: Any, field: str) -> str:
    url = _optional_web_url(value)
    if not url:
        raise ValueError(f"{field} must be an HTTP(S) URL")
    return url


def _optional_web_url(value: Any) -> str:
    text = _text(value)
    if not text or text.casefold() in _NO_VALUE:
        return ""
    canonical = canonicalize_url(text)
    parts = urlsplit(canonical)
    return canonical if parts.scheme in {"http", "https"} and parts.hostname else ""


def _web_url(value: Any, source: str) -> str:
    return _required_web_url(value, f"{source} API URL").rstrip("/")


def _unique_identifiers(values: Sequence[Identifier]) -> tuple[Identifier, ...]:
    return tuple(dict.fromkeys(values))


def _unique_links(values: Sequence[Link]) -> tuple[Link, ...]:
    seen: set[tuple[str, str]] = set()
    result = []
    for value in values:
        key = (value.url, value.relation)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = ["OsfPreprintSourceAdapter"]
