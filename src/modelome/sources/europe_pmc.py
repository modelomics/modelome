from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EuropePmcSourceAdapter:
    """Enumerate Europe PMC article updates across every source and subject."""

    def __init__(
        self,
        *,
        name: str = "europe-pmc",
        url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        page_size: int = 1_000,
        initial_lookback_days: int = 7,
        overlap_days: int = 2,
        email: str | None = None,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name)
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.page_size = int(page_size)
        self.initial_lookback_days = int(initial_lookback_days)
        self.overlap_days = int(overlap_days)
        self.email = _text(email) or None
        self.client = client or HttpClient()
        self.clock = clock
        if not 1 <= self.page_size <= 1_000:
            raise ValueError("Europe PMC page_size must be between 1 and 1000")
        if self.initial_lookback_days < 1:
            raise ValueError("Europe PMC initial_lookback_days must be positive")
        if self.overlap_days < 1:
            raise ValueError("Europe PMC overlap_days must be positive")
        self.checkpoint_signature = content_hash(
            {
                "adapter": "europe-pmc-v1",
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
        window_start, window_end = self._window(state, prior_watermark, closed_through)
        cursor = _text(state.get("cursor_mark")) or "*"
        resuming = cursor != "*"
        if resuming:
            raw_items_seen = _state_count(state, "raw_items_seen", self.name)
            scan_total = _state_count(state, "scan_total", self.name)
            if raw_items_seen is None or scan_total is None:
                raise ValueError(
                    f"{self.name}: cursor checkpoint requires raw_items_seen and scan_total"
                )
        else:
            raw_items_seen = 0
            scan_total = None
        seen_cursor_hashes = _state_hashes(state.get("seen_cursor_hashes"), self.name)
        started_at = _text(state.get("started_at")) or _isoformat(now)
        retry_state = self._scan_state(
            cursor=cursor,
            window_start=window_start,
            window_end=window_end,
            raw_items_seen=raw_items_seen,
            scan_total=scan_total,
            seen_cursor_hashes=seen_cursor_hashes,
            started_at=started_at,
            prior_watermark=prior_watermark,
        )
        query = (
            f"UPDATE_DATE:[{window_start.isoformat()} TO {window_end.isoformat()}]"
        )
        params: dict[str, str | int] = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "cursorMark": cursor,
            "pageSize": self.page_size,
            "synonym": "false",
        }
        if self.email:
            params["email"] = self.email
        response: HttpResponse = self.client.get(
            self.url,
            params=params,
            headers={"Accept": "application/json"},
        )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: response must be a JSON object")
        response_total = _nonnegative_integer(payload.get("hitCount"))
        if response_total is None:
            raise ValueError(f"{self.name}: response is missing a valid hitCount")
        raw_results = _results(payload, self.name)

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, item in enumerate(raw_results):
            if not isinstance(item, Mapping):
                issues.append(self._item_issue(index, item, "result is not a JSON object"))
                continue
            try:
                records.append(self._record(item))
            except (TypeError, ValueError) as error:
                issues.append(
                    self._item_issue(index, item, f"{type(error).__name__}: {error}")
                )
        raw_items_seen += len(raw_results)

        if scan_total is None:
            scan_total = response_total
        elif scan_total != response_total:
            issues.append(
                self._pagination_issue(
                    retry_state,
                    f"hitCount changed from {scan_total} to {response_total}",
                    raw_items_seen,
                    scan_total,
                )
            )
        reached_total = raw_items_seen == scan_total
        next_cursor = _text(payload.get("nextCursorMark"))
        next_hash = content_hash(next_cursor) if next_cursor else ""
        if not reached_total and next_cursor and (
            next_cursor == cursor or next_hash in seen_cursor_hashes
        ):
            issues.append(
                self._pagination_issue(
                    retry_state,
                    "cursor cycle detected",
                    raw_items_seen,
                    scan_total,
                )
            )
            next_cursor = ""
        if raw_items_seen > scan_total:
            issues.append(
                self._pagination_issue(
                    retry_state,
                    f"received {raw_items_seen} results for declared hitCount {scan_total}",
                    raw_items_seen,
                    scan_total,
                )
            )
        if not issues and not reached_total and (
            len(raw_results) < self.page_size or not next_cursor
        ):
            issues.append(
                self._pagination_issue(
                    retry_state,
                    f"pagination ended after {raw_items_seen} result(s), before "
                    f"the declared hitCount of {scan_total}",
                    raw_items_seen,
                    scan_total,
                )
            )

        pagination_failed = any(issue.stage == "source_pagination" for issue in issues)
        if pagination_failed:
            # A changed result set or broken cursor can shift every later page.
            # Replay the immutable interval from its first page so no record is skipped.
            retry_state = self._scan_state(
                cursor="*",
                window_start=window_start,
                window_end=window_end,
                raw_items_seen=0,
                scan_total=None,
                seen_cursor_hashes=(),
                started_at=started_at,
                prior_watermark=prior_watermark,
            )

        complete = reached_total and not issues
        if complete:
            next_state: dict[str, Any] = {
                "watermark": window_end.isoformat(),
                "completed_at": _isoformat(self.clock()),
            }
        elif next_cursor and not issues:
            next_state = self._scan_state(
                cursor=next_cursor,
                window_start=window_start,
                window_end=window_end,
                raw_items_seen=raw_items_seen,
                scan_total=scan_total,
                seen_cursor_hashes=(*seen_cursor_hashes, next_hash),
                started_at=started_at,
                prior_watermark=prior_watermark,
            )
        else:
            next_state = retry_state

        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=scan_total,
            issues=tuple(issues),
            retry_state=retry_state,
        )

    def _window(
        self,
        state: Mapping[str, Any],
        prior_watermark: date | None,
        closed_through: date,
    ) -> tuple[date, date]:
        raw_start = state.get("window_start")
        raw_end = state.get("window_end")
        if raw_start is not None or raw_end is not None:
            if raw_start is None or raw_end is None:
                raise ValueError(f"{self.name}: frozen window requires both boundaries")
            start = _required_date(raw_start, "window_start", self.name)
            end = _required_date(raw_end, "window_end", self.name)
            if start > end:
                raise ValueError(f"{self.name}: window_start must not follow window_end")
            if end > closed_through:
                raise ValueError(f"{self.name}: window_end must be a closed UTC day")
            return start, end
        if "cursor_mark" in state:
            raise ValueError(f"{self.name}: cursor checkpoint is missing its frozen window")
        if prior_watermark is None:
            start = closed_through - timedelta(days=self.initial_lookback_days - 1)
        else:
            if prior_watermark > closed_through:
                raise ValueError(f"{self.name}: watermark is later than the last closed UTC day")
            start = prior_watermark - timedelta(days=self.overlap_days - 1)
        return start, closed_through

    def _scan_state(
        self,
        *,
        cursor: str,
        window_start: date,
        window_end: date,
        raw_items_seen: int,
        scan_total: int | None,
        seen_cursor_hashes: Sequence[str],
        started_at: str,
        prior_watermark: date | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "cursor_mark": cursor,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "raw_items_seen": raw_items_seen,
            "started_at": started_at,
        }
        if scan_total is not None:
            result["scan_total"] = scan_total
        if seen_cursor_hashes:
            result["seen_cursor_hashes"] = list(dict.fromkeys(seen_cursor_hashes))
        if prior_watermark is not None:
            result["watermark"] = prior_watermark.isoformat()
        return result

    def _record(self, item: Mapping[str, Any]) -> SourceRecord:
        source = _required_text(item.get("source"), "Europe PMC result source").upper()
        record_id = _required_text(
            item.get("id") or item.get("extId"), "Europe PMC result id"
        )
        source_record_id = f"{source}:{record_id}"
        canonical_url = canonicalize_url(
            f"https://europepmc.org/article/{quote(source, safe='')}/"
            f"{quote(record_id, safe='._-')}"
        )
        title = _text(item.get("title")) or source_record_id
        abstract = _text(item.get("abstractText"))
        keywords = _keyword_terms(item.get("keywordList"))
        mesh_terms = _mesh_terms(item.get("meshHeadingList"))
        identifiers = list(_identifiers(item, source, record_id))
        links = list(_links(item, canonical_url))
        text_parts = [value for value in (title, abstract) if value]
        if keywords:
            text_parts.append(f"Keywords: {'; '.join(keywords)}")
        if mesh_terms:
            text_parts.append(f"MeSH: {'; '.join(mesh_terms)}")
        return SourceRecord(
            source_record_id=source_record_id,
            kind=self.artifact_kind,
            canonical_url=canonical_url,
            title=title,
            raw=dict(item),
            text="\n\n".join(text_parts),
            published_at=_text(item.get("firstPublicationDate")) or None,
            modified_at=(
                _text(item.get("dateOfRevision"))
                or _text(item.get("firstIndexDate"))
                or None
            ),
            identifiers=_unique_identifiers(identifiers),
            links=_unique_links(links),
        )

    def _item_issue(self, index: int, value: Any, error: str) -> SourceIssue:
        raw = dict(value) if isinstance(value, Mapping) else repr(value)[:1_000]
        return SourceIssue(
            source_record_id=f"{self.name}:malformed:{content_hash(raw)[:32]}",
            stage="source_normalize",
            error=f"{self.name}: {error}",
            summary={"index": index, "raw": raw},
        )

    def _pagination_issue(
        self,
        retry_state: Mapping[str, Any],
        error: str,
        raw_items_seen: int,
        known_total: int,
    ) -> SourceIssue:
        return SourceIssue(
            source_record_id=(
                f"{self.name}:pagination:"
                f"{content_hash({**retry_state, 'error': error})[:32]}"
            ),
            stage="source_pagination",
            error=f"{self.name}: {error}",
            summary={
                "window_start": retry_state["window_start"],
                "window_end": retry_state["window_end"],
                "raw_items_seen": raw_items_seen,
                "known_total": known_total,
            },
        )


def _results(payload: Mapping[str, Any], source: str) -> Sequence[Any]:
    result_list = payload.get("resultList")
    if result_list is None and payload.get("hitCount") == 0:
        return ()
    if not isinstance(result_list, Mapping):
        raise ValueError(f"{source}: response.resultList must be a JSON object")
    results = result_list.get("result", [])
    if not _is_sequence(results):
        raise ValueError(f"{source}: response.resultList.result must be an array")
    return results


def _identifiers(
    item: Mapping[str, Any], source: str, record_id: str
) -> Iterable[Identifier]:
    yield Identifier("europepmc", f"{source}:{record_id}")
    for field, namespace in (
        ("pmid", "pmid"),
        ("pmcid", "pmcid"),
        ("doi", "doi"),
    ):
        value = _text(item.get(field))
        if not value:
            continue
        if namespace == "doi":
            value = _normalize_doi(value)
            if not value:
                continue
        yield Identifier(namespace, value)


def _keyword_terms(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(
        dict.fromkeys(
            text
            for item in _sequence(value.get("keyword"))
            if (text := _text(item))
        )
    )


def _mesh_terms(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    terms: list[str] = []
    for heading in _sequence(value.get("meshHeading")):
        if not isinstance(heading, Mapping):
            continue
        descriptor = _text(heading.get("descriptorName"))
        if descriptor:
            terms.append(descriptor)
        qualifiers = heading.get("qualifierName")
        qualifier_values = _sequence(qualifiers)
        if not qualifier_values and (qualifier := _text(qualifiers)):
            qualifier_values = (qualifier,)
        terms.extend(text for item in qualifier_values if (text := _text(item)))
    return tuple(dict.fromkeys(terms))


def _links(item: Mapping[str, Any], canonical_url: str) -> Iterable[Link]:
    yield Link(canonical_url, relation="europe_pmc_record", crawl=False)
    full_texts = item.get("fullTextUrlList")
    if isinstance(full_texts, Mapping):
        for index, raw in enumerate(_sequence(full_texts.get("fullTextUrl"))):
            if not isinstance(raw, Mapping) or not (url := _safe_url(raw.get("url"))):
                continue
            availability = _text(raw.get("availabilityCode")).casefold()
            relation = "open_full_text" if availability == "oa" else "full_text"
            yield Link(
                url,
                relation=relation,
                locator=f"$.fullTextUrlList.fullTextUrl[{index}].url",
            )
    if doi := _normalize_doi(item.get("doi")):
        yield Link(
            canonicalize_url(f"https://doi.org/{quote(doi, safe='/():._-')}"),
            relation="doi",
            locator="$.doi",
        )


def _normalize_doi(value: Any) -> str:
    result = _text(value).casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if result.startswith(prefix):
            result = result[len(prefix) :]
            break
    result = result.strip().rstrip(".")
    return (
        result
        if result.startswith("10.")
        and "/" in result
        and not any(character.isspace() for character in result)
        else ""
    )


def _safe_url(value: Any) -> str:
    text = _text(value)
    if not text.casefold().startswith(("https://", "http://")):
        return ""
    try:
        canonical = canonicalize_url(text)
        parts = urlsplit(canonical)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return ""
        return canonical
    except ValueError:
        return ""


def _web_url(value: Any, source: str) -> str:
    url = _safe_url(value)
    if not url:
        raise ValueError(f"{source}: API URL must be an HTTP(S) URL")
    return url.rstrip("/")


def _unique_identifiers(values: Iterable[Identifier]) -> tuple[Identifier, ...]:
    return tuple(dict.fromkeys(values))


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    result: list[Link] = []
    seen: set[tuple[str, str]] = set()
    for link in values:
        key = (link.url, link.relation)
        if key not in seen:
            seen.add(key)
            result.append(link)
    return tuple(result)


def _optional_date(value: Any, field: str, source: str) -> date | None:
    if value is None or value == "":
        return None
    return _required_date(value, field, source)


def _required_date(value: Any, field: str, source: str) -> date:
    text = _text(value)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{source}: invalid checkpoint {field}: {value!r}") from None
    if parsed.isoformat() != text:
        raise ValueError(f"{source}: invalid checkpoint {field}: {value!r}")
    return parsed


def _state_count(state: Mapping[str, Any], field: str, source: str) -> int | None:
    if field not in state:
        return None
    value = state[field]
    result = _nonnegative_integer(value)
    if isinstance(value, bool) or result is None:
        raise ValueError(f"{source}: invalid checkpoint {field}: {value!r}")
    return result


def _state_hashes(value: Any, source: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not _is_sequence(value) or any(
        not isinstance(item, str) or len(item) != 64 for item in value
    ):
        raise ValueError(f"{source}: invalid checkpoint seen_cursor_hashes")
    return tuple(value)


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _sequence(value: Any) -> Sequence[Any]:
    if _is_sequence(value):
        return value
    return ()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _required_text(value: Any, field: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


__all__ = ["EuropePmcSourceAdapter"]
