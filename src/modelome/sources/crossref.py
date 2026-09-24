from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient, HttpFailure, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CrossrefSourceAdapter:
    """Enumerate Crossref works without venue, field, or model-name filters."""

    def __init__(
        self,
        *,
        name: str = "crossref",
        url: str = "https://api.crossref.org/works",
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        page_size: int = 1_000,
        initial_lookback_days: int = 7,
        overlap_days: int = 2,
        mailto: str | None = None,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name)
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.page_size = int(page_size)
        self.initial_lookback_days = int(initial_lookback_days)
        self.overlap_days = int(overlap_days)
        self.mailto = _text(mailto) or None
        self.client = client or HttpClient()
        self.clock = clock
        if not 1 <= self.page_size <= 1_000:
            raise ValueError("Crossref page_size must be between 1 and 1000")
        if self.initial_lookback_days < 1:
            raise ValueError("Crossref initial_lookback_days must be positive")
        if self.overlap_days < 1:
            raise ValueError("Crossref overlap_days must be positive")
        self.checkpoint_signature = content_hash(
            {
                "adapter": "crossref-v1",
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
        window_start, window_end = self._window(
            state,
            prior_watermark=prior_watermark,
            closed_through=closed_through,
        )
        cursor = _text(state.get("cursor")) or "*"
        resuming = cursor != "*"
        raw_items_seen = _state_count(state, "raw_items_seen", self.name) if resuming else 0
        scan_total = _state_count(state, "scan_total", self.name) if resuming else None
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

        params: dict[str, str | int] = {
            "cursor": cursor,
            "rows": self.page_size,
            "filter": (
                f"from-index-date:{window_start.isoformat()},"
                f"until-index-date:{window_end.isoformat()}"
            ),
        }
        if self.mailto:
            params["mailto"] = self.mailto
        try:
            response: HttpResponse = self.client.get(
                self.url,
                params=params,
                headers={"Accept": "application/json"},
            )
        except HttpFailure as error:
            # Crossref cursors expire after a few minutes and the API reports an
            # expired or invalid cursor as HTTP 404. Replay the immutable date
            # window from its first page instead of retaining a dead checkpoint.
            if not resuming or "HTTP Error 404" not in str(error):
                raise
            restart_state = self._scan_state(
                cursor="*",
                window_start=window_start,
                window_end=window_end,
                raw_items_seen=0,
                scan_total=None,
                seen_cursor_hashes=(),
                started_at=started_at,
                prior_watermark=prior_watermark,
            )
            issue = self._pagination_issue(
                retry_state,
                "cursor expired or was rejected; restarting the frozen window",
                raw_items_seen=raw_items_seen,
                known_total=scan_total,
            )
            return SourcePage(
                records=(),
                next_state=restart_state,
                complete=False,
                upstream_count=scan_total,
                issues=(issue,),
                retry_state=restart_state,
            )
        payload = response.json()
        message = _response_message(payload, self.name)
        raw_items = message.get("items")
        if not _is_sequence(raw_items):
            raise ValueError(f"{self.name}: response.message.items must be an array")

        issues: list[SourceIssue] = []
        records: list[SourceRecord] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, Mapping):
                issues.append(
                    self._item_issue(
                        index,
                        item,
                        "TypeError: work item is not a JSON object",
                    )
                )
                continue
            try:
                records.append(self._record(item))
            except (TypeError, ValueError) as error:
                issues.append(
                    self._item_issue(
                        index,
                        item,
                        f"{type(error).__name__}: {error}",
                    )
                )

        raw_items_seen += len(raw_items)
        response_total = _nonnegative_integer(message.get("total-results"))
        if response_total is None:
            raise ValueError(f"{self.name}: response is missing a valid total-results count")
        if scan_total is None:
            scan_total = response_total
        elif response_total != scan_total:
            issues.append(
                self._pagination_issue(
                    retry_state,
                    f"total-results changed from {scan_total} to {response_total}",
                    raw_items_seen=raw_items_seen,
                    known_total=scan_total,
                )
            )

        reached_total = raw_items_seen == scan_total
        next_cursor = _text(message.get("next-cursor"))
        next_hash = content_hash(next_cursor) if next_cursor else ""
        if not reached_total and next_cursor and (
            next_cursor == cursor or next_hash in seen_cursor_hashes
        ):
            issues.append(
                self._pagination_issue(
                    retry_state,
                    "cursor cycle detected",
                    raw_items_seen=raw_items_seen,
                    known_total=scan_total,
                )
            )
            next_cursor = ""
        if raw_items_seen > scan_total:
            issues.append(
                self._pagination_issue(
                    retry_state,
                    f"received {raw_items_seen} items for declared total {scan_total}",
                    raw_items_seen=raw_items_seen,
                    known_total=scan_total,
                )
            )

        short_page = len(raw_items) < self.page_size
        if not issues and not reached_total and (short_page or not next_cursor):
            issues.append(
                self._pagination_issue(
                    retry_state,
                    f"pagination ended after {raw_items_seen} item(s), before "
                    f"the declared total of {scan_total}",
                    raw_items_seen=raw_items_seen,
                    known_total=scan_total,
                )
            )

        pagination_failed = any(issue.stage == "source_pagination" for issue in issues)
        if pagination_failed:
            # A changed result set or broken cursor can shift every later page.
            # Restarting the frozen interval is slower but cannot silently skip.
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
        *,
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
        if "cursor" in state:
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
            "cursor": cursor,
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

    def _record(self, work: Mapping[str, Any]) -> SourceRecord:
        doi = _doi(work.get("DOI"), self.name)
        doi_url = canonicalize_url(f"https://doi.org/{quote(doi, safe='/():._-')}")
        title = _first_text(work.get("title")) or doi
        subtitle = _first_text(work.get("subtitle"))
        abstract = _markup_text(work.get("abstract"))
        # Crossref subject headings often carry the useful model/domain terms
        # omitted from publisher abstracts. Keep them in searchable text while
        # retaining their original, field-addressable values in raw evidence.
        subjects = tuple(
            dict.fromkeys(
                text
                for value in _sequence(work.get("subject"))
                if (text := _text(value))
            )
        )
        text_parts = [value for value in (title, subtitle, abstract) if value]
        if subjects:
            text_parts.append(f"Subjects: {'; '.join(subjects)}")
        text = "\n\n".join(text_parts)
        links = list(_work_links(work, doi_url))
        if not any(link.url == doi_url for link in links):
            links.insert(0, Link(doi_url, relation="doi", locator="$.DOI"))
        return SourceRecord(
            source_record_id=doi,
            kind=self.artifact_kind,
            canonical_url=doi_url,
            title=title,
            raw=dict(work),
            text=text,
            published_at=_publication_date(work),
            modified_at=_indexed_timestamp(work),
            identifiers=(Identifier("doi", doi),),
            links=_unique_links(links),
        )

    def _item_issue(self, index: int, value: Any, error: str) -> SourceIssue:
        raw = dict(value) if isinstance(value, Mapping) else repr(value)[:1_000]
        doi = _text(value.get("DOI")) if isinstance(value, Mapping) else ""
        return SourceIssue(
            source_record_id=doi or f"{self.name}:malformed:{content_hash(raw)[:32]}",
            stage="source_normalize",
            error=error,
            summary={"index": index, "raw": raw},
        )

    def _pagination_issue(
        self,
        retry_state: Mapping[str, Any],
        error: str,
        *,
        raw_items_seen: int,
        known_total: int | None,
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


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if value := data.strip():
            self.parts.append(value)


def _response_message(payload: Any, source: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: response must be a JSON object")
    if payload.get("status") != "ok" or payload.get("message-type") != "work-list":
        raise ValueError(f"{source}: response is not a successful work-list")
    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise ValueError(f"{source}: response.message must be a JSON object")
    return message


def _work_links(work: Mapping[str, Any], doi_url: str) -> Iterable[Link]:
    yield Link(doi_url, relation="doi", locator="$.DOI")
    if url := _safe_url(work.get("URL")):
        yield Link(url, relation="landing_page", locator="$.URL")
    resource = work.get("resource")
    if (
        isinstance(resource, Mapping)
        and isinstance(resource.get("primary"), Mapping)
        and (url := _safe_url(resource["primary"].get("URL")))
    ):
        yield Link(url, relation="landing_page", locator="$.resource.primary.URL")
    for index, raw in enumerate(_sequence(work.get("link"))):
        if not isinstance(raw, Mapping) or not (url := _safe_url(raw.get("URL"))):
            continue
        # MIME parameters are valid (for example, application/pdf; version=1.7)
        # and do not change the representation's media type.
        media_type = _text(raw.get("content-type")).casefold().split(";", 1)[0].strip()
        intended = _text(raw.get("intended-application")).casefold()
        relation = (
            "full_text"
            if media_type == "application/pdf" or intended == "text-mining"
            else "related_content"
        )
        yield Link(url, relation=relation, locator=f"$.link[{index}].URL")
    relations = work.get("relation")
    if isinstance(relations, Mapping):
        for predicate, values in relations.items():
            for index, raw in enumerate(_sequence(values)):
                if not isinstance(raw, Mapping):
                    continue
                identifier = _text(raw.get("id"))
                id_type = _text(raw.get("id-type")).casefold()
                if id_type == "doi" and identifier:
                    try:
                        relation_doi = _doi(identifier, "Crossref relation")
                    except ValueError:
                        # One malformed optional relationship must not discard an
                        # otherwise valid primary DOI record. The raw value remains
                        # available in SourceRecord.raw for later repair.
                        continue
                    target = canonicalize_url(
                        f"https://doi.org/{quote(relation_doi, safe='/():._-')}"
                    )
                    yield Link(
                        target,
                        relation=_text(predicate) or "related_work",
                        locator=f"$.relation.{predicate}[{index}]",
                    )
                elif id_type in {"uri", "url", "purl"}:
                    target = _safe_url(identifier)
                    if target:
                        yield Link(
                            target,
                            relation=_text(predicate) or "related_work",
                            locator=f"$.relation.{predicate}[{index}]",
                        )
    # Crossref's structured reference list is a useful citation graph for
    # publisher-hosted papers whose full text is not openly reusable.  A cited
    # DOI is evidence about the bibliographic relationship, not a request to
    # crawl every cited paper through a publisher landing page.  The complete
    # Crossref/OpenAlex acquisition paths remain responsible for discovery.
    for index, raw in enumerate(_sequence(work.get("reference"))):
        if not isinstance(raw, Mapping):
            continue
        value = raw.get("DOI") if "DOI" in raw else raw.get("doi")
        try:
            reference_doi = _doi(value, "Crossref reference")
        except ValueError:
            # References often omit a DOI or contain a malformed optional DOI.
            # Preserve their raw Crossref evidence without losing the work.
            continue
        target = canonicalize_url(
            f"https://doi.org/{quote(reference_doi, safe='/():._-')}"
        )
        field = "DOI" if "DOI" in raw else "doi"
        yield Link(
            target,
            relation="cites",
            locator=f"$.reference[{index}].{field}",
            crawl=False,
        )


def _publication_date(work: Mapping[str, Any]) -> str | None:
    for field in (
        "published-online",
        "published-print",
        "published",
        "issued",
        "created",
    ):
        value = work.get(field)
        if not isinstance(value, Mapping):
            continue
        if date_value := _date_parts(value.get("date-parts")):
            return date_value
        if timestamp := _text(value.get("date-time")):
            return timestamp
    return None


def _indexed_timestamp(work: Mapping[str, Any]) -> str | None:
    for field in ("indexed", "deposited"):
        value = work.get(field)
        if isinstance(value, Mapping) and (timestamp := _text(value.get("date-time"))):
            return timestamp
    return None


def _date_parts(value: Any) -> str | None:
    outer = _sequence(value)
    parts = _sequence(outer[0]) if outer else ()
    if not parts or len(parts) > 3:
        return None
    integers = [_nonnegative_integer(item) for item in parts]
    if any(item is None for item in integers):
        return None
    year = int(integers[0])
    if year < 1:
        return None
    if len(integers) == 1:
        return f"{year:04d}"
    month = int(integers[1])
    if not 1 <= month <= 12:
        return None
    if len(integers) == 2:
        return f"{year:04d}-{month:02d}"
    day = int(integers[2])
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _markup_text(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parser = _TextParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return text
    return " ".join(parser.parts)


def _doi(value: Any, source: str) -> str:
    result = _text(value).casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if result.startswith(prefix):
            result = result[len(prefix) :]
            break
    result = result.strip().rstrip(".")
    if not result.startswith("10.") or "/" not in result or any(char.isspace() for char in result):
        raise ValueError(f"{source}: work is missing a valid DOI")
    return result


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


def _first_text(value: Any) -> str:
    return next((_text(item) for item in _sequence(value) if _text(item)), "")


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
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
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


__all__ = ["CrossrefSourceAdapter"]
