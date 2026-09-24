from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]
_MAX_CURSOR_LENGTH = 8_192
_MAX_URL_LENGTH = 32_768
_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DataCiteSourceAdapter:
    """Enumerate every public Findable DataCite DOI without subject filters."""

    def __init__(
        self,
        *,
        name: str = "datacite",
        url: str = "https://api.datacite.org/dois",
        artifact_kind: str | ArtifactKind | None = None,
        page_size: int = 1_000,
        initial_lookback_days: int = 7,
        overlap_days: int = 2,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name)
        self.artifact_kind = ArtifactKind(artifact_kind) if artifact_kind is not None else None
        self.page_size = int(page_size)
        self.initial_lookback_days = int(initial_lookback_days)
        self.overlap_days = int(overlap_days)
        self.client = client or HttpClient()
        self.clock = clock
        if not 1 <= self.page_size <= 1_000:
            raise ValueError("DataCite page_size must be between 1 and 1000")
        if self.initial_lookback_days < 1:
            raise ValueError("DataCite initial_lookback_days must be positive")
        if self.overlap_days < 1:
            raise ValueError("DataCite overlap_days must be positive")
        self.checkpoint_signature = content_hash(
            {
                "adapter": "datacite-v1",
                "url": self.url,
                "artifact_kind": (
                    self.artifact_kind.value if self.artifact_kind is not None else "resource_type"
                ),
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
        cursor = _state_cursor(state, self.name)
        resuming = cursor != "1"
        seen_cursor_hashes = _state_hashes(state.get("seen_cursor_hashes"), self.name)
        if resuming:
            raw_items_seen = _state_count(state, "raw_items_seen", self.name)
            scan_total = _state_count(state, "scan_total", self.name)
            if raw_items_seen is None or scan_total is None:
                raise ValueError(
                    f"{self.name}: cursor checkpoint requires raw_items_seen and scan_total"
                )
            if raw_items_seen > scan_total:
                raise ValueError(f"{self.name}: checkpoint raw_items_seen exceeds scan_total")
            if content_hash(cursor) not in seen_cursor_hashes:
                raise ValueError(
                    f"{self.name}: checkpoint cursor is absent from seen_cursor_hashes"
                )
        else:
            raw_items_seen = _state_count(state, "raw_items_seen", self.name)
            scan_total = _state_count(state, "scan_total", self.name)
            if raw_items_seen not in {None, 0} or scan_total is not None:
                raise ValueError(f"{self.name}: first-page checkpoint has pagination counts")
            if seen_cursor_hashes:
                raise ValueError(f"{self.name}: first-page checkpoint has cursor history")
            raw_items_seen = 0
            scan_total = None

        started_at = _state_started_at(state, now, self.name)
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
        query = f"updated:[{window_start.isoformat()} TO {window_end.isoformat()}]"
        params: dict[str, str | int] = {
            "query": query,
            "page[cursor]": cursor,
            "page[size]": self.page_size,
            "affiliation": "true",
            "publisher": "true",
        }
        response: HttpResponse = self.client.get(
            self.url,
            params=params,
            headers={"Accept": "application/vnd.api+json"},
        )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: response must be a JSON object")
        raw_items = payload.get("data")
        if not _is_sequence(raw_items):
            raise ValueError(f"{self.name}: response.data must be an array")
        meta = payload.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError(f"{self.name}: response.meta must be a JSON object")
        response_total = _response_count(meta.get("total"))
        if response_total is None:
            raise ValueError(f"{self.name}: response is missing a valid meta.total")
        links = payload.get("links")
        if not isinstance(links, Mapping):
            raise ValueError(f"{self.name}: response.links must be a JSON object")

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, Mapping):
                issues.append(self._item_issue(index, item, "resource is not a JSON object"))
                continue
            try:
                records.append(self._record(item))
            except (TypeError, ValueError) as error:
                issues.append(self._item_issue(index, item, f"{type(error).__name__}: {error}"))
        raw_items_seen += len(raw_items)

        if scan_total is None:
            scan_total = response_total
        elif scan_total != response_total:
            issues.append(
                self._pagination_issue(
                    retry_state,
                    f"meta.total changed from {scan_total} to {response_total}",
                    raw_items_seen,
                    scan_total,
                )
            )

        reached_total = raw_items_seen == scan_total
        next_cursor = ""
        if not reached_total:
            try:
                next_cursor = _next_cursor(
                    links.get("next"),
                    endpoint=self.url,
                    expected_query=query,
                    page_size=self.page_size,
                )
            except ValueError as error:
                issues.append(
                    self._pagination_issue(
                        retry_state,
                        f"invalid links.next: {error}",
                        raw_items_seen,
                        scan_total,
                    )
                )

        next_hash = content_hash(next_cursor) if next_cursor else ""
        if (
            not reached_total
            and next_cursor
            and (next_cursor == cursor or next_hash in seen_cursor_hashes)
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
        if len(raw_items) > self.page_size:
            issues.append(
                self._pagination_issue(
                    retry_state,
                    f"page contained {len(raw_items)} resources, above page_size {self.page_size}",
                    raw_items_seen,
                    scan_total,
                )
            )
        if raw_items_seen > scan_total:
            issues.append(
                self._pagination_issue(
                    retry_state,
                    f"received {raw_items_seen} resources for declared total {scan_total}",
                    raw_items_seen,
                    scan_total,
                )
            )

        pagination_failed = any(issue.stage == "source_pagination" for issue in issues)
        if (
            not pagination_failed
            and not reached_total
            and (len(raw_items) < self.page_size or not next_cursor)
        ):
            issues.append(
                self._pagination_issue(
                    retry_state,
                    f"pagination ended after {raw_items_seen} resource(s), before "
                    f"the declared total of {scan_total}",
                    raw_items_seen,
                    scan_total,
                )
            )
            pagination_failed = True

        if pagination_failed:
            # Cursor drift can move every later page. Replaying the same closed
            # interval from its first page is slower, but it cannot silently skip.
            retry_state = self._scan_state(
                cursor="1",
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

    def _record(self, item: Mapping[str, Any]) -> SourceRecord:
        if _required_text(item.get("type"), "DataCite resource type") != "dois":
            raise ValueError("DataCite resource type must be 'dois'")
        attributes = item.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError("DataCite resource attributes must be a JSON object")

        top_level_doi = _required_doi(item.get("id"), "DataCite resource id")
        attribute_doi = _required_doi(attributes.get("doi"), "DataCite attributes.doi")
        if top_level_doi != attribute_doi:
            raise ValueError("DataCite resource id conflicts with attributes.doi")
        doi = top_level_doi
        if _text(attributes.get("state")).casefold() != "findable":
            raise ValueError("DataCite public resource is not in Findable state")
        if attributes.get("isActive") is False:
            raise ValueError("DataCite Findable resource is inactive")

        doi_url = canonicalize_url(f"https://doi.org/{quote(doi, safe='/():._-')}")
        primary_url = _safe_url(attributes.get("url"))
        canonical_url = primary_url or doi_url
        titles = _titles(attributes.get("titles"))
        descriptions = _descriptions(attributes.get("descriptions"))
        subjects = _subjects(attributes.get("subjects"))
        creators = _creators(attributes.get("creators"))
        resource_types = _resource_types(attributes.get("types"))
        title = titles[0] if titles else doi
        text_parts = list(dict.fromkeys((*titles, *descriptions)))
        if subjects:
            text_parts.append(f"Subjects: {'; '.join(subjects)}")
        if creators:
            text_parts.append(f"Creators: {'; '.join(creators)}")
        if resource_types:
            text_parts.append(f"DataCite resource type: {'; '.join(resource_types)}")

        return SourceRecord(
            source_record_id=doi,
            kind=self.artifact_kind or _artifact_kind(attributes),
            canonical_url=canonical_url,
            title=title,
            raw=dict(item),
            text="\n\n".join(text_parts),
            published_at=_published_date(attributes),
            modified_at=_scalar_text(attributes.get("updated")) or None,
            identifiers=tuple(
                dict.fromkeys(
                    (Identifier("doi", doi), *_identical_related_dois(attributes))
                )
            ),
            links=_unique_links(_links(attributes, doi_url, primary_url)),
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
                f"{self.name}:pagination:{content_hash({**retry_state, 'error': error})[:32]}"
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


def _links(
    attributes: Mapping[str, Any],
    doi_url: str,
    primary_url: str,
) -> Iterable[Link]:
    yield Link(doi_url, relation="doi", locator="$.attributes.doi")
    if primary_url:
        yield Link(primary_url, relation="landing_page", locator="$.attributes.url")

    content_urls = attributes.get("contentUrl")
    values = (content_urls,) if isinstance(content_urls, str) else _sequence(content_urls)
    for index, raw_url in enumerate(values):
        if url := _safe_url(raw_url):
            yield Link(
                url,
                relation="content",
                locator=f"$.attributes.contentUrl[{index}]",
            )

    for index, raw in enumerate(_sequence(attributes.get("relatedIdentifiers"))):
        if not isinstance(raw, Mapping):
            continue
        url = _related_url(
            raw.get("relatedIdentifier"),
            raw.get("relatedIdentifierType"),
        )
        if not url:
            continue
        yield Link(
            url,
            relation=_relation_name(raw.get("relationType")),
            locator=f"$.attributes.relatedIdentifiers[{index}].relatedIdentifier",
        )


def _identical_related_dois(attributes: Mapping[str, Any]) -> tuple[Identifier, ...]:
    """Expose only DataCite DOIs that the depositor marks as identical."""

    identifiers: list[Identifier] = []
    for related in _sequence(attributes.get("relatedIdentifiers")):
        if not isinstance(related, Mapping):
            continue
        if _relation_name(related.get("relationType")) != "is_identical_to":
            continue
        if _text(related.get("relatedIdentifierType")).casefold() != "doi":
            continue
        if doi := _optional_doi(related.get("relatedIdentifier")):
            identifiers.append(Identifier("doi", doi))
    return tuple(dict.fromkeys(identifiers))


def _next_cursor(
    value: Any,
    *,
    endpoint: str,
    expected_query: str,
    page_size: int,
) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or value != value.strip():
        raise ValueError("URL must be an unpadded string")
    text = _required_text(value, "links.next")
    if len(text) > _MAX_URL_LENGTH or any(character.isspace() for character in text):
        raise ValueError("URL is too long or contains control characters")
    try:
        parts = urlsplit(text)
        base = urlsplit(endpoint)
        _ = parts.port
    except ValueError as error:
        raise ValueError("URL is malformed") from error
    if (
        parts.scheme.casefold() not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise ValueError("URL must be an absolute HTTP(S) URL without credentials")
    if _origin(parts) != _origin(base) or parts.path.rstrip("/") != base.path.rstrip("/"):
        raise ValueError("URL changed the configured API endpoint")
    if _PERCENT_ESCAPE.search(parts.query):
        raise ValueError("URL contains invalid percent encoding")
    try:
        pairs = parse_qsl(
            parts.query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=20,
        )
    except ValueError as error:
        raise ValueError("URL query is malformed") from error
    values: dict[str, list[str]] = {}
    for key, item in pairs:
        values.setdefault(key, []).append(item)
    allowed = {
        "query",
        "page[cursor]",
        "page[size]",
        "affiliation",
        "publisher",
    }
    if unexpected := sorted(set(values) - allowed):
        raise ValueError(f"URL added unexpected parameter {unexpected[0]!r}")
    if any(len(items) != 1 for items in values.values()):
        raise ValueError("URL contains duplicate parameters")
    if values.get("query") != [expected_query]:
        raise ValueError("URL changed or omitted the frozen update query")
    if values.get("page[size]") != [str(page_size)]:
        raise ValueError("URL changed or omitted page[size]")
    if values.get("affiliation") != ["true"] or values.get("publisher") != ["true"]:
        raise ValueError("URL changed or omitted metadata detail parameters")
    cursor_values = values.get("page[cursor]")
    if cursor_values is None:
        raise ValueError("URL omitted page[cursor]")
    return _validate_cursor(cursor_values[0], "links.next page[cursor]")


def _titles(value: Any) -> tuple[str, ...]:
    result = []
    for item in _sequence(value):
        if isinstance(item, Mapping) and (title := _plain_text(item.get("title"))):
            result.append(title)
    return tuple(dict.fromkeys(result))


def _descriptions(value: Any) -> tuple[str, ...]:
    result = []
    for item in _sequence(value):
        if isinstance(item, Mapping) and (description := _plain_text(item.get("description"))):
            result.append(description)
    return tuple(dict.fromkeys(result))


def _subjects(value: Any) -> tuple[str, ...]:
    """Expose DataCite subject headings as discovery text, preserving raw data."""
    result = []
    for item in _sequence(value):
        if isinstance(item, Mapping) and (subject := _plain_text(item.get("subject"))):
            result.append(subject)
    return tuple(dict.fromkeys(result))


def _creators(value: Any) -> tuple[str, ...]:
    result = []
    for item in _sequence(value):
        if not isinstance(item, Mapping):
            continue
        name = _plain_text(item.get("name"))
        if not name:
            name = " ".join(
                part
                for part in (
                    _plain_text(item.get("givenName")),
                    _plain_text(item.get("familyName")),
                )
                if part
            )
        if name:
            result.append(name)
    return tuple(dict.fromkeys(result))


def _resource_types(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(
        dict.fromkeys(
            text
            for key in ("resourceTypeGeneral", "resourceType")
            if (text := _plain_text(value.get(key)))
        )
    )


def _artifact_kind(attributes: Mapping[str, Any]) -> ArtifactKind:
    """Map DataCite's controlled resource vocabulary without a subject allowlist."""

    types = attributes.get("types")
    if not isinstance(types, Mapping):
        return ArtifactKind.OTHER
    resource_type = _plain_text(types.get("resourceTypeGeneral")).casefold()
    if resource_type in {"computationalnotebook", "software", "workflow"}:
        return ArtifactKind.CODE_REPOSITORY
    if resource_type == "model":
        return ArtifactKind.MODEL_CARD
    if resource_type in {
        "bookchapter",
        "conferencepaper",
        "datapaper",
        "dissertation",
        "journalarticle",
        "peerreview",
        "preprint",
        "report",
        "text",
    }:
        return ArtifactKind.PAPER
    return ArtifactKind.OTHER


def _published_date(attributes: Mapping[str, Any]) -> str | None:
    if published := _scalar_text(attributes.get("published")):
        return published
    priorities = ("issued", "available", "created", "submitted")
    dates = [item for item in _sequence(attributes.get("dates")) if isinstance(item, Mapping)]
    for wanted in priorities:
        for item in dates:
            if _text(item.get("dateType")).casefold() == wanted and (
                value := _scalar_text(item.get("date"))
            ):
                return value
    return _scalar_text(attributes.get("publicationYear")) or None


def _related_url(identifier: Any, identifier_type: Any) -> str:
    value = _text(identifier)
    category = _text(identifier_type).casefold()
    if not value or len(value) > 2_048 or any(character.isspace() for character in value):
        return ""
    if category == "doi":
        doi = _optional_doi(value)
        return canonicalize_url(f"https://doi.org/{quote(doi, safe='/():._-')}") if doi else ""
    if category in {"url", "purl", "w3id"}:
        return _safe_url(value)
    if category == "pmid" and value.isascii() and value.isdigit():
        return canonicalize_url(f"https://pubmed.ncbi.nlm.nih.gov/{value}/")
    if category == "arxiv":
        lowered = value.casefold()
        for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv:"):
            if lowered.startswith(prefix):
                value = value[len(prefix) :]
                break
        if not value or any(character.isspace() for character in value):
            return ""
        return canonicalize_url(f"https://arxiv.org/abs/{quote(value, safe='._-/')}")
    if category == "handle":
        lowered = value.casefold()
        for prefix in ("https://hdl.handle.net/", "http://hdl.handle.net/", "hdl:"):
            if lowered.startswith(prefix):
                value = value[len(prefix) :]
                break
        if not value:
            return ""
        return canonicalize_url(f"https://hdl.handle.net/{quote(value, safe='._-/')}")
    if category == "bibcode":
        return canonicalize_url(
            f"https://ui.adsabs.harvard.edu/abs/{quote(value, safe='._-')}/abstract"
        )
    if category in {"ark", "urn"}:
        return canonicalize_url(f"https://n2t.net/{quote(value, safe=':._-/')}")
    return ""


def _relation_name(value: Any) -> str:
    text = _text(value)[:256]
    if not text:
        return "related_identifier"
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").casefold()
    return text[:100] or "related_identifier"


def _optional_doi(value: Any) -> str:
    result = _text(value).casefold()
    if result.startswith(("https://", "http://")):
        try:
            parts = urlsplit(result)
            if (
                parts.hostname not in {"doi.org", "dx.doi.org"}
                or parts.username is not None
                or parts.password is not None
                or parts.query
                or parts.fragment
            ):
                return ""
            result = parts.path.lstrip("/")
        except ValueError:
            return ""
    elif result.startswith("doi:"):
        result = result[4:]
    result = result.strip().rstrip(".")
    if re.fullmatch(r"10\.\d{4,9}/[^\s]+", result) is None or any(
        character.isspace() for character in result
    ):
        return ""
    return result


def _required_doi(value: Any, field: str) -> str:
    doi = _optional_doi(value)
    if not doi:
        raise ValueError(f"{field} is missing a valid DOI")
    return doi


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    result: list[Link] = []
    seen: set[tuple[str, str]] = set()
    for link in values:
        key = (link.url, link.relation)
        if key not in seen:
            seen.add(key)
            result.append(link)
    return tuple(result)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if text := data.strip():
            self.parts.append(text)


def _plain_text(value: Any) -> str:
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


def _safe_url(value: Any) -> str:
    text = _text(value)
    if not text or len(text) > _MAX_URL_LENGTH or any(character.isspace() for character in text):
        return ""
    try:
        original = urlsplit(text)
        _ = original.port
        if (
            original.scheme.casefold() not in {"http", "https"}
            or not original.hostname
            or original.username is not None
            or original.password is not None
        ):
            return ""
        canonical = canonicalize_url(text)
        parts = urlsplit(canonical)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return ""
        return canonical
    except ValueError:
        return ""


def _web_url(value: Any, source: str) -> str:
    raw = _text(value)
    if not raw or raw != value:
        raise ValueError(f"{source}: API URL must be an HTTP(S) URL")
    try:
        raw_parts = urlsplit(raw)
    except ValueError:
        raise ValueError(f"{source}: API URL must be an HTTP(S) URL") from None
    if raw_parts.query or raw_parts.fragment:
        raise ValueError(f"{source}: API URL must not contain a query or fragment")
    url = _safe_url(value)
    if not url:
        raise ValueError(f"{source}: API URL must be an HTTP(S) URL")
    return url.rstrip("/")


def _origin(parts: Any) -> tuple[str, str, int | None]:
    scheme = parts.scheme.casefold()
    port = parts.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (parts.hostname or "").casefold(), port


def _state_cursor(state: Mapping[str, Any], source: str) -> str:
    if "cursor" not in state:
        return "1"
    return _validate_cursor(state["cursor"], f"{source} checkpoint cursor")


def _validate_cursor(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    cursor = value.strip()
    if (
        not cursor
        or cursor != value
        or len(cursor) > _MAX_CURSOR_LENGTH
        or any(character.isspace() or ord(character) < 32 for character in cursor)
    ):
        raise ValueError(f"{field} is malformed")
    return cursor


def _state_started_at(state: Mapping[str, Any], now: datetime, source: str) -> str:
    if "started_at" not in state:
        return _isoformat(now)
    value = _text(state["started_at"])
    if not value or len(value) > 128:
        raise ValueError(f"{source}: invalid checkpoint started_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{source}: invalid checkpoint started_at") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{source}: invalid checkpoint started_at")
    return value


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
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}: invalid checkpoint {field}: {value!r}")
    return value


def _state_hashes(value: Any, source: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not _is_sequence(value) or any(
        not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in value
    ):
        raise ValueError(f"{source}: invalid checkpoint seen_cursor_hashes")
    return tuple(dict.fromkeys(value))


def _response_count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _sequence(value: Any) -> Sequence[Any]:
    return value if _is_sequence(value) else ()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _required_text(value: Any, field: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _scalar_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return ""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


__all__ = ["DataCiteSourceAdapter"]
