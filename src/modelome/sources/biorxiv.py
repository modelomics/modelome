from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]
_SERVERS = frozenset({"biorxiv", "medrxiv"})
_NO_VALUE = frozenset({"", "na", "n/a", "none", "not available", "null"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _BioRxivWindowSourceAdapter:
    endpoint = ""
    format_suffix = ""

    def __init__(
        self,
        *,
        name: str,
        url: str,
        server: str,
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        initial_lookback_days: int = 7,
        overlap_days: int = 1,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name)
        self.server = _server(server)
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.initial_lookback_days = int(initial_lookback_days)
        self.overlap_days = int(overlap_days)
        if self.initial_lookback_days < 0:
            raise ValueError(f"{self.name}: initial lookback days must be nonnegative")
        if self.overlap_days < 0:
            raise ValueError(f"{self.name}: overlap days must be nonnegative")
        self.client = client or HttpClient()
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": f"biorxiv-{self.endpoint}-v1",
                "url": self.url,
                "server": self.server,
                "artifact_kind": self.artifact_kind.value,
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
        cursor = _state_cursor(state, frozen=frozen, source=self.name)

        # With no overlap, a second run on the same UTC day has no newly closed
        # date to query. Do not turn an empty interval into an upstream request.
        if window_start > window_end:
            next_state: dict[str, Any] = {"completed_at": _isoformat(now)}
            if prior_watermark is not None:
                next_state["watermark"] = prior_watermark.isoformat()
            return SourcePage(records=(), next_state=next_state, complete=True, upstream_count=0)

        retry_state = self._retry_state(
            state,
            cursor=cursor,
            window_start=window_start,
            window_end=window_end,
            now=now,
            prior_watermark=prior_watermark,
        )
        request_url = self._request_url(window_start, window_end, cursor)
        response: HttpResponse = self.client.get(
            request_url,
            headers={"Accept": "application/json"},
        )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: expected a JSON object from {response.url}")
        raw_collection = payload.get("collection")
        if not _is_sequence(raw_collection):
            raise ValueError(f"{self.name}: response.collection must be an array")
        collection = tuple(raw_collection)
        message = _response_message(payload, self.name)
        no_articles = _is_no_articles(message)

        if no_articles:
            if collection:
                raise ValueError(
                    f"{self.name}: no-articles response included {len(collection)} record(s)"
                )
            response_count = 0
            response_total = 0
        else:
            status = _text(message.get("status")).casefold()
            if status != "ok":
                raise ValueError(f"{self.name}: upstream status was {status or 'missing'!r}")
            response_count = _required_count(message.get("count"), "count", self.name)
            response_total = _required_count(message.get("total"), "total", self.name)

        issues: list[SourceIssue] = []
        declared_cursor = _optional_count(message.get("cursor"))
        if not no_articles and declared_cursor is None:
            raise ValueError(f"{self.name}: response message is missing cursor")
        if declared_cursor is not None and declared_cursor != cursor:
            issues.append(
                self._pagination_issue(
                    cursor,
                    window_start,
                    window_end,
                    f"response declared cursor {declared_cursor}, expected {cursor}",
                    response_total,
                    len(collection),
                )
            )
        if response_count != len(collection):
            issues.append(
                self._pagination_issue(
                    cursor,
                    window_start,
                    window_end,
                    f"response declared count {response_count}, received {len(collection)}",
                    response_total,
                    len(collection),
                )
            )

        prior_total = _state_count(state, "scan_total", self.name) if frozen else None
        if prior_total is not None and prior_total != response_total:
            issues.append(
                self._pagination_issue(
                    cursor,
                    window_start,
                    window_end,
                    f"window total changed from {prior_total} to {response_total}",
                    response_total,
                    len(collection),
                )
            )
            # Offset pagination is unsafe after an insertion or deletion. Replay
            # the frozen interval from zero rather than resuming a shifted offset.
            retry_state = self._retry_state(
                {},
                cursor=0,
                window_start=window_start,
                window_end=window_end,
                now=now,
                prior_watermark=prior_watermark,
            )

        records: list[SourceRecord] = []
        for index, item in enumerate(collection):
            if not isinstance(item, Mapping):
                issues.append(
                    SourceIssue(
                        source_record_id=f"{self.name}:item:{cursor + index}",
                        stage="source_normalize",
                        error="TypeError: preprint result is not a JSON object",
                        summary={"index": cursor + index, "value": repr(item)[:1000]},
                    )
                )
                continue
            try:
                records.append(self._record(item))
            except (KeyError, TypeError, ValueError) as error:
                raw = dict(item)
                issues.append(
                    SourceIssue(
                        source_record_id=self._malformed_id(item, cursor + index),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": cursor + index, "raw": raw},
                    )
                )

        next_cursor = cursor + len(collection)
        if next_cursor > response_total:
            issues.append(
                self._pagination_issue(
                    cursor,
                    window_start,
                    window_end,
                    f"page ended at offset {next_cursor}, beyond declared total {response_total}",
                    response_total,
                    len(collection),
                )
            )
        premature = next_cursor < response_total and not collection
        if premature:
            issues.append(
                self._pagination_issue(
                    cursor,
                    window_start,
                    window_end,
                    f"pagination returned no records before declared total {response_total}",
                    response_total,
                    0,
                )
            )

        complete = next_cursor >= response_total and not premature
        if complete:
            next_state = {
                "watermark": window_end.isoformat(),
                "completed_at": _isoformat(now),
            }
        elif collection:
            next_state = {
                "cursor": next_cursor,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "started_at": _text(state.get("started_at")) or _isoformat(now),
                "raw_items_seen": next_cursor,
                "scan_total": response_total,
            }
            if prior_watermark is not None:
                next_state["watermark"] = prior_watermark.isoformat()
        else:
            next_state = dict(retry_state)

        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=response_total,
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

        if "cursor" in state:
            raise ValueError(f"{self.name}: cursor checkpoint is missing its frozen window")
        if prior_watermark is None:
            lookback = max(self.initial_lookback_days, 1)
            window_start = closed_through - timedelta(days=lookback - 1)
        else:
            if prior_watermark > closed_through:
                raise ValueError(f"{self.name}: watermark is later than the last closed UTC day")
            window_start = prior_watermark + timedelta(days=1 - self.overlap_days)
        return window_start, closed_through, False

    def _request_url(self, window_start: date, window_end: date, cursor: int) -> str:
        suffix = f"/{self.format_suffix}" if self.format_suffix else ""
        return (
            f"{self.url}/{self.server}/{window_start.isoformat()}/"
            f"{window_end.isoformat()}/{cursor}{suffix}"
        )

    def _retry_state(
        self,
        state: Mapping[str, Any],
        *,
        cursor: int,
        window_start: date,
        window_end: date,
        now: datetime,
        prior_watermark: date | None,
    ) -> dict[str, Any]:
        retry_state: dict[str, Any] = {
            "cursor": cursor,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "started_at": _text(state.get("started_at")) or _isoformat(now),
            "raw_items_seen": cursor,
        }
        if total := _state_count(state, "scan_total", self.name):
            retry_state["scan_total"] = total
        if prior_watermark is not None:
            retry_state["watermark"] = prior_watermark.isoformat()
        return retry_state

    def _pagination_issue(
        self,
        cursor: int,
        window_start: date,
        window_end: date,
        message: str,
        total: int,
        received: int,
    ) -> SourceIssue:
        summary = {
            "cursor": cursor,
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

    def _record(self, item: Mapping[str, Any]) -> SourceRecord:
        raise NotImplementedError

    def _malformed_id(self, item: Mapping[str, Any], index: int) -> str:
        return f"{self.name}:malformed:{content_hash(dict(item))[:32]}:{index}"


class BioRxivSourceAdapter(_BioRxivWindowSourceAdapter):
    """Enumerate every version posted to bioRxiv or medRxiv in closed UTC days."""

    endpoint = "details"
    format_suffix = "json"

    def _record(self, item: Mapping[str, Any]) -> SourceRecord:
        _validate_record_server(item.get("server"), self.server, self.name)
        doi = _required_doi(item.get("doi"), self.name, "doi")
        version = _required_version(item.get("version"), self.name)
        source_record_id = f"{self.server}:{doi}:v{version}"
        version_identifier = Identifier(f"{self.server}:version", f"{doi}v{version}")
        preprint_url = _preprint_url(self.server, doi, version)
        doi_url = _doi_url(doi)

        identifiers = [Identifier("doi", doi), version_identifier]
        links = [
            Link(preprint_url, relation="preprint", locator="$.doi"),
            Link(doi_url, relation="doi", locator="$.doi"),
        ]
        if jats_url := _optional_web_url(item.get("jatsxml")):
            links.append(Link(jats_url, relation="full_text", locator="$.jatsxml"))
        if published_doi := _optional_doi(item.get("published")):
            links.append(
                Link(
                    _doi_url(published_doi),
                    relation="published_as",
                    locator="$.published",
                )
            )

        return SourceRecord(
            source_record_id=source_record_id,
            kind=self.artifact_kind,
            canonical_url=preprint_url,
            title=_text(item.get("title")) or doi,
            raw=dict(item),
            text=_text(item.get("abstract")),
            published_at=_text(item.get("date")) or None,
            modified_at=_text(item.get("date")) or None,
            identifiers=_unique_identifiers(identifiers),
            links=_unique_links(links),
        )

    def _malformed_id(self, item: Mapping[str, Any], index: int) -> str:
        doi = _optional_doi(item.get("doi"))
        version = _optional_count(item.get("version"))
        if doi and version is not None and version > 0:
            return f"{self.server}:{doi}:v{version}"
        return super()._malformed_id(item, index)


class BioRxivPublicationSourceAdapter(_BioRxivWindowSourceAdapter):
    """Enumerate preprint-to-journal DOI links as they are added upstream."""

    endpoint = "publications"

    def __init__(
        self,
        *,
        name: str,
        url: str,
        server: str,
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        initial_lookback_days: int = 7,
        overlap_days: int = 90,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        super().__init__(
            name=name,
            url=url,
            server=server,
            artifact_kind=artifact_kind,
            initial_lookback_days=initial_lookback_days,
            overlap_days=overlap_days,
            client=client,
            clock=clock,
        )

    def _record(self, item: Mapping[str, Any]) -> SourceRecord:
        _validate_record_server(item.get("preprint_platform"), self.server, self.name)
        preprint_doi = _required_doi(
            item.get("preprint_doi") or item.get("biorxiv_doi"),
            self.name,
            "preprint_doi/biorxiv_doi",
        )
        published_doi = _required_doi(item.get("published_doi"), self.name, "published_doi")
        source_record_id = f"{self.server}:{preprint_doi}:published:{published_doi}"
        preprint_url = _preprint_url(self.server, preprint_doi)
        published_url = _doi_url(published_doi)

        return SourceRecord(
            source_record_id=source_record_id,
            kind=self.artifact_kind,
            canonical_url=preprint_url,
            title=_text(item.get("preprint_title")) or preprint_doi,
            raw=dict(item),
            text=_text(item.get("preprint_abstract")),
            published_at=_text(item.get("preprint_date")) or None,
            modified_at=_text(item.get("published_date")) or None,
            # The journal DOI is the target of ``published_as``, not another
            # identifier for this preprint-link evidence artifact. Treating both
            # DOIs as co-identifiers would assert that two distinct works are one.
            identifiers=(Identifier("doi", preprint_doi),),
            links=(
                Link(preprint_url, relation="preprint", locator="$.preprint_doi"),
                Link(
                    published_url,
                    relation="published_as",
                    locator="$.published_doi",
                ),
            ),
        )

    def _malformed_id(self, item: Mapping[str, Any], index: int) -> str:
        preprint_doi = _optional_doi(item.get("preprint_doi") or item.get("biorxiv_doi"))
        published_doi = _optional_doi(item.get("published_doi"))
        if preprint_doi and published_doi:
            return f"{self.server}:{preprint_doi}:published:{published_doi}"
        return super()._malformed_id(item, index)


def _response_message(payload: Mapping[str, Any], source: str) -> Mapping[str, Any]:
    messages = payload.get("messages")
    if not _is_sequence(messages) or not messages:
        raise ValueError(f"{source}: response.messages must be a nonempty array")
    message = messages[0]
    if not isinstance(message, Mapping):
        raise ValueError(f"{source}: response.messages[0] must be an object")
    return message


def _is_no_articles(message: Mapping[str, Any]) -> bool:
    return _text(message.get("status")).casefold().startswith("no articles found")


def _server(value: Any) -> str:
    server = _text(value).casefold()
    if server not in _SERVERS:
        raise ValueError("bioRxiv server must be 'biorxiv' or 'medrxiv'")
    return server


def _validate_record_server(value: Any, expected: str, source: str) -> None:
    actual = _text(value).casefold()
    if actual and actual != expected:
        raise ValueError(f"{source}: record server {actual!r} does not match {expected!r}")


def _preprint_url(server: str, doi: str, version: int | None = None) -> str:
    host = "www.biorxiv.org" if server == "biorxiv" else "www.medrxiv.org"
    version_suffix = f"v{version}" if version is not None else ""
    encoded = quote(f"{doi}{version_suffix}", safe="/.:_-()")
    return canonicalize_url(f"https://{host}/content/{encoded}")


def _doi_url(doi: str) -> str:
    return canonicalize_url(f"https://doi.org/{quote(doi, safe='/.:_-()')}")


def _required_doi(value: Any, source: str, field: str) -> str:
    doi = _optional_doi(value)
    if not doi:
        raise ValueError(f"{source}: record is missing a valid {field}")
    return doi


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


def _required_version(value: Any, source: str) -> int:
    version = _optional_count(value)
    if version is None or version < 1:
        raise ValueError(f"{source}: record version must be a positive integer")
    return version


def _required_count(value: Any, field: str, source: str) -> int:
    result = _optional_count(value)
    if result is None:
        raise ValueError(f"{source}: response message has invalid {field}")
    return result


def _optional_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _state_cursor(state: Mapping[str, Any], *, frozen: bool, source: str) -> int:
    if "cursor" not in state:
        return 0
    if not frozen:
        raise ValueError(f"{source}: cursor checkpoint is missing its frozen window")
    cursor = _optional_count(state.get("cursor"))
    if cursor is None:
        raise ValueError(f"{source}: invalid cursor checkpoint")
    raw_items_seen = _state_count(state, "raw_items_seen", source)
    if raw_items_seen is not None and raw_items_seen != cursor:
        raise ValueError(f"{source}: raw_items_seen does not match cursor")
    return cursor


def _state_count(state: Mapping[str, Any], key: str, source: str) -> int | None:
    if key not in state:
        return None
    count = _optional_count(state.get(key))
    if count is None:
        raise ValueError(f"{source}: invalid {key} checkpoint")
    return count


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


def _optional_web_url(value: Any) -> str:
    text = _text(value)
    if not text or text.casefold() in _NO_VALUE:
        return ""
    canonical = canonicalize_url(text)
    parts = urlsplit(canonical)
    return canonical if parts.scheme in {"http", "https"} and parts.hostname else ""


def _web_url(value: Any, source: str) -> str:
    url = _optional_web_url(value)
    if not url:
        raise ValueError(f"{source}: API URL must be an HTTP(S) URL")
    return url.rstrip("/")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


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


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = ["BioRxivPublicationSourceAdapter", "BioRxivSourceAdapter"]
