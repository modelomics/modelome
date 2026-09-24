from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash, identifier_from_url

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OpenAlexSourceAdapter:
    """Read an all-domain, date-bounded stream of OpenAlex works."""

    def __init__(
        self,
        *,
        name: str = "openalex",
        artifact_source: str | None = None,
        url: str = "https://api.openalex.org/works",
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        page_size: int = 100,
        initial_lookback_days: int = 7,
        overlap_days: int = 2,
        filter: str = "",
        corpus: str = "all",
        sync_mode: str = "published",
        api_key: str | None = None,
        mailto: str | None = None,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = name
        self.artifact_source = (artifact_source or name).strip()
        if not self.artifact_source:
            raise ValueError("OpenAlex artifact_source must not be empty")
        self.url = url
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.page_size = int(page_size)
        self.initial_lookback_days = int(initial_lookback_days)
        self.overlap_days = int(overlap_days)
        self.filter = filter.strip()
        self.corpus = corpus.strip() or "all"
        self.sync_mode = sync_mode.strip().casefold()
        if self.sync_mode not in {"published", "updated"}:
            raise ValueError("OpenAlex sync_mode must be 'published' or 'updated'")
        if self.sync_mode == "updated" and not api_key:
            raise ValueError("OpenAlex sync_mode='updated' requires an API key")
        self.api_key = api_key
        self.mailto = mailto
        self.client = client or HttpClient()
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openalex-v1",
                "artifact_source": self.artifact_source,
                "url": self.url,
                "artifact_kind": self.artifact_kind.value,
                "page_size": self.page_size,
                "initial_lookback_days": self.initial_lookback_days,
                "overlap_days": self.overlap_days,
                "filter": self.filter,
                "corpus": self.corpus,
                "sync_mode": self.sync_mode,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        cursor = _text(state.get("cursor")) or "*"
        resuming_scan = cursor != "*"
        raw_items_seen = _state_count(state, "raw_items_seen") if resuming_scan else 0
        scan_total = _state_count(state, "scan_total") if resuming_scan else None
        count_is_complete = not resuming_scan or (
            "raw_items_seen" in state and state.get("raw_count_incomplete") is not True
        )
        prior_watermark = _parse_timestamp(state.get("watermark"))
        if _text(state.get("window_start")) and _text(state.get("window_end")):
            window_start = _required_timestamp(state["window_start"], "window_start")
            window_end = _required_timestamp(state["window_end"], "window_end")
            scan_high = _parse_timestamp(state.get("scan_high_watermark"))
            started_at = _text(state.get("started_at")) or _isoformat(self.clock())
        else:
            window_end = _as_utc(self.clock())
            window_start = (
                prior_watermark - timedelta(days=self.overlap_days)
                if prior_watermark is not None
                else window_end - timedelta(days=self.initial_lookback_days)
            )
            scan_high = None
            started_at = _isoformat(window_end)

        retry_state = dict(state)
        retry_state.pop("completed_at", None)
        retry_state.update(
            {
                "cursor": cursor,
                "window_start": _isoformat(window_start),
                "window_end": _isoformat(window_end),
                "started_at": started_at,
                "raw_items_seen": raw_items_seen or 0,
            }
        )
        if scan_total is not None:
            retry_state["scan_total"] = scan_total
        if not count_is_complete:
            retry_state["raw_count_incomplete"] = True
        else:
            retry_state.pop("raw_count_incomplete", None)

        date_field = "updated_date" if self.sync_mode == "updated" else "publication_date"
        filters = [
            f"from_{date_field}:{window_start.date().isoformat()}",
            f"to_{date_field}:{window_end.date().isoformat()}",
        ]
        if self.filter:
            filters.insert(0, self.filter.strip(","))
        params: dict[str, str | int] = {
            "cursor": cursor,
            "per_page": self.page_size,
            "corpus": self.corpus,
            "filter": ",".join(filters),
            "sort": f"{date_field}:asc",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto

        response: HttpResponse = self.client.get(
            self.url,
            params=params,
            headers={"Accept": "application/json"},
        )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: expected a JSON object from {response.url}")
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, Sequence) or isinstance(
            raw_results, (str, bytes, bytearray)
        ):
            raise ValueError(f"{self.name}: response.results must be an array")
        raw_items_seen = (raw_items_seen or 0) + len(raw_results)

        records = []
        issues: list[SourceIssue] = []
        for index, item in enumerate(raw_results):
            if not isinstance(item, Mapping):
                issues.append(
                    SourceIssue(
                        source_record_id=f"{self.name}:item:{index}",
                        stage="source_normalize",
                        error="TypeError: work result is not a JSON object",
                        summary={"index": index, "value": repr(item)[:1000]},
                    )
                )
                continue
            try:
                records.append(self._record(item))
            except (KeyError, TypeError, ValueError) as error:
                raw = dict(item)
                openalex_id = _text(item.get("id")).rstrip("/").rsplit("/", maxsplit=1)[-1]
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            openalex_id
                            or f"{self.name}:malformed:{content_hash(raw)[:32]}"
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index, "raw": raw},
                    )
                )
            scan_value = (
                item.get("updated_date")
                if self.sync_mode == "updated"
                else item.get("publication_date")
            )
            modified = _parse_timestamp(scan_value)
            if modified is not None and (scan_high is None or modified > scan_high):
                scan_high = modified

        meta = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else {}
        response_total = _integer(meta.get("count"))
        if response_total is not None and response_total < 0:
            response_total = None
        if response_total is not None and count_is_complete:
            scan_total = max(scan_total or 0, response_total)
        next_cursor = _text(meta.get("next_cursor"))
        pagination_truncated = (
            not next_cursor
            and scan_total is not None
            and raw_items_seen < scan_total
        )
        if pagination_truncated:
            message = (
                f"{self.name}: pagination ended after {raw_items_seen} raw item(s), "
                f"before the known total of {scan_total}"
            )
            issues.append(
                SourceIssue(
                    source_record_id=(
                        f"{self.name}:pagination:"
                        f"{content_hash(dict(retry_state))[:32]}"
                    ),
                    stage="source_pagination",
                    error=message,
                    summary={
                        "cursor": cursor,
                        "window_start": _isoformat(window_start),
                        "window_end": _isoformat(window_end),
                        "raw_items_seen": raw_items_seen,
                        "known_total": scan_total,
                    },
                )
            )
        complete = not next_cursor and not pagination_truncated
        if complete:
            next_state: dict[str, Any] = {
                "watermark": _isoformat(window_end),
                "completed_at": _isoformat(self.clock()),
            }
        elif next_cursor:
            next_state = {
                "cursor": next_cursor,
                "window_start": _isoformat(window_start),
                "window_end": _isoformat(window_end),
                "started_at": started_at,
                "raw_items_seen": raw_items_seen,
            }
            if scan_total is not None:
                next_state["scan_total"] = scan_total
            if not count_is_complete:
                next_state["raw_count_incomplete"] = True
            if prior_watermark is not None:
                next_state["watermark"] = _isoformat(prior_watermark)
            if scan_high is not None:
                next_state["scan_high_watermark"] = _isoformat(scan_high)
        else:
            next_state = dict(retry_state)

        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=response_total if response_total is not None else scan_total,
            issues=tuple(issues),
            retry_state=retry_state,
        )

    def _record(self, work: Mapping[str, Any]) -> SourceRecord:
        openalex_url = _text(work.get("id"))
        openalex_id = openalex_url.rstrip("/").rsplit("/", maxsplit=1)[-1]
        if not openalex_id:
            raise ValueError(f"{self.name}: work result is missing id")

        identifiers = list(_work_identifiers(work, openalex_id))
        links = list(_work_links(work))
        primary = work.get("primary_location")
        primary_landing = _location_url(primary, "landing_page_url")
        doi_url = _doi_url(work)
        canonical_url = primary_landing or doi_url or openalex_url
        canonical_url = canonicalize_url(canonical_url)
        if not any(link.url == canonical_url for link in links):
            links.insert(0, Link(canonical_url, relation="landing_page", locator="$.id"))

        return SourceRecord(
            source_record_id=openalex_id,
            kind=self.artifact_kind,
            canonical_url=canonical_url,
            title=_text(work.get("title") or work.get("display_name")) or openalex_id,
            raw=dict(work),
            text=reconstruct_abstract(work.get("abstract_inverted_index")),
            published_at=_text(work.get("publication_date")) or None,
            modified_at=_text(work.get("updated_date")) or None,
            identifiers=_unique_identifiers(identifiers),
            links=_unique_links(links),
        )


def reconstruct_abstract(value: Any) -> str:
    """Rebuild OpenAlex's abstract text from its token-to-position index."""

    if not isinstance(value, Mapping):
        return ""
    positions: dict[int, str] = {}
    for token, offsets in value.items():
        if not isinstance(token, str):
            continue
        for offset in _sequence(offsets):
            position = _integer(offset)
            if position is not None and position >= 0:
                positions.setdefault(position, token)
    return " ".join(token for _, token in sorted(positions.items()))


def _work_identifiers(work: Mapping[str, Any], openalex_id: str) -> Iterable[Identifier]:
    yield Identifier("openalex", openalex_id)
    ids = work.get("ids") if isinstance(work.get("ids"), Mapping) else {}
    candidates = dict(ids)
    if work.get("doi"):
        candidates.setdefault("doi", work.get("doi"))

    for raw_namespace, raw_value in candidates.items():
        value = _text(raw_value)
        if not value:
            continue
        namespace = str(raw_namespace).casefold()
        if namespace == "openalex":
            continue
        if identifier := identifier_from_url(value):
            yield identifier
            continue
        if namespace == "doi":
            yield Identifier("doi", value.removeprefix("doi:").casefold())
        elif namespace in {"mag", "pmid", "pmcid", "arxiv", "wikidata"}:
            yield Identifier(namespace, value.rstrip("/").rsplit("/", maxsplit=1)[-1])


def _work_links(work: Mapping[str, Any]) -> Iterable[Link]:
    if url := _text(work.get("id")):
        yield Link(canonicalize_url(url), relation="openalex_record", locator="$.id")
    if url := _doi_url(work):
        yield Link(canonicalize_url(url), relation="doi", locator="$.doi")

    for index, value in enumerate(_sequence(work.get("referenced_works"))):
        url = _text(value)
        if re.fullmatch(r"https?://openalex\.org/W\d+", url):
            yield Link(
                canonicalize_url(url),
                relation="cites",
                locator=f"$.referenced_works[{index}]",
                crawl=False,
            )

    locations = []
    for key in ("primary_location", "best_oa_location"):
        if isinstance(location := work.get(key), Mapping):
            locations.append((f"$.{key}", location))
    for index, location in enumerate(_sequence(work.get("locations"))):
        if isinstance(location, Mapping):
            locations.append((f"$.locations[{index}]", location))
    for locator, location in locations:
        if url := _location_url(location, "landing_page_url"):
            yield Link(canonicalize_url(url), relation="landing_page", locator=locator)
        if url := _location_url(location, "pdf_url"):
            yield Link(canonicalize_url(url), relation="full_text", locator=locator)


def _doi_url(work: Mapping[str, Any]) -> str:
    doi = _text(work.get("doi"))
    if not doi and isinstance(ids := work.get("ids"), Mapping):
        doi = _text(ids.get("doi"))
    if not doi:
        return ""
    if doi.casefold().startswith(("http://", "https://")):
        return doi
    return f"https://doi.org/{doi.removeprefix('doi:')}"


def _location_url(value: Any, key: str) -> str:
    return _text(value.get(key)) if isinstance(value, Mapping) else ""


def _required_timestamp(value: Any, field: str) -> datetime:
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise ValueError(f"invalid OpenAlex {field}: {value!r}")
    return parsed


def _parse_timestamp(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _state_count(state: Mapping[str, Any], key: str) -> int | None:
    if key not in state:
        return None
    value = state.get(key)
    if isinstance(value, bool):
        raise ValueError(f"invalid OpenAlex checkpoint {key}: {value!r}")
    count = _integer(value)
    if count is None or count < 0:
        raise ValueError(f"invalid OpenAlex checkpoint {key}: {value!r}")
    return count


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _unique_identifiers(values: Iterable[Identifier]) -> tuple[Identifier, ...]:
    return tuple(dict.fromkeys(values))


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    seen: set[tuple[str, str]] = set()
    result = []
    for value in values:
        key = (value.url, value.relation)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = ["OpenAlexSourceAdapter", "reconstruct_abstract"]
