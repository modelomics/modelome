from __future__ import annotations

import html
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]
_DOI_RE = re.compile(r"10\.1371/[a-z0-9.-]+", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]*>")
_JOURNAL_SLUGS = {
    "pbio": "plosbiology",
    "pcbi": "ploscompbiol",
    "pctr": "plosclinicaltrials",
    "pgen": "plosgenetics",
    "pmed": "plosmedicine",
    "pntd": "plosntds",
    "pone": "plosone",
    "ppat": "plospathogens",
}
_FIELDS = (
    "id,title_display,abstract,publication_date,author,article_type,subject,journal,"
    "eissn,pissn,copyright"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PlosSourceAdapter:
    """Enumerate PLOS's complete first-party journal article metadata API.

    The source retains full Solr documents only and makes no selection by
    journal, topic, author, or model vocabulary.  PLOS exposes
    offset pagination rather than opaque cursors, so a frozen publication-date
    window is validated against its declared result total.  If that total moves
    during a scan, the entire window is replayed from offset zero before its
    watermark can advance.
    """

    def __init__(
        self,
        *,
        name: str = "plos",
        url: str = "https://api.plos.org/search",
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        page_size: int = 100,
        initial_start_date: str | date = "2003-08-18",
        overlap_days: int = 14,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name)
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.page_size = _bounded_positive(page_size, "page_size", maximum=100)
        self.initial_start_date = _required_date(
            initial_start_date, "initial_start_date", self.name
        )
        self.overlap_days = _nonnegative(overlap_days, "overlap_days")
        self.client = client or HttpClient()
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "plos-solr-v1",
                "url": self.url,
                "artifact_kind": self.artifact_kind.value,
                "page_size": self.page_size,
                "initial_start_date": self.initial_start_date.isoformat(),
                "overlap_days": self.overlap_days,
                "query": "publication_date:[closed window]",
                "filter": "doc_type:full",
                "sort": "publication_date asc,id asc",
                "fields": _FIELDS,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        now = _as_utc(self.clock())
        closed_through = now.date() - timedelta(days=1)
        prior_watermark = _optional_date(state.get("watermark"), "watermark", self.name)
        window_start, window_end, frozen = self._window(
            state, prior_watermark=prior_watermark, closed_through=closed_through
        )
        offset = _nonnegative(state.get("offset", 0), "offset")
        previous_total = _optional_nonnegative(state.get("scan_total"), "scan_total")
        if offset and not frozen:
            raise ValueError(f"{self.name}: offset checkpoint is missing its frozen window")
        if previous_total is not None and not frozen:
            raise ValueError(f"{self.name}: scan_total checkpoint is missing its frozen window")
        if previous_total is not None and offset > previous_total:
            raise ValueError(f"{self.name}: offset exceeds scan_total")
        if window_start > window_end:
            next_state: dict[str, Any] = {"completed_at": _isoformat(now)}
            if prior_watermark is not None:
                next_state["watermark"] = prior_watermark.isoformat()
            return SourcePage(records=(), next_state=next_state, complete=True, upstream_count=0)

        retry_state = self._scan_state(
            state,
            offset=offset,
            total=previous_total,
            window_start=window_start,
            window_end=window_end,
            prior_watermark=prior_watermark,
            now=now,
        )
        response: HttpResponse = self.client.get(
            self.url,
            params={
                "q": _date_range(window_start, window_end),
                "fq": "doc_type:full",
                "fl": _FIELDS,
                "sort": "publication_date asc,id asc",
                "rows": str(self.page_size),
                "start": str(offset),
                "wt": "json",
            },
            headers={"Accept": "application/json"},
        )
        payload = _json_object(response.body, self.name, response.url)
        listing = payload.get("response")
        if not isinstance(listing, Mapping):
            raise ValueError(f"{self.name}: response is missing its result object")
        total = _required_nonnegative(listing.get("numFound"), "response.numFound", self.name)
        response_start = _required_nonnegative(
            listing.get("start"), "response.start", self.name
        )
        if response_start != offset:
            return self._restart_page(
                state,
                window_start=window_start,
                window_end=window_end,
                prior_watermark=prior_watermark,
                now=now,
                total=total,
                error=(
                    f"{self.name}: response start {response_start} does not match "
                    f"requested offset {offset}"
                ),
            )
        if previous_total is not None and total != previous_total:
            return self._restart_page(
                state,
                window_start=window_start,
                window_end=window_end,
                prior_watermark=prior_watermark,
                now=now,
                total=total,
                error=(
                    f"{self.name}: result total changed from {previous_total} to {total}; "
                    "restarting the frozen window"
                ),
            )
        documents = listing.get("docs")
        if not isinstance(documents, list):
            raise ValueError(f"{self.name}: response.docs must be an array")
        if len(documents) > self.page_size:
            raise ValueError(f"{self.name}: response exceeds configured page_size")
        if offset > total:
            return self._restart_page(
                state,
                window_start=window_start,
                window_end=window_end,
                prior_watermark=prior_watermark,
                now=now,
                total=total,
                error=f"{self.name}: offset {offset} exceeds result total {total}",
            )

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        page_ids: set[str] = set()
        last_key: tuple[str, str] | None = None
        for index, document in enumerate(documents):
            try:
                record = self._record(document)
                if record.source_record_id in page_ids:
                    raise ValueError(f"duplicate PLOS DOI {record.source_record_id!r} on one page")
                published_day = date.fromisoformat((record.published_at or "")[:10])
                if not window_start <= published_day <= window_end:
                    raise ValueError(
                        "document publication_date is outside the frozen source window"
                    )
                key = (record.published_at or "", record.source_record_id)
                if last_key is not None and key < last_key:
                    raise ValueError("response does not follow publication_date,id ascending order")
                page_ids.add(record.source_record_id)
                last_key = key
                records.append(record)
            except (TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=_malformed_id(document, index, self.name),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index},
                    )
                )
        if issues:
            return SourcePage(
                records=tuple(records),
                next_state=retry_state,
                complete=False,
                upstream_count=total,
                issues=tuple(issues),
                retry_state=retry_state,
            )

        next_offset = offset + len(documents)
        if next_offset > total:
            return self._restart_page(
                state,
                window_start=window_start,
                window_end=window_end,
                prior_watermark=prior_watermark,
                now=now,
                total=total,
                error=f"{self.name}: page ends at {next_offset}, beyond result total {total}",
            )
        if next_offset < total and not documents:
            return self._restart_page(
                state,
                window_start=window_start,
                window_end=window_end,
                prior_watermark=prior_watermark,
                now=now,
                total=total,
                error=f"{self.name}: empty page before result total {total}",
            )
        if next_offset < total and len(documents) < self.page_size:
            return self._restart_page(
                state,
                window_start=window_start,
                window_end=window_end,
                prior_watermark=prior_watermark,
                now=now,
                total=total,
                error=(
                    f"{self.name}: short page at offset {offset} before result total {total}"
                ),
            )
        if next_offset == total:
            return SourcePage(
                records=tuple(records),
                next_state={
                    "watermark": window_end.isoformat(),
                    "completed_at": _isoformat(now),
                },
                complete=True,
                upstream_count=total,
            )
        return SourcePage(
            records=tuple(records),
            next_state=self._scan_state(
                state,
                offset=next_offset,
                total=total,
                window_start=window_start,
                window_end=window_end,
                prior_watermark=prior_watermark,
                now=now,
            ),
            complete=False,
            upstream_count=total,
        )

    def _window(
        self,
        state: Mapping[str, Any],
        *,
        prior_watermark: date | None,
        closed_through: date,
    ) -> tuple[date, date, bool]:
        raw_start, raw_end = state.get("window_start"), state.get("window_end")
        if raw_start is not None or raw_end is not None:
            if raw_start is None or raw_end is None:
                raise ValueError(f"{self.name}: frozen window requires both boundaries")
            start = _required_date(raw_start, "window_start", self.name)
            end = _required_date(raw_end, "window_end", self.name)
            if start > end:
                raise ValueError(f"{self.name}: window_start must not follow window_end")
            if end > closed_through:
                raise ValueError(f"{self.name}: window_end must be a closed UTC day")
            return start, end, True
        if prior_watermark is None:
            return self.initial_start_date, closed_through, False
        if prior_watermark > closed_through:
            raise ValueError(f"{self.name}: watermark is later than the last closed UTC day")
        return prior_watermark + timedelta(days=1 - self.overlap_days), closed_through, False

    def _scan_state(
        self,
        state: Mapping[str, Any],
        *,
        offset: int,
        total: int | None,
        window_start: date,
        window_end: date,
        prior_watermark: date | None,
        now: datetime,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "offset": offset,
            "started_at": _text(state.get("started_at")) or _isoformat(now),
        }
        if total is not None:
            result["scan_total"] = total
        if prior_watermark is not None:
            result["watermark"] = prior_watermark.isoformat()
        return result

    def _restart_page(
        self,
        state: Mapping[str, Any],
        *,
        window_start: date,
        window_end: date,
        prior_watermark: date | None,
        now: datetime,
        total: int,
        error: str,
    ) -> SourcePage:
        retry_state = self._scan_state(
            state,
            offset=0,
            total=None,
            window_start=window_start,
            window_end=window_end,
            prior_watermark=prior_watermark,
            now=now,
        )
        issue = SourceIssue(
            source_record_id=f"{self.name}:pagination:{window_start}:{window_end}",
            stage="source_pagination",
            error=error,
            summary={
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "total": total,
            },
        )
        return SourcePage(
            records=(),
            next_state=retry_state,
            complete=False,
            upstream_count=total,
            issues=(issue,),
            retry_state=retry_state,
        )

    def _record(self, document: Any) -> SourceRecord:
        if not isinstance(document, Mapping):
            raise ValueError("PLOS document must be an object")
        doi = _doi(document.get("id"))
        publication_date = _timestamp(document.get("publication_date"), "publication_date")
        title = _plain_text(_required_text(document.get("title_display"), "title_display"))
        if not title:
            raise ValueError("PLOS title_display contains no text")
        canonical_url = canonicalize_url(f"https://doi.org/{doi}")
        article_root = _article_url(doi)
        links = [Link(canonical_url, relation="doi", locator="id", crawl=False)]
        if article_root:
            article_url = canonicalize_url(
                f"{article_root}?id={quote(doi, safe='')}"
            )
            xml_url = canonicalize_url(
                f"{article_root}/file?id={quote(doi, safe='')}&type=manuscript"
            )
            links.extend(
                (
                    Link(article_url, relation="publisher_article", locator="id", crawl=False),
                    Link(
                        xml_url,
                        relation="full_text",
                        locator="id",
                        crawl=False,
                    ),
                )
            )
        abstracts = _texts(document.get("abstract"))
        article_types = _texts(document.get("article_type"))
        is_issue_image = any(value.casefold() == "issue image" for value in article_types)
        return SourceRecord(
            source_record_id=f"plos:{doi}",
            kind=ArtifactKind.OTHER if is_issue_image else self.artifact_kind,
            canonical_url=canonical_url,
            title=title,
            raw={
                "doi": doi,
                "publication_date": publication_date,
                "authors": _texts(document.get("author")),
                "abstracts": abstracts,
                "article_types": article_types,
                "subjects": _texts(document.get("subject")),
                "journal": _texts(document.get("journal")),
                "issn": {
                    "electronic": _texts(document.get("eissn")),
                    "print": _texts(document.get("pissn")),
                },
                "rights": _texts(document.get("copyright")),
            },
            text="\n\n".join(abstracts),
            published_at=publication_date,
            modified_at=publication_date,
            identifiers=(Identifier("doi", doi), Identifier("plos:document", doi)),
            links=tuple(links),
        )


def _date_range(start: date, end: date) -> str:
    return (
        "publication_date:["
        f"{start.isoformat()}T00:00:00Z TO {end.isoformat()}T23:59:59Z]"
    )


def _article_url(doi: str) -> str:
    prefix, separator, suffix = doi.partition("/")
    parts = suffix.split(".")
    if separator != "/" or prefix != "10.1371" or len(parts) < 3 or parts[0] != "journal":
        return ""
    journal = _JOURNAL_SLUGS.get(parts[1])
    return canonicalize_url(f"https://journals.plos.org/{journal}/article") if journal else ""


def _doi(value: Any) -> str:
    text = _text(value).casefold()
    if not _DOI_RE.fullmatch(text):
        raise ValueError(f"invalid PLOS DOI {value!r}")
    return text


def _plain_text(value: str) -> str:
    return " ".join(html.unescape(_HTML_TAG.sub("", value)).split())


def _texts(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [_plain_text(text) for item in values if (text := _text(item))]


def _malformed_id(document: Any, index: int, source: str) -> str:
    value = document.get("id") if isinstance(document, Mapping) else None
    return f"{source}:{_text(value)}" if _text(value) else f"{source}:malformed:{index}"


def _json_object(body: bytes, source: str, url: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(body)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source}: invalid JSON from {url}: {error}") from None
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: response must be a JSON object")
    return payload


def _required_nonnegative(value: Any, field: str, source: str) -> int:
    result = _optional_nonnegative(value, field)
    if result is None:
        raise ValueError(f"{source}: {field} is required")
    return result


def _optional_nonnegative(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a nonnegative integer") from None
    if result < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return result


def _required_date(value: Any, field: str, source: str) -> date:
    parsed = _optional_date(value, field, source)
    if parsed is None:
        raise ValueError(f"{source}: {field} is required")
    return parsed


def _optional_date(value: Any, field: str, source: str) -> date | None:
    if value is None or _text(value) == "":
        return None
    if isinstance(value, datetime):
        return _as_utc(value).date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(_text(value))
    except ValueError:
        raise ValueError(f"{source}: invalid {field}: {value!r}") from None


def _timestamp(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"invalid {field}: {value!r}") from None
    return _isoformat(parsed)


def _bounded_positive(value: Any, field: str, *, maximum: int) -> int:
    result = _nonnegative(value, field)
    if result == 0 or result > maximum:
        raise ValueError(f"{field} must be between 1 and {maximum}")
    return result


def _nonnegative(value: Any, field: str) -> int:
    result = _optional_nonnegative(value, field)
    if result is None:
        raise ValueError(f"{field} is required")
    return result


def _web_url(value: Any, source: str) -> str:
    url = canonicalize_url(_required_text(value, "URL"))
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"{source}: URL must be an HTTP(S) URL")
    return url.rstrip("/")


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


__all__ = ["PlosSourceAdapter"]
