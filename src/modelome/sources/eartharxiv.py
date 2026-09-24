from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]
_OAI = "http://www.openarchives.org/OAI/2.0/"
_DC = "http://purl.org/dc/elements/1.1/"
_NO_VALUE = frozenset({"", "na", "n/a", "none", "not available", "null"})
_OBJECT_ID = re.compile(r":id:(\d+)$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EarthArxivSourceAdapter:
    """Enumerate the first-party EarthArXiv Janeway OAI-PMH change stream."""

    def __init__(
        self,
        *,
        name: str = "eartharxiv",
        url: str = "https://eartharxiv.org/api/oai/",
        web_base_url: str = "https://eartharxiv.org",
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        initial_lookback_days: int = 7,
        overlap_days: int = 2,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name)
        self.web_base_url = _web_url(web_base_url, self.name)
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
                "adapter": "eartharxiv-oai-dc-v1",
                "url": self.url,
                "web_base_url": self.web_base_url,
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
        token = _state_token(state, frozen=frozen, source=self.name)
        retry_state = self._scan_state(
            state,
            token=token,
            window_start=window_start,
            window_end=window_end,
            now=now,
            prior_watermark=prior_watermark,
        )
        if window_start > window_end:
            next_state: dict[str, Any] = {"completed_at": _isoformat(now)}
            if prior_watermark is not None:
                next_state["watermark"] = prior_watermark.isoformat()
            return SourcePage(records=(), next_state=next_state, complete=True)

        params: dict[str, str] = {"verb": "ListRecords"}
        if token is None:
            params.update(self._list_request_parameters(window_start, window_end))
        else:
            params["resumptionToken"] = token
        response: HttpResponse = self.client.get(
            self.url,
            params=params,
            headers={"Accept": "application/xml, text/xml;q=0.9"},
        )
        root = _parse_oai(response.body, self.name, response.url)
        errors = _errors(root)
        if errors:
            if token is None and {code for code, _ in errors} == {"noRecordsMatch"}:
                return SourcePage(
                    records=(),
                    next_state={
                        "watermark": window_end.isoformat(),
                        "completed_at": _isoformat(now),
                    },
                    complete=True,
                )
            detail = "; ".join(f"{code}: {message}" for code, message in errors)
            raise ValueError(f"{self.name}: OAI-PMH ListRecords error: {detail}")

        listing = root.find(f"{{{_OAI}}}ListRecords")
        if listing is None:
            raise ValueError(f"{self.name}: OAI-PMH response is missing ListRecords")
        raw_records = tuple(listing.findall(f"{{{_OAI}}}record"))
        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, record in enumerate(raw_records):
            try:
                records.append(self._record(record))
            except (TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=self._malformed_id(record, index),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index},
                    )
                )

        next_token = _resumption_token(listing, self.name)
        if next_token and next_token == token:
            issues.append(
                SourceIssue(
                    source_record_id=f"{self.name}:pagination:{content_hash(next_token)[:32]}",
                    stage="source_pagination",
                    error=f"{self.name}: OAI-PMH resumption token repeated",
                    summary={"token_hash": content_hash(next_token)},
                )
            )
            next_token = None
        if next_token and not raw_records:
            issues.append(
                SourceIssue(
                    source_record_id=(
                        f"{self.name}:pagination:{content_hash(next_token)[:32]}"
                    ),
                    stage="source_pagination",
                    error=f"{self.name}: OAI-PMH returned an empty page with a resumption token",
                    summary={"token_hash": content_hash(next_token)},
                )
            )
            next_token = None

        if next_token is None and not issues:
            next_state = {
                "watermark": window_end.isoformat(),
                "completed_at": _isoformat(now),
            }
            complete = True
        elif next_token is not None:
            next_state = self._scan_state(
                state,
                token=next_token,
                window_start=window_start,
                window_end=window_end,
                now=now,
                prior_watermark=prior_watermark,
            )
            complete = False
        else:
            next_state = retry_state
            complete = False
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            issues=tuple(issues),
            retry_state=retry_state,
        )

    def _list_request_parameters(
        self, window_start: date, window_end: date
    ) -> dict[str, str]:
        """Build an initial OAI-PMH request for a frozen change window.

        Janeway's EarthArXiv endpoint accepts second-granularity bounds. Other
        OAI-PMH providers can reuse the state and resumption-token machinery
        while overriding only this method for date-granularity repositories.
        """

        return {
            "metadataPrefix": "oai_dc",
            "from": _start_of_day(window_start),
            "until": _end_of_day(window_end),
        }

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
            window_start = _required_date(raw_start, "window_start", self.name)
            window_end = _required_date(raw_end, "window_end", self.name)
            if window_start > window_end:
                raise ValueError(f"{self.name}: window_start must not follow window_end")
            if window_end > closed_through:
                raise ValueError(f"{self.name}: window_end must be a closed UTC day")
            return window_start, window_end, True
        if "resumption_token" in state:
            raise ValueError(f"{self.name}: token checkpoint is missing its frozen window")
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
        token: str | None,
        window_start: date,
        window_end: date,
        now: datetime,
        prior_watermark: date | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "started_at": _text(state.get("started_at")) or _isoformat(now),
        }
        if token:
            result["resumption_token"] = token
        if prior_watermark is not None:
            result["watermark"] = prior_watermark.isoformat()
        return result

    def _record(self, element: ET.Element) -> SourceRecord:
        header = element.find(f"{{{_OAI}}}header")
        if header is None:
            raise ValueError("OAI record is missing its header")
        oai_identifier = _required_element_text(header, "identifier", _OAI, self.name)
        datestamp = _required_timestamp(
            _required_element_text(header, "datestamp", _OAI, self.name), self.name
        )
        object_id = _object_id(oai_identifier)
        preprint_url = canonicalize_url(
            f"{self.web_base_url}/repository/object/{quote(object_id, safe='')}"
        )
        if header.get("status") == "deleted":
            return SourceRecord(
                source_record_id=f"eartharxiv:{oai_identifier}",
                kind=self.artifact_kind,
                canonical_url=preprint_url,
                title=f"Withdrawn EarthArXiv preprint {object_id}",
                raw={"oai_identifier": oai_identifier, "datestamp": datestamp},
                modified_at=datestamp,
                identifiers=(
                    Identifier("oai:eartharxiv", oai_identifier),
                    Identifier("eartharxiv:object", object_id),
                ),
                links=(Link(preprint_url, relation="preprint", locator="header.identifier"),),
                deleted=True,
            )
        metadata = element.find(f"{{{_OAI}}}metadata/{{http://www.openarchives.org/OAI/2.0/oai_dc/}}dc")
        if metadata is None:
            raise ValueError("OAI record is missing Dublin Core metadata")
        title = _first_text(metadata, "title", self.name)
        descriptions = _texts(metadata, "description")
        dates = _texts(metadata, "date")
        relations = _texts(metadata, "relation")
        identifiers = [
            Identifier("oai:eartharxiv", oai_identifier),
            Identifier("eartharxiv:object", object_id),
        ]
        links = [Link(preprint_url, relation="preprint", locator="header.identifier")]
        for index, value in enumerate(_texts(metadata, "identifier")):
            if doi := _optional_doi(value):
                identifiers.append(Identifier("doi", doi))
                links.append(Link(_doi_url(doi), relation="doi", locator=f"dc.identifier[{index}]"))
            elif url := _optional_web_url(value):
                links.append(Link(url, relation="full_text", locator=f"dc.identifier[{index}]"))
        # Dublin Core relation values can name an associated work, dataset, or
        # other resource. Preserve their exact values and expose resolvable
        # targets as references without asserting that two records are the same
        # work or that the target is a journal publication.
        for index, value in enumerate(relations):
            if doi := _optional_doi(value):
                links.append(
                    Link(_doi_url(doi), relation="references", locator=f"dc.relation[{index}]")
                )
            elif url := _optional_web_url(value):
                links.append(Link(url, relation="references", locator=f"dc.relation[{index}]"))
        return SourceRecord(
            source_record_id=f"eartharxiv:{oai_identifier}",
            kind=self.artifact_kind,
            canonical_url=preprint_url,
            title=title,
            raw={
                "oai_identifier": oai_identifier,
                "datestamp": datestamp,
                "creators": _texts(metadata, "creator"),
                "dates": dates,
                "relations": relations,
                "identifiers": _texts(metadata, "identifier"),
                "subjects": _texts(metadata, "subject"),
                "rights": _texts(metadata, "rights"),
            },
            text="\n\n".join(descriptions),
            published_at=_optional_timestamp(dates[0]) if dates else None,
            modified_at=datestamp,
            identifiers=tuple(dict.fromkeys(identifiers)),
            links=_unique_links(links),
        )

    def _malformed_id(self, element: ET.Element, index: int) -> str:
        identifier = element.findtext(f"{{{_OAI}}}header/{{{_OAI}}}identifier")
        if identifier:
            return f"eartharxiv:{identifier.strip()}"
        return f"{self.name}:malformed:{index}"


def _parse_oai(body: bytes, source: str, url: str) -> ET.Element:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise ValueError(f"{source}: invalid OAI-PMH XML from {url}: {error}") from None
    if root.tag != f"{{{_OAI}}}OAI-PMH":
        raise ValueError(f"{source}: response is not an OAI-PMH document")
    return root


def _errors(root: ET.Element) -> tuple[tuple[str, str], ...]:
    return tuple(
        ((element.get("code") or "unknown").strip(), _text(element.text) or "no detail")
        for element in root.findall(f"{{{_OAI}}}error")
    )


def _resumption_token(listing: ET.Element, source: str) -> str | None:
    element = listing.find(f"{{{_OAI}}}resumptionToken")
    if element is None:
        return None
    token = _text(element.text)
    return token or None


def _state_token(state: Mapping[str, Any], *, frozen: bool, source: str) -> str | None:
    token = _text(state.get("resumption_token"))
    if token and not frozen:
        raise ValueError(f"{source}: token checkpoint is missing its frozen window")
    return token or None


def _object_id(value: str) -> str:
    match = _OBJECT_ID.search(value)
    if match is None:
        raise ValueError(f"EarthArXiv OAI identifier has no object ID: {value!r}")
    return match.group(1)


def _first_text(element: ET.Element, name: str, source: str) -> str:
    values = _texts(element, name)
    if not values:
        raise ValueError(f"{source}: Dublin Core record has no {name}")
    return values[0]


def _texts(element: ET.Element, name: str) -> list[str]:
    return [text for child in element.findall(f"{{{_DC}}}{name}") if (text := _text(child.text))]


def _required_element_text(element: ET.Element, name: str, namespace: str, source: str) -> str:
    return _required_text(element.findtext(f"{{{namespace}}}{name}"), f"{source} {name}")


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


def _required_timestamp(value: str, source: str) -> str:
    result = _optional_timestamp(value)
    if result is None:
        raise ValueError(f"{source}: missing timestamp")
    return result


def _optional_timestamp(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"invalid timestamp: {value!r}") from None
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
        raise ValueError(f"{source}: URL must be an HTTP(S) URL")
    return url.rstrip("/")


def _unique_links(values: Sequence[Link]) -> tuple[Link, ...]:
    return tuple({(link.url, link.relation): link for link in values}.values())


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


__all__ = ["EarthArxivSourceAdapter"]
