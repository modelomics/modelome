"""Grand Challenge's anonymous public algorithm-card listing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qs, urlsplit

from modelome.http import HttpClient
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_API = "https://grand-challenge.org/api/v1/algorithms/"
_MAX_PAGE_SIZE = 100
_MAX_ENTRIES = 5000


class GrandChallengeAlgorithmsSourceAdapter:
    """Page publicly listed medical algorithm cards without claiming weights."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public algorithm cards returned by Grand Challenge's anonymous paginated API. "
        "Algorithm containers may include trained weights, but container contents and direct "
        "checkpoint files are not exposed by this listing. Private algorithms are excluded."
    )

    def __init__(
        self,
        *,
        name: str = "grand-challenge-public-algorithms",
        page_size: int = 100,
        max_response_bytes: int = 8 * 1024 * 1024,
        max_entries: int = _MAX_ENTRIES,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not name.strip() or not 1 <= page_size <= _MAX_PAGE_SIZE:
            raise ValueError("name and page_size from 1 to 100 are required")
        if max_response_bytes <= 0 or not 1 <= max_entries <= _MAX_ENTRIES:
            raise ValueError("positive response limit and max_entries up to 5000 are required")
        self.name = name
        self.page_size = page_size
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "grand-challenge-public-algorithm-cards-v1",
                "endpoint": _API,
                "page_size": page_size,
                "max_entries": max_entries,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        offset = state.get("offset", 0)
        expected_count = state.get("count")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError(f"{self.name}: invalid offset checkpoint")
        if expected_count is not None and (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count < 0
        ):
            raise ValueError(f"{self.name}: invalid expected count checkpoint")
        response = self.client.get(
            _API,
            params={"limit": self.page_size, "offset": offset},
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: response exceeds configured byte limit")
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: malformed algorithm listing")
        count = payload.get("count")
        rows = payload.get("results")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or not isinstance(rows, list)
        ):
            raise ValueError(f"{self.name}: listing lacks a valid count or results")
        if expected_count is not None and count != expected_count:
            raise ValueError(f"{self.name}: upstream total changed during pagination")
        if len(rows) > self.page_size or offset + len(rows) > count:
            raise ValueError(f"{self.name}: listing exceeded requested page or total")
        if count > self.max_entries:
            raise ValueError(f"{self.name}: listing exceeds configured entry limit")
        next_url = payload.get("next")
        expected_next = offset + len(rows)
        if next_url is None:
            if expected_next != count:
                raise ValueError(f"{self.name}: listing ended before its declared total")
        elif _next_offset(next_url, self.page_size) != expected_next or expected_next >= count:
            raise ValueError(f"{self.name}: malformed or discontinuous next link")
        records = tuple(_record(row, self.name) for row in rows)
        complete = next_url is None
        return SourcePage(
            records,
            {"offset": expected_next, "count": count},
            complete=complete,
            upstream_count=count,
            authoritative_snapshot=False,
        )


def _next_offset(value: Any, page_size: int) -> int:
    if not isinstance(value, str):
        raise ValueError("Grand Challenge next link is not a URL")
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or parts.netloc != "grand-challenge.org"
        or parts.path != "/api/v1/algorithms/"
    ):
        raise ValueError("Grand Challenge next link escaped its API endpoint")
    query = parse_qs(parts.query)
    try:
        limit = int(query["limit"][0])
        offset = int(query["offset"][0])
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError("Grand Challenge next link lacks limit or offset") from error
    if limit != page_size or offset < 0:
        raise ValueError("Grand Challenge next link changed page size or has invalid offset")
    return offset


def _record(row: Any, source_name: str) -> SourceRecord:
    if not isinstance(row, Mapping):
        raise ValueError(f"{source_name}: algorithm row is not an object")
    pk = row.get("pk")
    slug = row.get("slug")
    title = row.get("title")
    card_url = row.get("url")
    api_url = row.get("api_url")
    if not all(isinstance(value, str) and value.strip() for value in (pk, slug, title)):
        raise ValueError(f"{source_name}: algorithm row lacks its ID, slug, or title")
    if not _valid_url(card_url) or not _valid_url(api_url):
        raise ValueError(f"{source_name}: algorithm row has invalid card or API URL")
    if urlsplit(api_url).netloc != "grand-challenge.org" or not urlsplit(api_url).path.startswith(
        "/api/v1/algorithms/"
    ):
        raise ValueError(f"{source_name}: algorithm API URL is outside Grand Challenge")
    model_id = f"model:{pk}"
    description = row.get("description")
    interfaces = row.get("interfaces")
    raw = {
        "pk": pk,
        "slug": slug,
        "description": description if isinstance(description, str) else None,
        "interfaces": interfaces if isinstance(interfaces, list) else [],
    }
    return SourceRecord(
        source_record_id=f"grand-challenge:algorithm:{pk}",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url=canonicalize_url(card_url),
        title=title.strip(),
        raw=raw,
        text=description.strip() if isinstance(description, str) else "",
        identifiers=(Identifier("grand-challenge:algorithm", pk),),
        links=(
            Link(card_url, "provider_model_card", crawl=False),
            Link(api_url, "provider_api_record", crawl=False),
        ),
        models=(
            ModelHint(
                model_id,
                title.strip(),
                identifiers=(Identifier("grand-challenge:algorithm", pk),),
                aliases=(slug,),
                status=ModelStatus.DOCUMENTED,
                locator="Grand Challenge public algorithm listing",
            ),
        ),
    )


def _valid_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    return parsed.scheme == "https" and bool(parsed.netloc)
