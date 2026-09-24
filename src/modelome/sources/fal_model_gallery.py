from __future__ import annotations

import re
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url

_ORIGIN = "https://fal.ai"
_MODEL_PATH = re.compile(r"^/models/(?P<id>[A-Za-z0-9][A-Za-z0-9._/-]{0,400})/?$")
_SHOWING = re.compile(
    r"Showing\s+(?:(?P<first>[\d,]+)\s+to\s+(?P<last>[\d,]+)\s+of\s+)?"
    r"(?P<total>[\d,]+)\s+results?",
    re.IGNORECASE,
)


class _GalleryParser(HTMLParser):
    def __init__(self, max_anchors: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_anchors = max_anchors
        self.items: list[tuple[str, str, bool]] = []
        self.page_text: list[str] = []
        self._href = ""
        self._is_model_card = False
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        if self._href:
            self._finish_anchor()
        values = {key.casefold(): value or "" for key, value in attrs}
        self._href = values.get("href", "")
        self._is_model_card = "page-model-card" in values.get("class", "").split()
        self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a" and self._href:
            self._finish_anchor()

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)
        self.page_text.append(data)

    def close(self) -> None:
        super().close()
        if self._href:
            self._finish_anchor()

    def _finish_anchor(self) -> None:
        if len(self.items) >= self.max_anchors:
            raise ValueError("fal model gallery page exceeds anchor limit")
        self.items.append(
            (self._href, " ".join(" ".join(self._text).split()), self._is_model_card)
        )
        self._href = ""
        self._is_model_card = False
        self._text = []


class FalModelGalleryAdapter:
    """Read paginated exact endpoint IDs from fal's public model gallery.

    This records provider endpoint declarations only. It does not infer a
    weight artifact or claim that each endpoint exposes reusable checkpoints.
    """

    def __init__(
        self,
        *,
        name: str = "fal-model-gallery",
        url: str = "https://fal.ai/explore/search",
        client: HttpClient | Any | None = None,
        page_size: int = 24,
        max_pages: int = 1_000,
        max_anchors: int = 10_000,
        max_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.name = name
        self.url = url
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        if page_size < 1 or max_pages < 1 or max_anchors < 1 or max_response_bytes < 1:
            raise ValueError("fal gallery bounds must be positive")
        self.page_size = page_size
        self.max_pages = max_pages
        self.max_anchors = max_anchors
        self.max_response_bytes = max_response_bytes

    def fetch_page(self, state: Mapping[str, Any] | None = None) -> SourcePage:
        state = state or {}
        page_number = _state_int(state.get("page", 1), "page")
        if page_number > self.max_pages:
            raise ValueError(f"{self.name}: page limit exceeded")
        seen_ids = _seen_ids(state.get("seen_ids", []), self.max_pages * self.page_size)

        page_url = self.url if page_number == 1 else f"{self.url}?page={page_number}"
        response: HttpResponse = self.client.get(page_url, headers={"Accept": "text/html"})
        if not 200 <= response.status < 300:
            raise ValueError(f"{self.name}: gallery returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: gallery response exceeds byte limit")
        final_url = response.url or page_url
        parsed_url = urlsplit(final_url)
        expected_path = urlsplit(page_url)
        if (
            parsed_url.scheme != "https"
            or parsed_url.netloc != "fal.ai"
            or parsed_url.path != expected_path.path
        ):
            raise ValueError(f"{self.name}: gallery redirected outside the requested page")

        parser = _GalleryParser(self.max_anchors)
        parser.feed(response.text())
        parser.close()
        match = _SHOWING.search(" ".join(parser.page_text))
        if match is None:
            raise ValueError(f"{self.name}: response has no gallery result range")
        total = _number(match.group("total"))
        first = _number(match.group("first")) if match.group("first") else 0
        last = _number(match.group("last")) if match.group("last") else 0
        records_by_id: dict[str, SourceRecord] = {}
        for href, text, is_model_card in parser.items:
            if not is_model_card:
                continue
            path = urlsplit(href).path
            item_match = _MODEL_PATH.fullmatch(path)
            if item_match is None:
                continue
            endpoint_id = item_match.group("id").strip("/")
            if not endpoint_id:
                continue
            if endpoint_id in records_by_id:
                raise ValueError(f"{self.name}: duplicate model card on gallery page")
            item_url = f"{_ORIGIN}/models/{quote(endpoint_id, safe='/._-')}"
            records_by_id[endpoint_id] = SourceRecord(
                source_record_id=f"fal:model-endpoint:{endpoint_id}",
                kind=ArtifactKind.PROVIDER_PAGE,
                canonical_url=canonicalize_url(item_url),
                title=endpoint_id,
                raw={"provider": "fal", "endpoint_id": endpoint_id, "gallery_text": text},
                text=text,
                identifiers=(Identifier("fal:endpoint", endpoint_id),),
                links=(Link(item_url, relation="documents_model", crawl=False),),
                models=(
                    ModelHint(
                        local_id=endpoint_id,
                        name=endpoint_id,
                        identifiers=(Identifier("fal:endpoint", endpoint_id),),
                        status=ModelStatus.DOCUMENTED,
                        locator=f"gallery:page[{page_number}]",
                    ),
                ),
            )

        page_card_count = len(records_by_id)
        if total == 0:
            if page_card_count:
                raise ValueError(f"{self.name}: zero-result page contains model cards")
            complete = True
        else:
            if first < 1 or last < first or last > total:
                raise ValueError(f"{self.name}: invalid gallery result range")
            if page_card_count != last - first + 1:
                raise ValueError(f"{self.name}: card count does not match reported page range")
            if first != (page_number - 1) * self.page_size + 1:
                raise ValueError(f"{self.name}: page range does not match requested page")
            if last < total and page_card_count != self.page_size:
                raise ValueError(f"{self.name}: short non-terminal gallery page")
            complete = last == total

        # A live gallery can reorder or add/remove cards between requests. Keep
        # walking by page number, but emit each endpoint only once per run.
        novel_ids = set(records_by_id) - set(seen_ids)
        if page_card_count and not novel_ids:
            raise ValueError(f"{self.name}: gallery page made no model-ID progress")
        page_records = tuple(
            record for endpoint_id, record in records_by_id.items() if endpoint_id in novel_ids
        )
        next_seen_ids = [
            *seen_ids,
            *(
                endpoint_id
                for endpoint_id in records_by_id
                if endpoint_id in novel_ids
            ),
        ]
        if len(next_seen_ids) > self.max_pages * self.page_size:
            raise ValueError(f"{self.name}: seen model-ID limit exceeded")

        next_state: dict[str, Any] = {
            "page": page_number + 1,
            "seen_ids": next_seen_ids,
        }
        if complete:
            next_state = {"page": page_number, "seen_ids": next_seen_ids, "complete": True}
        return SourcePage(
            records=page_records,
            next_state=next_state,
            complete=complete,
            upstream_count=total,
            authoritative_snapshot=False,
        )


def _number(value: str) -> int:
    return int(value.replace(",", ""))


def _state_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"fal gallery {label} state must be a positive integer")
    return value


def _seen_ids(value: Any, maximum: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError("fal gallery seen_ids state must be a bounded list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError("fal gallery seen_ids state must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError("fal gallery seen_ids state contains duplicates")
    return value


__all__ = ["FalModelGalleryAdapter"]
