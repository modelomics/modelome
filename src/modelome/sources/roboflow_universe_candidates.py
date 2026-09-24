"""Bounded public-UI discovery of Roboflow Universe model candidates.

This is deliberately not a registry snapshot. Search is query-scoped, ranking and
visibility are controlled by Roboflow, and the UI reports a finite result window.
For each listed project the adapter reads the public project page and records only
the exact deployed ``model_id`` rendered there; it does not infer IDs from project
names or claim downloadable weights.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlencode, urlsplit

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

_ORIGIN = "https://universe.roboflow.com"
_PROJECT_PATH = re.compile(
    r"^/(?P<workspace>[A-Za-z0-9][A-Za-z0-9._-]{0,99})/"
    r"(?P<project>[A-Za-z0-9][A-Za-z0-9._-]{0,199})/?$"
)
_MODEL_ID = re.compile(
    r"model_id\s*[:=]\s*[\"']"
    r"(?P<id>[A-Za-z0-9][A-Za-z0-9._-]{0,199}/[1-9][0-9]{0,9})[\"']",
    re.IGNORECASE,
)
_MODEL_TYPE = re.compile(r"Model type:\s*([^\r\n<]{1,160})", re.IGNORECASE)
_PAGE_RANGE = re.compile(
    r"Showing\s+(?P<first>[\d,]+)\s*[-–]\s*"
    r"(?P<last>[\d,]+)\s+of\s+(?P<total>[\d,]+)",
    re.IGNORECASE,
)
_RESULTS_PER_PAGE = re.compile(r"(?P<count>[\d,]+)\s+results per page", re.IGNORECASE)


class _LinksAndText(HTMLParser):
    def __init__(self, max_anchors: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_anchors = max_anchors
        self.hrefs: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            if len(self.hrefs) >= self.max_anchors:
                raise ValueError("Roboflow response exceeds anchor limit")
            self.hrefs.append(href)

    def handle_data(self, data: str) -> None:
        self.text.append(data)


class RoboflowUniverseCandidatesAdapter:
    """Find explicit deployed model IDs from a bounded Roboflow search query.

    Results are non-authoritative observations. The UI can rank, filter, and cap
    results; a project's page exposes one selected deployment, not necessarily all
    model versions attached to that project.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Query-scoped candidates from Roboflow Universe's public search UI only. "
        "Search visibility/ranking and its finite result window can omit projects; "
        "each project page exposes only the selected deployed model ID. This is not "
        "an exhaustive model/version snapshot and does not establish weight access."
    )

    def __init__(
        self,
        *,
        name: str = "roboflow-universe-candidates",
        search_url: str = "https://universe.roboflow.com/search",
        query: str,
        client: HttpClient | Any | None = None,
        max_pages: int = 20,
        max_projects_per_page: int = 50,
        max_anchors: int = 10_000,
        max_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if not query.strip():
            raise ValueError("Roboflow search query must not be empty")
        bounds = (
            (max_pages, "max_pages"),
            (max_projects_per_page, "max_projects_per_page"),
            (max_anchors, "max_anchors"),
            (max_response_bytes, "max_response_bytes"),
        )
        for value, label in bounds:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        parts = urlsplit(search_url)
        if (
            parts.scheme != "https"
            or parts.netloc != "universe.roboflow.com"
            or parts.path != "/search"
        ):
            raise ValueError("search_url must be Roboflow's HTTPS Universe search page")
        self.name = name
        self.search_url = search_url
        self.query = query.strip()
        self.max_pages = max_pages
        self.max_projects_per_page = max_projects_per_page
        self.max_anchors = max_anchors
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)

    def fetch_page(self, state: Mapping[str, Any] | None = None) -> SourcePage:
        state = state or {}
        page = _positive_state_int(state.get("page", 1), "page")
        if page > self.max_pages:
            raise ValueError(f"{self.name}: page limit exceeded")
        query = urlencode({"p": page - 1, "q": self.query})
        page_url = f"{self.search_url}?{query}"
        listing = self._get_html(page_url)
        parser = _parse(listing.text(), self.max_anchors)
        text = " ".join(parser.text)
        range_match = _PAGE_RANGE.search(text)
        if range_match is None:
            raise ValueError(f"{self.name}: search page has no displayed result range")
        first, last, total = (_number(range_match.group(key)) for key in ("first", "last", "total"))
        page_size_match = _RESULTS_PER_PAGE.search(text)
        if page_size_match is None:
            raise ValueError(f"{self.name}: search page has no displayed page size")
        page_size = _number(page_size_match.group("count"))
        page_start = (page - 1) * page_size
        expected_last = min(page * page_size, total)
        # Roboflow's UI displays the previous page's last index as the next
        # page's first index (e.g. 1–50 followed by 50–100), despite showing
        # only 50 projects on each page. Accept that boundary convention and
        # the usual one-based convention, while validating against its explicit
        # page size to avoid rejecting valid pages as having one extra result.
        valid_firsts = {page_start + 1}
        if page > 1:
            valid_firsts.add(page_start)
        if (
            page_size < 1
            or total < 1
            or first not in valid_firsts
            or last != expected_last
            or last < first
        ):
            raise ValueError(f"{self.name}: invalid displayed result range")

        project_urls: list[str] = []
        for href in parser.hrefs:
            parsed = urlsplit(href)
            if parsed.netloc and parsed.netloc != "universe.roboflow.com":
                continue
            match = _PROJECT_PATH.fullmatch(parsed.path)
            if match is None:
                continue
            project_url = f"{_ORIGIN}/{match.group('workspace')}/{match.group('project')}"
            if project_url not in project_urls:
                project_urls.append(project_url)
            if len(project_urls) > self.max_projects_per_page:
                raise ValueError(f"{self.name}: project count exceeds per-page limit")

        expected = min(page_size, total - page_start)
        if len(project_urls) != expected:
            raise ValueError(f"{self.name}: displayed result range does not match project links")
        records: list[SourceRecord] = []
        for project_url in project_urls:
            detail = self._get_html(project_url)
            detail_text = " ".join(_parse(detail.text(), self.max_anchors).text)
            model_match = _MODEL_ID.search(detail_text)
            if model_match is None:
                # Search results may include projects whose deployed model became
                # unavailable between the listing and detail request.
                continue
            model_id = model_match.group("id")
            model_type_match = _MODEL_TYPE.search(detail_text)
            model_type = model_type_match.group(1).strip() if model_type_match else None
            records.append(_record(project_url, model_id, model_type, page))

        complete = last == total
        next_state = {"page": page + 1} if not complete else {"page": page, "complete": True}
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=total,
            authoritative_snapshot=False,
        )

    def _get_html(self, url: str) -> HttpResponse:
        response: HttpResponse = self.client.get(url, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: public page returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: response exceeds byte limit")
        final_url = urlsplit(response.url or url)
        if final_url.scheme != "https" or final_url.netloc != "universe.roboflow.com":
            raise ValueError(f"{self.name}: response redirected outside Roboflow Universe")
        return response


def _record(project_url: str, model_id: str, model_type: str | None, page: int) -> SourceRecord:
    return SourceRecord(
        source_record_id=f"roboflow-universe:model:{model_id}",
        kind=ArtifactKind.PROVIDER_PAGE,
        canonical_url=canonicalize_url(project_url),
        title=model_id,
        raw={"provider": "roboflow-universe", "model_id": model_id, "model_type": model_type,
             "candidate_scope": "public-search-selected-deployment"},
        text=f"{model_id}" + (f"; model type: {model_type}" if model_type else ""),
        identifiers=(Identifier("roboflow:model", model_id),),
        links=(Link(project_url, relation="documents_model", crawl=False),),
        models=(ModelHint(
            local_id=model_id,
            name=model_id,
            identifiers=(Identifier("roboflow:model", model_id),),
            status=ModelStatus.DOCUMENTED,
            locator=f"search:page[{page}] selected deployment",
        ),),
    )


def _parse(html: str, max_anchors: int) -> _LinksAndText:
    parser = _LinksAndText(max_anchors)
    parser.feed(html)
    parser.close()
    return parser


def _number(value: str) -> int:
    return int(value.replace(",", ""))


def _positive_state_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} state must be a positive integer")
    return value


__all__ = ["RoboflowUniverseCandidatesAdapter"]
