"""Opt-in public JATS follow-up for supplementary model-resource links.

bioRxiv/medRxiv's details API exposes a per-record ``jatsxml`` URL. This
adapter composes with the ordinary details adapter and follows that URL for
each paper in a bounded page. It is deliberately not registered in the default
catalog because it adds one sequential HTTP request per eligible record.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Link, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash
from modelome.sources.biorxiv import BioRxivSourceAdapter

_ALLOWED_HOSTS = {
    "biorxiv": frozenset({"www.biorxiv.org", "biorxiv.org"}),
    "medrxiv": frozenset({"www.medrxiv.org", "medrxiv.org"}),
}
_RESOURCE_CUE_RE = re.compile(
    r"\b(?:pre[- ]?trained|trained|downloadable|released)\s+(?:deep[- ]learning\s+)?models?\b|"
    r"\b(?:model\s+)?(?:weights?|checkpoints?|parameters?)\b|"
    r"\bweights?\s+for\s+(?:the\s+)?(?:trained\s+)?models?\b",
    re.IGNORECASE,
)
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_XML_BASE = "{http://www.w3.org/XML/1998/namespace}base"
_NO_VALUE = frozenset({"", "na", "n/a", "none", "not available", "null"})


class BioRxivJatsSupplementSourceAdapter:
    """Fetch public JATS only for papers in one bounded details-API page.

    ``source`` must be the regular details adapter. The maximum JATS fetches is
    capped at the documented 30-record details response size. Each fetch is
    sequential and rate limited. A failed XML fetch raises before the wrapped
    metadata checkpoint advances, so the same API page can be retried.
    """

    def __init__(
        self,
        *,
        source: BioRxivSourceAdapter,
        client: HttpClient | Any | None = None,
        max_jats_fetches_per_page: int = 30,
        max_jats_bytes: int = 32 * 1024 * 1024,
        max_jats_elements: int = 250_000,
        max_supplement_links_per_record: int = 2_000,
        minimum_request_interval_seconds: float = 0.34,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(source, BioRxivSourceAdapter):
            raise TypeError("source must be a BioRxivSourceAdapter")
        self.source = source
        self.name = f"{source.name}-jats-supplementary"
        self.server = source.server
        self.max_jats_fetches_per_page = _positive(
            max_jats_fetches_per_page, "max_jats_fetches_per_page"
        )
        if self.max_jats_fetches_per_page > 30:
            raise ValueError("max_jats_fetches_per_page must not exceed 30")
        self.max_jats_bytes = _positive(max_jats_bytes, "max_jats_bytes")
        self.client = client or HttpClient(max_response_bytes=self.max_jats_bytes)
        client_body_limit = getattr(self.client, "max_response_bytes", None)
        if not isinstance(client_body_limit, int) or isinstance(client_body_limit, bool):
            raise ValueError("JATS HTTP client must expose its max_response_bytes limit")
        if client_body_limit > self.max_jats_bytes:
            raise ValueError("JATS HTTP client's max_response_bytes must not exceed max_jats_bytes")
        self.max_jats_elements = _positive(max_jats_elements, "max_jats_elements")
        self.max_supplement_links_per_record = _positive(
            max_supplement_links_per_record, "max_supplement_links_per_record"
        )
        if not 0.1 <= minimum_request_interval_seconds <= 60:
            raise ValueError("minimum_request_interval_seconds must be between 0.1 and 60")
        self.minimum_request_interval_seconds = float(minimum_request_interval_seconds)
        self.monotonic = monotonic
        self.sleep = sleep
        self.checkpoint_signature = content_hash(
            {
                "adapter": "biorxiv-public-jats-supplementary-v1",
                "metadata_source": source.checkpoint_signature,
                "max_jats_fetches_per_page": self.max_jats_fetches_per_page,
                "max_jats_bytes": self.max_jats_bytes,
                "max_jats_elements": self.max_jats_elements,
                "max_supplement_links_per_record": self.max_supplement_links_per_record,
                "minimum_request_interval_seconds": self.minimum_request_interval_seconds,
            }
        )
        self._next_request_at: float | None = None

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if state.get("complete") is True:
            return SourcePage(records=(), next_state=dict(state), complete=True, upstream_count=0)
        metadata_state = state.get("metadata_state", {})
        if not isinstance(metadata_state, Mapping):
            raise ValueError(f"{self.name}: metadata_state must be an object")
        self._rate_limit()
        page = self.source.fetch_page(metadata_state)
        if len(page.records) > self.max_jats_fetches_per_page:
            raise ValueError(
                f"{self.name}: metadata page has {len(page.records)} records, above the "
                f"JATS fetch budget of {self.max_jats_fetches_per_page}"
            )
        records: list[SourceRecord] = []
        for record in page.records:
            jats_url = _jats_url(record.raw.get("jatsxml"), self.server)
            if not jats_url:
                continue
            self._rate_limit()
            response: HttpResponse = self.client.get(
                jats_url,
                headers={"Accept": "application/xml, text/xml;q=0.9"},
                redirect_validator=lambda target: _jats_url(target, self.server),
            )
            article = _parse_article(response.body, self.name, self.max_jats_bytes)
            element_count = sum(1 for _ in article.iter())
            if element_count > self.max_jats_elements:
                raise ValueError(
                    f"{self.name}: JATS document exceeds {self.max_jats_elements} elements"
                )
            records.extend(
                _supplement_records(
                    record,
                    article,
                    jats_url,
                    server=self.server,
                    max_links=self.max_supplement_links_per_record,
                )
            )
        next_state = {"metadata_state": page.next_state, "complete": page.complete}
        retry_state = {"metadata_state": page.retry_state}
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=page.complete,
            upstream_count=page.upstream_count,
            issues=page.issues,
            retry_state=retry_state,
        )

    def _rate_limit(self) -> None:
        now = self.monotonic()
        if self._next_request_at is not None and now < self._next_request_at:
            self.sleep(self._next_request_at - now)
            now = self.monotonic()
        self._next_request_at = max(now, self._next_request_at or now) + (
            self.minimum_request_interval_seconds
        )


def _supplement_records(
    paper: SourceRecord,
    article: ET.Element,
    jats_url: str,
    *,
    server: str,
    max_links: int,
) -> tuple[SourceRecord, ...]:
    if _local_name(article.tag) != "article":
        raise ValueError("JATS full text root must be an article element")
    article_doi = next(
        (
            _normalize_doi(_element_text(node))
            for node in article.iter()
            if _local_name(node.tag) == "article-id"
            and _text(node.get("pub-id-type")).casefold() == "doi"
        ),
        "",
    )
    expected_doi = next((item.value for item in paper.identifiers if item.namespace == "doi"), "")
    if article_doi and article_doi != expected_doi:
        raise ValueError("JATS DOI does not match the metadata record DOI")
    links: list[Link] = []
    resources: list[dict[str, str]] = []
    supplement_nodes = [
        node
        for node in article.iter()
        if _local_name(node.tag) in {"supplementary-material", "inline-supplementary-material"}
    ]
    parents = {child: parent for parent in article.iter() for child in parent}
    seen_urls: set[str] = set()
    for supplement_index, node in enumerate(supplement_nodes):
        context = " ".join(" ".join(node.itertext()).split())
        if not _RESOURCE_CUE_RE.search(context):
            continue
        base_url = _element_base_url(node, jats_url, parents)
        hrefs = []
        for candidate in node.iter():
            href = _text(candidate.get(_XLINK_HREF) or candidate.get("href"))
            if href:
                hrefs.append(href)
        for href in dict.fromkeys(hrefs):
            url = _safe_resolved_url(href, base_url, server)
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            locator = f"jats.supplementary-material[{supplement_index}].{href[:240]}"
            resources.append({"url": url, "locator": locator})
            links.append(Link(url, relation="model_artifact", locator=locator, crawl=False))
            if len(links) > max_links:
                raise ValueError(f"supplementary model links exceed {max_links}")
    if not links:
        return ()
    paper_url = paper.canonical_url
    return (
        SourceRecord(
            source_record_id=f"{paper.source_record_id}:jats-model-resources",
            kind=ArtifactKind.PAPER,
            canonical_url=paper_url,
            title=paper.title,
            raw={
                "metadata_source_record_id": paper.source_record_id,
                "jatsxml": jats_url,
                "server": server,
                "supplementary_model_resources": resources,
            },
            identifiers=paper.identifiers,
            links=(
                Link(paper_url, relation="preprint", locator="metadata.jatsxml", crawl=False),
                *links,
            ),
        ),
    )


def _parse_article(body: Any, source: str, limit: int) -> ET.Element:
    if not isinstance(body, bytes):
        raise ValueError(f"{source}: JATS response body must be bytes")
    if len(body) > limit:
        raise ValueError(f"{source}: JATS response exceeds {limit} bytes")
    if re.search(rb"<!\s*(?:DOCTYPE|ENTITY)\b", body, flags=re.IGNORECASE):
        raise ValueError(f"{source}: unsafe XML declaration in JATS response")
    try:
        return ET.fromstring(body)
    except ET.ParseError as error:
        raise ValueError(f"{source}: malformed JATS XML: {error}") from None


def _jats_url(value: Any, server: str) -> str:
    text = _text(value)
    if text.casefold() in _NO_VALUE:
        return ""
    parts = urlsplit(text)
    if (
        parts.scheme not in {"http", "https"}
        or (parts.hostname or "").casefold() not in _ALLOWED_HOSTS[server]
        or not parts.path.casefold().endswith(".xml")
        or parts.username
        or parts.password
    ):
        raise ValueError("details API jatsxml URL is not a first-party source.xml URL")
    return canonicalize_url(text)


def _safe_resolved_url(href: str, base_url: str, server: str) -> str:
    if href.startswith("#"):
        return ""
    try:
        url = canonicalize_url(urljoin(base_url, href))
    except ValueError:
        return ""
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        return ""
    explicit_url = urlsplit(href).scheme.casefold() in {"http", "https"} or href.startswith("//")
    if not explicit_url and (parts.hostname or "").casefold() not in _ALLOWED_HOSTS[server]:
        return ""
    if parts.path.casefold().endswith(".source.xml") or canonicalize_url(base_url) == url:
        return ""
    return url


def _element_base_url(
    node: ET.Element,
    document_url: str,
    parents: Mapping[ET.Element, ET.Element],
) -> str:
    ancestry: list[ET.Element] = []
    current: ET.Element | None = node
    while current is not None:
        ancestry.append(current)
        current = parents.get(current)
    base = document_url
    for ancestor in reversed(ancestry):
        xml_base = _text(ancestor.get(_XML_BASE))
        if xml_base:
            base = urljoin(base, xml_base)
    return base


def _normalize_doi(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
    return text.casefold().rstrip(".,;:")


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _element_text(node: ET.Element) -> str:
    return " ".join("".join(node.itertext()).split())


def _positive(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a positive integer") from None
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
