from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from html.parser import HTMLParser
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_ORIGIN = "https://ollama.com"
_LIBRARY = re.compile(r"^/library/(?P<slug>[A-Za-z0-9][A-Za-z0-9._-]{0,180})$")
_TAG_PATH = re.compile(r"^/library/(?P<tag>[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255})$")
_SIZE = re.compile(r"\b(?P<size>\d+(?:\.\d+)?\s*[KMGT]B)\b", re.IGNORECASE)
_DIGEST = re.compile(r"\b[a-f0-9]{12,64}\b", re.IGNORECASE)
_NEXT_TEXT = re.compile(r"^(?:next|next page|load more)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _Anchor:
    href: str
    text: str
    title: str
    aria_label: str
    rel: str


class _AnchorParser(HTMLParser):
    def __init__(self, *, max_anchors: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_anchors = max_anchors
        self.anchors: list[_Anchor] = []
        self._href = ""
        self._title = ""
        self._aria_label = ""
        self._rel = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        if self._href:
            self._finish_anchor()
        values = {key.casefold(): value or "" for key, value in attrs}
        self._href = values.get("href", "")
        self._title = values.get("title", "")
        self._aria_label = values.get("aria-label", "")
        self._rel = values.get("rel", "")
        self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a" and self._href:
            self._finish_anchor()

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        if self._href:
            self._finish_anchor()

    def _finish_anchor(self) -> None:
        if len(self.anchors) >= self.max_anchors:
            raise ValueError("Ollama library page exceeds anchor limit")
        self.anchors.append(
            _Anchor(
                self._href,
                " ".join(" ".join(self._text).split()),
                self._title,
                self._aria_label,
                self._rel,
            )
        )
        self._href = ""
        self._title = self._aria_label = self._rel = ""
        self._text = []


class OllamaLibraryTagCatalogAdapter:
    """Enumerate exact publicly listed Ollama tags from first-party library pages.

    It follows the library's family cards, each card's declared “View all” tag
    link, tag links found on those pages, and explicit next-page links. It never
    requests model manifests, config, layers, or weights.
    """

    def __init__(
        self,
        *,
        name: str = "ollama-library-tags",
        url: str = "https://ollama.com/library",
        client: HttpClient | Any | None = None,
        max_response_bytes: int = 8 * 1024 * 1024,
        max_families: int = 5_000,
        max_tags_per_family: int = 10_000,
        max_pages_per_family: int = 100,
        max_index_pages: int = 100,
        max_anchors_per_page: int = 100_000,
    ) -> None:
        self.name = name
        self.url = canonicalize_url(url)
        if self.url != f"{_ORIGIN}/library":
            raise ValueError("Ollama library URL must be the first-party library index")
        limits = (
            max_response_bytes,
            max_families,
            max_tags_per_family,
            max_pages_per_family,
            max_index_pages,
            max_anchors_per_page,
        )
        if any(value < 1 for value in limits):
            raise ValueError("Ollama library adapter limits must be positive")
        self.max_response_bytes = max_response_bytes
        self.max_families = max_families
        self.max_tags_per_family = max_tags_per_family
        self.max_pages_per_family = max_pages_per_family
        self.max_index_pages = max_index_pages
        self.max_anchors_per_page = max_anchors_per_page
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if state:
            raise ValueError(f"{self.name}: this catalog does not accept pagination state")
        index_anchors: list[tuple[_Anchor, str]] = []
        revision_material: list[str] = []
        visited_index_pages: set[str] = set()
        index_url: str | None = self.url
        while index_url:
            if index_url in visited_index_pages:
                raise ValueError(f"{self.name}: library index pagination repeated")
            visited_index_pages.add(index_url)
            if len(visited_index_pages) > self.max_index_pages:
                raise ValueError(f"{self.name}: library index pages exceed limit")
            index_response = self._get(index_url)
            revision_material.append(content_hash(index_response.body))
            page_anchors = self._parse(index_response.text())
            base_url = index_response.url or index_url
            index_anchors.extend((anchor, base_url) for anchor in page_anchors)
            next_links = [anchor for anchor in page_anchors if _is_next(anchor)]
            if next_links:
                next_urls = {
                    self._safe_url(anchor.href, index_response.url or index_url)
                    for anchor in next_links
                }
                if None in next_urls:
                    raise ValueError(
                        f"{self.name}: library index declares an unusable next-page link"
                    )
                if len(next_urls) != 1:
                    raise ValueError(
                        f"{self.name}: library index declares conflicting next-page links"
                    )
                index_url = next_urls.pop()
            else:
                index_url = None
        family_pages: dict[str, str] = {}
        for anchor, base_url in index_anchors:
            url = self._safe_url(anchor.href, base_url)
            if not url:
                continue
            slug = _family_slug(url)
            if slug:
                family_pages.setdefault(slug, url)
        if not family_pages:
            raise ValueError(f"{self.name}: no first-party family-card links found")
        if len(family_pages) > self.max_families:
            raise ValueError(f"{self.name}: library exceeds {self.max_families} families")

        records: list[SourceRecord] = []
        for slug, family_url in sorted(family_pages.items()):
            family_response = self._get(family_url)
            revision_material.append(content_hash(family_response.body))
            family_anchors = self._parse(family_response.text())
            tags_url = self._declared_tags_url(
                slug, family_anchors, family_response.url or family_url
            )
            tags: dict[str, dict[str, Any]] = {}
            visited_pages: set[str] = set()
            page_url: str | None = tags_url
            pages_read = 0
            while page_url:
                if page_url in visited_pages:
                    raise ValueError(f"{self.name}: tag pagination repeated for {slug}")
                visited_pages.add(page_url)
                pages_read += 1
                if pages_read > self.max_pages_per_family:
                    raise ValueError(f"{self.name}: tag pages exceed limit for {slug}")
                tags_response = self._get(page_url)
                revision_material.append(content_hash(tags_response.body))
                tag_anchors = self._parse(tags_response.text())
                next_page: str | None = None
                for anchor in tag_anchors:
                    tag_url = self._safe_url(anchor.href, tags_response.url or page_url)
                    tag_id = _tag_id(tag_url, slug) if tag_url else None
                    if tag_id:
                        title = anchor.text or anchor.aria_label or anchor.title or tag_id
                        evidence_text = " ".join(
                            part for part in (anchor.text, anchor.aria_label, anchor.title) if part
                        )
                        tags.setdefault(
                            tag_id,
                            {
                                "url": tag_url,
                                "title": title,
                                "evidence_text": evidence_text,
                                **_tag_metadata(evidence_text),
                            },
                        )
                    if _is_next(anchor):
                        candidate = self._safe_url(anchor.href, tags_response.url or page_url)
                        if candidate is None:
                            raise ValueError(
                                f"{self.name}: tag page declares an unusable next-page link "
                                f"for {slug}"
                            )
                        if next_page is not None and candidate != next_page:
                            raise ValueError(
                                f"{self.name}: tag page declares conflicting next-page links "
                                f"for {slug}"
                            )
                        next_page = candidate
                page_url = next_page
                if len(tags) > self.max_tags_per_family:
                    raise ValueError(f"{self.name}: tags for {slug} exceed configured limit")
            if not tags:
                raise ValueError(f"{self.name}: no exact tag links found for family {slug}")
            records.append(self._record(slug, family_url, tags))

        revision = content_hash({"page_hashes": revision_material})
        # Bind a single complete retrieval revision to every family record.
        records = [
            _with_revision(record, revision)
            for record in records
        ]
        return SourcePage(
            records=tuple(records),
            next_state={"entry_count": len(records), "content_hash": revision},
            complete=True,
            upstream_count=sum(len(record.releases) for record in records),
            authoritative_snapshot=True,
        )

    def _get(self, url: str) -> HttpResponse:
        response: HttpResponse = self.client.get(url, headers={"Accept": "text/html"})
        if not 200 <= response.status < 300:
            raise ValueError(f"{self.name}: library page returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: library page exceeds {self.max_response_bytes} bytes")
        final_url = self._safe_url(response.url or url, url)
        if final_url != url:
            raise ValueError(f"{self.name}: library page redirected to a different URL")
        return response

    def _parse(self, body: str) -> tuple[_Anchor, ...]:
        parser = _AnchorParser(max_anchors=self.max_anchors_per_page)
        parser.feed(body)
        parser.close()
        return tuple(parser.anchors)

    def _safe_url(self, href: str, base_url: str) -> str | None:
        if not href:
            return None
        absolute = canonicalize_url(urljoin(base_url, href))
        parsed = urlsplit(absolute)
        if parsed.scheme != "https" or parsed.netloc != "ollama.com":
            return None
        return absolute

    def _declared_tags_url(
        self, slug: str, anchors: tuple[_Anchor, ...], base_url: str
    ) -> str:
        candidates = []
        for anchor in anchors:
            url = self._safe_url(anchor.href, base_url)
            if url and _is_tags_page(url, slug):
                candidates.append(url)
        if not candidates:
            raise ValueError(f"{self.name}: family page did not declare its tag-list link")
        return candidates[0]

    def _record(
        self, slug: str, family_url: str, tags: Mapping[str, Mapping[str, Any]]
    ) -> SourceRecord:
        model_local_id = f"ollama-family:{slug}#model"
        family_identifier = Identifier("ollama:model-family", slug)
        releases = []
        links = [
            Link(
                family_url,
                relation="model_family",
                crawl=False,
                model_local_ids=(model_local_id,),
            )
        ]
        raw_tags = []
        for tag_id, item in sorted(tags.items()):
            suffix = tag_id[len(slug) + 1 :]
            identifier = Identifier("ollama:model-tag", tag_id)
            metadata = {
                key: item[key]
                for key in ("size_label", "size_bytes", "digest", "context_window")
                if item.get(key) is not None
            }
            metadata["tag_url"] = item["url"]
            metadata["catalog_label"] = item["evidence_text"]
            releases.append(
                ReleaseHint(
                    local_id=f"{model_local_id}:tag:{suffix}",
                    model_local_id=model_local_id,
                    version=suffix,
                    identifiers=(identifier,),
                    metadata=metadata,
                    locator=f"tag-link:{tag_id}",
                )
            )
            links.append(
                Link(
                    str(item["url"]),
                    relation="model_tag",
                    crawl=False,
                    model_local_ids=(model_local_id,),
                )
            )
            raw_tags.append({"id": tag_id, **metadata})
        return SourceRecord(
            source_record_id=f"family:{slug}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=family_url,
            title=slug,
            raw={"family_id": slug, "tags": raw_tags},
            text="\n".join(f"{tag['id']} {tag.get('size_label', '')}" for tag in raw_tags),
            identifiers=(family_identifier,),
            links=tuple(links),
            models=(
                ModelHint(
                    local_id=model_local_id,
                    name=slug,
                    identifiers=(family_identifier,),
                    status=ModelStatus.RELEASED,
                    locator="library-family-card",
                ),
            ),
            releases=tuple(releases),
        )


def _family_slug(url: str) -> str | None:
    match = _LIBRARY.fullmatch(urlsplit(url).path)
    return match.group("slug") if match else None


def _is_tags_page(url: str, slug: str) -> bool:
    return urlsplit(url).path == f"/library/{slug}/tags"


def _tag_id(url: str, family_slug: str) -> str | None:
    path = unquote(urlsplit(url).path)
    match = _TAG_PATH.fullmatch(path)
    if match is None:
        return None
    tag_id = match.group("tag")
    if not tag_id.startswith(family_slug + ":"):
        return None
    return tag_id


def _is_next(anchor: _Anchor) -> bool:
    return "next" in anchor.rel.casefold().split() or _NEXT_TEXT.fullmatch(
        anchor.aria_label or anchor.text
    ) is not None


def _tag_metadata(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    size_match = _SIZE.search(text)
    if size_match:
        size_label = re.sub(r"\s+", "", size_match.group("size")).upper()
        result["size_label"] = size_label
        number, unit = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]B)", size_label).groups()
        multiplier = {"KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}[unit]
        result["size_bytes"] = int(Decimal(number) * multiplier)
    digest = _DIGEST.search(text)
    if digest:
        result["digest"] = digest.group(0).lower()
    context = re.search(r"\b(?P<context>\d+(?:\.\d+)?\s*[KMG])\s+context window", text, re.I)
    if context:
        result["context_window"] = re.sub(r"\s+", "", context.group("context")).upper()
    return result


def _with_revision(record: SourceRecord, revision: str) -> SourceRecord:
    return SourceRecord(
        source_record_id=record.source_record_id,
        kind=record.kind,
        canonical_url=record.canonical_url,
        title=record.title,
        raw={**dict(record.raw), "catalog_revision_sha256": revision},
        text=record.text,
        published_at=record.published_at,
        modified_at=record.modified_at,
        identifiers=record.identifiers,
        links=record.links,
        models=record.models,
        model_relations=record.model_relations,
        releases=record.releases,
        deleted=record.deleted,
    )
