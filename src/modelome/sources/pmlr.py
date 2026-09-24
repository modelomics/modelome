from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _PmlrParser(HTMLParser):
    """Keep text and anchors in document order for the generated PMLR pages."""

    def __init__(self, *, max_anchors: int, max_text_chars: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_anchors = max_anchors
        self.max_text_chars = max_text_chars
        self.events: list[tuple[str, str, str]] = []
        self._anchor: tuple[str, list[str]] | None = None
        self._text_chars = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "a":
            if self._anchor is not None:
                self._finish_anchor()
            values = {key.casefold(): value or "" for key, value in attrs}
            self._anchor = (values.get("href", ""), [])

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a" and self._anchor is not None:
            self._finish_anchor()
        elif tag.casefold() in {"p", "div", "h1", "h2", "h3", "li", "br"}:
            self.events.append(("text", "\n", ""))

    def handle_data(self, data: str) -> None:
        if self._anchor is not None:
            self._anchor[1].append(data)
        else:
            self._text_chars += len(data)
            if self._text_chars > self.max_text_chars:
                raise ValueError("PMLR page text exceeds configured limit")
            self.events.append(("text", data, ""))

    def _finish_anchor(self) -> None:
        assert self._anchor is not None
        href, parts = self._anchor
        self._anchor = None
        if len(self.events) >= self.max_anchors:
            raise ValueError("PMLR page exceeds configured anchor limit")
        self.events.append(("anchor", " ".join(parts).strip(), href))


class PmlrSourceAdapter:
    """Traverse the first-party PMLR volume index and each volume listing.

    Each checkpoint holds the complete volume manifest and advances one volume
    page at a time. Paper links are taken only from the corresponding PMLR row;
    supplements remain generic references and are never treated as weights.
    """

    def __init__(
        self,
        *,
        name: str = "pmlr",
        index_url: str = "https://proceedings.mlr.press/",
        max_response_bytes: int = 16 * 1024 * 1024,
        max_volumes: int = 2000,
        max_papers_per_volume: int = 5000,
        client: HttpClient | Any | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.name = name.strip()
        if not self.name:
            raise ValueError("source name must not be empty")
        if not _https(index_url):
            raise ValueError("index_url must be an HTTPS URL")
        self.index_url = index_url
        self.max_response_bytes = _positive(max_response_bytes, "max_response_bytes")
        self.max_volumes = _positive(max_volumes, "max_volumes")
        self.max_papers_per_volume = _positive(max_papers_per_volume, "max_papers_per_volume")
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "pmlr-volume-index-v1",
                "index_url": self.index_url,
                "max_response_bytes": max_response_bytes,
                "max_volumes": max_volumes,
                "max_papers_per_volume": max_papers_per_volume,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        volumes = state.get("volumes")
        if volumes is None:
            response = self.client.get(self.index_url, headers={"Accept": "text/html"})
            body = _body(response, self.max_response_bytes, "index")
            parser = _parse(body, max_anchors=100_000)
            volumes = []
            for kind, label, href in parser.events:
                if kind != "anchor":
                    continue
                # The live index also lists its separate reissue series at /rN.
                match = re.fullmatch(r"/([vr]\d+)/?", urlsplit(urljoin(self.index_url, href)).path)
                if match:
                    volume_id = match.group(1)
                    volumes.append(
                        {
                            "id": volume_id,
                            "url": f"{self.index_url.rstrip('/')}/{volume_id}/",
                            "title": label,
                        }
                    )
            if not volumes or len(volumes) > self.max_volumes:
                raise ValueError(
                    f"{self.name}: index volume count is empty or exceeds configured limit"
                )
            if len({volume["id"] for volume in volumes}) != len(volumes):
                raise ValueError(f"{self.name}: index repeats a volume ID")
            state = {"volumes": volumes, "index_digest": content_hash(volumes), "volume_index": 0}
        else:
            volumes = _validated_volumes(volumes, self.max_volumes, self.name)
        index = state.get("volume_index", 0)
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= len(volumes):
            raise ValueError(f"{self.name}: invalid volume_index checkpoint")
        if index == len(volumes):
            return SourcePage(
                records=(),
                next_state={"completed_index_digest": state.get("index_digest")},
                complete=True,
                upstream_count=state.get("paper_count"),
            )
        volume = volumes[index]
        response = self.client.get(volume["url"], headers={"Accept": "text/html"})
        parser = _parse(
            _body(response, self.max_response_bytes, f"volume {volume['id']}"), max_anchors=100_000
        )
        records = _records(parser.events, volume, self.name, self.max_papers_per_volume)
        next_state = {
            "volumes": volumes,
            "index_digest": state.get("index_digest"),
            "volume_index": index + 1,
            "paper_count": state.get("paper_count", 0) + len(records),
        }
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=index + 1 == len(volumes),
            upstream_count=next_state["paper_count"] if index + 1 == len(volumes) else None,
        )


def _records(
    events: list[tuple[str, str, str]], volume: Mapping[str, str], source: str, limit: int
) -> list[SourceRecord]:
    parsed: list[SourceRecord] = []
    text_before_abs: list[tuple[str, str, str]] = []
    active_id: str | None = None
    active_title = ""
    active_links: list[tuple[str, str, str]] = []
    for event in events:
        if event[0] == "anchor" and event[1].casefold() == "abs":
            if active_id is not None:
                record = _record(active_id, active_title, active_links, volume)
                if record is not None:
                    parsed.append(record)
                    if len(parsed) > limit:
                        raise ValueError(f"{source}: volume exceeds max_papers_per_volume {limit}")
            active_id = event[2]
            title_lines = [
                value.strip()
                for kind, value, _ in text_before_abs
                if kind == "text"
                and value.strip()
                and "proceedings of" not in value.casefold()
                and not value.strip().casefold().startswith(("editors:", "series editors:"))
            ]
            active_title = title_lines[-1] if title_lines else ""
            text_before_abs = []
            active_links = []
        else:
            if event[0] == "anchor":
                active_links.append(event)
            else:
                text_before_abs.append(event)
    if active_id is not None:
        record = _record(active_id, active_title, active_links, volume)
        if record is not None:
            parsed.append(record)
    if len(parsed) > limit:
        raise ValueError(f"{source}: volume exceeds max_papers_per_volume {limit}")
    return parsed


def _record(
    abs_href: str, title: str, anchors: list[tuple[str, str, str]], volume: Mapping[str, str]
) -> SourceRecord | None:
    url = urljoin(volume["url"], abs_href)
    path = urlsplit(url).path
    match = re.fullmatch(r"/(v\d+|r\d+)/([A-Za-z0-9_-]+)\.html", path)
    if not match or match.group(1) != volume["id"]:
        return None
    paper_id = match.group(2)
    if not title:
        return None
    raw_links: list[tuple[str, str]] = []
    for kind, label, href in anchors:
        if kind != "anchor" or not href:
            continue
        lowered = label.casefold()
        if lowered in {"download pdf", "pdf"}:
            relation = "full_text"
        elif "supplement" in lowered:
            relation = "supplementary_material"
        elif lowered == "code":
            relation = "implementation"
        else:
            continue
        raw_links.append((relation, urljoin(volume["url"], href)))
    canonical = canonicalize_url(url)
    links = tuple(
        Link(url=u, relation=relation, crawl=(relation == "implementation"))
        for relation, u in raw_links
        if _https(u)
    )
    return SourceRecord(
        source_record_id=f"{volume['id']}/{paper_id}",
        kind=ArtifactKind.PAPER,
        canonical_url=canonical,
        title=title,
        raw={
            "volume_id": volume["id"],
            "paper_id": paper_id,
            "volume_title": volume.get("title", ""),
            "declared_links": [{"relation": rel, "url": href} for rel, href in raw_links],
        },
        identifiers=(Identifier("pmlr", f"{volume['id']}/{paper_id}"),),
        links=links,
    )


def _parse(body: bytes, *, max_anchors: int) -> _PmlrParser:
    parser = _PmlrParser(max_anchors=max_anchors, max_text_chars=4_000_000)
    parser.feed(body.decode("utf-8", errors="replace"))
    parser.close()
    if parser._anchor is not None:
        parser._finish_anchor()
    return parser


def _body(response: HttpResponse, maximum: int, label: str) -> bytes:
    if response.status != 200:
        raise ValueError(f"PMLR {label} returned HTTP {response.status}")
    if len(response.body) > maximum:
        raise ValueError(f"PMLR {label} exceeds max_response_bytes {maximum}")
    return response.body


def _validated_volumes(value: Any, maximum: int, source: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise ValueError(f"{source}: invalid frozen volume manifest")
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{source}: invalid volume checkpoint")
        vol_id, url = item.get("id"), item.get("url")
        if (
            not isinstance(vol_id, str)
            or not re.fullmatch(r"(?:v|r)?\d+", vol_id)
            or not isinstance(url, str)
            or not _https(url)
        ):
            raise ValueError(f"{source}: invalid volume checkpoint")
        if vol_id.isdigit():
            vol_id = f"v{vol_id}"
        result.append({"id": vol_id, "url": url, "title": str(item.get("title", ""))})
    return result


def _https(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "https" and bool(parts.netloc)


def _positive(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


__all__ = ["PmlrSourceAdapter"]
