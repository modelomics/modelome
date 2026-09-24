"""Bounded reader for Kaldi's first-party downloadable model index."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

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

Clock = Callable[[], datetime]
_RESOURCE = re.compile(r"^m(?P<number>[1-9][0-9]{0,3})$")
_ARCHIVE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,240}\.(?:tar\.gz|tar\.bz2|tar\.xz|tgz)$", re.I
)
_SAFE_LABEL = re.compile(r"^[^\x00-\x1f]{1,240}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _valid_resource_href(value: str) -> bool:
    parsed = urlsplit(value)
    path = parsed.path.rstrip("/")
    pieces = path.split("/")
    return (
        parsed.scheme in {"", "https"}
        and (parsed.hostname is None or parsed.hostname in {"kaldi-asr.org", "www.kaldi-asr.org"})
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and len(pieces) == 3
        and pieces[1] == "models"
        and _RESOURCE.fullmatch(pieces[-1]) is not None
    )


@dataclass(frozen=True, slots=True)
class _Resource:
    number: int
    name: str
    category: str
    summary: str
    url: str


@dataclass(frozen=True, slots=True)
class _Download:
    name: str
    url: str
    filename: str


class _IndexParser(HTMLParser):
    """Extract resource table rows, accepting only official model-detail paths."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.resources: list[_Resource] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []
        self._row_links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        if tag == "tr":
            self._row, self._row_links = [], []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "a" and self._row is not None:
            self._anchor_href = attrs_map.get("href")
            self._anchor_text = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        if self._anchor_href is not None:
            self._anchor_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor_href is not None:
            self._row_links.append((self._anchor_href, " ".join(self._anchor_text).strip()))
            self._anchor_href = None
            self._anchor_text = []
        elif tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join(" ".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if len(self._row) >= 4:
                resource_link = next(
                    (
                        (href, label)
                        for href, label in self._row_links
                        if _valid_resource_href(href)
                    ),
                    None,
                )
                if resource_link:
                    href, name = resource_link
                    number_match = _RESOURCE.fullmatch(
                        urlsplit(href).path.rstrip("/").split("/")[-1]
                    )
                    assert number_match is not None
                    self.resources.append(
                        _Resource(
                            int(number_match.group("number")),
                            name,
                            self._row[2],
                            self._row[3],
                            href,
                        )
                    )
            self._row = None
            self._cell = None
            self._row_links = []


class _DetailParser(HTMLParser):
    """Collect download anchors associated with their nearest h1/h2 heading."""

    def __init__(self, resource_number: int) -> None:
        super().__init__(convert_charrefs=True)
        self.resource_number = resource_number
        self.downloads: list[_Download] = []
        self.heading = ""
        self._heading_level: int | None = None
        self._heading_parts: list[str] = []
        self._anchor_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2"}:
            self._heading_level = int(tag[1])
            self._heading_parts = []
        elif tag == "a":
            self._anchor_href = dict(attrs).get("href")

    def handle_data(self, data: str) -> None:
        if self._heading_level is not None:
            self._heading_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2"} and self._heading_level == int(tag[1]):
            heading = " ".join(" ".join(self._heading_parts).split())
            if heading:
                self.heading = heading
            self._heading_level = None
            self._heading_parts = []
        elif tag == "a" and self._anchor_href is not None:
            absolute = urljoin("https://kaldi-asr.org/", self._anchor_href)
            parsed = urlsplit(absolute)
            prefix = f"/models/{self.resource_number}/"
            filename = parsed.path.removeprefix(prefix)
            if (
                parsed.scheme == "https"
                and parsed.hostname in {"kaldi-asr.org", "www.kaldi-asr.org"}
                and parsed.port is None
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and parsed.path.startswith(prefix)
                and "/" not in filename
                and _ARCHIVE.fullmatch(filename)
                and _SAFE_LABEL.fullmatch(self.heading)
            ):
                self.downloads.append(_Download(self.heading, absolute, filename))
            self._anchor_href = None


class KaldiModelIndexSourceAdapter:
    """Enumerate Kaldi's explicit M1.. model table and linked archive pages.

    The index promises downloadable .tar.gz models. Individual detail pages may
    additionally link .tar.bz2 or .tar.xz archives; only links under the matching
    `/models/<resource-id>/` path are admitted. Archive contents are not fetched.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only resources listed by Kaldi's official models.html and archive "
        "links listed on those resource pages. It does not crawl the separate older "
        "downloads tree or Kaldi recipe pages, and never downloads archive bytes."
    )

    def __init__(
        self,
        *,
        name: str = "kaldi-model-index",
        index_url: str = "https://www.kaldi-asr.org/models.html",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_resources: int = 100,
        max_archives: int = 2_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        parsed = urlsplit(index_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in {"kaldi-asr.org", "www.kaldi-asr.org"}
            or parsed.path != "/models.html"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("index_url must be the official Kaldi models.html index")
        limits = (
            ("max_response_bytes", max_response_bytes),
            ("max_resources", max_resources),
            ("max_archives", max_archives),
        )
        for field, value in limits:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        self.name = name.strip()
        self.index_url = index_url
        self.max_response_bytes = max_response_bytes
        self.max_resources = max_resources
        self.max_archives = max_archives
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "kaldi-model-index-v1",
                "index_url": index_url,
                "max_response_bytes": max_response_bytes,
                "max_resources": max_resources,
                "max_archives": max_archives,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        index_response = self._get(self.index_url)
        index = _IndexParser()
        index.feed(index_response.text())
        resources = index.resources
        if not resources:
            raise ValueError(f"{self.name}: official model index contains no resources")
        if len(resources) > self.max_resources:
            raise ValueError(f"{self.name}: model index exceeds {self.max_resources} resources")
        numbers = [resource.number for resource in resources]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f"{self.name}: duplicate resource identifiers")

        records: list[SourceRecord] = []
        seen_archives: set[tuple[int, str]] = set()
        for resource in resources:
            detail_url = f"https://www.kaldi-asr.org/models/m{resource.number}"
            detail_response = self._get(detail_url)
            detail = _DetailParser(resource.number)
            detail.feed(detail_response.text())
            if not detail.downloads:
                raise ValueError(f"{self.name}: resource M{resource.number} has no archive links")
            for ordinal, download in enumerate(detail.downloads, start=1):
                archive_key = (resource.number, download.filename)
                if archive_key in seen_archives:
                    raise ValueError(
                        f"{self.name}: duplicate archive in resource M{resource.number}"
                    )
                seen_archives.add(archive_key)
                records.append(self._record(resource, download, ordinal, detail_response.body))
                if len(records) > self.max_archives:
                    raise ValueError(f"{self.name}: archive list exceeds {self.max_archives} rows")
        return SourcePage(
            records=tuple(records),
            next_state={
                "checked_at": _isoformat(self.clock()),
                "index_sha256": content_hash(index_response.body),
                "resource_count": len(resources),
                "archive_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _get(self, url: str) -> HttpResponse:
        response: HttpResponse = self.client.get(url, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: page returned HTTP {response.status}: {url}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: page exceeds {self.max_response_bytes} bytes")
        return response

    def _record(
        self, resource: _Resource, download: _Download, ordinal: int, detail_body: bytes
    ) -> SourceRecord:
        handle = f"M{resource.number}/{download.filename}"
        local_id = f"model:{handle}"
        identifier = Identifier("kaldi:model-archive", handle)
        locator = f"models/m{resource.number}"
        detail_url = f"https://www.kaldi-asr.org/models/m{resource.number}"
        model = ModelHint(
            local_id=local_id,
            name=download.name,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{handle}",
            model_local_id=local_id,
            version=download.filename,
            identifiers=(Identifier("kaldi:archive", download.filename),),
            metadata={
                "resource_id": f"M{resource.number}",
                "resource_name": resource.name,
                "category": resource.category,
                "summary": resource.summary,
                "archive_filename": download.filename,
                "declared_download_url": download.url,
            },
            locator=locator,
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(detail_url),
            title=download.name,
            raw={
                "resource_id": f"M{resource.number}",
                "resource_name": resource.name,
                "category": resource.category,
                "summary": resource.summary,
                "archive_filename": download.filename,
                "detail_sha256": content_hash(detail_body),
                "archive_ordinal": ordinal,
            },
            text="\n".join((download.name, resource.name, resource.category, resource.summary)),
            identifiers=(identifier,),
            links=(
                Link(detail_url, relation="model_card", locator=locator, crawl=False),
                Link(download.url, relation="weights", locator=locator, crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
