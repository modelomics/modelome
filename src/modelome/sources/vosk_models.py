"""Index Vosk's first-party downloadable speech-model table."""

from __future__ import annotations

from collections.abc import Mapping
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

_INDEX_URL = "https://alphacephei.com/vosk/models"
_ARCHIVE_ROOT = "https://alphacephei.com/vosk/models/"
_REPOSITORY_URL = "https://github.com/alphacep/vosk-api"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class VoskModelsSourceAdapter:
    """Read all versioned ZIP archives listed in Vosk's official model tables."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only downloadable .zip model rows present on Vosk's official models "
        "page, including ASR, speaker-identification, and punctuation archives. "
        "It excludes models mentioned only in prose or linked external catalogs."
    )

    def __init__(
        self,
        *,
        name: str = "vosk-models",
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
        min_models: int = 20,
        max_models: int = 500,
        max_response_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        for label, value in (
            ("min_models", min_models),
            ("max_models", max_models),
            ("max_response_bytes", max_response_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if min_models > max_models:
            raise ValueError("min_models cannot exceed max_models")
        self.name = name.strip()
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.min_models = min_models
        self.max_models = max_models
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "vosk-model-table-v1",
                "index_url": _INDEX_URL,
                "archive_root": _ARCHIVE_ROOT,
                "min_models": min_models,
                "max_models": max_models,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(_INDEX_URL, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: model index returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: model index exceeds response limit")
        html = response.body.decode("utf-8", errors="strict")
        if "</html>" not in html.lower():
            raise ValueError(f"{self.name}: model index HTML is incomplete")
        digest = content_hash(response.body)
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if digest == state.get("index_sha256"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=state.get("model_count"),
            )
        rows = _parse_model_rows(html, self.name)
        if len(rows) < self.min_models:
            raise ValueError(
                f"{self.name}: model index exposed only {len(rows)} archives; "
                f"expected at least {self.min_models}"
            )
        if len(rows) > self.max_models:
            raise ValueError(f"{self.name}: model index exceeds {self.max_models} archive limit")
        records = tuple(self._record(row, digest) for row in rows)
        return SourcePage(
            records=records,
            next_state={
                "index_sha256": digest,
                "checked_at": checked_at,
                "model_count": len(records),
                "source_url": _INDEX_URL,
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, row: _VoskRow, digest: str) -> SourceRecord:
        filename = urlsplit(row.url).path.rsplit("/", 1)[-1]
        slug = filename.removesuffix(".zip")
        category = _category(slug)
        model_id = f"model:{slug}"
        identifier = Identifier("vosk:model", slug)
        model = ModelHint(
            local_id=model_id,
            name=slug,
            aliases=() if row.label == slug else (row.label,),
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=filename,
        )
        release = ReleaseHint(
            local_id=f"release:{slug}",
            model_local_id=model_id,
            identifiers=(Identifier("vosk:model-archive", filename),),
            metadata={
                "archive_url": row.url,
                "archive_filename": filename,
                "category": category,
                "index_row_text": row.text,
                "index_sha256": digest,
            },
            locator=filename,
        )
        return SourceRecord(
            source_record_id=f"vosk:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(row.url),
            title=slug,
            raw={
                "archive_url": row.url,
                "archive_filename": filename,
                "category": category,
                "index_row_text": row.text,
            },
            text=f"Vosk {category.replace('_', ' ')} model archive: {row.text}",
            identifiers=(identifier,),
            links=(
                Link(_INDEX_URL, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(
                    _REPOSITORY_URL,
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
                Link(row.url, "weights", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


class _VoskRowParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[_VoskRow] = []
        self._in_row = False
        self._in_cell = False
        self._cell_text: list[str] = []
        self._cells: list[str] = []
        self._links: list[tuple[str, str]] = []
        self._active_link: str | None = None
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "tr":
            self._in_row = True
            self._cells = []
            self._links = []
        elif self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_text = []
        elif self._in_cell and tag == "a":
            href = attributes.get("href")
            self._active_link = href if isinstance(href, str) else None
            self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_link is not None:
            self._links.append((self._active_link, "".join(self._link_text).strip()))
            self._active_link = None
        elif tag in {"td", "th"} and self._in_cell:
            self._cells.append(" ".join("".join(self._cell_text).split()))
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            row_text = " | ".join(cell for cell in self._cells if cell)
            for href, label in self._links:
                url = urljoin(_INDEX_URL, href)
                if urlsplit(url).path.lower().endswith(".zip"):
                    self.rows.append(_VoskRow(url=url, label=label, text=row_text))
            self._in_row = False
            self._in_cell = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text.append(data)
        if self._active_link is not None:
            self._link_text.append(data)


class _VoskRow:
    __slots__ = ("url", "label", "text")

    def __init__(self, *, url: str, label: str, text: str) -> None:
        self.url = url
        self.label = label
        self.text = text


def _parse_model_rows(html: str, source: str) -> tuple[_VoskRow, ...]:
    parser = _VoskRowParser()
    parser.feed(html)
    parser.close()
    rows = parser.rows
    seen_urls: set[str] = set()
    seen_slugs: set[str] = set()
    for row in rows:
        _validate_archive_url(row.url, source)
        slug = urlsplit(row.url).path.rsplit("/", 1)[-1].removesuffix(".zip")
        if slug in seen_slugs or row.url in seen_urls:
            raise ValueError(f"{source}: duplicate Vosk model archive {slug}")
        seen_slugs.add(slug)
        seen_urls.add(row.url)
    return tuple(rows)


def _validate_archive_url(url: str, source: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "alphacephei.com"
        or parsed.path.split("/")[:3] != ["", "vosk", "models"]
        or not parsed.path.lower().endswith(".zip")
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"{source}: invalid Vosk model archive URL")


def _category(slug: str) -> str:
    if slug.startswith("vosk-recasepunc-"):
        return "punctuation_and_case_restoration"
    if slug.startswith("vosk-model-spk-"):
        return "speaker_identification"
    return "automatic_speech_recognition"


__all__ = ["VoskModelsSourceAdapter"]
