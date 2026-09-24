"""Argus embodied-robot policy inventory published with its Dryad release."""

from __future__ import annotations

import re
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

_PAGE_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.3j9kd520k"
_ARCHIVE_URL = "https://datadryad.org/downloads/file_stream/4802810"
_REPOSITORY_URL = "https://github.com/generalroboticslab/Argus"
_DOI = "10.5061/dryad.3j9kd520k"
_PATH = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.pt$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _InventoryParser(HTMLParser):
    """Extract `.pt` rows from the release's checkpoint table and its archive link."""

    def __init__(self, maximum: int) -> None:
        super().__init__(convert_charrefs=True)
        self.maximum = maximum
        self.rows: list[tuple[str, ...]] = []
        self.archive_href: str | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {key.casefold(): value or "" for key, value in attrs}
        tag = tag.casefold()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "a":
            self._anchor_href = attrs_map.get("href", "")
            self._anchor_text = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)
        if self._anchor_href is not None:
            self._anchor_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "a" and self._anchor_href is not None:
            text = " ".join("".join(self._anchor_text).split())
            href = self._anchor_href
            if text == "Argus.zip" and href:
                self.archive_href = href
            self._anchor_href = None
            self._anchor_text = []
        elif tag in {"td", "th"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if len(self._row) >= 3 and self._row[0].endswith(".pt"):
                self.rows.append(tuple(self._row))
                if len(self.rows) > self.maximum:
                    raise ValueError(f"Argus registry exceeds {self.maximum} checkpoint rows")
            self._row = None


class ArgusCheckpointInventorySourceAdapter:
    """Enumerate checkpoint members declared by the Argus Dryad release.

    Dryad publishes the checkpoints in one ZIP archive, not as individual
    downloadable files. Each model record therefore preserves its exact archive
    member path and links to the single exact archive URL; it does not fabricate
    per-member download URLs or fetch checkpoint bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only PyTorch checkpoint paths listed in the Argus Dryad release's "
        "checkpoint table. All members are distributed inside one ZIP archive; "
        "the adapter does not inspect archive bytes, infer omitted models, or "
        "download weights."
    )

    def __init__(
        self,
        *,
        name: str = "argus-robotics-checkpoints",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 100,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        if isinstance(max_response_bytes, bool) or max_response_bytes < 1:
            raise ValueError("max_response_bytes must be a positive integer")
        if isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "argus-checkpoint-inventory-v1",
                "page_url": _PAGE_URL,
                "archive_url": _ARCHIVE_URL,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
            }
        )

    @property
    def repository_url(self) -> str:
        return _REPOSITORY_URL

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            _PAGE_URL,
            headers={"Accept": "text/html"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: Dryad page returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: Dryad page exceeds {self.max_response_bytes} bytes")
        parser = _InventoryParser(self.max_entries)
        parser.feed(response.text())
        parser.close()
        if not parser.rows:
            raise ValueError(f"{self.name}: Dryad page contains no checkpoint rows")
        if parser.archive_href:
            archive_url = _absolute_https(urljoin(_PAGE_URL, parser.archive_href), self.name)
            if archive_url != _ARCHIVE_URL:
                raise ValueError(f"{self.name}: Argus.zip link changed unexpectedly")
        else:
            raise ValueError(f"{self.name}: Dryad page has no Argus.zip download link")

        checked_at = _isoformat(self.clock())
        records = tuple(self._record(row, response.body) for row in parser.rows)
        digest = content_hash(response.body)
        return SourcePage(
            records=records,
            next_state={
                "checked_at": checked_at,
                "page_sha256": digest,
                "model_count": len(records),
                "archive_url": _ARCHIVE_URL,
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, cells: tuple[str, ...], page_body: bytes) -> SourceRecord:
        member, variant, task = cells[:3]
        notes = cells[3] if len(cells) > 3 else ""
        if not _PATH.fullmatch(member):
            raise ValueError(f"{self.name}: invalid checkpoint member path {member!r}")
        identity = Identifier("argus:checkpoint", member)
        local_id = f"model:{member}"
        locator = f"dryad:checkpoint-table/{member}"
        model = ModelHint(
            local_id=local_id,
            name=member,
            identifiers=(identity,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        metadata = {
            "dataset_doi": _DOI,
            "archive_url": _ARCHIVE_URL,
            "archive_member_path": member,
            "robot_variant": variant,
            "task": task,
            "notes": notes,
        }
        release = ReleaseHint(
            local_id=f"release:{member}",
            model_local_id=local_id,
            identifiers=(Identifier("argus:checkpoint-release", member),),
            metadata=metadata,
            locator=locator,
        )
        links = (
            Link(_PAGE_URL, relation="model_card", locator=locator, crawl=False),
            Link(_REPOSITORY_URL, relation="source_implementation", crawl=False),
            Link(
                _ARCHIVE_URL,
                relation="model_artifact",
                locator=locator,
                crawl=False,
                model_local_ids=(local_id,),
            ),
        )
        return SourceRecord(
            source_record_id=f"argus:{member}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(_PAGE_URL),
            title=f"Argus {task}: {member}",
            raw=metadata | {"dryad_page_sha256": content_hash(page_body)},
            text="\n".join(part for part in (member, variant, task, notes) if part),
            identifiers=(identity,),
            links=links,
            models=(model,),
            releases=(release,),
        )


def _absolute_https(value: str, source: str) -> str:
    value = value.strip()
    parts = urlsplit(value)
    if parts.scheme != "https" or parts.netloc != "datadryad.org":
        raise ValueError(f"{source}: unexpected Argus.zip link")
    return value


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
