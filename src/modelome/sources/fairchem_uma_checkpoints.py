"""FAIR Chemistry's UMA checkpoint file table and access-gated file links."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpClient
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
_MODEL_URL = "https://huggingface.co/facebook/UMA"
_FILE_BASE = "https://huggingface.co/facebook/UMA/blob/main/checkpoints/"
_CHECKPOINTS = {
    "uma-s-1.1": ("uma-s-1p1.pt", "36a2f071350be0ee4c15e7ebdd16dde1", False),
    "uma-s-1.2": ("uma-s-1p2.pt", "26ac47f57e7d68af9f031077cdc2cbe9", False),
    "uma-s-1.2.1": ("uma-s-1p2p1.pt", "3497615fd30a24c5b35cd3b41a682e6e", False),
    "uma-m-1.1": ("uma-m-1p1.pt", "8936aecf2eb101089af85934d3e881d6", False),
    "uma-s-1": ("uma-s-1.pt", "dc9964d66d54746652a352f74ead19b6", True),
}
_MD5 = re.compile(r"^[0-9a-f]{32}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _UmaRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, str, str]] = []
        self.page_text: list[str] = []
        self._row = False
        self._cell = False
        self._cell_text: list[str] = []
        self._cells: list[str] = []
        self._cell_links: list[list[str]] = []
        self._links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "tr":
            self._row = True
            self._cells = []
            self._cell_links = []
        elif self._row and tag in {"td", "th"}:
            self._cell = True
            self._cell_text = []
            self._links = []
        elif self._row and self._cell and tag == "a" and attributes.get("href"):
            self._links.append(attributes["href"] or "")

    def handle_data(self, data: str) -> None:
        self.page_text.append(data)
        if self._row and self._cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._row and tag in {"td", "th"} and self._cell:
            self._cells.append(" ".join(" ".join(self._cell_text).split()))
            self._cell_links.append(self._links)
            self._cell = False
        elif tag == "tr" and self._row:
            if len(self._cells) >= 3 and len(self._cell_links) >= 2:
                checkpoint_links = self._cell_links[1]
                if checkpoint_links:
                    self.rows.append(
                        (
                            self._cells[0],
                            checkpoint_links[0],
                            self._cells[2],
                        )
                    )
            self._row = False


class FairChemUMACheckpointSourceAdapter:
    """Parse the first-party UMA model card without downloading checkpoint bytes."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the four current UMA checkpoints and one archived UMA checkpoint in the "
        "first-party model card. Hugging Face requires agreement and contact information "
        "before file access; the adapter records exact file routes and checksums but does "
        "not verify or download model bytes."
    )

    def __init__(
        self,
        *,
        name: str = "fairchem-uma-checkpoints",
        page_url: str = _MODEL_URL,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if page_url != _MODEL_URL or not name.strip() or max_response_bytes <= 0:
            raise ValueError("page URL is fixed; name and positive response limit are required")
        self.name, self.page_url = name, page_url
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "fairchem-uma-checkpoints-v1",
                "page_url": page_url,
                "checkpoints": _CHECKPOINTS,
                "file_base": _FILE_BASE,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(self.page_url, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: source returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source exceeds {self.max_response_bytes} bytes")
        source_hash = content_hash(response.body)
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if source_hash == state.get("source_sha256"):
            return SourcePage(
                (), {**state, "checked_at": checked_at}, True,
                upstream_count=state.get("model_count"),
            )
        rows, page_text = _parse_rows(response.text(), self.name)
        if not re.search(
            r"agree to share your contact information to access this model", page_text,
            flags=re.IGNORECASE,
        ):
            raise ValueError(f"{self.name}: expected the documented Hugging Face access gate")
        records = tuple(self._record(handle, filename, checksum, archived, source_hash)
                        for handle, filename, checksum, archived in rows)
        return SourcePage(
            records,
            {
                "checked_at": checked_at,
                "source_url": self.page_url,
                "source_sha256": source_hash,
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self, handle: str, filename: str, checksum: str, archived: bool, source_hash: str
    ) -> SourceRecord:
        model_id = f"model:{handle}"
        namespace = "fairchem:uma-checkpoint"
        model = ModelHint(
            model_id,
            f"FAIR Chemistry {handle} checkpoint",
            identifiers=(Identifier(namespace, handle),),
            aliases=(filename,),
            status=ModelStatus.RELEASED,
        )
        file_url = f"{_FILE_BASE}{filename}"
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={
                "checkpoint_filename": filename,
                "checkpoint_url": file_url,
                "checksum_md5": checksum,
                "archived": archived,
                "access_url": self.page_url,
                "access_restricted": True,
                "binary_reachability_checked": False,
                "source_sha256": source_hash,
            },
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{handle}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(file_url),
            title=model.name,
            raw={
                "model_handle": handle,
                "checkpoint_filename": filename,
                "checkpoint_url": file_url,
                "checksum_md5": checksum,
                "archived": archived,
                "access_restricted": True,
            },
            text=(
                f"The FAIR Chemistry UMA model card lists {filename} with MD5 {checksum}. "
                "The Hub requires users to accept its access conditions before file access."
            ),
            links=(
                Link(file_url, "checkpoint", crawl=False, model_local_ids=(model_id,)),
                Link(self.page_url, "access_and_source", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_rows(document: str, source: str) -> tuple[tuple[tuple[str, str, str, bool], ...], str]:
    parser = _UmaRows()
    parser.feed(document)
    found: dict[str, tuple[str, str]] = {}
    for handle, url, checksum in parser.rows:
        if handle not in _CHECKPOINTS:
            raise ValueError(f"{source}: unexpected UMA checkpoint row {handle!r}")
        filename, expected_checksum, _ = _CHECKPOINTS[handle]
        expected_url = f"{_FILE_BASE}{filename}"
        if url != expected_url or checksum != expected_checksum or handle in found:
            raise ValueError(f"{source}: unexpected or duplicate checkpoint row {handle}")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
            raise ValueError(f"{source}: checkpoint URL must use HTTPS Hugging Face")
        if not _MD5.fullmatch(checksum):
            raise ValueError(f"{source}: invalid checkpoint MD5 for {handle}")
        found[handle] = (url, checksum)
    if set(found) != set(_CHECKPOINTS):
        raise ValueError(f"{source}: expected exactly the five UMA checkpoint rows")
    rows = tuple(
        (handle, filename, checksum, archived)
        for handle, (filename, checksum, archived) in _CHECKPOINTS.items()
    )
    return rows, " ".join(parser.page_text)


__all__ = ["FairChemUMACheckpointSourceAdapter"]
