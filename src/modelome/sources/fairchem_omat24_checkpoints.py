"""FAIR Chemistry's documented legacy OMat24 and MPTrj checkpoint files."""

from __future__ import annotations

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
_DOCS_URL = "https://facebookresearch.github.io/fairchem/models-2/"
_ACCESS_URL = "https://huggingface.co/facebook/OMAT24"
_HF_BASE = "https://huggingface.co/fairchem/OMAT24/blob/main/"
_CHECKPOINTS = {
    "EquiformerV2-31M-OMat": "eqV2_31M_omat.pt",
    "EquiformerV2-86M-OMat": "eqV2_86M_omat.pt",
    "EquiformerV2-153M-OMat": "eqV2_153M_omat.pt",
    "EquiformerV2-31M-MP": "eqV2_31M_mp.pt",
    "EquiformerV2-31M-DeNS-MP": "eqV2_dens_31M_mp.pt",
    "EquiformerV2-86M-DeNS-MP": "eqV2_dens_86M_mp.pt",
    "EquiformerV2-153M-DeNS-MP": "eqV2_dens_153M_mp.pt",
    "EquiformerV2-31M-OMat-Alex-MP": "eqV2_31M_omat_mp_salex.pt",
    "EquiformerV2-86M-OMat-Alex-MP": "eqV2_86M_omat_mp_salex.pt",
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _CheckpointRows(HTMLParser):
    """Collect checkpoint anchors from their enclosing model table rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, str]] = []
        self._in_row = False
        self._in_cell = False
        self._cell_text: list[str] = []
        self._cells: list[str] = []
        self._links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        if tag == "tr":
            self._in_row = True
            self._cells = []
            self._links = []
        elif self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_text = []
        elif self._in_row and tag == "a" and (href := attrs_map.get("href")):
            self._links.append(href)

    def handle_data(self, data: str) -> None:
        if self._in_row and self._in_cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._in_row and tag in {"td", "th"} and self._in_cell:
            self._cells.append(" ".join(" ".join(self._cell_text).split()))
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if self._cells:
                name = self._cells[0]
                for url in self._links:
                    if url.startswith(_HF_BASE) and url.lower().endswith(".pt"):
                        self.rows.append((name, url))
            self._in_row = False


class FairChemOMat24CheckpointSourceAdapter:
    """Index exact checkpoint links from FAIR Chemistry's legacy OMat page."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the nine model rows with explicit per-checkpoint Hugging Face file links "
        "on FAIR Chemistry's OMat24 documentation page. The page requires users to request "
        "or accept access before retrieving files. The generic-linked 153M Alexandria row, "
        "UMA, and model bytes are excluded."
    )

    def __init__(
        self,
        *,
        name: str = "fairchem-omat24-legacy-checkpoints",
        page_url: str = _DOCS_URL,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if page_url != _DOCS_URL or not name.strip() or max_response_bytes <= 0:
            raise ValueError("page URL is fixed; name and positive response limit are required")
        self.name, self.page_url = name, page_url
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "fairchem-omat24-checkpoints-v1",
                "page_url": page_url,
                "access_url": _ACCESS_URL,
                "checkpoints": _CHECKPOINTS,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(
            self.page_url, headers={"Accept": "text/html,application/xhtml+xml"}
        )
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
        rows = _parse_checkpoint_rows(response.text(), self.name)
        records = tuple(self._record(name, url, source_hash) for name, url in rows)
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

    def _record(self, name: str, checkpoint_url: str, source_hash: str) -> SourceRecord:
        filename = _CHECKPOINTS[name]
        handle = name.lower()
        namespace = "fairchem:omat24-checkpoint"
        model_id = f"model:{handle}"
        model = ModelHint(
            model_id,
            name,
            identifiers=(Identifier(namespace, handle),),
            aliases=(filename,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={
                "checkpoint_filename": filename,
                "checkpoint_url": checkpoint_url,
                "access_url": _ACCESS_URL,
                "access_restricted": True,
                "binary_reachability_checked": False,
                "source_url": self.page_url,
                "source_sha256": source_hash,
            },
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{handle}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(checkpoint_url),
            title=f"FAIR Chemistry {name} checkpoint",
            raw={
                "model_name": name,
                "checkpoint_filename": filename,
                "checkpoint_url": checkpoint_url,
                "access_restricted": True,
            },
            text=(
                f"FAIR Chemistry documents {filename} for {name}; access must be requested "
                "or accepted on the OMat24 Hub page."
            ),
            links=(
                Link(checkpoint_url, "checkpoint", crawl=False, model_local_ids=(model_id,)),
                Link(_ACCESS_URL, "access_instructions", crawl=False, model_local_ids=(model_id,)),
                Link(
                    self.page_url,
                    "source_documentation",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_checkpoint_rows(document: str, source: str) -> tuple[tuple[str, str], ...]:
    parser = _CheckpointRows()
    parser.feed(document)
    found: dict[str, str] = {}
    for name, url in parser.rows:
        if name not in _CHECKPOINTS:
            raise ValueError(f"{source}: unexpected checkpoint row {name!r}")
        expected = f"{_HF_BASE}{_CHECKPOINTS[name]}"
        if url != expected or name in found:
            raise ValueError(f"{source}: unexpected or duplicate checkpoint for {name}")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
            raise ValueError(f"{source}: checkpoint URL must use the official HTTPS Hub")
        found[name] = url
    if set(found) != set(_CHECKPOINTS):
        raise ValueError(f"{source}: expected exactly the nine direct checkpoint rows")
    return tuple((name, found[name]) for name in _CHECKPOINTS)


__all__ = ["FairChemOMat24CheckpointSourceAdapter"]
