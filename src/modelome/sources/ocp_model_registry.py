"""Official legacy Open Catalyst checkpoint table published by FAIR Chemistry."""

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
_DOCS_URL = "https://facebookresearch.github.io/fairchem/models-1/"
_CHECKPOINT_HOST = "dl.fbaipublicfiles.com"
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+>-]*$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _CheckpointRows(HTMLParser):
    """Read model names and direct .pt links only from documentation table rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, str]] = []
        self._in_row = False
        self._in_cell = False
        self._cell_text: list[str] = []
        self._cells: list[str] = []
        self._row_links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        if tag == "tr":
            self._in_row = True
            self._cells = []
            self._row_links = []
        elif self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_text = []
        elif self._in_row and tag == "a" and (href := attrs_map.get("href")):
            self._row_links.append(href)

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
                for link in self._row_links:
                    if link.lower().split("?", 1)[0].endswith(".pt"):
                        self.rows.append((name, link))
            self._in_row = False


class OCPModelRegistrySourceAdapter:
    """Index the exact checkpoint links enumerated in FAIR Chemistry's docs.

    The docs page is the first-party inventory for legacy OCP/OC20/OC22 models.
    It is parsed as documentation, with no model code execution or weight fetch.
    Current UMA checkpoints are Hugging Face hosted and are outside this adapter.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only legacy OCP checkpoint rows with direct .pt links in the "
        "FAIR Chemistry pretrained-model documentation. It excludes current UMA, "
        "non-checkpoint links, and models omitted from that page; weight bytes are not fetched."
    )

    def __init__(
        self,
        *,
        name: str = "fairchem-ocp-legacy-checkpoints",
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if not name.strip() or max_response_bytes <= 0:
            raise ValueError("name and positive max_response_bytes are required")
        self.name = name
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "fairchem-ocp-legacy-doc-table-v1",
                "docs_url": _DOCS_URL,
                "max_response_bytes": max_response_bytes,
                "admission": "table row name + exact direct dl.fbaipublicfiles.com .pt link",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(_DOCS_URL, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: docs returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: docs exceed {self.max_response_bytes} bytes")
        parser = _CheckpointRows()
        parser.feed(response.text())
        entries: dict[str, str] = {}
        for name, url in parser.rows:
            parsed = urlsplit(url)
            if (
                not _MODEL.fullmatch(name)
                or parsed.scheme != "https"
                or parsed.hostname != _CHECKPOINT_HOST
                or not parsed.path.startswith("/opencatalystproject/models/")
                or not parsed.path.lower().endswith(".pt")
                or parsed.query
                or parsed.fragment
            ):
                continue
            if name in entries and entries[name] != url:
                raise ValueError(f"{self.name}: conflicting URLs for {name}")
            entries[name] = url
        if not entries:
            raise ValueError(f"{self.name}: no first-party checkpoint rows found")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        doc_hash = content_hash(response.body)
        if state.get("completed_sha256") == doc_hash:
            return SourcePage(
                (),
                {**state, "checked_at": checked_at},
                True,
                upstream_count=state.get("model_count"),
            )
        records = tuple(self._record(name, url, doc_hash) for name, url in sorted(entries.items()))
        return SourcePage(
            records,
            {
                "completed_sha256": doc_hash,
                "checked_at": checked_at,
                "model_count": len(records),
                "source_sha256": doc_hash,
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, name: str, url: str, doc_hash: str) -> SourceRecord:
        namespace = "fairchem:ocp-checkpoint"
        model_id = f"model:{name}"
        model = ModelHint(
            model_id, name, identifiers=(Identifier(namespace, name),), status=ModelStatus.RELEASED
        )
        release = ReleaseHint(
            f"release:{name}",
            model_id,
            identifiers=(Identifier(f"{namespace}:release", name),),
            metadata={
                "checkpoint_handle": name,
                "weight_url": url,
                "documentation_sha256": doc_hash,
            },
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{name}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(_DOCS_URL),
            title=f"FAIR Chemistry {name}",
            raw={"checkpoint_handle": name, "weight_url": url, "documentation_url": _DOCS_URL},
            text=f"Legacy Open Catalyst checkpoint {name} documented by FAIR Chemistry.",
            links=(
                Link(_DOCS_URL, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


__all__ = ["OCPModelRegistrySourceAdapter"]
