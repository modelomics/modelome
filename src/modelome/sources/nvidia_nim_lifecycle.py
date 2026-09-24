"""Dated NVIDIA NIM component lifecycle notices from AI Enterprise docs."""

from __future__ import annotations

from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any

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
from modelome.normalize import canonicalize_url, content_hash

_URL = "https://docs.nvidia.com/ai-enterprise/lifecycle/latest/eol-notices.html"
_HEADERS = ("Component", "Current State", "Action Required By")


class NvidiaNimLifecycle:
    """Capture NIM component rows in NVIDIA's public AI Enterprise notices."""

    def __init__(
        self,
        *,
        name: str = "nvidia-ai-enterprise-nim-lifecycle",
        url: str = _URL,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 200,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = name
        self.url = canonicalize_url(url)
        if self.url != _URL:
            raise ValueError("URL must use NVIDIA's public AI Enterprise EOL notices")
        self.max_response_bytes = int(max_response_bytes)
        self.max_entries = int(max_entries)
        if self.max_response_bytes < 1 or self.max_entries < 1:
            raise ValueError("response and entry limits must be positive")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.checkpoint_signature = content_hash({
            "adapter": "nvidia-ai-enterprise-nim-lifecycle-v1",
            "url": self.url,
            "max_response_bytes": self.max_response_bytes,
            "max_entries": self.max_entries,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.url, headers={"Accept": "text/html"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: lifecycle page returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: lifecycle page exceeds {self.max_response_bytes} bytes"
            )
        parser = _TableParser()
        parser.feed(response.text())
        tables = [table for table in parser.tables if table and tuple(table[0]) == _HEADERS]
        if not tables:
            raise ValueError(f"{self.name}: lifecycle tables were not found")
        rows: list[tuple[str, str, str]] = []
        for table in tables:
            for row in table[1:]:
                if len(row) != 3:
                    raise ValueError(f"{self.name}: lifecycle row has unexpected cell count")
                component, status, action_date = row
                if component.startswith("NVIDIA NIM"):
                    if not status or not action_date:
                        raise ValueError(f"{self.name}: NIM row lacks lifecycle fields")
                    rows.append((component, status, action_date))
        if not rows:
            raise ValueError(f"{self.name}: lifecycle tables contain no NIM notices")
        if len(rows) > self.max_entries:
            raise ValueError(f"{self.name}: lifecycle tables exceed {self.max_entries} NIM entries")
        ids = [row[0] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.name}: lifecycle tables contain duplicate NIM components")
        snapshot = content_hash(response.body)
        records = tuple(self._record(*row, snapshot=snapshot) for row in rows)
        return SourcePage(
            records=records,
            next_state={"entry_count": len(records), "content_hash": snapshot},
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=False,
        )

    def _record(
        self, component: str, status: str, action_date: str, *, snapshot: str
    ) -> SourceRecord:
        identifier = Identifier("nvidia:ai-enterprise-component", component)
        model = ModelHint(
            local_id=f"{component}#component",
            name=component,
            identifiers=(identifier,),
            status=ModelStatus.DOCUMENTED,
        )
        return SourceRecord(
            source_record_id=f"lifecycle:{component}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self.url,
            title=f"NVIDIA AI Enterprise lifecycle: {component}",
            raw={
                "component_name": component,
                "provider_lifecycle_status": status,
                "action_required_by": action_date,
                "catalog_sha256": snapshot,
            },
            text=f"NVIDIA lists {component} as {status}; action required by {action_date}.",
            identifiers=(identifier,),
            links=(Link(self.url, relation="lifecycle_documentation"),),
            models=(model,),
        )


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None
