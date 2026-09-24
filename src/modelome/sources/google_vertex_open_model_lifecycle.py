"""Public Vertex AI lifecycle records for managed open-model offerings."""

from __future__ import annotations

import re
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

_URL = "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/deprecations/open-models"
_HEADER = ("Model ID", "Deprecation date", "Retirement date", "Self-deploy alternative")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,180}$")
_DATE = re.compile(r"^(?:[A-Z][a-z]+ \d{1,2}, \d{4}|No retirement date announced)$")


class GoogleVertexOpenModelLifecycle:
    """Read Google Cloud's public lifecycle table for Vertex managed open models."""

    def __init__(
        self,
        *,
        name: str = "google-vertex-managed-open-model-lifecycle",
        url: str = _URL,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 500,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = name
        self.url = canonicalize_url(url)
        if self.url != _URL:
            raise ValueError("URL must use Google's managed open-model lifecycle page")
        self.max_response_bytes = int(max_response_bytes)
        self.max_entries = int(max_entries)
        if self.max_response_bytes < 1 or self.max_entries < 1:
            raise ValueError("response and entry limits must be positive")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.checkpoint_signature = content_hash({
            "adapter": "google-vertex-open-model-lifecycle-v1",
            "url": self.url,
            "max_response_bytes": self.max_response_bytes,
            "max_entries": self.max_entries,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.url, headers={"Accept": "text/markdown"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: lifecycle page returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: lifecycle page exceeds {self.max_response_bytes} bytes"
            )
        parser = _TableParser()
        parser.feed(response.text())
        matching_table = next(
            (table for table in parser.tables if table and tuple(table[0]) == _HEADER),
            None,
        )
        if matching_table is None:
            raise ValueError(f"{self.name}: lifecycle table header was not found")
        records_data: list[tuple[str, str, str, str]] = []
        for row in matching_table[1:]:
            if len(row) != 4:
                raise ValueError(f"{self.name}: lifecycle row has {len(row)} cells, expected 4")
            model_id, dep_date, retire_date, alternative = row
            if not _MODEL_ID.fullmatch(model_id):
                raise ValueError(f"{self.name}: invalid model ID in lifecycle row: {model_id!r}")
            if not _DATE.fullmatch(dep_date) or not _DATE.fullmatch(retire_date):
                raise ValueError(f"{self.name}: invalid lifecycle date for {model_id}")
            records_data.append((model_id, dep_date, retire_date, alternative))
            if len(records_data) > self.max_entries:
                raise ValueError(f"{self.name}: lifecycle table exceeds {self.max_entries} entries")
        if not records_data:
            raise ValueError(f"{self.name}: lifecycle table contains no model rows")
        ids = [row[0] for row in records_data]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.name}: lifecycle table contains duplicate model IDs")
        snapshot = content_hash(response.body)
        records = tuple(self._record(*row, snapshot=snapshot) for row in records_data)
        return SourcePage(
            records=records,
            next_state={"entry_count": len(records), "content_hash": snapshot},
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=False,
        )

    def _record(
        self,
        model_id: str,
        deprecation_date: str,
        retirement_date: str,
        alternative: str,
        *,
        snapshot: str,
    ) -> SourceRecord:
        identifier = Identifier("google:vertex-managed-open-model", model_id)
        model = ModelHint(
            local_id=f"{model_id}#model",
            name=model_id,
            identifiers=(identifier,),
            status=ModelStatus.DOCUMENTED,
        )
        return SourceRecord(
            source_record_id=f"lifecycle:{model_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self.url,
            title=f"Vertex AI managed open model lifecycle: {model_id}",
            raw={
                "model_id": model_id,
                "provider_lifecycle_status": "deprecated",
                "deprecation_date": deprecation_date,
                "retirement_date": retirement_date,
                "self_deploy_alternative": alternative,
                "catalog_sha256": snapshot,
            },
            text=(
                f"Google lists {model_id} as deprecated on {deprecation_date} "
                f"with retirement on {retirement_date}."
            ),
            identifiers=(identifier,),
            links=(Link(self.url, relation="lifecycle_documentation"),),
            models=(model,),
        )


class _TableParser(HTMLParser):
    """Extract bounded table cells from the docs page's HTML response."""

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
