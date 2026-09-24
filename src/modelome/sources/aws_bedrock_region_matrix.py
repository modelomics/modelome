from __future__ import annotations

import re
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

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

_CARD_PATH = re.compile(
    r"^/bedrock/latest/userguide/model-card-(?P<id>[a-z0-9][a-z0-9-]*)\.html$"
)
_CARD_ORIGIN = "https://docs.aws.amazon.com"
_HEADERS = ("region", "in-region", "geo", "global")
_MAX_MODEL_ID = 180


class _MatrixParser(HTMLParser):
    """Collect model-card-linked Region matrices from AWS documentation HTML."""

    def __init__(self, page_url: str, *, max_cells: int) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.max_cells = max_cells
        self.card_id: str | None = None
        self.card_title = ""
        self.card_url: str | None = None
        self.records: dict[str, dict[str, Any]] = {}
        self._anchor_href = ""
        self._anchor_parts: list[str] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._cells_seen = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {key.casefold(): value or "" for key, value in attrs}
        if tag.casefold() == "a":
            self._anchor_href = attrs_map.get("href", "")
            self._anchor_parts = []
        elif tag.casefold() == "table":
            self._table = []
            self._row = None
            self._cell = None
        elif tag.casefold() == "tr" and self._table is not None:
            self._row = []
            self._cell = None
        elif tag.casefold() in {"td", "th"} and self._row is not None:
            self._cells_seen += 1
            if self._cells_seen > self.max_cells:
                raise ValueError("AWS Bedrock region matrix exceeds cell limit")
            self._cell = []
        elif tag.casefold() == "img" and self._cell is not None:
            alt = attrs_map.get("alt", "")
            if alt:
                self._cell.append(alt)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "a":
            self._select_model_card()
            self._anchor_href = ""
            self._anchor_parts = []
        elif tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            self._table.append(self._row)
            self._row = None
            self._cell = None
        elif tag == "table" and self._table is not None:
            self._save_table(self._table)
            self._table = None
            self._row = None
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._anchor_href:
            self._anchor_parts.append(data)
        if self._cell is not None:
            self._cell.append(data)

    def close(self) -> None:
        super().close()
        if self._table is not None:
            self._save_table(self._table)
            self._table = None

    def _select_model_card(self) -> None:
        absolute = canonicalize_url(urljoin(self.page_url, self._anchor_href))
        split = absolute.split(_CARD_ORIGIN, 1)
        if len(split) != 2:
            return
        match = _CARD_PATH.fullmatch(split[1].split("?", 1)[0])
        if match is None:
            return
        card_id = match.group("id")
        if len(card_id) > _MAX_MODEL_ID:
            raise ValueError("AWS Bedrock model-card ID exceeds limit")
        self.card_id = card_id
        self.card_title = " ".join(" ".join(self._anchor_parts).split()) or card_id
        self.card_url = absolute

    def _save_table(self, rows: list[list[str]]) -> None:
        if not self.card_id or not self.card_url or not rows:
            return
        header = tuple(_cell_value(value).casefold() for value in rows[0])
        if header[:4] != _HEADERS:
            return
        matrix_rows = []
        for row in rows[1:]:
            if len(row) < 4:
                continue
            region = _cell_value(row[0])
            if not region:
                continue
            matrix_rows.append({
                "region": region,
                "in_region": _availability(row[1]),
                "geo": _availability(row[2]),
                "global": _availability(row[3]),
            })
        if not matrix_rows:
            return
        model = self.records.setdefault(
            self.card_id,
            {"title": self.card_title, "url": self.card_url, "availability": []},
        )
        model["availability"].extend(matrix_rows)


class AwsBedrockRegionMatrixAdapter:
    """Capture AWS's public per-model Region and lifecycle matrix.

    The matrix's model names link directly to the same exact ``model-card-*``
    identifiers emitted by the existing Bedrock card catalog. This adapter uses
    that linked card identity and never derives it from a display name.
    """

    def __init__(
        self,
        *,
        name: str = "aws-bedrock-region-matrix",
        url: str = "https://docs.aws.amazon.com/bedrock/latest/userguide/models-region-compatibility.html",
        client: HttpClient | Any | None = None,
        max_response_bytes: int = 8 * 1024 * 1024,
        max_entries: int = 5_000,
        max_cells: int = 250_000,
    ) -> None:
        self.name = name
        self.url = canonicalize_url(url)
        if not self.url.startswith(_CARD_ORIGIN + "/bedrock/latest/userguide/"):
            raise ValueError("AWS Bedrock matrix URL must use the official user guide")
        if min(max_response_bytes, max_entries, max_cells) < 1:
            raise ValueError("response and entry limits must be positive")
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.max_cells = max_cells
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if state:
            raise ValueError(f"{self.name}: this catalog does not accept pagination state")
        response: HttpResponse = self.client.get(
            self.url, headers={"Accept": "text/html"}
        )
        if not 200 <= response.status < 300:
            raise ValueError(f"{self.name}: matrix returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: matrix exceeds {self.max_response_bytes} bytes")
        page_url = canonicalize_url(response.url or self.url)
        if not page_url.startswith(_CARD_ORIGIN + "/bedrock/latest/userguide/"):
            raise ValueError(f"{self.name}: response escaped AWS Bedrock documentation")
        parser = _MatrixParser(page_url, max_cells=self.max_cells)
        parser.feed(response.text())
        parser.close()
        if not parser.records:
            raise ValueError(f"{self.name}: no model-card-linked region matrices found")
        if len(parser.records) > self.max_entries:
            raise ValueError(f"{self.name}: matrix exceeds {self.max_entries} models")
        revision = content_hash(response.body)
        records = tuple(
            self._record(card_id, item, revision)
            for card_id, item in sorted(parser.records.items())
        )
        return SourcePage(
            records=records,
            next_state={"entry_count": len(records), "content_hash": revision},
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, card_id: str, item: Mapping[str, Any], revision: str) -> SourceRecord:
        identifier = Identifier("aws:bedrock-model-card", card_id)
        local_id = f"aws-bedrock:{card_id}#model"
        card_url = str(item["url"])
        title = str(item["title"])
        availability = list(item["availability"])
        return SourceRecord(
            source_record_id=f"aws-bedrock-region:{card_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self.url,
            title=title,
            raw={
                "model_card_id": card_id,
                "availability": availability,
                "catalog_revision_sha256": revision,
            },
            text="\n".join(
                (
                    f"{row['region']}: In-Region={row['in_region']}; "
                    f"Geo={row['geo']}; Global={row['global']}"
                )
                for row in availability
            ),
            identifiers=(identifier,),
            links=(
                Link(
                    card_url,
                    relation="model_card",
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
            ),
            models=(
                ModelHint(
                    local_id=local_id,
                    name=title,
                    identifiers=(identifier,),
                    status=ModelStatus.DOCUMENTED,
                    locator=f"model-card:{card_id}",
                ),
            ),
        )


def _cell_value(value: str) -> str:
    return " ".join(value.replace("`", "").split())


def _availability(value: str) -> str:
    value = _cell_value(value)
    normalized = value.casefold().replace("_", "-")
    if "not-supported" in normalized or "not supported" in normalized:
        return "not_supported"
    if "legacy" in normalized:
        return value
    if normalized == "supported" or normalized.endswith(": supported"):
        return "supported"
    if not value:
        return "unknown"
    return value
