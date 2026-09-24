"""First-party NeMo checkpoint-table ingestion with provider-identity bridges."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
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
from modelome.sources.html_catalog import _CatalogHtmlParser

Clock = Callable[[], datetime]

_MODEL_HEADER = frozenset({"model", "model name"})
_HF_REPOSITORY = re.compile(
    r"^https://huggingface\.co/(?P<id>[A-Za-z0-9][A-Za-z0-9_.-]*/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*)$"
)
_NGC_NEMO = re.compile(
    r"^https://ngc\.nvidia\.com/catalog/models/nvidia:nemo:(?P<name>[A-Za-z0-9_.-]+)$"
)
_NGC_COLLECTION = re.compile(
    r"^https://ngc\.nvidia\.com/catalog/models/nvidia:(?P<name>[A-Za-z0-9_.-]+)$"
)
_NGC_TEAM = re.compile(
    r"^https://(?:catalog\.nvidia\.com|catalog\.ngc\.nvidia\.com)/"
    r"orgs/nvidia/teams/nemo/models/"
    r"(?P<name>[A-Za-z0-9_.-]+)$"
)
_SAFE_MODEL_NAME = re.compile(r"^[^\x00-\x1f]{1,512}$")
_WEIGHT_URL = re.compile(
    r"(?:https://api\.ngc\.nvidia\.com/v2/models/nvidia/nemo/"
    r"[A-Za-z0-9_.-]+/versions/[A-Za-z0-9_.-]+/files/[A-Za-z0-9_.-]+"
    r"|https://huggingface\.co/[A-Za-z0-9][A-Za-z0-9_.-]*/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*/resolve/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"(?=$|[\s)`>])"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    name: str
    card_url: str
    identifiers: tuple[Identifier, ...]
    weight_urls: tuple[str, ...]
    headers: tuple[str, ...]
    cells: tuple[str, ...]
    locator: str


class NemoCheckpointCatalogSourceAdapter:
    """Read only rows in a NeMo documentation table that declare a model card.

    An entry is admitted only if its first table column is a model field and the
    same row exposes an exact Hugging Face or NVIDIA NGC model-card URL. Provider
    IDs are parsed from those URLs when unambiguous, allowing a source-declared
    NeMo row to enrich the corresponding Hub or NGC observation rather than make
    a name-based duplicate.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only first-party NeMo documentation table rows that directly declare "
        "a Hugging Face or NVIDIA NGC model-card URL. It does not invoke "
        "list_available_models(), infer checkpoint URLs, fetch provider cards, or "
        "treat citations and prose mentions as releases. Literal versioned NeMo NGC "
        "file URLs and Hugging Face resolve URLs in an admitted row are preserved "
        "as weight links."
    )

    def __init__(
        self,
        *,
        name: str = "nemo-checkpoints",
        url: str,
        max_response_bytes: int = 8 * 1024 * 1024,
        max_entries: int = 10_000,
        max_anchors: int = 100_000,
        max_tables: int = 10_000,
        max_cells: int = 100_000,
        max_text_chars: int = 16 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, "catalog URL")
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.max_anchors = _positive_int(max_anchors, "max_anchors")
        self.max_tables = _positive_int(max_tables, "max_tables")
        self.max_cells = _positive_int(max_cells, "max_cells")
        self.max_text_chars = _positive_int(max_text_chars, "max_text_chars")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "nemo-checkpoint-catalog-v1",
                "url": self.url,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "model table row with direct Hugging Face or NGC card",
                "weights": "literal NeMo NGC or Hugging Face file URLs in admitted rows",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        headers = {"Accept": "text/html,application/xhtml+xml"}
        if etag := _text(state.get("etag")):
            headers["If-None-Match"] = etag
        if last_modified := _text(state.get("http_last_modified")):
            headers["If-Modified-Since"] = last_modified
        response: HttpResponse = self.client.get(self.url, headers=headers)
        checked_at = _isoformat(self.clock())
        if response.status == 304:
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("entry_count")),
            )
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog exceeds {self.max_response_bytes} bytes"
            )
        response_url = _same_origin_response_url(response.url or self.url, self.url, self.name)
        parser = _CatalogHtmlParser(
            max_anchors=self.max_anchors,
            max_tables=self.max_tables,
            max_cells=self.max_cells,
            max_text_chars=self.max_text_chars,
        )
        parser.feed(response.text())
        parser.close()
        parser.finish()
        checkpoints = _checkpoints(parser, response_url, self.name)
        if not checkpoints:
            raise ValueError(f"{self.name}: catalog contains no direct checkpoint-card rows")
        if len(checkpoints) > self.max_entries:
            raise ValueError(f"{self.name}: catalog exceeds {self.max_entries} checkpoint rows")
        document_hash = content_hash(response.body)
        records = tuple(
            self._record(checkpoint, document_hash, response_url)
            for checkpoint in checkpoints
        )
        next_state: dict[str, Any] = {
            "checked_at": checked_at,
            "content_hash": document_hash,
            "entry_count": len(records),
        }
        if etag := _header(response.headers, "etag"):
            next_state["etag"] = etag
        if last_modified := _header(response.headers, "last-modified"):
            next_state["http_last_modified"] = last_modified
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        checkpoint: _Checkpoint,
        document_hash: str,
        response_url: str,
    ) -> SourceRecord:
        key = f"{checkpoint.name}\n{checkpoint.card_url}"
        model = ModelHint(
            local_id=f"model:{content_hash(key)[:24]}",
            name=checkpoint.name,
            identifiers=checkpoint.identifiers,
            status=ModelStatus.RELEASED,
            locator=checkpoint.locator,
        )
        release = ReleaseHint(
            local_id=f"release:{content_hash(key)[:24]}",
            model_local_id=model.local_id,
            identifiers=(Identifier("nemo:checkpoint-card", key),),
            metadata={
                "model_name": checkpoint.name,
                "model_card_url": checkpoint.card_url,
                "weight_urls": list(checkpoint.weight_urls),
                "headers": list(checkpoint.headers),
                "row_cells": list(checkpoint.cells),
            },
            locator=checkpoint.locator,
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{content_hash(key)[:24]}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=checkpoint.card_url,
            title=checkpoint.name,
            raw={
                "catalog_url": self.url,
                "response_url": response_url,
                "catalog_sha256": document_hash,
                "model_name": checkpoint.name,
                "model_card_url": checkpoint.card_url,
                "weight_urls": list(checkpoint.weight_urls),
                "headers": list(checkpoint.headers),
                "row_cells": list(checkpoint.cells),
                "locator": checkpoint.locator,
            },
            text="\n".join((checkpoint.name, *checkpoint.headers, *checkpoint.cells)),
            identifiers=checkpoint.identifiers,
            links=(
                Link(
                    response_url,
                    relation="documentation",
                    locator=checkpoint.locator,
                    crawl=False,
                    model_local_ids=(model.local_id,),
                ),
                Link(
                    checkpoint.card_url,
                    relation="model_card",
                    locator=checkpoint.locator,
                    crawl=False,
                    model_local_ids=(model.local_id,),
                ),
                *(
                    Link(
                        weight_url,
                        relation="weights",
                        locator=checkpoint.locator,
                        crawl=False,
                        model_local_ids=(model.local_id,),
                    )
                    for weight_url in checkpoint.weight_urls
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _checkpoints(parser: _CatalogHtmlParser, base_url: str, source: str) -> tuple[_Checkpoint, ...]:
    result: list[_Checkpoint] = []
    seen: set[tuple[str, str]] = set()
    for table in parser.tables:
        model_header = _model_header_index(table.rows)
        if model_header is None:
            continue
        header_index, model_column = model_header
        headers = tuple(cell.text for cell in table.rows[header_index].cells)
        for row in table.rows[header_index + 1 :]:
            if (
                not row.cells
                or model_column >= len(row.cells)
                or any(cell.is_header for cell in row.cells)
            ):
                continue
            name = _model_name(row.cells[model_column].text)
            if not name:
                continue
            cards = []
            weight_urls = []
            for cell in row.cells:
                weight_urls.extend(
                    canonicalize_url(match.group(0))
                    for match in _WEIGHT_URL.finditer(cell.text)
                )
                for anchor in cell.anchors:
                    card_url = canonicalize_url(urljoin(base_url, anchor.href))
                    if _provider_identifiers(card_url, name) is not None:
                        cards.append(card_url)
            for card_url in dict.fromkeys(cards):
                key = (name, card_url)
                if key in seen:
                    continue
                seen.add(key)
                provider_identifiers = _provider_identifiers(card_url, name)
                assert provider_identifiers is not None
                result.append(
                    _Checkpoint(
                        name=name,
                        card_url=card_url,
                        identifiers=(
                            Identifier("nemo:checkpoint", f"{name}\n{card_url}"),
                            *provider_identifiers,
                        ),
                        weight_urls=tuple(dict.fromkeys(weight_urls)),
                        headers=headers,
                        cells=tuple(cell.text for cell in row.cells),
                        locator=row.locator,
                    )
                )
    return tuple(result)


def _model_header_index(rows: tuple[Any, ...]) -> tuple[int, int] | None:
    for index, row in enumerate(rows):
        if not row.cells or not any(cell.is_header for cell in row.cells):
            continue
        for column, cell in enumerate(row.cells):
            if _text(cell.text).casefold() in _MODEL_HEADER:
                return index, column
    return None


def _model_name(value: str) -> str:
    name = _text(value)
    if not _SAFE_MODEL_NAME.fullmatch(name) or name.casefold() in {"n/a", "none"}:
        return ""
    return name


def _provider_identifiers(url: str, name: str) -> tuple[Identifier, ...] | None:
    if match := _HF_REPOSITORY.fullmatch(url):
        return (Identifier("huggingface:model", match.group("id")),)
    if match := _NGC_NEMO.fullmatch(url):
        return (Identifier("ngc:model", f"nvidia/nemo/{match.group('name')}"),)
    if match := _NGC_TEAM.fullmatch(url):
        return (Identifier("ngc:model", f"nvidia/nemo/{match.group('name')}"),)
    if _NGC_COLLECTION.fullmatch(url):
        # A collection page can be the documented card for several distinct
        # checkpoints. Preserve it as a resource, but never make it an identity
        # bridge that would merge those distinct source-declared model names.
        return ()
    return None


def _same_origin_response_url(value: str, expected: str, source: str) -> str:
    response_url = _web_url(value, "response URL")
    if _origin(response_url) != _origin(expected):
        raise ValueError(f"{source}: catalog response changed origin")
    return response_url


def _origin(value: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(value)
    scheme = parsed.scheme.casefold()
    host = (parsed.hostname or "").casefold()
    port = parsed.port
    if (scheme, port) in {("https", 443), ("http", 80)}:
        port = None
    return scheme, host, port


def _web_url(value: Any, field: str) -> str:
    url = canonicalize_url(_required_text(value, field))
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError(f"{field} must be an absolute HTTP(S) URL")
    return url


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _header(headers: Mapping[str, Any], key: str) -> str:
    wanted = key.casefold()
    for header, value in headers.items():
        if isinstance(header, str) and header.casefold() == wanted:
            return _text(value)
    return ""


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
