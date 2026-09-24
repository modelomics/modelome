"""PaddleNLP's explicit pre-trained TokenEmbedding checkpoint catalog."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
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

Clock = Callable[[], datetime]
_EMBEDDING_NAME = re.compile(r"^(?:w2v|glove|fasttext)\.[A-Za-z0-9_.-]+$")
_FILE_SIZE = re.compile(r"^\d+(?:\.\d+)?\s*(?:B|KB|MB|GB|TB)$", re.I)
_INTEGER = re.compile(r"^\d[\d,]*$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _ModelInfoTableParser(HTMLParser):
    """Read rows from the documentation's exact Model Information section."""

    def __init__(self, *, max_rows: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_rows = max_rows
        self.rows: list[tuple[str, ...]] = []
        self._heading_depth = 0
        self._heading_parts: list[str] = []
        self._in_model_section = False
        self._table_depth = 0
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_depth = int(tag[1])
            self._heading_parts = []
            if self._table_depth == 0:
                self._in_model_section = False
        elif tag == "table" and self._in_model_section:
            self._table_depth += 1
        elif tag == "tr" and self._table_depth and self._row is None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_parts = []
        elif tag in {"br", "p", "div"} and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._heading_depth:
            title = " ".join("".join(self._heading_parts).split()).casefold()
            self._in_model_section = title == "model information"
            self._heading_depth = 0
            self._heading_parts = []
        elif tag in {"td", "th"} and self._cell_parts is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell_parts).split()))
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(tuple(self._row))
                if len(self.rows) > self.max_rows:
                    raise ValueError(f"Model Information exceeds {self.max_rows} rows")
            self._row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._heading_depth:
            self._heading_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)


def _parse_model_rows(document: str, *, max_rows: int) -> tuple[tuple[str, str, str], ...]:
    parser = _ModelInfoTableParser(max_rows=max_rows)
    parser.feed(document)
    rows = parser.rows
    if not rows or tuple(_cell_text(cell).casefold() for cell in rows[0][:3]) != (
        "model",
        "file size",
        "vocabulary size",
    ):
        raise ValueError("paddlenlp-token-embeddings: Model Information table is missing")
    entries: dict[str, tuple[str, str]] = {}
    for row in rows[1:]:
        if len(row) < 3:
            continue
        name = _cell_text(row[0])
        size = _cell_text(row[1])
        vocabulary = _cell_text(row[2])
        if not _EMBEDDING_NAME.fullmatch(name):
            continue
        if not _FILE_SIZE.fullmatch(size) or not _INTEGER.fullmatch(vocabulary):
            continue
        details = (size, vocabulary.replace(",", ""))
        if name in entries and entries[name] != details:
            raise ValueError(f"paddlenlp-token-embeddings: conflicting metadata for {name}")
        entries[name] = details
    if not entries:
        raise ValueError("paddlenlp-token-embeddings: Model Information table has no entries")
    return tuple((name, *entries[name]) for name in sorted(entries))


def _cell_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split()).strip()


class PaddleNlpTokenEmbeddingSourceAdapter:
    """Enumerate PaddleNLP's documented downloadable TokenEmbedding checkpoints.

    The upstream page publishes exact embedding identifiers, archive sizes, and
    vocabulary sizes. It does not publish each archive URL in the catalog table,
    so records remain documented models and do not claim direct artifact links.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only exact Word2Vec, GloVe, and FastText entries in PaddleNLP's "
        "TokenEmbedding Model Information table. It does not cover PaddleNLP "
        "language models, infer artifact URLs, or download checkpoint files."
    )

    def __init__(
        self,
        *,
        name: str = "paddlenlp-token-embeddings",
        document_url: str = "https://paddlenlp.readthedocs.io/en/stable/model_zoo/embeddings.html",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_rows: int = 10_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name is required")
        if not isinstance(document_url, str) or not document_url.startswith("https://"):
            raise ValueError("document_url must be an HTTPS URL")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
            raise ValueError("max_rows must be a positive integer")
        self.name = name.strip()
        self.document_url = document_url
        self.max_response_bytes = max_response_bytes
        self.max_rows = max_rows
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paddlenlp-token-embeddings-v1",
                "document_url": self.document_url,
                "max_response_bytes": self.max_response_bytes,
                "max_rows": self.max_rows,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.document_url,
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: documentation returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: documentation exceeds {self.max_response_bytes} bytes")
        document_hash = content_hash(response.body)
        checked_at = _isoformat(self.clock())
        if document_hash == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )
        entries = _parse_model_rows(response.text(), max_rows=self.max_rows)
        records = tuple(
            self._record(name, file_size, vocabulary_size, document_hash)
            for name, file_size, vocabulary_size in entries
        )
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": document_hash,
                "checked_at": checked_at,
                "document_url": self.document_url,
                "document_sha256": document_hash,
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        name: str,
        file_size: str,
        vocabulary_size: str,
        document_hash: str,
    ) -> SourceRecord:
        identifier = Identifier("paddlenlp:token-embedding", name)
        model = ModelHint(
            local_id=f"model:{name}",
            name=name,
            identifiers=(identifier,),
            status=ModelStatus.DOCUMENTED,
            locator=f"Model Information:{name}",
        )
        return SourceRecord(
            source_record_id=f"model:{name}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(self.document_url),
            title=name,
            raw={
                "registry": "PaddleNLP TokenEmbedding",
                "file_size": file_size,
                "vocabulary_size": int(vocabulary_size),
                "document_sha256": document_hash,
            },
            text=(
                f"{name}\n"
                "task: word embedding\n"
                f"file size: {file_size}\n"
                f"vocabulary size: {vocabulary_size}"
            ),
            identifiers=(identifier,),
            links=(
                Link(self.document_url, relation="model_card", crawl=False),
                Link(
                    "https://github.com/PaddlePaddle/PaddleNLP",
                    relation="source_repository",
                    crawl=False,
                ),
            ),
            models=(model,),
        )


def _isoformat(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise ValueError("clock must return a datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


__all__ = ["PaddleNlpTokenEmbeddingSourceAdapter"]
