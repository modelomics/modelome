"""Pinned Markdown-table model catalogs with explicit artifact admission."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urlsplit

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
from modelome.normalize import canonicalize_url, content_hash, extract_urls

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HEADING = re.compile(r"^#{1,6}[ \t]+(?P<title>.+?)\s*$")
_MD_LINK = re.compile(r"\[(?P<label>[^\]\r\n]+)\]\((?P<url>https?://[^)\s]+)\)")
_ANY_MD_LINK = re.compile(r"\[+\s*(?P<label>[^\]\r\n]+?)\s*\]+\((?P<url>[^)\s]+)\)")
_TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")
_MARKDOWN = re.compile(r"[`*~]|<[^>]+>")
_TEX_DISPLAY_CELL = re.compile(r"\^\{_\{(?P<value>.*?)\}\}")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<tail>.*)$")
_MODEL_HEADER = re.compile(r"(?:^|\s)(?:model|models|model type|acoustic model)(?:\s|$)", re.I)
_WEIGHT_LABEL = re.compile(
    r"pre.?train|train(?:ed)?|weight|parameter|checkpoint|download|model|static|onnx|"
    r"paddle.?lite|fp32|int8|推理|训练|模型",
    re.I,
)
_PAPER_LABEL = re.compile(r"\b(?:arxiv|doi|paper|preprint|publication)\b", re.I)
_INFERENCE_LABEL = re.compile(r"infer|deploy|static|onnx|fp32|int8|推理", re.I)
_METRICS_LABEL = re.compile(r"\b(?:metric|evaluation|benchmark|score|result)s?\b", re.I)
_LOG_LABEL = re.compile(r"\b(?:log|logs|training[- ]?log)s?\b", re.I)
_WEIGHT_SUFFIX = re.compile(
    r"\.(?:bin|ckpt|zip|tar|tgz|gz|bz2|xz|7z|pdparams|pkl|pth|pt|onnx|nb|nemo|safetensors)(?:$|/)",
    re.I,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _TableRow:
    name: str
    dataset: str | None
    heading: str
    description: str
    links: tuple[tuple[str, str], ...]
    locator: str


@dataclass(frozen=True, slots=True)
class _HtmlTable:
    header: tuple[str, ...]
    rows: tuple[tuple[tuple[str, ...], int], ...]


class MarkdownModelTableSourceAdapter:
    """Enumerate explicitly downloadable models from a first-party Markdown table.

    This deliberately accepts only rows whose header declares a model column and
    whose source row declares at least one direct downloadable artifact. It is a
    compact complement to machine APIs: rendered prose, relative links, and
    tables without a model/artifact pair do not become model assertions.
    """

    disable_derived_extraction = True

    def __init__(
        self,
        *,
        name: str,
        repository: str,
        branch: str,
        document_path: str,
        provider_namespace: str,
        model_column: int = 0,
        model_header_pattern: str | None = None,
        model_name_pattern: str | None = None,
        dataset_column: int | None = None,
        identity_context_columns: Sequence[int] | None = None,
        checkpoint_row_pattern: str | None = None,
        identity_include_heading: bool = True,
        excluded_headings: Sequence[str] = (),
        max_response_bytes: int = 8 * 1024 * 1024,
        max_rows: int = 100_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.document_path = _safe_path(document_path)
        self.provider_namespace = _namespace(provider_namespace)
        self.model_column = _nonnegative_int(model_column, "model_column")
        if model_header_pattern is None:
            self.model_header_pattern = _MODEL_HEADER
        else:
            pattern = _required_text(model_header_pattern, "model_header_pattern")
            try:
                self.model_header_pattern = re.compile(pattern, re.I)
            except re.error as exc:
                raise ValueError(
                    f"model_header_pattern is not a valid regular expression: {exc}"
                ) from exc
        if model_name_pattern is None:
            self.model_name_pattern = None
        else:
            pattern = _required_text(model_name_pattern, "model_name_pattern")
            try:
                self.model_name_pattern = re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"model_name_pattern is not a valid regular expression: {exc}"
                ) from exc
        self.dataset_column = (
            None if dataset_column is None else _nonnegative_int(dataset_column, "dataset_column")
        )
        if identity_context_columns is None:
            self.identity_context_columns = ()
        elif isinstance(identity_context_columns, (str, bytes, bytearray)):
            raise ValueError("identity_context_columns must be a sequence of columns")
        else:
            self.identity_context_columns = tuple(
                _nonnegative_int(column, "identity context column")
                for column in identity_context_columns
            )
            if not self.identity_context_columns:
                raise ValueError("identity_context_columns cannot be empty")
            if len(set(self.identity_context_columns)) != len(self.identity_context_columns):
                raise ValueError("identity_context_columns cannot contain duplicates")
        if self.dataset_column is not None and self.identity_context_columns:
            raise ValueError(
                "dataset_column and identity_context_columns cannot both be configured"
            )
        if not isinstance(identity_include_heading, bool):
            raise ValueError("identity_include_heading must be a boolean")
        self.identity_include_heading = identity_include_heading
        if checkpoint_row_pattern is None:
            self.checkpoint_row_pattern = None
        else:
            pattern = _required_text(checkpoint_row_pattern, "checkpoint_row_pattern")
            try:
                self.checkpoint_row_pattern = re.compile(pattern, re.I)
            except re.error as exc:
                raise ValueError(
                    f"checkpoint_row_pattern is not a valid regular expression: {exc}"
                ) from exc
        self.excluded_headings = tuple(
            re.compile(_required_text(item, "excluded heading"), re.I) for item in excluded_headings
        )
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_rows = _positive_int(max_rows, "max_rows")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "markdown-model-table-v1",
                "repository": self.repository,
                "branch": self.branch,
                "document_path": self.document_path,
                "provider_namespace": self.provider_namespace,
                "model_column": self.model_column,
                "model_header_pattern": self.model_header_pattern.pattern,
                "model_name_pattern": (
                    None if self.model_name_pattern is None else self.model_name_pattern.pattern
                ),
                "dataset_column": self.dataset_column,
                "identity_context_columns": self.identity_context_columns,
                "identity_include_heading": self.identity_include_heading,
                "checkpoint_row_pattern": (
                    None
                    if self.checkpoint_row_pattern is None
                    else self.checkpoint_row_pattern.pattern
                ),
                "excluded_headings": [pattern.pattern for pattern in self.excluded_headings],
                "max_response_bytes": self.max_response_bytes,
                "max_rows": self.max_rows,
            }
        )

    @property
    def coverage_limitation(self) -> str:
        return (
            "Covers explicit downloadable model rows or configured checkpoint-table "
            f"columns in {self.repository}/{self.document_path} at one observed commit. "
            "It does not establish "
            "coverage outside that first-party document, infer papers, resolve "
            "relative links, or download model artifacts."
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(self.document_path, safe='/')}"
        )

    def blob_url(self, revision: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.document_path, safe='/')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            if etag := _header(commit_response.headers, "etag"):
                next_state["commit_etag"] = etag
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_from_state(state.get("model_count")),
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision),
            headers={"Accept": "text/markdown,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog document returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog document exceeds {self.max_response_bytes} bytes"
            )
        rows, skipped_rows = self._rows(response.text())
        records = tuple(self._record(row, revision, response.body) for row in rows)
        if not records:
            raise ValueError(f"{self.name}: catalog document contains no downloadable model rows")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "document_url": self.raw_url(revision),
            "document_sha256": content_hash(response.body),
            "model_count": len(records),
            "skipped_nonmodel_rows": skipped_rows,
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _revision(self) -> tuple[str, HttpResponse]:
        response: HttpResponse = self.client.get(
            self.commit_url,
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _rows(self, document: str) -> tuple[tuple[_TableRow, ...], int]:
        heading = ""
        rows: list[_TableRow] = []
        skipped = 0
        visible_document = _visible_markdown(document)
        lines = visible_document.splitlines()
        headings = _headings_by_line(lines)
        index = 0
        while index < len(lines):
            line = lines[index]
            if match := _HEADING.match(line):
                heading = _plain(match.group("title"))
                index += 1
                continue
            cells = _table_cells(line)
            if not cells:
                index += 1
                continue
            if index + 1 < len(lines) and _table_separator(_table_cells(lines[index + 1])):
                header = cells
                index += 2
                table_rows: list[tuple[tuple[str, ...], int]] = []
                while index < len(lines):
                    row_cells = _table_cells(lines[index])
                    if not row_cells:
                        break
                    table_rows.append((row_cells, index + 1))
                    index += 1
                table_records, table_skipped = self._table_records(
                    header,
                    table_rows,
                    heading,
                )
                rows.extend(table_records)
                skipped += table_skipped
                continue
            index += 1
        for table in _html_tables(visible_document, self.name):
            table_records, table_skipped = self._table_records(
                table.header,
                table.rows,
                headings.get(table.rows[0][1], "") if table.rows else "",
            )
            rows.extend(table_records)
            skipped += table_skipped
        if len(rows) > self.max_rows:
            raise ValueError(f"{self.name}: catalog document exceeds {self.max_rows} model rows")
        return (
            _deduplicate_rows(
                rows,
                self.name,
                include_heading=self.identity_include_heading,
            ),
            skipped,
        )

    def _table_records(
        self,
        header: tuple[str, ...],
        table_rows: Sequence[tuple[tuple[str, ...], int]],
        heading: str,
    ) -> tuple[tuple[_TableRow, ...], int]:
        if self.checkpoint_row_pattern is not None:
            return self._transposed_table_rows(header, table_rows, heading)
        rows: list[_TableRow] = []
        skipped = 0
        for cells, line_number in table_rows:
            row = self._table_row(cells, header, heading, line_number)
            if row is None:
                skipped += 1
            else:
                rows.append(row)
        return tuple(rows), skipped

    def _transposed_table_rows(
        self,
        header: tuple[str, ...],
        table_rows: Sequence[tuple[tuple[str, ...], int]],
        heading: str,
    ) -> tuple[tuple[_TableRow, ...], int]:
        if any(pattern.search(heading) for pattern in self.excluded_headings):
            return (), len(table_rows)
        if len(header) < 2:
            return (), len(table_rows)
        rows: list[_TableRow] = []
        skipped = 0
        for cells, line_number in table_rows:
            if not cells or not self.checkpoint_row_pattern.search(_plain(cells[0])):
                skipped += 1
                continue
            for column in range(1, min(len(header), len(cells))):
                name = _plain(header[column])
                if not _model_name(name):
                    continue
                links = _row_links((cells[column],))
                if not any(
                    relation in {"weights", "inference_artifact", "model_artifact"}
                    for _, relation in links
                ):
                    continue
                label = _plain(cells[0])
                description = " | ".join(
                    item for item in (name, label, _plain(cells[column])) if item
                )
                rows.append(
                    _TableRow(
                        name=name,
                        dataset=None,
                        heading=heading,
                        description=description[:8_192],
                        links=links,
                        locator=f"{self.document_path}:line:{line_number}",
                    )
                )
        return tuple(rows), skipped

    def _table_row(
        self,
        cells: tuple[str, ...],
        header: tuple[str, ...],
        heading: str,
        line_number: int,
    ) -> _TableRow | None:
        if self.model_column >= len(cells) or self.model_column >= len(header):
            return None
        if not self.model_header_pattern.search(_plain(header[self.model_column])):
            return None
        if any(pattern.search(heading) for pattern in self.excluded_headings):
            return None
        name = _plain(cells[self.model_column])
        if self.model_name_pattern is not None:
            match = self.model_name_pattern.search(name)
            if match is None:
                return None
            name = match.group(1) if match.lastindex else match.group()
        if not _model_name(name):
            return None
        dataset = _row_identity_context(
            cells,
            dataset_column=self.dataset_column,
            context_columns=self.identity_context_columns,
        )
        links = _row_links(cells, header)
        has_artifact = any(
            relation in {"weights", "inference_artifact", "model_artifact"} for _, relation in links
        )
        if not has_artifact:
            return None
        text = " | ".join(_plain(cell) for cell in cells if _plain(cell))
        return _TableRow(
            name=name,
            dataset=dataset,
            heading=heading,
            description=text[:8_192],
            links=links,
            locator=f"{self.document_path}:line:{line_number}",
        )

    def _record(self, row: _TableRow, revision: str, document: bytes) -> SourceRecord:
        identity_parts = []
        if self.identity_include_heading:
            identity_parts.append(row.heading or "root")
        identity_parts.append(row.name)
        if row.dataset:
            identity_parts.append(row.dataset)
        identity = " / ".join(identity_parts)
        model_identifier = Identifier(self.provider_namespace, identity)
        release_identifier = Identifier(f"{self.provider_namespace}-release", identity)
        doc_url = self.blob_url(revision)
        links = [
            Link(doc_url, relation="model_card", locator=row.locator, crawl=False),
            Link(self.repository_url, relation="source_repository", crawl=False),
        ]
        for url, relation in row.links:
            links.append(Link(url, relation=relation, locator=row.locator, crawl=False))
        model = ModelHint(
            local_id=f"model:{identity}",
            name=row.name,
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=row.locator,
        )
        release = ReleaseHint(
            local_id=f"release:{identity}",
            model_local_id=model.local_id,
            revision=revision,
            identifiers=(release_identifier,),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "document_path": self.document_path,
                "heading": row.heading,
                "dataset": row.dataset,
                "artifacts": [
                    {"url": url, "relation": relation}
                    for url, relation in row.links
                    if relation in {"weights", "inference_artifact", "model_artifact"}
                ],
            },
            locator=row.locator,
        )
        return SourceRecord(
            source_record_id=f"model:{identity}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(doc_url),
            title=row.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "document_path": self.document_path,
                "document_sha256": content_hash(document),
                "heading": row.heading,
                "dataset": row.dataset,
                "row": row.description,
                "links": [{"url": url, "relation": relation} for url, relation in row.links],
            },
            text="\n".join(
                item for item in (row.name, row.heading, row.dataset, row.description) if item
            ),
            identifiers=(model_identifier,),
            links=tuple(dict.fromkeys(links)),
            models=(model,),
            releases=(release,),
        )


def _row_links(
    cells: tuple[str, ...],
    headers: Sequence[str] = (),
) -> tuple[tuple[str, str], ...]:
    links: list[tuple[str, str]] = []
    for index, cell in enumerate(cells):
        # Authors often use a generic ``link`` label. The source-declared table
        # header is then the only precise relation evidence, so retain it with
        # the label rather than guessing from the URL or table heading.
        header = headers[index] if index < len(headers) else ""
        matches = tuple(_MD_LINK.finditer(cell))
        for match in matches:
            links.append(
                (
                    match.group("url"),
                    _relation(
                        match.group("url"),
                        _label_with_header_evidence(match.group("label"), header),
                    ),
                )
            )
        for url in extract_urls(_MD_LINK.sub("", cell)):
            # This fallback also sees malformed/nested Markdown labels. It has
            # no trustworthy label evidence, so preserve its historic URL-only
            # classification instead of letting a broad column header override
            # a label the lightweight pattern could not parse.
            links.append((url, _relation(url, "")))
    return tuple(dict.fromkeys(links))


def _label_with_header_evidence(label: str, header: str) -> str:
    """Use a table header only when the author supplied no link semantics."""

    # A named label such as ``source``, ``GMM``, or ``model`` is already the
    # most specific evidence the source gives us. Generic ``link`` labels are
    # common in result tables, where ``weight`` or ``log`` only appears above
    # the column. In that one case the header disambiguates without overriding
    # a source-provided label.
    return f"{label} {header}" if label.strip().casefold() == "link" else label


def _row_identity_context(
    cells: Sequence[str],
    *,
    dataset_column: int | None,
    context_columns: Sequence[int],
) -> str | None:
    """Return the configured source-declared context used to distinguish variants."""

    columns = context_columns or (() if dataset_column is None else (dataset_column,))
    values = tuple(
        value for column in columns if column < len(cells) if (value := _plain(cells[column]))
    )
    return " / ".join(values) or None


def _relation(url: str, label: str) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    if host in {"arxiv.org", "export.arxiv.org", "doi.org", "dx.doi.org", "openreview.net"}:
        return "paper_reference"
    if _PAPER_LABEL.search(label):
        return "paper_reference"
    if host == "github.com" and re.search(
        r"(?:^|/)releases/download/[^/]+/[^/]+(?:/|$)", parsed.path, re.I
    ):
        # Release download URLs point to downloadable assets, not repository
        # source pages. They often omit a file extension, so URL-suffix
        # classification alone misses them.
        if _INFERENCE_LABEL.search(label):
            return "inference_artifact"
        if _WEIGHT_LABEL.search(label) or _WEIGHT_SUFFIX.search(parsed.path):
            return "weights"
        return "model_artifact"
    if host == "github.com" and not _WEIGHT_LABEL.search(label):
        return "source_implementation"
    if _INFERENCE_LABEL.search(label):
        return "inference_artifact"
    if _METRICS_LABEL.search(label):
        return "benchmark_results"
    if _LOG_LABEL.search(label):
        return "training_log"
    if _WEIGHT_LABEL.search(label) or _WEIGHT_SUFFIX.search(urlsplit(url).path):
        return "weights"
    return "model_artifact"


def _deduplicate_rows(
    rows: list[_TableRow],
    source: str,
    *,
    include_heading: bool,
) -> tuple[_TableRow, ...]:
    result: dict[tuple[str, str, str | None], _TableRow] = {}
    for row in rows:
        key = (row.heading if include_heading else "", row.name, row.dataset)
        existing = result.get(key)
        if existing is None:
            result[key] = row
            continue
        # A documentation page can repeat one named model for different
        # format/version rows. They are explicit source-local declarations of
        # the same table identity, so retain every direct artifact on the one
        # unified row rather than discarding an older release or inventing a
        # separate model from a version string.
        result[key] = _TableRow(
            name=existing.name,
            dataset=existing.dataset,
            heading=existing.heading,
            description=(
                existing.description
                if row.description == existing.description
                else f"{existing.description}\n{row.description}"
            ),
            links=tuple(dict.fromkeys([*existing.links, *row.links])),
            locator=existing.locator,
        )
    return tuple(result.values())


def _table_cells(line: str) -> tuple[str, ...]:
    body = line.strip()
    if "|" not in body:
        return ()
    cells: list[str] = []
    current: list[str] = []
    code_delimiter = 0
    index = 0
    while index < len(body):
        character = body[index]
        if (
            character == "\\"
            and index + 1 < len(body)
            and body[index + 1] in r"|\`*{}[]()#+-.!_>"
        ):
            current.append(body[index + 1])
            index += 2
            continue
        if character == "`":
            end = index + 1
            while end < len(body) and body[end] == "`":
                end += 1
            run = end - index
            if code_delimiter == 0:
                code_delimiter = run
            elif run == code_delimiter:
                code_delimiter = 0
            current.extend(body[index:end])
            index = end
            continue
        if character == "|" and code_delimiter == 0:
            cells.append("".join(current).strip())
            current.clear()
        else:
            current.append(character)
        index += 1
    cells.append("".join(current).strip())
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return tuple(cells)


def _table_separator(cells: tuple[str, ...]) -> bool:
    return bool(cells) and all(_TABLE_SEPARATOR.fullmatch(cell.replace(" ", "")) for cell in cells)


def _plain(value: str) -> str:
    value = _ANY_MD_LINK.sub(lambda match: match.group("label"), value)
    value = _TEX_DISPLAY_CELL.sub(lambda match: match.group("value"), value)
    return _MARKDOWN.sub("", value).strip()


def _visible_markdown(document: str) -> str:
    """Remove comments and fenced code without shifting Markdown line locators."""

    document = _HTML_COMMENT.sub(lambda match: "\n" * match.group().count("\n"), document)
    lines = document.splitlines(keepends=True)
    visible: list[str] = []
    fence: tuple[str, int] | None = None
    for line in lines:
        match = _FENCE.match(line.rstrip("\r\n"))
        if match:
            marker = match.group("fence")
            if fence is None:
                if marker[0] != "`" or "`" not in match.group("tail"):
                    fence = (marker[0], len(marker))
            elif (
                marker[0] == fence[0]
                and len(marker) >= fence[1]
                and not match.group("tail").strip()
            ):
                fence = None
            visible.append("\n" if line.endswith("\n") else "")
        elif fence is not None:
            visible.append("\n" if line.endswith("\n") else "")
        else:
            visible.append(line)
    return "".join(visible)


def _headings_by_line(lines: Sequence[str]) -> dict[int, str]:
    heading = ""
    result: dict[int, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if match := _HEADING.match(line):
            heading = _plain(match.group("title"))
        result[line_number] = heading
    return result


def _html_tables(document: str, source: str) -> tuple[_HtmlTable, ...]:
    parser = _HtmlTableParser()
    try:
        parser.feed(document)
        parser.close()
    except Exception as error:
        raise ValueError(f"{source}: cannot parse embedded HTML tables") from error
    return tuple(parser.tables)


class _HtmlTableParser(HTMLParser):
    """Extract visible, non-nested HTML table cells as Markdown-compatible text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[_HtmlTable] = []
        self._table_rows: list[tuple[tuple[str, ...], tuple[bool, ...], int]] | None = None
        self._table_depth = 0
        self._row_cells: list[str] | None = None
        self._row_headers: list[bool] | None = None
        self._row_line = 0
        self._cell_parts: list[str] | None = None
        self._cell_is_header = False
        self._anchors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized == "table":
            if self._table_rows is None:
                self._table_rows = []
                self._table_depth = 1
            else:
                self._table_depth += 1
            return
        if self._table_rows is None or self._table_depth != 1:
            return
        if normalized == "tr":
            if self._row_cells is not None:
                self._finish_row()
            self._begin_row()
            return
        if normalized in {"td", "th"} and self._cell_parts is None:
            if self._row_cells is None:
                if normalized != "th":
                    return
                self._begin_row()
            self._cell_parts = []
            self._cell_is_header = normalized == "th"
            return
        if normalized == "a" and self._cell_parts is not None:
            href = next((value for key, value in attrs if key.casefold() == "href"), None)
            if href:
                self._cell_parts.append("[")
                self._anchors.append(href)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized == "table" and self._table_rows is not None:
            if self._table_depth > 1:
                self._table_depth -= 1
                return
            self._finish_table()
            return
        if self._table_rows is None or self._table_depth != 1:
            return
        if normalized == "a" and self._cell_parts is not None and self._anchors:
            self._cell_parts.append(f"]({self._anchors.pop()})")
            return
        if (
            normalized in {"td", "th"}
            and self._cell_parts is not None
            and self._row_cells is not None
        ):
            self._row_cells.append("".join(self._cell_parts).strip())
            assert self._row_headers is not None
            self._row_headers.append(self._cell_is_header)
            self._cell_parts = None
            self._cell_is_header = False
            self._anchors.clear()
            return
        if normalized == "tr" and self._row_cells is not None:
            self._finish_row()
            return

    def handle_data(self, data: str) -> None:
        if self._table_depth == 1 and self._cell_parts is not None:
            self._cell_parts.append(data)

    def _finish_table(self) -> None:
        assert self._table_rows is not None
        if self._row_cells is not None:
            self._finish_row()
        header_index = next(
            (
                index
                for index, (_, cell_headers, _) in enumerate(self._table_rows)
                if cell_headers and all(cell_headers)
            ),
            None,
        )
        if header_index is not None:
            header = self._table_rows[header_index][0]
            rows = tuple(
                (cells, line_number)
                for cells, _, line_number in self._table_rows[header_index + 1 :]
                if cells
            )
            if header and rows:
                self.tables.append(_HtmlTable(header=header, rows=rows))
        self._table_rows = None
        self._table_depth = 0
        self._row_cells = None
        self._row_headers = None

    def _begin_row(self) -> None:
        self._row_cells = []
        self._row_headers = []
        self._row_line = self.getpos()[0]

    def _finish_row(self) -> None:
        assert self._table_rows is not None
        assert self._row_cells is not None
        if self._cell_parts is not None:
            self._row_cells.append("".join(self._cell_parts).strip())
            assert self._row_headers is not None
            self._row_headers.append(self._cell_is_header)
        self._table_rows.append(
            (tuple(self._row_cells), tuple(self._row_headers or ()), self._row_line)
        )
        self._row_cells = None
        self._row_headers = None
        self._cell_parts = None
        self._cell_is_header = False
        self._anchors.clear()
        self._cell_parts = None
        self._anchors.clear()


def _model_name(value: str) -> bool:
    return bool(
        value
        and len(value) <= 300
        and value.casefold() not in {"model", "models", "model type", "acoustic model"}
        and not value.startswith("#")
    )


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _namespace(value: str) -> str:
    namespace = _required_text(value, "provider namespace")
    if any(character.isspace() for character in namespace):
        raise ValueError("provider namespace cannot contain whitespace")
    return namespace


def _safe_path(value: str) -> str:
    path = _required_text(value, "document path")
    invalid_part = any(part in {"", ".", ".."} for part in path.split("/"))
    if path.startswith("/") or "\\" in path or invalid_part:
        raise ValueError("document path is not a safe relative path")
    return path


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


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _nonnegative_from_state(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _header(headers: Mapping[str, Any], key: str) -> str:
    value = headers.get(key) or headers.get(key.casefold()) or headers.get(key.title())
    return _text(value)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
