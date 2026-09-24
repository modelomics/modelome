from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit

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
_IGNORED_ELEMENTS = frozenset({"noscript", "script", "style", "template"})
_RULE_KEYS = frozenset(
    {
        "kind",
        "href_pattern",
        "header_pattern",
        "entry_pattern",
        "url_template",
        "name_column",
        "identity_column",
        "name_pattern",
        "skip_nonmatching_names",
        "link_column",
        "identity_pattern",
        "identity_source",
        "raw_href_pattern",
        "name_source",
        "name_template",
        "relation",
        "crawl",
        "require_url",
        "row_link_rules",
    }
)
_SHARED_LINK_RULE_KEYS = frozenset({"href_pattern", "text_pattern", "relation", "crawl"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Anchor:
    href: str
    text: str
    title: str
    locator: str
    attributes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _Cell:
    text: str
    text_before_first_anchor: str
    is_header: bool
    anchors: tuple[_Anchor, ...]
    locator: str


@dataclass(frozen=True, slots=True)
class _Row:
    cells: tuple[_Cell, ...]
    locator: str


@dataclass(frozen=True, slots=True)
class _Table:
    rows: tuple[_Row, ...]
    locator: str


@dataclass(slots=True)
class _MutableAnchor:
    href: str
    title: str
    locator: str
    attributes: dict[str, str]
    text_parts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _MutableCell:
    is_header: bool
    locator: str
    text_parts: list[str] = field(default_factory=list)
    text_before_first_anchor_parts: list[str] = field(default_factory=list)
    first_anchor_seen: bool = False
    anchors: list[_Anchor] = field(default_factory=list)


@dataclass(slots=True)
class _MutableRow:
    locator: str
    cells: list[_Cell] = field(default_factory=list)


@dataclass(slots=True)
class _MutableTable:
    locator: str
    rows: list[_Row] = field(default_factory=list)


class _CatalogHtmlParser(HTMLParser):
    def __init__(
        self,
        *,
        max_anchors: int,
        max_tables: int,
        max_cells: int,
        max_text_chars: int,
    ) -> None:
        super().__init__(convert_charrefs=True)
        self.max_anchors = max_anchors
        self.max_tables = max_tables
        self.max_cells = max_cells
        self.max_text_chars = max_text_chars
        self.anchors: list[_Anchor] = []
        self.tables: list[_Table] = []
        self.title_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False
        self._anchor: _MutableAnchor | None = None
        self._table: _MutableTable | None = None
        self._row: _MutableRow | None = None
        self._cell: _MutableCell | None = None
        # Documentation themes occasionally use an inner layout table inside a
        # table cell. Link extraction is document-global and remains valid there,
        # but treating the inner rows as outer-catalog rows corrupts table rules.
        # Keep parsing anchors while suppressing only the nested table's row/cell
        # structure.
        self._nested_table_depth = 0
        self._anchor_count = 0
        self._cell_count = 0
        self._captured_text_chars = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.casefold()
        if tag in _IGNORED_ELEMENTS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        attributes = {key.casefold(): value or "" for key, value in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "a":
            if self._anchor is not None:
                self._finish_anchor()
            if self._cell is not None:
                self._cell.first_anchor_seen = True
            self._anchor_count += 1
            if self._anchor_count > self.max_anchors:
                raise ValueError(f"HTML document exceeds {self.max_anchors} anchors")
            self._anchor = _MutableAnchor(
                href=attributes.get("href", ""),
                title=attributes.get("title", ""),
                locator=f"html:a[{self._anchor_count - 1}]",
                attributes=attributes,
            )
        elif tag == "table":
            if self._table is not None:
                self._nested_table_depth += 1
                return
            if len(self.tables) >= self.max_tables:
                raise ValueError(f"HTML document exceeds {self.max_tables} tables")
            self._table = _MutableTable(locator=f"html:table[{len(self.tables)}]")
        elif (
            tag == "tr"
            and self._table is not None
            and self._nested_table_depth == 0
        ):
            if self._row is not None:
                self._finish_row()
            self._row = _MutableRow(
                locator=f"{self._table.locator}/tr[{len(self._table.rows)}]"
            )
        elif (
            tag in {"td", "th"}
            and self._row is not None
            and self._nested_table_depth == 0
        ):
            if self._cell is not None:
                self._finish_cell()
            self._cell_count += 1
            if self._cell_count > self.max_cells:
                raise ValueError(f"HTML document exceeds {self.max_cells} table cells")
            self._cell = _MutableCell(
                is_header=tag == "th",
                locator=f"{self._row.locator}/{tag}[{len(self._row.cells)}]",
            )

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in _IGNORED_ELEMENTS:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "a" and self._anchor is not None:
            self._finish_anchor()
        elif (
            tag in {"td", "th"}
            and self._cell is not None
            and self._nested_table_depth == 0
        ):
            self._finish_cell()
        elif tag == "tr" and self._row is not None and self._nested_table_depth == 0:
            self._finish_row()
        elif tag == "table":
            if self._nested_table_depth:
                self._nested_table_depth -= 1
            elif self._table is not None:
                self._finish_table()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth or not data:
            return
        self._captured_text_chars += len(data)
        if self._captured_text_chars > self.max_text_chars:
            raise ValueError(
                f"HTML document exceeds {self.max_text_chars} captured text characters"
            )
        if self._in_title:
            self.title_parts.append(data)
        if self._anchor is not None:
            self._anchor.text_parts.append(data)
        if self._cell is not None and self._nested_table_depth == 0:
            self._cell.text_parts.append(data)
            if not self._cell.first_anchor_seen:
                self._cell.text_before_first_anchor_parts.append(data)

    def finish(self) -> None:
        if self._anchor is not None:
            self._finish_anchor()
        if self._cell is not None:
            self._finish_cell()
        if self._row is not None:
            self._finish_row()
        if self._table is not None:
            self._finish_table()

    def _finish_anchor(self) -> None:
        assert self._anchor is not None
        anchor = _Anchor(
            href=self._anchor.href.strip(),
            text=_normalize_text("".join(self._anchor.text_parts)),
            title=_normalize_text(self._anchor.title),
            locator=self._anchor.locator,
            attributes=dict(self._anchor.attributes),
        )
        self.anchors.append(anchor)
        if self._cell is not None:
            self._cell.anchors.append(anchor)
        self._anchor = None

    def _finish_cell(self) -> None:
        assert self._cell is not None
        assert self._row is not None
        self._row.cells.append(
            _Cell(
                text=_normalize_text("".join(self._cell.text_parts)),
                text_before_first_anchor=_normalize_text(
                    "".join(self._cell.text_before_first_anchor_parts)
                ),
                is_header=self._cell.is_header,
                anchors=tuple(self._cell.anchors),
                locator=self._cell.locator,
            )
        )
        self._cell = None

    def _finish_row(self) -> None:
        if self._cell is not None:
            self._finish_cell()
        assert self._row is not None
        assert self._table is not None
        self._table.rows.append(_Row(cells=tuple(self._row.cells), locator=self._row.locator))
        self._row = None

    def _finish_table(self) -> None:
        if self._row is not None:
            self._finish_row()
        assert self._table is not None
        self.tables.append(
            _Table(rows=tuple(self._table.rows), locator=self._table.locator)
        )
        self._table = None


@dataclass(frozen=True, slots=True)
class _CompiledScopedLinkRule:
    """A source-declared resource link, scoped to one catalog row or document."""

    href_pattern: re.Pattern[str]
    text_pattern: re.Pattern[str] | None
    relation: str
    crawl: bool
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _CompiledRule:
    kind: Literal["link", "table", "text"]
    href_pattern: re.Pattern[str] | None
    header_pattern: re.Pattern[str] | None
    entry_pattern: re.Pattern[str] | None
    url_template: str
    name_template: str
    name_column: int
    identity_column: int | None
    name_pattern: re.Pattern[str] | None
    skip_nonmatching_names: bool
    link_column: int | None
    identity_pattern: re.Pattern[str] | None
    identity_source: Literal["href", "id", "name", "url"]
    raw_href_pattern: re.Pattern[str] | None
    name_source: Literal["text", "identity", "before_first_anchor"]
    relation: str
    crawl: bool
    require_url: bool
    row_link_rules: tuple[_CompiledScopedLinkRule, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _EntryOccurrence:
    name: str
    identity: str
    url: str | None
    relation: str
    crawl: bool
    locator: str
    evidence: Mapping[str, Any]
    links: tuple[Link, ...] = ()


@dataclass(slots=True)
class _EntryGroup:
    identity: str
    name: str
    url: str | None
    locator: str
    page_relation: str
    page_crawl: bool
    aliases: list[str] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    occurrences: list[Mapping[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _DetailTask:
    """One exact, model-scoped documentation page selected by a catalog rule.

    Detail tasks are held in the source checkpoint, rather than inferred later
    from a general web crawl.  That preserves the catalog's exact model identity
    and makes each selected detail page independently restartable.
    """

    identity: str
    name: str
    aliases: tuple[str, ...]
    url: str
    relation: str
    locator: str

    def as_state(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "name": self.name,
            "aliases": list(self.aliases),
            "url": self.url,
            "relation": self.relation,
            "locator": self.locator,
        }


class HtmlCatalogSourceAdapter:
    """Extract official model declarations from a configured document catalog.

    The adapter contains no model or architecture vocabulary. Declarative rules
    select either structurally matched links or rows from a table identified by
    its header. A named-group text rule also handles machine-readable navigation
    manifests and Markdown catalogs. A successful response is treated as one
    complete catalog snapshot; a rule that stops matching fails closed instead
    of deleting prior evidence silently.
    """

    def __init__(
        self,
        *,
        name: str,
        url: str,
        provider_namespace: str,
        rules: Sequence[Mapping[str, Any]],
        shared_link_rules: Sequence[Mapping[str, Any]] = (),
        detail_resource_rules: Sequence[Mapping[str, Any]] = (),
        detail_batch_size: int = 25,
        artifact_kind: str | ArtifactKind = ArtifactKind.CATALOG_RECORD,
        model_status: str | ModelStatus = ModelStatus.DOCUMENTED,
        allowed_origins: Sequence[str] | None = None,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 16 * 1024 * 1024,
        max_entries: int = 20_000,
        max_anchors: int = 100_000,
        max_tables: int = 2_000,
        max_cells: int = 500_000,
        max_text_chars: int = 16 * 1024 * 1024,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name, "catalog URL")
        self.provider_namespace = _namespace(provider_namespace, self.name)
        if isinstance(rules, (str, bytes, bytearray)) or not isinstance(rules, Sequence):
            raise ValueError(f"{self.name}: rules must be an array of tables")
        self.rules = tuple(_compile_rule(rule, self.name) for rule in rules)
        if not self.rules:
            raise ValueError(f"{self.name}: at least one extraction rule is required")
        if isinstance(shared_link_rules, (str, bytes, bytearray)) or not isinstance(
            shared_link_rules, Sequence
        ):
            raise ValueError(f"{self.name}: shared_link_rules must be an array of tables")
        self.shared_link_rules = tuple(
            _compile_shared_link_rule(rule, self.name) for rule in shared_link_rules
        )
        if isinstance(detail_resource_rules, (str, bytes, bytearray)) or not isinstance(
            detail_resource_rules, Sequence
        ):
            raise ValueError(
                f"{self.name}: detail_resource_rules must be an array of tables"
            )
        self.detail_resource_rules = tuple(
            _compile_scoped_link_rule(
                rule,
                self.name,
                label="model detail resource",
            )
            for rule in detail_resource_rules
        )
        self.detail_batch_size = _positive_int(
            detail_batch_size,
            "detail_batch_size",
            self.name,
        )
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.model_status = ModelStatus(model_status)
        configured_origins = tuple(allowed_origins or (self.url,))
        self.allowed_origins = frozenset(
            _origin(_web_url(value, self.name, "allowed origin"))
            for value in configured_origins
        )
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes", self.name)
        self.max_entries = _positive_int(max_entries, "max_entries", self.name)
        self.max_anchors = _positive_int(max_anchors, "max_anchors", self.name)
        self.max_tables = _positive_int(max_tables, "max_tables", self.name)
        self.max_cells = _positive_int(max_cells, "max_cells", self.name)
        self.max_text_chars = _positive_int(max_text_chars, "max_text_chars", self.name)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "html-catalog-v1",
                "url": self.url,
                "provider_namespace": self.provider_namespace,
                "rules": [dict(rule.raw) for rule in self.rules],
                "shared_link_rules": [
                    dict(rule.raw) for rule in self.shared_link_rules
                ],
                "detail_resource_rules": [
                    dict(rule.raw) for rule in self.detail_resource_rules
                ],
                "detail_batch_size": self.detail_batch_size,
                "artifact_kind": self.artifact_kind.value,
                "model_status": self.model_status.value,
                "allowed_origins": sorted(
                    f"{scheme}://{host}:{port}"
                    for scheme, host, port in self.allowed_origins
                ),
                "limits": {
                    "max_response_bytes": self.max_response_bytes,
                    "max_entries": self.max_entries,
                    "max_anchors": self.max_anchors,
                    "max_tables": self.max_tables,
                    "max_cells": self.max_cells,
                    "max_text_chars": self.max_text_chars,
                },
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if "detail_tasks" in state:
            return self._fetch_detail_batch(state)

        # Some documentation hosts redirect any request that advertises Markdown
        # or a generic text representation to a rendered ``.md`` endpoint. Table
        # and link rules operate on HTML structure, so request HTML only; a direct
        # Markdown URL still returns its own text representation to text rules.
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
            return SourcePage(records=(), next_state=next_state, complete=True)
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog exceeds {self.max_response_bytes} bytes"
            )

        document = response.text()
        parser = _CatalogHtmlParser(
            max_anchors=self.max_anchors,
            max_tables=self.max_tables,
            max_cells=self.max_cells,
            max_text_chars=self.max_text_chars,
        )
        parser.feed(document)
        parser.close()
        parser.finish()
        response_base_url = response.url or self.url
        response_url = _web_url(response_base_url, self.name, "response URL")

        occurrences: list[_EntryOccurrence] = []
        for index, rule in enumerate(self.rules):
            matches = self._apply_rule(
                rule,
                parser,
                response_base_url,
                document=document,
            )
            if not matches:
                raise ValueError(
                    f"{self.name}: extraction rule {index} matched no catalog entries"
                )
            occurrences.extend(matches)
            if len(occurrences) > self.max_entries:
                raise ValueError(
                    f"{self.name}: catalog exceeds {self.max_entries} matched entries"
                )

        groups = self._group_entries(occurrences)
        if len(groups) > self.max_entries:
            raise ValueError(
                f"{self.name}: catalog exceeds {self.max_entries} unique entries"
            )
        document_hash = content_hash(response.body)
        shared_links = self._shared_links(parser, response_base_url)
        record = self._record(
            groups,
            shared_links=shared_links,
            parser=parser,
            response=response,
            response_url=response_url,
            document_hash=document_hash,
            document=document,
        )
        next_state: dict[str, Any] = {
            "checked_at": checked_at,
            "content_hash": document_hash,
            "entry_count": len(groups),
        }
        if etag := _header(response.headers, "etag"):
            next_state["etag"] = etag
        if last_modified := _header(response.headers, "last-modified"):
            next_state["http_last_modified"] = last_modified
        if self.detail_resource_rules:
            tasks = self._detail_tasks(groups)
            next_state.update(
                {
                    "detail_tasks": [task.as_state() for task in tasks],
                    "detail_offset": 0,
                    "detail_rule_matches": [0] * len(self.detail_resource_rules),
                }
            )
            return SourcePage(
                records=(record,),
                next_state=next_state,
                complete=False,
                upstream_count=len(groups),
                authoritative_snapshot=False,
            )
        return SourcePage(
            records=(record,),
            next_state=next_state,
            complete=True,
            upstream_count=len(groups),
            authoritative_snapshot=True,
        )

    def _detail_tasks(self, groups: Sequence[_EntryGroup]) -> tuple[_DetailTask, ...]:
        """Freeze the source-declared model-page worklist in checkpoint state."""

        tasks = []
        for group in groups:
            if group.url is None:
                raise ValueError(
                    f"{self.name}: detail resource rules require a model page for "
                    f"{group.identity!r}"
                )
            tasks.append(
                _DetailTask(
                    identity=group.identity,
                    name=group.name,
                    aliases=tuple(group.aliases),
                    url=group.url,
                    relation=group.page_relation,
                    locator=group.locator,
                )
            )
        return tuple(tasks)

    def _fetch_detail_batch(self, state: Mapping[str, Any]) -> SourcePage:
        """Resolve a bounded, resumable slice of exact catalog model pages.

        This is intentionally serial and preserves HTTP retry-after behavior in
        :class:`HttpClient`. It does not infer resource links from an unrelated
        frontier record: every emitted resource is selected by this source's
        configured rule and scoped to the one model whose catalog link named the
        page.
        """

        tasks = self._detail_tasks_from_state(state)
        offset = _nonnegative_int(state.get("detail_offset", 0), "detail_offset", self.name)
        if offset >= len(tasks):
            raise ValueError(f"{self.name}: detail task offset is outside the worklist")
        raw_matches = state.get("detail_rule_matches", [])
        if not isinstance(raw_matches, list) or len(raw_matches) != len(
            self.detail_resource_rules
        ):
            raise ValueError(f"{self.name}: detail rule match state is invalid")
        rule_matches = [
            _nonnegative_int(value, "detail_rule_matches", self.name)
            for value in raw_matches
        ]
        batch = tasks[offset : offset + self.detail_batch_size]
        records: list[SourceRecord] = []
        for task in batch:
            record, matches = self._detail_record(task)
            records.append(record)
            rule_matches = [
                prior + matched
                for prior, matched in zip(rule_matches, matches, strict=True)
            ]

        next_offset = offset + len(batch)
        next_state = {
            key: value
            for key, value in state.items()
            if key not in {"detail_offset", "detail_rule_matches"}
        }
        complete = next_offset == len(tasks)
        if complete:
            next_state.pop("detail_tasks", None)
        else:
            next_state["detail_offset"] = next_offset
            next_state["detail_rule_matches"] = rule_matches
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=_nonnegative_int(
                state.get("entry_count", len(tasks)),
                "entry_count",
                self.name,
            ),
            authoritative_snapshot=complete,
        )

    def _detail_tasks_from_state(self, state: Mapping[str, Any]) -> tuple[_DetailTask, ...]:
        raw_tasks = state.get("detail_tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError(f"{self.name}: detail task state is invalid")
        if len(raw_tasks) > self.max_entries:
            raise ValueError(f"{self.name}: detail task state exceeds {self.max_entries} entries")
        tasks = []
        for index, raw_task in enumerate(raw_tasks):
            if not isinstance(raw_task, Mapping):
                raise ValueError(f"{self.name}: detail task {index} is not an object")
            aliases = raw_task.get("aliases", [])
            if not isinstance(aliases, list) or not all(
                isinstance(alias, str) and alias.strip() for alias in aliases
            ):
                raise ValueError(f"{self.name}: detail task {index} aliases are invalid")
            url = _web_url(raw_task.get("url"), self.name, f"detail task {index} URL")
            self._assert_allowed_model_url(url)
            tasks.append(
                _DetailTask(
                    identity=_required_text(
                        raw_task.get("identity"),
                        f"detail task {index} identity",
                    ),
                    name=_required_text(raw_task.get("name"), f"detail task {index} name"),
                    aliases=tuple(_normalize_text(alias) for alias in aliases),
                    url=url,
                    relation=_required_relation(
                        raw_task.get("relation"),
                        f"detail task {index} relation",
                        self.name,
                    ),
                    locator=_required_text(
                        raw_task.get("locator"),
                        f"detail task {index} locator",
                    ),
                )
            )
        return tuple(tasks)

    def _detail_record(
        self,
        task: _DetailTask,
    ) -> tuple[SourceRecord, tuple[int, ...]]:
        response: HttpResponse = self.client.get(
            task.url,
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        if response.status != 200:
            raise ValueError(
                f"{self.name}: model detail page {task.url} returned HTTP {response.status}"
            )
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: model detail page {task.url} exceeds "
                f"{self.max_response_bytes} bytes"
            )
        response_url = _web_url(response.url or task.url, self.name, "detail response URL")
        self._assert_allowed_model_url(response_url)
        parser = _CatalogHtmlParser(
            max_anchors=self.max_anchors,
            max_tables=self.max_tables,
            max_cells=self.max_cells,
            max_text_chars=self.max_text_chars,
        )
        parser.feed(response.text())
        parser.close()
        parser.finish()
        resources, matches = self._detail_links(parser, response.url or task.url)
        local_id = _model_local_id(self.provider_namespace, task.identity)
        links = [
            Link(
                task.url,
                relation=task.relation,
                locator=task.locator,
                crawl=False,
                model_local_ids=(local_id,),
            )
        ]
        links.extend(
            Link(
                resource.url,
                relation=resource.relation,
                locator=resource.locator,
                crawl=resource.crawl,
                model_local_ids=(local_id,),
            )
            for resource in resources
        )
        title = _normalize_text("".join(parser.title_parts)) or task.name
        return (
            SourceRecord(
                source_record_id=(
                    f"{self.name}:model-page:{content_hash(task.identity)[:24]}"
                ),
                kind=self.artifact_kind,
                canonical_url=response_url,
                title=title,
                raw={
                    "catalog_url": self.url,
                    "model_identity": task.identity,
                    "model_name": task.name,
                    "source_declared_page_url": task.url,
                    "response_url": response_url,
                    "content_hash": content_hash(response.body),
                    "content_type": _header(response.headers, "content-type"),
                    "content_bytes": len(response.body),
                    "resource_links": [
                        {
                            "url": link.url,
                            "relation": link.relation,
                            "locator": link.locator,
                            "crawl": link.crawl,
                        }
                        for link in resources
                    ],
                },
                modified_at=_header(response.headers, "last-modified"),
                identifiers=(Identifier(self.provider_namespace, task.identity),),
                links=tuple(links),
                models=(
                    ModelHint(
                        local_id=local_id,
                        name=task.name,
                        aliases=task.aliases,
                        identifiers=(Identifier(self.provider_namespace, task.identity),),
                        status=self.model_status,
                        locator=f"detail:{task.locator}",
                    ),
                ),
            ),
            matches,
        )

    def _detail_links(
        self,
        parser: _CatalogHtmlParser,
        response_url: str,
    ) -> tuple[tuple[Link, ...], tuple[int, ...]]:
        links: list[Link] = []
        seen: set[tuple[str, str, bool]] = set()
        matches: list[int] = []
        for rule in self.detail_resource_rules:
            matched = 0
            for anchor in parser.anchors:
                url = self._resolved_url(anchor.href, response_url)
                if url is None or not rule.href_pattern.search(url):
                    continue
                if rule.text_pattern is not None and not rule.text_pattern.search(
                    anchor.text or anchor.title
                ):
                    continue
                self._assert_allowed_model_url(url)
                matched += 1
                key = (url, rule.relation, rule.crawl)
                if key not in seen:
                    seen.add(key)
                    links.append(
                        Link(
                            url,
                            relation=rule.relation,
                            locator=anchor.locator,
                            crawl=rule.crawl,
                        )
                    )
            matches.append(matched)
        return tuple(links), tuple(matches)

    def _apply_rule(
        self,
        rule: _CompiledRule,
        parser: _CatalogHtmlParser,
        response_url: str,
        *,
        document: str,
    ) -> tuple[_EntryOccurrence, ...]:
        if rule.kind == "text":
            assert rule.entry_pattern is not None
            result = []
            for match in rule.entry_pattern.finditer(document):
                groups = {
                    key: _normalize_text(value or "")
                    for key, value in match.groupdict().items()
                }
                name = groups["name"]
                if rule.name_template:
                    name = _normalize_text(
                        _render_template(
                            rule.name_template,
                            groups,
                            source=self.name,
                            label="name_template",
                        )
                    )
                if not name:
                    raise ValueError(
                        f"{self.name}: text rule produced an empty name at "
                        f"text:{match.start()}-{match.end()}"
                    )
                upstream_id = groups.get("id") or None
                captured_url = groups.get("url") or ""
                rendered_url = _render_template(
                    rule.url_template,
                    groups,
                    source=self.name,
                    label="url_template",
                )
                raw_url = rendered_url or captured_url
                url = self._resolved_url(raw_url, response_url) if raw_url else None
                if url is not None:
                    self._assert_allowed_model_url(url)
                if rule.require_url and url is None:
                    raise ValueError(
                        f"{self.name}: selected text entry at "
                        f"text:{match.start()}-{match.end()} has no URL"
                    )
                identity = _entry_identity(
                    rule,
                    name=name,
                    url=url,
                    upstream_id=upstream_id,
                    source=self.name,
                )
                locator = f"text:{match.start()}-{match.end()}"
                result.append(
                    _EntryOccurrence(
                        name=name,
                        identity=identity,
                        url=url,
                        relation=rule.relation,
                        crawl=rule.crawl,
                        locator=locator,
                        evidence={
                            "kind": "text_match",
                            "locator": locator,
                            "match": match.group(0),
                            "groups": groups,
                            "url": url,
                        },
                    )
                )
            return tuple(result)

        if rule.kind == "link":
            assert rule.href_pattern is not None
            result = []
            for anchor in parser.anchors:
                raw_href = self._resolved_href(anchor.href, response_url)
                url = self._resolved_url(anchor.href, response_url)
                if url is None or not rule.href_pattern.search(url):
                    continue
                if (
                    rule.raw_href_pattern is not None
                    and (raw_href is None or not rule.raw_href_pattern.search(raw_href))
                ):
                    continue
                self._assert_allowed_model_url(url)
                name = anchor.text or anchor.title
                if not name and rule.name_source == "text":
                    raise ValueError(
                        f"{self.name}: selected link at {anchor.locator} has no name"
                    )
                identity = _entry_identity(
                    rule,
                    name=name,
                    url=raw_href if rule.identity_source == "href" else url,
                    upstream_id=None,
                    source=self.name,
                )
                if rule.name_source == "identity":
                    name = identity
                result.append(
                    _EntryOccurrence(
                        name=name,
                        identity=identity,
                        url=url,
                        relation=rule.relation,
                        crawl=rule.crawl,
                        locator=anchor.locator,
                        evidence={
                            "kind": "link",
                            "locator": anchor.locator,
                            "href": anchor.href,
                            "url": url,
                            "text": anchor.text,
                            "title": anchor.title,
                            "attributes": dict(anchor.attributes),
                        },
                    )
                )
            return tuple(result)

        assert rule.header_pattern is not None
        result = []
        row_link_matches = [0 for _ in rule.row_link_rules]
        for table in parser.tables:
            header_index = _matching_header_row(table, rule.header_pattern)
            if header_index is None:
                continue
            for row in table.rows[header_index + 1 :]:
                if rule.name_column >= len(row.cells):
                    continue
                name_cell = row.cells[rule.name_column]
                name = (
                    name_cell.text_before_first_anchor
                    if rule.name_source == "before_first_anchor"
                    else name_cell.text
                )
                if not name or all(cell.is_header for cell in row.cells):
                    continue
                name = _table_entry_name(rule, name, self.name)
                if name is None:
                    continue
                link_cell = (
                    row.cells[rule.link_column]
                    if rule.link_column is not None and rule.link_column < len(row.cells)
                    else name_cell
                )
                url = self._selected_table_url(rule, link_cell.anchors, response_url)
                if rule.require_url and url is None:
                    raise ValueError(
                        f"{self.name}: selected table row at {row.locator} has no matching URL"
                    )
                upstream_id = (
                    row.cells[rule.identity_column].text
                    if rule.identity_column is not None
                    and rule.identity_column < len(row.cells)
                    else None
                )
                if rule.identity_column is not None and not upstream_id:
                    raise ValueError(
                        f"{self.name}: selected table row at {row.locator} "
                        "has no identity-column value"
                    )
                identity = _entry_identity(
                    rule,
                    name=name,
                    url=url,
                    upstream_id=upstream_id,
                    source=self.name,
                )
                row_links, matches = self._row_links(
                    row,
                    rule.row_link_rules,
                    response_url,
                )
                row_link_matches = [
                    prior + matched
                    for prior, matched in zip(row_link_matches, matches, strict=True)
                ]
                result.append(
                    _EntryOccurrence(
                        name=name,
                        identity=identity,
                        url=url,
                        relation=rule.relation,
                        crawl=rule.crawl,
                        locator=name_cell.locator,
                        evidence={
                            "kind": "table_row",
                            "locator": row.locator,
                            "cells": [cell.text for cell in row.cells],
                            "cell_locators": [cell.locator for cell in row.cells],
                            "url": url,
                            "scoped_links": [
                                {
                                    "url": link.url,
                                    "relation": link.relation,
                                    "locator": link.locator,
                                    "crawl": link.crawl,
                                }
                                for link in row_links
                            ],
                        },
                        links=row_links,
                    )
                )
        for index, matched in enumerate(row_link_matches):
            if matched == 0:
                raise ValueError(
                    f"{self.name}: table row link rule {index} matched no catalog links"
                )
        return tuple(result)

    def _row_links(
        self,
        row: _Row,
        rules: Sequence[_CompiledScopedLinkRule],
        response_url: str,
    ) -> tuple[tuple[Link, ...], tuple[int, ...]]:
        """Collect direct resource declarations from one selected catalog row.

        A row rule is deliberately evaluated only within the selected row. This
        prevents a paper, code repository, or weight link for one model from
        being broadcast to every entry in a catalogue table.
        """

        links: list[Link] = []
        seen: set[tuple[str, str, bool]] = set()
        matches: list[int] = []
        for rule in rules:
            matched = 0
            for cell in row.cells:
                for anchor in cell.anchors:
                    url = self._resolved_url(anchor.href, response_url)
                    if url is None or not rule.href_pattern.search(url):
                        continue
                    if rule.text_pattern is not None and not rule.text_pattern.search(
                        anchor.text or anchor.title
                    ):
                        continue
                    self._assert_allowed_model_url(url)
                    matched += 1
                    key = (url, rule.relation, rule.crawl)
                    if key not in seen:
                        seen.add(key)
                        links.append(
                            Link(
                                url,
                                relation=rule.relation,
                                locator=anchor.locator,
                                crawl=rule.crawl,
                            )
                        )
            matches.append(matched)
        return tuple(links), tuple(matches)

    def _shared_links(
        self,
        parser: _CatalogHtmlParser,
        response_url: str,
    ) -> tuple[Link, ...]:
        links: list[Link] = []
        seen: set[tuple[str, str, bool]] = set()
        for index, rule in enumerate(self.shared_link_rules):
            matched = False
            for anchor in parser.anchors:
                url = self._resolved_url(anchor.href, response_url)
                if url is None or not rule.href_pattern.search(url):
                    continue
                if rule.text_pattern is not None and not rule.text_pattern.search(
                    anchor.text or anchor.title
                ):
                    continue
                self._assert_allowed_model_url(url)
                matched = True
                key = (url, rule.relation, rule.crawl)
                if key not in seen:
                    seen.add(key)
                    links.append(
                        Link(
                            url,
                            relation=rule.relation,
                            locator=anchor.locator,
                            crawl=rule.crawl,
                        )
                    )
            if not matched:
                raise ValueError(
                    f"{self.name}: shared link rule {index} matched no catalog links"
                )
        return tuple(links)

    def _selected_table_url(
        self,
        rule: _CompiledRule,
        anchors: Sequence[_Anchor],
        response_url: str,
    ) -> str | None:
        for anchor in anchors:
            url = self._resolved_url(anchor.href, response_url)
            if url is None:
                continue
            if rule.href_pattern is not None and not rule.href_pattern.search(url):
                continue
            self._assert_allowed_model_url(url)
            return url
        return None

    def _resolved_url(self, href: str, response_url: str) -> str | None:
        if not href:
            return None
        candidate = canonicalize_url(urljoin(response_url, href))
        parts = urlsplit(candidate)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        return candidate

    def _resolved_href(self, href: str, response_url: str) -> str | None:
        """Resolve an original link without discarding its fragment.

        Canonical resource URLs intentionally discard fragments because two anchors on
        one document are one crawl target. Some documentation hosts, however, place a
        stable fully-qualified API identifier in that fragment. A link rule can opt in
        to use it only as source identity while retaining the canonical page URL as the
        resource reference.
        """

        if not href:
            return None
        candidate = urljoin(response_url, href.strip())
        parts = urlsplit(candidate)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        return candidate

    def _assert_allowed_model_url(self, url: str) -> None:
        if _origin(url) not in self.allowed_origins:
            raise ValueError(f"{self.name}: selected model URL is outside allowed origins")

    def _group_entries(
        self, occurrences: Sequence[_EntryOccurrence]
    ) -> tuple[_EntryGroup, ...]:
        grouped: dict[str, _EntryGroup] = {}
        for occurrence in occurrences:
            group = grouped.get(occurrence.identity)
            if group is None:
                group = _EntryGroup(
                    identity=occurrence.identity,
                    name=occurrence.name,
                    url=occurrence.url,
                    locator=occurrence.locator,
                    page_relation=occurrence.relation,
                    page_crawl=occurrence.crawl,
                )
                grouped[occurrence.identity] = group
            elif occurrence.name != group.name and occurrence.name not in group.aliases:
                group.aliases.append(occurrence.name)
            if group.url is None and occurrence.url is not None:
                group.url = occurrence.url
            if occurrence.url is not None:
                group.links.append(
                    Link(
                        occurrence.url,
                        relation=occurrence.relation,
                        locator=occurrence.locator,
                        crawl=occurrence.crawl,
                    )
                )
            group.links.extend(occurrence.links)
            group.occurrences.append(dict(occurrence.evidence))
        return tuple(grouped.values())

    def _record(
        self,
        groups: Sequence[_EntryGroup],
        *,
        shared_links: Sequence[Link],
        parser: _CatalogHtmlParser,
        response: HttpResponse,
        response_url: str,
        document_hash: str,
        document: str,
    ) -> SourceRecord:
        models: list[ModelHint] = []
        links: list[Link] = []
        seen_links: set[tuple[str, str, bool, tuple[str, ...]]] = set()
        raw_entries = []
        text_lines = []
        for group in groups:
            identifier = Identifier(self.provider_namespace, group.identity)
            local_id = _model_local_id(self.provider_namespace, group.identity)
            models.append(
                ModelHint(
                    local_id=local_id,
                    name=group.name,
                    identifiers=(identifier,),
                    aliases=tuple(group.aliases),
                    status=self.model_status,
                    locator=group.locator,
                )
            )
            text_lines.append(group.name)
            for link in group.links:
                scoped_link = Link(
                    link.url,
                    relation=link.relation,
                    locator=link.locator,
                    crawl=link.crawl,
                    model_local_ids=(local_id,),
                )
                link_key = (
                    scoped_link.url,
                    scoped_link.relation,
                    scoped_link.crawl,
                    scoped_link.model_local_ids,
                )
                if link_key not in seen_links:
                    seen_links.add(link_key)
                    links.append(scoped_link)
            raw_entries.append(
                {
                    "identity": group.identity,
                    "name": group.name,
                    "aliases": list(group.aliases),
                    "url": group.url,
                    "occurrences": list(group.occurrences),
                }
            )
        for link in shared_links:
            link_key = (link.url, link.relation, link.crawl, link.model_local_ids)
            if link_key not in seen_links:
                seen_links.add(link_key)
                links.append(link)
        page_title = _normalize_text("".join(parser.title_parts)) or self.name
        return SourceRecord(
            source_record_id=f"{self.name}:catalog",
            kind=self.artifact_kind,
            canonical_url=self.url,
            title=page_title,
            raw={
                "catalog_url": self.url,
                "response_url": response_url,
                "content_hash": document_hash,
                "content_type": _header(response.headers, "content-type"),
                "content_bytes": len(response.body),
                "document_text": document,
                "etag": _header(response.headers, "etag"),
                "last_modified": _header(response.headers, "last-modified"),
                "entries": raw_entries,
                "shared_links": [
                    {
                        "url": link.url,
                        "relation": link.relation,
                        "locator": link.locator,
                        "crawl": link.crawl,
                    }
                    for link in shared_links
                ],
            },
            text="\n".join(text_lines),
            modified_at=_header(response.headers, "last-modified"),
            links=tuple(links),
            models=tuple(models),
        )


def _compile_rule(value: Mapping[str, Any], source: str) -> _CompiledRule:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: each extraction rule must be a table")
    unknown = sorted(set(value) - _RULE_KEYS)
    if unknown:
        raise ValueError(f"{source}: unknown extraction rule field(s): {', '.join(unknown)}")
    raw = dict(value)
    kind = _text(value.get("kind")).casefold()
    if kind not in {"link", "table", "text"}:
        raise ValueError(f"{source}: rule.kind must be 'link', 'table', or 'text'")
    href_pattern = _optional_pattern(value.get("href_pattern"), source, "href_pattern")
    header_pattern = _optional_pattern(
        value.get("header_pattern"), source, "header_pattern"
    )
    entry_pattern = _optional_pattern(
        value.get("entry_pattern"), source, "entry_pattern"
    )
    if kind == "link" and href_pattern is None:
        raise ValueError(f"{source}: link rule requires href_pattern")
    if kind == "table" and header_pattern is None:
        raise ValueError(f"{source}: table rule requires header_pattern")
    if kind == "text":
        if entry_pattern is None:
            raise ValueError(f"{source}: text rule requires entry_pattern")
        if "name" not in entry_pattern.groupindex:
            raise ValueError(
                f"{source}: text rule entry_pattern requires a named 'name' group"
            )
    url_template = _text(value.get("url_template"))
    if len(url_template) > 8_192:
        raise ValueError(f"{source}: url_template is too long")
    if (
        kind == "text"
        and entry_pattern is not None
        and url_template
        and "url" in entry_pattern.groupindex
    ):
        raise ValueError(
            f"{source}: text rule must use either a URL capture or url_template"
        )
    name_template = _text(value.get("name_template"))
    if name_template and kind != "text":
        raise ValueError(f"{source}: name_template is supported only by text rules")
    if len(name_template) > 8_192:
        raise ValueError(f"{source}: name_template is too long")
    default_identity = (
        "id"
        if kind == "text" and entry_pattern is not None and "id" in entry_pattern.groupindex
        else "name"
    )
    identity_source = _text(value.get("identity_source")).casefold() or default_identity
    if identity_source not in {"href", "id", "name", "url"}:
        raise ValueError(
            f"{source}: identity_source must be 'href', 'id', 'name', or 'url'"
        )
    if "identity_column" in value and kind != "table":
        raise ValueError(f"{source}: identity_column is supported only by table rules")
    identity_column = (
        _nonnegative_int(value.get("identity_column"), "identity_column", source)
        if "identity_column" in value
        else None
    )
    has_text_id = (
        kind == "text"
        and entry_pattern is not None
        and "id" in entry_pattern.groupindex
    )
    if identity_source == "id" and not (has_text_id or identity_column is not None):
        raise ValueError(
            f"{source}: id identity requires a text rule named 'id' group "
            "or a table identity_column"
        )
    if identity_source == "href" and kind != "link":
        raise ValueError(f"{source}: href identity is supported only by link rules")
    name_source = _text(value.get("name_source")).casefold() or "text"
    if name_source not in {"text", "identity", "before_first_anchor"}:
        raise ValueError(
            f"{source}: name_source must be 'text', 'identity', or 'before_first_anchor'"
        )
    if name_source == "identity" and kind != "link":
        raise ValueError(f"{source}: identity name_source is supported only by link rules")
    if name_source == "before_first_anchor" and kind != "table":
        raise ValueError(
            f"{source}: before_first_anchor name_source is supported only by table rules"
        )
    relation = _text(value.get("relation")) or "model_page"
    if (
        len(relation) > 128
        or any(character.isspace() or ord(character) < 32 for character in relation)
    ):
        raise ValueError(f"{source}: relation must be a compact relation label")
    require_url = _boolean(value.get("require_url"), default=identity_source == "url")
    name_column = _nonnegative_int(value.get("name_column", 0), "name_column", source)
    name_pattern = _optional_pattern(value.get("name_pattern"), source, "name_pattern")
    if name_pattern is not None:
        if kind != "table":
            raise ValueError(f"{source}: name_pattern is supported only by table rules")
        if "name" not in name_pattern.groupindex:
            raise ValueError(f"{source}: name_pattern requires a named 'name' group")
    skip_nonmatching_names = _boolean(
        value.get("skip_nonmatching_names"), default=False
    )
    if skip_nonmatching_names and name_pattern is None:
        raise ValueError(
            f"{source}: skip_nonmatching_names requires name_pattern"
        )
    raw_href_pattern = _optional_pattern(
        value.get("raw_href_pattern"), source, "raw_href_pattern"
    )
    if raw_href_pattern is not None and kind != "link":
        raise ValueError(f"{source}: raw_href_pattern is supported only by link rules")
    link_column = (
        _nonnegative_int(value.get("link_column"), "link_column", source)
        if "link_column" in value
        else name_column
    )
    raw_row_link_rules = value.get("row_link_rules", ())
    if isinstance(raw_row_link_rules, (str, bytes, bytearray)) or not isinstance(
        raw_row_link_rules, Sequence
    ):
        raise ValueError(f"{source}: row_link_rules must be an array of tables")
    if raw_row_link_rules and kind != "table":
        raise ValueError(f"{source}: row_link_rules are supported only by table rules")
    row_link_rules = tuple(
        _compile_scoped_link_rule(
            item,
            source,
            label="table row link",
        )
        for item in raw_row_link_rules
    )
    return _CompiledRule(
        kind=kind,
        href_pattern=href_pattern,
        header_pattern=header_pattern,
        entry_pattern=entry_pattern,
        url_template=url_template,
        name_template=name_template,
        name_column=name_column,
        identity_column=identity_column,
        name_pattern=name_pattern,
        skip_nonmatching_names=skip_nonmatching_names,
        link_column=link_column,
        identity_pattern=_optional_pattern(
            value.get("identity_pattern"), source, "identity_pattern"
        ),
        identity_source=identity_source,
        raw_href_pattern=raw_href_pattern,
        name_source=name_source,
        relation=relation,
        crawl=_boolean(value.get("crawl"), default=True),
        require_url=require_url,
        row_link_rules=row_link_rules,
        raw=raw,
    )


def _compile_shared_link_rule(
    value: Mapping[str, Any], source: str
) -> _CompiledScopedLinkRule:
    return _compile_scoped_link_rule(value, source, label="shared link")


def _compile_scoped_link_rule(
    value: Mapping[str, Any],
    source: str,
    *,
    label: str,
) -> _CompiledScopedLinkRule:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: each {label} rule must be a table")
    unknown = sorted(set(value) - _SHARED_LINK_RULE_KEYS)
    if unknown:
        raise ValueError(
            f"{source}: unknown {label} rule field(s): {', '.join(unknown)}"
        )
    href_pattern = _optional_pattern(value.get("href_pattern"), source, "href_pattern")
    if href_pattern is None:
        raise ValueError(f"{source}: {label} rule requires href_pattern")
    text_pattern = _optional_pattern(value.get("text_pattern"), source, "text_pattern")
    relation = _text(value.get("relation")) or "references"
    if (
        len(relation) > 128
        or any(character.isspace() or ord(character) < 32 for character in relation)
    ):
        raise ValueError(f"{source}: relation must be a compact relation label")
    return _CompiledScopedLinkRule(
        href_pattern=href_pattern,
        text_pattern=text_pattern,
        relation=relation,
        crawl=_boolean(value.get("crawl"), default=True),
        raw=dict(value),
    )


def _matching_header_row(table: _Table, pattern: re.Pattern[str]) -> int | None:
    for index, row in enumerate(table.rows):
        if not any(cell.is_header for cell in row.cells):
            continue
        header = " | ".join(cell.text for cell in row.cells)
        if pattern.search(header):
            return index
    return None


def _entry_identity(
    rule: _CompiledRule,
    *,
    name: str,
    url: str | None,
    upstream_id: str | None,
    source: str,
) -> str:
    if rule.identity_source == "id":
        if upstream_id is None:
            raise ValueError(f"{source}: ID identity requested for an entry without an ID")
        value = upstream_id
    elif rule.identity_source in {"href", "url"}:
        if url is None:
            label = "href" if rule.identity_source == "href" else "URL"
            raise ValueError(f"{source}: {label} identity requested for an entry without a URL")
        value = url
    else:
        value = name
    if rule.identity_pattern is None:
        return value
    match = rule.identity_pattern.search(value)
    if match is None:
        raise ValueError(f"{source}: selected entry did not match identity_pattern")
    if "id" in match.groupdict():
        identity = match.group("id")
    elif match.lastindex:
        identity = match.group(1)
    else:
        identity = match.group(0)
    identity = _text(identity)
    if not identity:
        raise ValueError(f"{source}: identity_pattern produced an empty identity")
    return identity


def _table_entry_name(rule: _CompiledRule, value: str, source: str) -> str | None:
    """Optionally remove documented table decorations from a model label.

    Sphinx-style model tables commonly append a numeric footnote reference to
    the visible model name. A configured full-match pattern can retain the
    source label's named ``name`` capture without treating the footnote marker
    as part of the entry identity. A source can opt to skip cells that do not
    match that pattern, which lets one structural matrix omit explicit ``N/A``
    cells without turning them into false model declarations.
    """

    if rule.name_pattern is None:
        return value
    match = rule.name_pattern.fullmatch(value)
    if match is None:
        if rule.skip_nonmatching_names:
            return None
        raise ValueError(f"{source}: selected table name did not match name_pattern")
    result = _normalize_text(match.group("name") or "")
    if not result:
        raise ValueError(f"{source}: name_pattern produced an empty name")
    return result


def _render_template(
    template: str,
    groups: Mapping[str, str],
    *,
    source: str,
    label: str,
) -> str:
    if not template:
        return ""
    try:
        return template.format_map(groups)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{source}: {label} references an unavailable group") from error


def _optional_pattern(value: Any, source: str, label: str) -> re.Pattern[str] | None:
    text = _text(value)
    if not text:
        return None
    if len(text) > 2_048:
        raise ValueError(f"{source}: {label} is too long")
    try:
        return re.compile(text)
    except re.error as error:
        raise ValueError(f"{source}: invalid {label}: {error}") from error


def _namespace(value: Any, source: str) -> str:
    result = _required_text(value, "provider namespace")
    if any(character.isspace() or ord(character) < 32 for character in result):
        raise ValueError(f"{source}: invalid provider namespace")
    return result.casefold()


def _model_local_id(provider_namespace: str, identity: str) -> str:
    return f"{provider_namespace}:{content_hash(identity)[:24]}#model"


def _required_relation(value: Any, label: str, source: str) -> str:
    relation = _required_text(value, label)
    if (
        len(relation) > 128
        or any(character.isspace() or ord(character) < 32 for character in relation)
    ):
        raise ValueError(f"{source}: {label} must be a compact relation label")
    return relation


def _web_url(value: Any, source: str, label: str) -> str:
    result = canonicalize_url(_required_text(value, label))
    parts = urlsplit(result)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"{source}: {label} must be an absolute HTTP(S) URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"{source}: {label} must not contain credentials")
    return result


def _origin(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    scheme = parts.scheme.casefold()
    port = parts.port if parts.port is not None else (443 if scheme == "https" else 80)
    return scheme, (parts.hostname or "").casefold(), port


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == wanted), None)


def _positive_int(value: Any, label: str, source: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: {label} must be an integer") from None
    if result < 1:
        raise ValueError(f"{source}: {label} must be positive")
    return result


def _nonnegative_int(value: Any, label: str, source: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: {label} must be an integer") from None
    if result < 0:
        raise ValueError(f"{source}: {label} must be nonnegative")
    return result


def _boolean(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("source boolean settings must be true or false")
    return value


def _required_text(value: Any, label: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{label} is required")
    return result


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["HtmlCatalogSourceAdapter"]
