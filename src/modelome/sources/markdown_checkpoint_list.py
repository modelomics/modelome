"""Pinned checkpoint lists with explicit Markdown-section and handle contracts.

Some projects publish a compact checkpoint list instead of a table.  This
reader is intentionally narrower than a general Markdown scraper: one
configured heading activates the list, every admitted bullet must expose one
configured source-native handle, and it must declare exactly one direct,
recognised checkpoint URL.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpResponse
from modelome.models import SourcePage
from modelome.normalize import canonicalize_url, content_hash
from modelome.sources.static_json_checkpoint_registry import (
    _HANDLE,
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _checkpoint_url,
    _header,
    _isoformat,
    _nonnegative_int,
    _required_text,
    _text,
)

_HEADING = re.compile(r"^(?P<level>#{1,6})[ \t]+(?P<title>.+?)\s*$")
_ITEM = re.compile(r"^[ \t]*[-*+][ \t]+(?P<body>.+?)\s*$")
_LINK = re.compile(r"\[[^\]\r\n]+\]\((?P<url>https?://[^)\s]+)\)")
_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<tail>.*)$")


@dataclass(frozen=True, slots=True)
class _ListCheckpoint(_Checkpoint):
    pass


class MarkdownCheckpointListSourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Read one configured Markdown checkpoint-list section at a pinned commit."""

    coverage_limitation = (
        "Covers explicit checkpoint-list bullets under one configured heading in one "
        "first-party Markdown document at a pinned commit. It does not parse arbitrary "
        "lists, infer model handles or papers, follow artifact URLs, or download weights."
    )

    def __init__(
        self,
        *,
        name: str,
        repository: str,
        branch: str,
        document_path: str,
        section_heading_pattern: str,
        model_handle_pattern: str,
        provider_namespace: str,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 100_000,
        **kwargs: Any,
    ) -> None:
        self.document_path = _required_text(document_path, "document_path")
        self.section_heading_pattern = _pattern(
            section_heading_pattern,
            "section_heading_pattern",
        )
        self.model_handle_pattern = _pattern(
            model_handle_pattern,
            "model_handle_pattern",
        )
        if "handle" not in self.model_handle_pattern.groupindex:
            raise ValueError("model_handle_pattern must declare a named 'handle' group")
        super().__init__(
            name=name,
            repository=repository,
            branch=branch,
            source_path=self.document_path,
            provider_namespace=provider_namespace,
            max_response_bytes=max_response_bytes,
            max_entries=max_entries,
            **kwargs,
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "markdown-checkpoint-list-v1",
                "repository": self.repository,
                "branch": self.branch,
                "document_path": self.document_path,
                "section_heading_pattern": self.section_heading_pattern.pattern,
                "model_handle_pattern": self.model_handle_pattern.pattern,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "one named direct checkpoint per configured list bullet",
            }
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
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision),
            headers={"Accept": "text/markdown,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: checkpoint list returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: checkpoint list exceeds {self.max_response_bytes} bytes"
            )
        checkpoints = _parse_checkpoint_list(
            response.text(),
            source=self.name,
            path=self.document_path,
            heading_pattern=self.section_heading_pattern,
            handle_pattern=self.model_handle_pattern,
            maximum=self.max_entries,
        )
        records = tuple(
            self._record(checkpoint, revision, response.body) for checkpoint in checkpoints
        )
        if not records:
            raise ValueError(f"{self.name}: checkpoint list contains no matching entries")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "document_url": self.raw_url(revision),
            "document_sha256": content_hash(response.body),
            "model_count": len(records),
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


def _parse_checkpoint_list(
    document: str,
    *,
    source: str,
    path: str,
    heading_pattern: re.Pattern[str],
    handle_pattern: re.Pattern[str],
    maximum: int,
) -> tuple[_ListCheckpoint, ...]:
    checkpoints: list[_ListCheckpoint] = []
    handles: set[str] = set()
    active_level: int | None = None
    fence: tuple[str, int] | None = None
    for line_number, line in enumerate(document.splitlines(), start=1):
        if match := _FENCE.match(line):
            marker = match.group("fence")
            if fence is None:
                # A backtick fence info string cannot itself contain backticks.
                if marker[0] != "`" or "`" not in match.group("tail"):
                    fence = (marker[0], len(marker))
            elif (
                marker[0] == fence[0]
                and len(marker) >= fence[1]
                and not match.group("tail").strip()
            ):
                fence = None
            continue
        if fence is not None:
            continue
        if heading := _HEADING.match(line):
            level = len(heading.group("level"))
            if active_level is not None and level <= active_level:
                active_level = None
            if heading_pattern.search(heading.group("title")):
                active_level = level
            continue
        if active_level is None or (item := _ITEM.match(line)) is None:
            continue
        body = item.group("body")
        handle_match = handle_pattern.search(body)
        if handle_match is None:
            raise ValueError(
                f"{source}: list item has no configured model handle at line {line_number}"
            )
        handle = handle_match.group("handle").strip()
        if not _HANDLE.fullmatch(handle) or handle != handle.strip() or ".." in handle.split("/"):
            raise ValueError(f"{source}: invalid checkpoint handle {handle!r}")
        if handle in handles:
            raise ValueError(f"{source}: duplicate checkpoint handle {handle!r}")
        urls = [match.group("url") for match in _LINK.finditer(body)]
        checkpoint_urls = []
        for url in urls:
            try:
                checkpoint_urls.append(_checkpoint_list_url(url, source, handle))
            except ValueError:
                continue
        if len(checkpoint_urls) != 1:
            raise ValueError(
                f"{source}: list item for {handle!r} must declare exactly one direct checkpoint"
            )
        handles.add(handle)
        checkpoints.append(
            _ListCheckpoint(
                handle=handle,
                url=checkpoint_urls[0],
                locator=f"{path}:line:{line_number}",
            )
        )
        if len(checkpoints) > maximum:
            raise ValueError(f"{source}: checkpoint list exceeds {maximum} entries")
    if not checkpoints:
        raise ValueError(f"{source}: checkpoint list contains no matching entries")
    return tuple(checkpoints)


def _checkpoint_list_url(url: str, source: str, handle: str) -> str:
    """Admit known direct object endpoints with no file suffix.

    The surrounding configured heading and literal model handle provide the
    checkpoint context. Keep suffixless admission limited to direct Zenodo
    content URLs and Google Cloud Storage object URLs; arbitrary page URLs stay
    rejected by the shared checkpoint-file validator.
    """

    try:
        return _checkpoint_url(url, source, handle)
    except ValueError:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold()
        segments = tuple(segment for segment in parsed.path.split("/") if segment)
        is_zenodo_content = (
            host in {"zenodo.org", "www.zenodo.org"}
            and len(segments) == 6
            and segments[:2] == ("api", "records")
            and segments[2].isdigit()
            and segments[3] == "files"
            and segments[4]
            and segments[5] == "content"
        )
        is_gcs_object = (
            host == "storage.googleapis.com"
            and len(segments) >= 2
        )
        if parsed.scheme.casefold() == "https" and (is_zenodo_content or is_gcs_object):
            return canonicalize_url(url)
        raise


def _pattern(value: str, field: str) -> re.Pattern[str]:
    try:
        return re.compile(_required_text(value, field), re.I)
    except re.error as error:
        raise ValueError(f"{field} is not a valid regular expression: {error}") from error
