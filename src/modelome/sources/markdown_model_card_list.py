"""Pinned Markdown sections that declare exact public model-card URLs.

This is deliberately narrower than a Markdown scraper. A configuration selects
one heading and one complete URL shape with a source-native ``handle`` capture.
Only matching public card links below that heading become released model records;
prose, binaries, and links outside the section remain ordinary source evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
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
from modelome.normalize import canonicalize_url, content_hash
from modelome.sources.static_json_checkpoint_registry import (
    _COMMIT,
    _HANDLE,
    _NAMESPACE,
    _REPOSITORY,
    _header,
    _isoformat,
    _nonnegative_int,
    _positive_int,
    _required_text,
    _text,
    _utcnow,
)

_HEADING = re.compile(r"^(?P<level>#{1,6})[ \t]+(?P<title>.+?)\s*$")
_LINK = re.compile(r"\[[^\]\r\n]+\]\((?P<url>https?://[^)\s]+)\)")
_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<tail>.*)$")


@dataclass(frozen=True, slots=True)
class _ModelCard:
    handle: str
    url: str
    locator: str


class MarkdownModelCardListSourceAdapter:
    """Read source-declared card links in one pinned Markdown document section."""

    disable_derived_extraction = True

    def __init__(
        self,
        *,
        name: str,
        repository: str,
        branch: str,
        document_path: str,
        section_heading_pattern: str,
        model_url_pattern: str,
        provider_namespace: str,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 100_000,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.document_path = _safe_path(document_path)
        self.section_heading_pattern = _pattern(
            section_heading_pattern,
            "section_heading_pattern",
        )
        self.model_url_pattern = _pattern(model_url_pattern, "model_url_pattern")
        if "handle" not in self.model_url_pattern.groupindex:
            raise ValueError("model_url_pattern must declare a named 'handle' group")
        self.provider_namespace = _namespace(provider_namespace)
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "markdown-model-card-list-v1",
                "repository": self.repository,
                "branch": self.branch,
                "document_path": self.document_path,
                "section_heading_pattern": self.section_heading_pattern.pattern,
                "model_url_pattern": self.model_url_pattern.pattern,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "one exact source-declared model-card URL shape",
            }
        )

    @property
    def coverage_limitation(self) -> str:
        return (
            "Covers only exact model-card URLs under one configured Markdown heading "
            f"in {self.repository}/{self.document_path} at one observed commit. It does "
            "not infer models from prose, follow cards, or download files."
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
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision),
            headers={"Accept": "text/markdown,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: model-card list returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: model-card list exceeds {self.max_response_bytes} bytes"
            )
        cards = _parse_model_cards(
            response.text(),
            source=self.name,
            path=self.document_path,
            heading_pattern=self.section_heading_pattern,
            url_pattern=self.model_url_pattern,
            maximum=self.max_entries,
        )
        records = tuple(self._record(card, revision, response.body) for card in cards)
        if not records:
            raise ValueError(f"{self.name}: model-card list contains no matching entries")
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

    def _record(self, card: _ModelCard, revision: str, document: bytes) -> SourceRecord:
        model_identifier = Identifier(self.provider_namespace, card.handle)
        identifiers = (model_identifier, *_huggingface_identifiers(card.url))
        model = ModelHint(
            local_id=f"model:{card.handle}",
            name=card.handle,
            identifiers=identifiers,
            status=ModelStatus.RELEASED,
            locator=card.locator,
        )
        release = ReleaseHint(
            local_id=f"release:{card.handle}",
            model_local_id=model.local_id,
            revision=revision,
            identifiers=(Identifier(f"{self.provider_namespace}-release", card.handle),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "document_path": self.document_path,
                "model_card": card.url,
            },
            locator=card.locator,
        )
        document_url = self.blob_url(revision)
        return SourceRecord(
            source_record_id=f"model:{card.handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(document_url),
            title=card.handle,
            raw={
                "repository": self.repository,
                "revision": revision,
                "document_path": self.document_path,
                "document_sha256": content_hash(document),
                "model_card": card.url,
            },
            text=card.handle,
            identifiers=identifiers,
            links=(
                Link(document_url, relation="model_card", locator=card.locator, crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
                Link(card.url, relation="model_card", locator=card.locator, crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_model_cards(
    document: str,
    *,
    source: str,
    path: str,
    heading_pattern: re.Pattern[str],
    url_pattern: re.Pattern[str],
    maximum: int,
) -> tuple[_ModelCard, ...]:
    cards: dict[str, _ModelCard] = {}
    active_level: int | None = None
    fence: tuple[str, int] | None = None
    for line_number, line in enumerate(document.splitlines(), start=1):
        if match := _FENCE.match(line):
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
        if active_level is None:
            continue
        for link in _LINK.finditer(line):
            url = canonicalize_url(link.group("url"))
            match = url_pattern.fullmatch(url)
            if match is None:
                continue
            handle = match.group("handle").strip()
            if not _HANDLE.fullmatch(handle) or handle != handle.strip():
                raise ValueError(f"{source}: invalid model-card handle {handle!r}")
            candidate = _ModelCard(
                handle=handle,
                url=url,
                locator=f"{path}:line:{line_number}",
            )
            existing = cards.get(handle)
            if existing is not None and existing.url != candidate.url:
                raise ValueError(f"{source}: model-card handle {handle!r} maps to multiple URLs")
            cards[handle] = candidate
            if len(cards) > maximum:
                raise ValueError(f"{source}: model-card list exceeds {maximum} entries")
    if not cards:
        raise ValueError(f"{source}: model-card list contains no matching entries")
    return tuple(cards.values())


def _huggingface_identifiers(url: str) -> tuple[Identifier, ...]:
    """Bridge an exact root Hub-card URL to its public repository identity."""

    parsed = urlsplit(url)
    if parsed.hostname is None or parsed.hostname.casefold() != "huggingface.co":
        return ()
    parts = tuple(part for part in parsed.path.split("/") if part)
    if len(parts) != 2:
        return ()
    owner, name = parts
    if any(not _HANDLE.fullmatch(part) for part in (owner, name)):
        return ()
    return (Identifier("huggingface:model", f"{owner}/{name}"),)


def _pattern(value: str, field: str) -> re.Pattern[str]:
    try:
        return re.compile(_required_text(value, field), re.I)
    except re.error as error:
        raise ValueError(f"{field} is not a valid regular expression: {error}") from error


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _namespace(value: str) -> str:
    namespace = _required_text(value, "provider namespace")
    if not _NAMESPACE.fullmatch(namespace):
        raise ValueError("provider namespace is invalid")
    return namespace


def _safe_path(value: str) -> str:
    path = _required_text(value, "document path")
    if (
        path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("document path is not a safe relative path")
    return path


__all__ = ["MarkdownModelCardListSourceAdapter"]
