"""Official WSA model-zoo checkpoints from the project README."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelRelationHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_REPOSITORY = "zaleni/WSA"
_DOCUMENT = "README.md"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^(?P<level>#{1,6})\s+(?P<title>.+?)\s*$")
_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_MODEL_URL = re.compile(
    r"^https://huggingface\.co/zaleni/(?P<slug>WSA-(?:Base|Large)(?:-(?:RoboTwin|LIBERO))?)$"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class WSAModelZooSourceAdapter:
    """Enumerate exact pretrained and benchmark-tuned WSA checkpoint refs."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only direct Hugging Face model links in the official README Model Zoo. "
        "It does not enumerate user fine-tunes or third-party mirrors."
    )

    def __init__(
        self,
        *,
        name: str = "wsa-model-zoo",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 8,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        for value, label in (
            (max_response_bytes, "max_response_bytes"),
            (max_entries, "max_entries"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "wsa-model-zoo-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "direct HF links in README Model Zoo table",
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{_REPOSITORY}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        commit_response: HttpResponse = self.client.get(
            f"https://api.github.com/repos/{_REPOSITORY}/commits/main",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        payload = commit_response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a full commit SHA")
        document_url = (
            f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/"
            f"{quote(_DOCUMENT, safe='/')}"
        )
        response: HttpResponse = self.client.get(
            document_url, headers={"Accept": "text/markdown,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds {self.max_response_bytes} bytes")
        entries = _parse_model_zoo(response.text(), source=self.name, maximum=self.max_entries)
        if not entries:
            raise ValueError(f"{self.name}: Model Zoo contains no WSA checkpoint refs")
        records = tuple(
            self._record(entry, revision, response.body, document_url) for entry in entries
        )
        checked_at = self.clock()
        if checked_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "document_url": document_url,
                "document_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        entry: tuple[str, str, str, int],
        revision: str,
        document: bytes,
        document_url: str,
    ) -> SourceRecord:
        slug, checkpoint_type, usage, line = entry
        repo = f"zaleni/{slug}"
        artifact_url = f"https://huggingface.co/{repo}"
        local_id = f"model:{slug.casefold()}"
        locator = f"{_DOCUMENT}:line:{line}"
        identifier = Identifier("huggingface:model", repo)
        model = ModelHint(
            local_id=local_id,
            name=f"WSA {slug.removeprefix('WSA-')}",
            identifiers=(identifier,),
            aliases=(slug,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        base_slug = "WSA-Base" if slug.startswith("WSA-Base") else "WSA-Large"
        base_repo = f"zaleni/{base_slug}"
        is_finetuned = slug != base_slug
        metadata = {
            "repository": _REPOSITORY,
            "revision": revision,
            "document_path": _DOCUMENT,
            "checkpoint_repo": repo,
            "checkpoint_type": checkpoint_type,
            "intended_usage": usage,
            "base_checkpoint_repo": base_repo if is_finetuned else None,
            "source_document_sha256": content_hash(document),
        }
        release = ReleaseHint(
            local_id=f"release:{slug.casefold()}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("wsa:checkpoint", repo),),
            metadata=metadata,
            locator=locator,
        )
        relations: tuple[ModelRelationHint, ...] = ()
        if is_finetuned:
            relations = (
                ModelRelationHint(
                    subject_local_id=local_id,
                    predicate="fine_tuned_from",
                    target=ModelHint(
                        local_id=f"model:{base_slug.casefold()}",
                        name=f"WSA {base_slug.removeprefix('WSA-')}",
                        identifiers=(Identifier("huggingface:model", base_repo),),
                        status=ModelStatus.RELEASED,
                    ),
                    locator=locator,
                ),
            )
        return SourceRecord(
            source_record_id=f"wsa:{slug.casefold()}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=f"{slug} checkpoint",
            raw=metadata,
            text=f"{repo}; {checkpoint_type}; {usage}",
            identifiers=(identifier,),
            links=(
                Link(
                    artifact_url,
                    relation="model_artifact",
                    locator=locator,
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
                Link(document_url, relation="model_catalog", locator=locator, crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=(model,),
            model_relations=relations,
            releases=(release,),
        )


def _parse_model_zoo(
    document: str,
    *,
    source: str,
    maximum: int,
) -> tuple[tuple[str, str, str, int], ...]:
    active = False
    section_level: int | None = None
    entries: dict[str, tuple[str, str, str, int]] = {}
    section_lines: list[tuple[int, str]] = []
    for line_number, line in enumerate(document.splitlines(), start=1):
        if heading := _HEADING.match(line):
            level = len(heading.group("level"))
            title = heading.group("title").strip().casefold()
            if active and section_level is not None and level <= section_level:
                active = False
                entries.update(
                    _parse_html_table("\n".join(value for _, value in section_lines), source)
                )
                section_lines = []
            if title == "model zoo":
                active = True
                section_level = level
            continue
        if not active:
            continue
        section_lines.append((line_number, line))
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        links = [
            item for item in _LINK.finditer(cells[0]) if _MODEL_URL.fullmatch(item.group("url"))
        ]
        if not links:
            continue
        if len(links) != 1:
            raise ValueError(f"{source}: expected one WSA model link on line {line_number}")
        match = _MODEL_URL.fullmatch(links[0].group("url"))
        assert match is not None
        slug = match.group("slug")
        if slug in entries:
            raise ValueError(f"{source}: duplicate checkpoint ref {slug}")
        entry = (slug, " ".join(cells[1].split()), " ".join(cells[2].split()), line_number)
        entries[slug] = entry
        if len(entries) > maximum:
            raise ValueError(f"{source}: model zoo exceeds {maximum} entries")
    if active:
        entries.update(_parse_html_table("\n".join(value for _, value in section_lines), source))
    if len(entries) > maximum:
        raise ValueError(f"{source}: model zoo exceeds {maximum} entries")
    return tuple(entries.values())


class _ModelZooHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[tuple[str, str, str, int]] = []
        self._row: list[list[str]] | None = None
        self._cell: list[str] | None = None
        self._href: str | None = None
        self._anchors: list[str] = []
        self._line = 1
        self._cell_line = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {key.casefold(): value or "" for key, value in attrs}
        if tag.casefold() == "tr":
            self._row = []
            self._line = self.getpos()[0]
        elif tag.casefold() in {"td", "th"} and self._row is not None:
            self._cell = []
            self._cell_line = self.getpos()[0]
        elif tag.casefold() == "a" and self._cell is not None:
            href = attrs_map.get("href", "")
            if _MODEL_URL.fullmatch(href):
                self._anchors.append(href)

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"td", "th"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag.casefold() == "tr" and self._row is not None:
            if len(self._row) >= 3 and self._anchors:
                if len(self._anchors) != 1:
                    raise ValueError("WSA Model Zoo row has multiple allowed checkpoint links")
                match = _MODEL_URL.fullmatch(self._anchors[-1])
                assert match is not None
                slug = match.group("slug")
                self.entries.append(
                    (slug, self._row[1], self._row[2], self._line or self._cell_line)
                )
            self._row = None
            self._anchors = []


def _parse_html_table(section: str, source: str) -> dict[str, tuple[str, str, str, int]]:
    parser = _ModelZooHtmlParser()
    parser.feed(section)
    parser.close()
    entries: dict[str, tuple[str, str, str, int]] = {}
    for entry in parser.entries:
        slug = entry[0]
        if slug in entries:
            raise ValueError(f"{source}: duplicate checkpoint ref {slug}")
        entries[slug] = entry
    return entries
