from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, quote, urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import (
    canonicalize_url,
    content_hash,
    extract_url_mentions,
    identifier_from_url,
    infer_url_relation,
)

Clock = Callable[[], datetime]

_API_V1 = "https://api.openreview.net/notes"
_API_V2 = "https://api2.openreview.net/notes"
_WEB_BASE = "https://openreview.net"
_STAGES = ("v1", "v2")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MAX_TIMESTAMP_MS = 9_999_999_999_999
_MAX_CURSOR_LENGTH = 4_096
_MAX_STATE_CURSOR_HASHES = 16_384
_MAX_URL_LENGTH = 32_768
_DOI_RE = re.compile(r"(?i)(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(10\.\d{4,9}/[^\s<>\"']+)")
_ARXIV_RE = re.compile(
    r"(?i)(?:arxiv:\s*|https?://(?:www\.)?arxiv\.org/(?:abs|pdf|html)/)"
    r"([a-z][a-z0-9.-]*/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?"
)
_ARXIV_BARE_RE = re.compile(r"(?i)([a-z][a-z0-9.-]*/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?")
_ARXIV_PATH_RE = re.compile(
    r"(?i)(?:abs|pdf|html)/([a-z][a-z0-9.-]*/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?"
)
_FIELD_KEY_RE = re.compile(r"[^a-z0-9]+")
_WITHDRAWN_RE = re.compile(r"(?i)\bwithdraw(?:n|al)?\b")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OpenReviewSourceAdapter:
    """Enumerate publication-shaped public notes across OpenReview API v1 and v2.

    The upstream query is intentionally global: it has no invitation, venue,
    domain, subject, keyword, or model-name constraint.  Every returned note is
    counted, validated, and retained.  Notes with publication structure (a
    title plus an abstract, authors, or a PDF) become paper records; replies and
    workflow notes remain evidence records and are not mislabeled as papers.

    Exact coverage boundary: this adapter can see only notes readable by an
    unauthenticated caller of the public ``GET /notes`` endpoints.  OpenReview's
    own documentation says current and most past venues use API v2, while some
    older venues still require API v1, so both endpoints are scanned.  Private
    or confidential notes are necessarily absent. API v1 has no documented
    minimum-modification parameter and is therefore rescanned from the start on
    every cycle; API v2 supports ``mintmdate``, which is sent again on every page
    alongside its id-based ``after`` cursor. The APIs also differ in note content:
    v1 exposes direct values, while v2 wraps each content field in a ``value``
    object. Neither endpoint documents a maximum-modification parameter; the
    adapter freezes an upper timestamp locally and stops the ascending scan when
    it reaches that boundary.
    """

    coverage_limitation = (
        "Only notes readable by an unauthenticated GET /notes request are visible; "
        "private/confidential notes are outside the public corpus. API v1 has no "
        "documented minimum-modification filter and is fully rescanned each cycle. "
        "Both APIs lack a documented maximum-modification filter, so the upper "
        "boundary is enforced client-side on a tmdate-ascending scan."
    )

    def __init__(
        self,
        *,
        name: str = "openreview",
        artifact_source: str | None = None,
        api_v1_url: str = _API_V1,
        api_v2_url: str = _API_V2,
        web_base_url: str = _WEB_BASE,
        page_size: int = 1_000,
        overlap_days: int = 2,
        consistency_lag_seconds: int = 300,
        include_revision_details: bool = True,
        max_response_bytes: int = 32 * 1024 * 1024,
        max_item_bytes: int = 8 * 1024 * 1024,
        max_item_fields: int = 256,
        max_content_fields: int = 1_024,
        max_container_items: int = 20_000,
        max_nesting_depth: int = 16,
        max_string_chars: int = 4 * 1024 * 1024,
        max_text_chars: int = 4 * 1024 * 1024,
        max_authors: int = 10_000,
        max_links: int = 2_048,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.artifact_source = _required_text(artifact_source or self.name, "artifact source")
        self.api_v1_url = _web_url(api_v1_url, self.name)
        self.api_v2_url = _web_url(api_v2_url, self.name)
        self.web_base_url = _web_url(web_base_url, self.name).rstrip("/")
        self.page_size = _positive_int(page_size, "page_size", self.name)
        if self.page_size > 1_000:
            raise ValueError(f"{self.name}: page_size must not exceed OpenReview's 1000 limit")
        self.overlap_days = _nonnegative_int(overlap_days, "overlap_days", self.name)
        self.consistency_lag_seconds = _nonnegative_int(
            consistency_lag_seconds, "consistency_lag_seconds", self.name
        )
        self.include_revision_details = bool(include_revision_details)
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes", self.name)
        self.max_item_bytes = _positive_int(max_item_bytes, "max_item_bytes", self.name)
        if self.max_item_bytes > self.max_response_bytes:
            raise ValueError(f"{self.name}: max_item_bytes must not exceed max_response_bytes")
        self.max_item_fields = _positive_int(max_item_fields, "max_item_fields", self.name)
        self.max_content_fields = _positive_int(max_content_fields, "max_content_fields", self.name)
        self.max_container_items = _positive_int(
            max_container_items, "max_container_items", self.name
        )
        self.max_nesting_depth = _positive_int(max_nesting_depth, "max_nesting_depth", self.name)
        self.max_string_chars = _positive_int(max_string_chars, "max_string_chars", self.name)
        self.max_text_chars = _positive_int(max_text_chars, "max_text_chars", self.name)
        self.max_authors = _positive_int(max_authors, "max_authors", self.name)
        self.max_links = _positive_int(max_links, "max_links", self.name)
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openreview-public-notes-v1",
                "artifact_source": self.artifact_source,
                "api_v1_url": self.api_v1_url,
                "api_v2_url": self.api_v2_url,
                "web_base_url": self.web_base_url,
                "page_size": self.page_size,
                "overlap_days": self.overlap_days,
                "consistency_lag_seconds": self.consistency_lag_seconds,
                "include_revision_details": self.include_revision_details,
                "max_response_bytes": self.max_response_bytes,
                "max_item_bytes": self.max_item_bytes,
                "max_item_fields": self.max_item_fields,
                "max_content_fields": self.max_content_fields,
                "max_container_items": self.max_container_items,
                "max_nesting_depth": self.max_nesting_depth,
                "max_string_chars": self.max_string_chars,
                "max_text_chars": self.max_text_chars,
                "max_authors": self.max_authors,
                "max_links": self.max_links,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        now = _as_utc(self.clock())
        prior_watermark = _optional_timestamp(state.get("watermark"), "watermark", self.name)
        stage, window_start, window_end, started_at = self._scan_boundary(
            state, now=now, prior_watermark=prior_watermark
        )
        after = _state_after(state, self.name)
        offset = _state_count(state, "offset", self.name) or 0
        raw_items_seen = _state_count(state, "raw_items_seen", self.name) or 0
        scan_total = _state_count(state, "scan_total", self.name)
        last_tmdate_ms = _state_timestamp_ms(state, "last_tmdate_ms", self.name)
        cursor_hashes = _state_cursor_hashes(state.get("seen_after_hashes"), self.name)
        if stage == "v1":
            if after:
                # Older checkpoints used the v2 cursor on both stages. API v1
                # has no `after` parameter, so safely replay its frozen window
                # from offset zero when resuming one of those checkpoints.
                after = ""
                offset = 0
                raw_items_seen = 0
                scan_total = None
                last_tmdate_ms = None
                cursor_hashes = ()
            if cursor_hashes:
                raise ValueError(f"{self.name}: v1 checkpoints cannot use an after cursor")
            if offset != raw_items_seen:
                raise ValueError(f"{self.name}: v1 offset does not match raw_items_seen")
            if offset and scan_total is None:
                raise ValueError(f"{self.name}: v1 offset checkpoint is missing scan_total")
            if not offset and (scan_total is not None or last_tmdate_ms is not None):
                raise ValueError(f"{self.name}: first-page checkpoint contains pagination state")
        else:
            if offset:
                raise ValueError(f"{self.name}: v2 checkpoints cannot use offset pagination")
            if after and scan_total is None:
                raise ValueError(f"{self.name}: after cursor checkpoint is missing scan_total")
            if after and content_hash(after) not in cursor_hashes:
                raise ValueError(
                    f"{self.name}: checkpoint after cursor is absent from seen_after_hashes"
                )
            if not after and (
                raw_items_seen
                or scan_total is not None
                or last_tmdate_ms is not None
                or cursor_hashes
            ):
                raise ValueError(f"{self.name}: first-page checkpoint contains pagination state")

        page_retry_state = self._page_state(
            stage=stage,
            after=after,
            window_start=window_start,
            window_end=window_end,
            started_at=started_at,
            prior_watermark=prior_watermark,
            raw_items_seen=raw_items_seen,
            scan_total=scan_total,
            last_tmdate_ms=last_tmdate_ms,
            cursor_hashes=cursor_hashes,
            offset=offset,
        )
        restart_state = self._page_state(
            stage=stage,
            after="",
            window_start=window_start,
            window_end=window_end,
            started_at=started_at,
            prior_watermark=prior_watermark,
            raw_items_seen=0,
            scan_total=None,
            last_tmdate_ms=None,
            cursor_hashes=(),
            offset=0,
        )

        params: dict[str, str | int] = {
            "limit": self.page_size,
            "sort": "tmdate:asc",
            "trash": "true",
        }
        if (stage == "v1" and offset == 0) or (stage == "v2" and not after):
            params["count"] = "true"
        if stage == "v2":
            params["mintmdate"] = _to_millis(window_start)
        elif offset:
            params["offset"] = offset
        if self.include_revision_details:
            params["details"] = "revisions"
        if stage == "v2" and after:
            params["after"] = after

        response: HttpResponse = self.client.get(
            self.api_v1_url if stage == "v1" else self.api_v2_url,
            params=params,
            headers={"Accept": "application/json"},
        )
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: response exceeds {self.max_response_bytes} bytes")
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: response must be a JSON object")
        if len(payload) > 64:
            raise ValueError(f"{self.name}: response has too many top-level fields")
        raw_notes = payload.get("notes")
        if not _is_sequence(raw_notes):
            raise ValueError(f"{self.name}: response.notes must be an array")
        if len(raw_notes) > self.page_size:
            return self._pagination_failure(
                stage=stage,
                retry_state=restart_state,
                error=(
                    f"page contained {len(raw_notes)} notes, above configured "
                    f"page_size {self.page_size}"
                ),
                received=len(raw_notes),
            )
        response_count = _optional_count(payload.get("count"), "response.count", self.name)
        first_page = (stage == "v1" and offset == 0) or (stage == "v2" and not after)
        if first_page and response_count is None:
            raise ValueError(f"{self.name}: first page is missing response.count")
        if response_count is not None and response_count < len(raw_notes):
            return self._pagination_failure(
                stage=stage,
                retry_state=restart_state,
                error=(
                    f"response count {response_count} is smaller than its "
                    f"{len(raw_notes)} returned notes"
                ),
                received=len(raw_notes),
            )

        if scan_total is None:
            scan_total = response_count
        elif response_count is not None:
            scan_total = max(scan_total, response_count)
        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        boundary_reached = False
        observed = last_tmdate_ms
        previous_page_cursor = after
        for index, raw_note in enumerate(raw_notes):
            absolute_index = raw_items_seen + index
            if not isinstance(raw_note, Mapping):
                issues.append(
                    self._normalization_issue(
                        stage,
                        absolute_index,
                        raw_note,
                        "TypeError: note is not a JSON object",
                    )
                )
                continue
            try:
                note = dict(raw_note)
                self._validate_note_bounds(note)
                note_id = _required_bounded_text(
                    note.get("id"), "note.id", self.name, _MAX_CURSOR_LENGTH
                )
                modified_ms = _note_modified_ms(note, self.name)
                if observed is not None and modified_ms < observed:
                    issues.append(
                        self._pagination_issue(
                            stage,
                            restart_state,
                            f"tmdate order moved backward at note {note_id!r}",
                            len(raw_notes),
                        )
                    )
                    continue
                observed = modified_ms
                if stage == "v2" and modified_ms < _to_millis(window_start):
                    issues.append(
                        self._pagination_issue(
                            stage,
                            restart_state,
                            f"v2 returned note {note_id!r} below mintmdate",
                            len(raw_notes),
                        )
                    )
                    continue
                if modified_ms > _to_millis(window_end):
                    boundary_reached = True
                    continue
                records.append(
                    self._record(
                        note,
                        stage=stage,
                        boundary_ms=_to_millis(window_end),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                issues.append(
                    self._normalization_issue(
                        stage,
                        absolute_index,
                        raw_note,
                        f"{type(error).__name__}: {error}",
                    )
                )

        raw_items_seen += len(raw_notes)
        pagination_issues = [issue for issue in issues if issue.stage == "source_pagination"]
        if pagination_issues:
            return SourcePage(
                records=tuple(records),
                next_state=restart_state,
                complete=False,
                upstream_count=scan_total,
                issues=tuple(issues),
                retry_state=restart_state,
            )
        if issues:
            return SourcePage(
                records=tuple(records),
                next_state=page_retry_state,
                complete=False,
                upstream_count=scan_total,
                issues=tuple(issues),
                retry_state=page_retry_state,
            )

        stream_complete = boundary_reached or not raw_notes
        if (
            not stream_complete
            and len(raw_notes) < self.page_size
            and scan_total is not None
            and raw_items_seen >= scan_total
        ):
            stream_complete = True
        if stream_complete:
            if stage == "v1":
                next_state = self._page_state(
                    stage="v2",
                    after="",
                    window_start=window_start,
                    window_end=window_end,
                    started_at=started_at,
                    prior_watermark=prior_watermark,
                    raw_items_seen=0,
                    scan_total=None,
                    last_tmdate_ms=None,
                    cursor_hashes=(),
                    offset=0,
                )
                return SourcePage(
                    records=tuple(records),
                    next_state=next_state,
                    complete=False,
                    upstream_count=scan_total,
                    retry_state=page_retry_state,
                )
            return SourcePage(
                records=tuple(records),
                next_state={
                    "watermark": _isoformat(window_end),
                    "completed_at": _isoformat(now),
                },
                complete=True,
                upstream_count=scan_total,
                retry_state=page_retry_state,
            )

        if stage == "v1":
            next_state = self._page_state(
                stage=stage,
                after="",
                window_start=window_start,
                window_end=window_end,
                started_at=started_at,
                prior_watermark=prior_watermark,
                raw_items_seen=raw_items_seen,
                scan_total=scan_total,
                last_tmdate_ms=observed,
                cursor_hashes=(),
                offset=raw_items_seen,
            )
        else:
            next_after = _required_bounded_text(
                raw_notes[-1].get("id") if isinstance(raw_notes[-1], Mapping) else None,
                "last note id",
                self.name,
                _MAX_CURSOR_LENGTH,
            )
            next_hash = content_hash(next_after)
            if next_after == previous_page_cursor or next_hash in cursor_hashes:
                return self._pagination_failure(
                    stage=stage,
                    retry_state=restart_state,
                    error=f"after cursor cycle detected at {next_after!r}",
                    received=len(raw_notes),
                    records=records,
                    upstream_count=scan_total,
                )
            next_hashes = (*cursor_hashes, next_hash)
            if len(next_hashes) > _MAX_STATE_CURSOR_HASHES:
                return self._pagination_failure(
                    stage=stage,
                    retry_state=restart_state,
                    error=(
                        "cursor history exceeded its bounded state capacity; increase page_size "
                        "or split this source into an explicit historical bootstrap"
                    ),
                    received=len(raw_notes),
                    records=records,
                    upstream_count=scan_total,
                )
            next_state = self._page_state(
                stage=stage,
                after=next_after,
                window_start=window_start,
                window_end=window_end,
                started_at=started_at,
                prior_watermark=prior_watermark,
                raw_items_seen=raw_items_seen,
                scan_total=scan_total,
                last_tmdate_ms=observed,
                cursor_hashes=next_hashes,
                offset=0,
            )
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=False,
            upstream_count=scan_total,
            retry_state=page_retry_state,
        )

    def _scan_boundary(
        self,
        state: Mapping[str, Any],
        *,
        now: datetime,
        prior_watermark: datetime | None,
    ) -> tuple[str, datetime, datetime, str]:
        raw_start = state.get("window_start")
        raw_end = state.get("window_end")
        raw_stage = state.get("stage")
        frozen = raw_start is not None or raw_end is not None or raw_stage is not None
        if frozen:
            if raw_start is None or raw_end is None or raw_stage is None:
                raise ValueError(
                    f"{self.name}: frozen scan requires stage, window_start, and window_end"
                )
            stage = _required_text(raw_stage, "checkpoint stage")
            if stage not in _STAGES:
                raise ValueError(f"{self.name}: unsupported checkpoint stage {stage!r}")
            start = _required_timestamp(raw_start, "window_start", self.name)
            end = _required_timestamp(raw_end, "window_end", self.name)
            if start > end:
                raise ValueError(f"{self.name}: window_start must not follow window_end")
            if end > now:
                raise ValueError(f"{self.name}: window_end must not be in the future")
            started_at = _optional_text(state.get("started_at")) or _isoformat(now)
            return stage, start, end, started_at

        if any(
            key in state
            for key in (
                "after",
                "offset",
                "raw_items_seen",
                "last_tmdate_ms",
                "seen_after_hashes",
            )
        ):
            raise ValueError(f"{self.name}: pagination checkpoint is missing frozen boundaries")
        end = now - timedelta(seconds=self.consistency_lag_seconds)
        if end < _EPOCH:
            raise ValueError(f"{self.name}: consistency lag places boundary before Unix epoch")
        if prior_watermark is None:
            start = _EPOCH
        else:
            if prior_watermark > end:
                raise ValueError(f"{self.name}: watermark is later than the frozen boundary")
            start = max(_EPOCH, prior_watermark - timedelta(days=self.overlap_days))
        return "v1", start, end, _isoformat(now)

    def _page_state(
        self,
        *,
        stage: str,
        after: str,
        window_start: datetime,
        window_end: datetime,
        started_at: str,
        prior_watermark: datetime | None,
        raw_items_seen: int,
        scan_total: int | None,
        last_tmdate_ms: int | None,
        cursor_hashes: Sequence[str],
        offset: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "stage": stage,
            "window_start": _isoformat(window_start),
            "window_end": _isoformat(window_end),
            "started_at": started_at,
        }
        if prior_watermark is not None:
            result["watermark"] = _isoformat(prior_watermark)
        if stage == "v1" and offset:
            result.update(
                {
                    "offset": offset,
                    "raw_items_seen": raw_items_seen,
                    "scan_total": scan_total,
                    "last_tmdate_ms": last_tmdate_ms,
                }
            )
        elif stage == "v2" and after:
            result.update(
                {
                    "after": after,
                    "raw_items_seen": raw_items_seen,
                    "scan_total": scan_total,
                    "last_tmdate_ms": last_tmdate_ms,
                    "seen_after_hashes": list(cursor_hashes),
                }
            )
        return result

    def _record(
        self,
        note: Mapping[str, Any],
        *,
        stage: str,
        boundary_ms: int,
    ) -> SourceRecord:
        note_id = _required_bounded_text(note.get("id"), "note.id", self.name, _MAX_CURSOR_LENGTH)
        content_raw = note.get("content", {})
        if not isinstance(content_raw, Mapping):
            raise TypeError(f"{self.name}: note.content must be a JSON object")
        if len(content_raw) > self.max_content_fields:
            raise ValueError(f"{self.name}: note.content exceeds {self.max_content_fields} fields")
        if stage == "v2" and any(
            not isinstance(value, Mapping) or "value" not in value for value in content_raw.values()
        ):
            raise ValueError(f"{self.name}: API v2 content field is missing its value wrapper")
        content = {str(key): _content_value(value) for key, value in content_raw.items()}
        deleted_at_ms = _optional_millis(note.get("ddate"), "note.ddate", self.name)
        deleted = deleted_at_ms is not None and deleted_at_ms <= boundary_ms
        identity_evidence = self._identity_evidence(note, content)
        lifecycle = self._lifecycle_evidence(note, content, deleted_at_ms, boundary_ms)
        forum_id = _optional_identifier_text(
            note.get("forum"), "note.forum", self.name, _MAX_CURSOR_LENGTH
        )
        forum_url = _openreview_note_url(self.web_base_url, forum_id or note_id)
        canonical_url = _openreview_note_url(
            self.web_base_url,
            forum_id or note_id,
            note_id=note_id if forum_id and forum_id != note_id else "",
        )
        identifiers = [Identifier("openreview", note_id)]
        links = [
            Link(
                canonical_url,
                relation="landing_page",
                locator="$.forum" if forum_id else "$.id",
                crawl=False,
            )
        ]
        if forum_id and forum_id != note_id:
            links.append(
                Link(
                    forum_url,
                    relation="forum",
                    locator="$.forum",
                    crawl=False,
                )
            )

        if deleted:
            return SourceRecord(
                source_record_id=note_id,
                kind=ArtifactKind.PAPER,
                canonical_url=canonical_url,
                title=f"[deleted OpenReview note] {note_id}",
                raw={
                    "api_version": stage,
                    "note": dict(note),
                    "identity_evidence": identity_evidence,
                    "lifecycle": lifecycle,
                },
                modified_at=_millis_iso(_note_modified_ms(note, self.name)),
                identifiers=tuple(identifiers),
                links=tuple(links),
                deleted=True,
            )

        title_field = _find_field(content, "title")
        abstract_field = _find_field(content, "abstract")
        authors_field = _find_field(content, "authors")
        pdf_field = _find_field(content, "pdf")
        title = _scalar_text(title_field[1]) if title_field is not None else ""
        abstract = _text_value(abstract_field[1]) if abstract_field is not None else ""
        authors = self._authors(authors_field[1]) if authors_field is not None else []
        publication_shaped = bool(title and (abstract or authors or pdf_field is not None))
        title = title or f"[OpenReview note] {note_id}"

        text = self._record_text(title, content)
        content_links = self._content_links(content)
        links.extend(content_links)
        for link in content_links:
            if identifier := identifier_from_url(link.url):
                identifiers.append(identifier)
        identifiers.extend(self._external_identifiers(note, content))

        published_ms = _first_millis(note, ("pdate", "odate", "cdate", "tcdate"), self.name)
        modified_ms = _note_modified_ms(note, self.name)
        raw = {
            "api_version": stage,
            "note": dict(note),
            "normalized_content": content,
            "identity_evidence": identity_evidence,
            "lifecycle": lifecycle,
            "authors": authors,
        }
        return SourceRecord(
            source_record_id=note_id,
            kind=ArtifactKind.PAPER if publication_shaped else ArtifactKind.OTHER,
            canonical_url=canonical_url,
            title=title,
            raw=raw,
            text=text,
            published_at=_millis_iso(published_ms) if published_ms is not None else None,
            modified_at=_millis_iso(modified_ms),
            identifiers=tuple(dict.fromkeys(identifiers)),
            links=_unique_links(links),
        )

    def _validate_note_bounds(self, note: Mapping[str, Any]) -> None:
        if len(note) > self.max_item_fields:
            raise ValueError(f"{self.name}: note exceeds {self.max_item_fields} fields")
        try:
            encoded = json.dumps(
                note,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        except (TypeError, ValueError) as error:
            raise TypeError(f"{self.name}: note is not JSON serializable") from error
        if len(encoded) > self.max_item_bytes:
            raise ValueError(f"{self.name}: note exceeds {self.max_item_bytes} bytes")
        _validate_json_bounds(
            note,
            source=self.name,
            max_items=self.max_container_items,
            max_depth=self.max_nesting_depth,
            max_string_chars=self.max_string_chars,
        )

    def _record_text(self, title: str, content: Mapping[str, Any]) -> str:
        chunks = [title]
        for key, value in content.items():
            for item in _flatten_text(value):
                item = item.strip()
                if item and item != title:
                    chunks.append(f"{key}: {item}")
        text = "\n\n".join(chunks)
        if len(text) > self.max_text_chars:
            raise ValueError(
                f"{self.name}: extracted text exceeds {self.max_text_chars} characters"
            )
        return text

    def _authors(self, value: Any) -> list[dict[str, str | None]]:
        values = _as_values(value)
        if len(values) > self.max_authors:
            raise ValueError(f"{self.name}: authors exceed {self.max_authors} entries")
        authors: list[dict[str, str | None]] = []
        for item in values:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    authors.append({"name": name, "id": None})
            elif isinstance(item, Mapping):
                name = _optional_text(item.get("name") or item.get("fullname"))
                author_id = _optional_text(item.get("id") or item.get("authorid"))
                if name or author_id:
                    authors.append({"name": name or author_id or "", "id": author_id or None})
            else:
                raise TypeError(f"{self.name}: author entry has unsupported type")
        return authors

    def _content_links(self, content: Mapping[str, Any]) -> list[Link]:
        result: list[Link] = []
        for raw_key, value in content.items():
            key = _field_key(raw_key)
            for value_index, scalar in enumerate(_flatten_text(value)):
                locator = f"$.content.{raw_key}"
                if value_index:
                    locator += f"[{value_index}]"
                for url, span in extract_url_mentions(scalar):
                    attachment_name = _openreview_attachment_name(
                        url, web_base_url=self.web_base_url
                    )
                    relation_key = _field_key(attachment_name) if attachment_name else key
                    self._append_link(
                        result,
                        url,
                        relation=_link_relation(relation_key, scalar, span),
                        locator=f"{locator}:{span}",
                    )
                stripped = scalar.strip()
                attachment_name = _openreview_attachment_name(
                    stripped, web_base_url=self.web_base_url
                )
                if stripped.startswith("/") and (
                    _attachment_field(key) or attachment_name is not None
                ):
                    attachment_key = _field_key(attachment_name) if attachment_name else key
                    self._append_link(
                        result,
                        urljoin(f"{self.web_base_url}/", stripped),
                        relation=_link_relation(attachment_key, stripped, "text:0-1"),
                        locator=locator,
                    )
                doi = _plain_doi(stripped)
                if doi:
                    self._append_link(
                        result,
                        f"https://doi.org/{quote(doi, safe='/():._-')}",
                        relation="published_as",
                        locator=locator,
                    )
                arxiv_id = _plain_arxiv(stripped)
                if arxiv_id:
                    self._append_link(
                        result,
                        f"https://arxiv.org/abs/{quote(arxiv_id, safe='/._-')}",
                        relation="preprint",
                        locator=locator,
                    )
        if len(result) > self.max_links:
            raise ValueError(f"{self.name}: note exceeds {self.max_links} external links")
        return result

    def _append_link(
        self,
        target: list[Link],
        url: str,
        *,
        relation: str,
        locator: str,
    ) -> None:
        if len(url) > _MAX_URL_LENGTH:
            raise ValueError(f"{self.name}: external URL exceeds {_MAX_URL_LENGTH} characters")
        canonical = canonicalize_url(url)
        parts = urlsplit(canonical)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError(f"{self.name}: invalid external HTTP(S) URL")
        crawl = relation in {
            "implementation",
            "official_implementation",
            "project_page",
            "model_card",
        }
        target.append(Link(canonical, relation=relation, locator=locator, crawl=crawl))

    def _external_identifiers(
        self, note: Mapping[str, Any], content: Mapping[str, Any]
    ) -> list[Identifier]:
        result: list[Identifier] = []
        candidates: list[str] = []
        for key in ("externalId", "externalIds"):
            candidates.extend(_text_scalars(note.get(key)))
        for raw_key, value in content.items():
            key = _field_key(raw_key)
            arxiv_field = (
                key in {"arxiv", "arxiv_id"}
                or key.endswith("_arxiv_id")
                or key.endswith("_arxiv_url")
            )
            if arxiv_field or key in {"doi", "external_id", "external_ids"}:
                for candidate in _text_scalars(value):
                    if arxiv_field:
                        arxiv_id = _plain_arxiv(candidate, allow_bare=True)
                        if arxiv_id:
                            result.append(Identifier("arxiv", arxiv_id))
                            continue
                    candidates.append(candidate)
            # First-party venues use many local field names for publication URLs.
            # Keep arXiv identifiers from those URLs even when the field is not
            # named externalId/arxiv; in particular, arXiv's HTML reader route is
            # a valid paper URL but is not understood by the generic URL helper.
            for scalar in _flatten_text(value):
                candidates.extend(url for url, _ in extract_url_mentions(scalar))
        for candidate in candidates:
            if identifier := identifier_from_url(candidate):
                result.append(identifier)
                continue
            if doi := _plain_doi(candidate):
                result.append(Identifier("doi", doi.casefold()))
                continue
            if arxiv_id := _plain_arxiv(candidate):
                result.append(Identifier("arxiv", arxiv_id))
        return list(dict.fromkeys(result))

    def _identity_evidence(
        self, note: Mapping[str, Any], content: Mapping[str, Any]
    ) -> dict[str, Any]:
        invitations = _invitation_ids(note, self.name)
        content_ids: list[dict[str, str]] = []
        for raw_key, value in content.items():
            key = _field_key(raw_key)
            if key in {"id", "venueid"} or key.endswith("_id") or key.endswith("_ids"):
                for item in _text_scalars(value):
                    if len(item) <= _MAX_CURSOR_LENGTH:
                        content_ids.append({"field": raw_key, "value": item})
        revisions: list[dict[str, Any]] = []
        details = note.get("details")
        if isinstance(details, Mapping):
            revision_values = details.get("revisions")
            if _is_sequence(revision_values):
                for revision in revision_values:
                    if isinstance(revision, Mapping):
                        revisions.append(
                            {
                                key: revision.get(key)
                                for key in ("id", "version", "tcdate", "tmdate", "cdate", "mdate")
                                if revision.get(key) is not None
                            }
                        )
        return {
            "note_id": _optional_text(note.get("id")) or None,
            "forum_id": _optional_text(note.get("forum")) or None,
            "original_id": _optional_text(note.get("original")) or None,
            "referent_id": _optional_text(note.get("referent")) or None,
            "invitation_ids": invitations,
            "external_ids": list(
                dict.fromkeys(
                    _text_scalars(note.get("externalId")) + _text_scalars(note.get("externalIds"))
                )
            ),
            "content_ids": content_ids,
            "version": note.get("version"),
            "revisions": revisions,
        }

    def _lifecycle_evidence(
        self,
        note: Mapping[str, Any],
        content: Mapping[str, Any],
        deleted_at_ms: int | None,
        boundary_ms: int,
    ) -> dict[str, Any]:
        withdrawal: list[dict[str, str]] = []
        for invitation in _invitation_ids(note, self.name):
            if _WITHDRAWN_RE.search(invitation):
                withdrawal.append({"locator": "$.invitations", "value": invitation})
        for raw_key, value in content.items():
            key = _field_key(raw_key)
            if "venue" not in key and "status" not in key and "decision" not in key:
                continue
            for item in _text_scalars(value):
                if _WITHDRAWN_RE.search(item):
                    withdrawal.append({"locator": f"$.content.{raw_key}", "value": item})
        return {
            "ddate_ms": deleted_at_ms,
            "deleted_at": _millis_iso(deleted_at_ms) if deleted_at_ms is not None else None,
            "deleted_at_boundary": (deleted_at_ms is not None and deleted_at_ms <= boundary_ms),
            "withdrawal_evidence": withdrawal,
        }

    def _normalization_issue(
        self,
        stage: str,
        index: int,
        raw: Any,
        error: str,
    ) -> SourceIssue:
        note_id = _optional_text(raw.get("id")) if isinstance(raw, Mapping) else ""
        return SourceIssue(
            source_record_id=(
                note_id or f"{self.name}:{stage}:malformed:{content_hash(repr(raw))[:32]}"
            ),
            stage="source_normalize",
            error=error,
            summary={"api_version": stage, "index": index, "raw": raw},
        )

    def _pagination_issue(
        self,
        stage: str,
        retry_state: Mapping[str, Any],
        error: str,
        received: int,
    ) -> SourceIssue:
        return SourceIssue(
            source_record_id=(
                f"{self.name}:{stage}:pagination:"
                f"{content_hash({**retry_state, 'error': error})[:32]}"
            ),
            stage="source_pagination",
            error=f"{self.name}: {error}",
            summary={
                "api_version": stage,
                "received_count": received,
                "retry_state": dict(retry_state),
            },
        )

    def _pagination_failure(
        self,
        *,
        stage: str,
        retry_state: Mapping[str, Any],
        error: str,
        received: int,
        records: Sequence[SourceRecord] = (),
        upstream_count: int | None = None,
    ) -> SourcePage:
        issue = self._pagination_issue(stage, retry_state, error, received)
        return SourcePage(
            records=tuple(records),
            next_state=dict(retry_state),
            complete=False,
            upstream_count=upstream_count,
            issues=(issue,),
            retry_state=dict(retry_state),
        )


def _validate_json_bounds(
    value: Any,
    *,
    source: str,
    max_items: int,
    max_depth: int,
    max_string_chars: int,
) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            raise ValueError(f"{source}: note JSON exceeds nesting depth {max_depth}")
        visited += 1
        if visited > max_items:
            raise ValueError(f"{source}: note JSON exceeds {max_items} values")
        if isinstance(current, str):
            if len(current) > max_string_chars:
                raise ValueError(f"{source}: note string exceeds {max_string_chars} characters")
        elif isinstance(current, Mapping):
            if len(current) > max_items:
                raise ValueError(f"{source}: note JSON object exceeds {max_items} entries")
            if any(not isinstance(key, str) for key in current):
                raise TypeError(f"{source}: note JSON object keys must be strings")
            stack.extend((key, depth + 1) for key in current)
            stack.extend((item, depth + 1) for item in current.values())
        elif _is_sequence(current):
            if len(current) > max_items:
                raise ValueError(f"{source}: note JSON array exceeds {max_items} entries")
            stack.extend((item, depth + 1) for item in current)
        elif current is not None and not isinstance(current, (bool, int, float)):
            raise TypeError(f"{source}: note contains a non-JSON value")


def _content_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _find_field(content: Mapping[str, Any], suffix: str) -> tuple[str, Any] | None:
    exact: tuple[str, Any] | None = None
    fallback: tuple[str, Any] | None = None
    for key, value in content.items():
        normalized = _field_key(key)
        if normalized == suffix:
            exact = (key, value)
            break
        if normalized.endswith(f"_{suffix}") and fallback is None:
            fallback = (key, value)
    return exact or fallback


def _field_key(value: Any) -> str:
    return _FIELD_KEY_RE.sub("_", str(value).casefold()).strip("_")


def _flatten_text(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, (str, int, float)) and not isinstance(item, bool):
                yield f"{key}: {item}"
            else:
                yield from _flatten_text(item)
    elif _is_sequence(value):
        for item in value:
            if isinstance(item, (str, int, float)) and not isinstance(item, bool):
                yield str(item)
            else:
                yield from _flatten_text(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)


def _text_value(value: Any) -> str:
    return "\n".join(item.strip() for item in _flatten_text(value) if item.strip())


def _scalar_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _as_values(value: Any) -> Sequence[Any]:
    if _is_sequence(value):
        return value
    if value is None:
        return ()
    return (value,)


def _text_scalars(value: Any) -> list[str]:
    result: list[str] = []
    for item in _as_values(value):
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _invitation_ids(note: Mapping[str, Any], source: str) -> list[str]:
    singular = note.get("invitation")
    if singular is not None and not isinstance(singular, str):
        raise TypeError(f"{source}: note.invitation must be a string")
    plural = note.get("invitations")
    if plural is not None and not _is_sequence(plural):
        raise TypeError(f"{source}: note.invitations must be an array")
    result = _text_scalars(singular)
    plural_values = _text_scalars(plural)
    if plural is not None and len(plural_values) != len(plural):
        raise TypeError(f"{source}: note.invitations entries must be nonempty strings")
    result.extend(plural_values)
    details = note.get("details")
    if isinstance(details, Mapping):
        invitation = details.get("invitation")
        if isinstance(invitation, Mapping):
            result.extend(_text_scalars(invitation.get("id")))
    return list(dict.fromkeys(result))


def _link_relation(key: str, text: str, locator: str) -> str:
    if "pdf" in key:
        return "full_text"
    if any(
        token in key
        for token in (
            "implementation",
            "repository",
            "source_code",
            "code",
            "github",
            "gitlab",
            "software",
        )
    ):
        return "implementation"
    if "weight" in key or "checkpoint" in key:
        return "weights"
    if "model_card" in key:
        return "model_card"
    if "project" in key or "demo" in key or "website" in key:
        return "project_page"
    if "supplement" in key:
        return "supplementary_material"
    if "dataset" in key or key == "data":
        return "dataset"
    if "license" in key:
        return "license"
    if locator != "text:0-1":
        match = re.fullmatch(r"text:(\d+)-(\d+)", locator)
        if match is not None:
            start, end = (int(value) for value in match.groups())
            identifier = identifier_from_url(text[start:end])
            if identifier is not None and identifier.namespace == "github:repository":
                return "implementation"
            if identifier is not None and identifier.namespace == "huggingface:model":
                return "model_card"
        return infer_url_relation(text, locator)
    return "attachment"


def _attachment_field(key: str) -> bool:
    return any(
        token in key
        for token in (
            "pdf",
            "supplement",
            "attachment",
            "code",
            "data",
            "material",
            "checkpoint",
            "weight",
        )
    )


def _openreview_attachment_name(value: str, *, web_base_url: str) -> str | None:
    """Return the field name from an OpenReview attachment route."""
    parts = urlsplit(value)
    if parts.path.rstrip("/") not in {
        "/attachment",
        "/notes/edits/attachment",
        "/groups/attachment",
        "/invitations/attachment",
    }:
        return None
    if parts.scheme or parts.netloc:
        base = urlsplit(web_base_url)
        known_hosts = {base.hostname}
        if base.hostname == "openreview.net":
            known_hosts.update({"api.openreview.net", "api2.openreview.net"})
        if parts.scheme not in {"http", "https"} or parts.hostname not in known_hosts:
            return None
    elif not value.startswith("/"):
        return None
    names = parse_qs(parts.query, keep_blank_values=False).get("name", [])
    if len(names) != 1 or not names[0].strip():
        return None
    return names[0].strip()


def _plain_doi(value: str) -> str:
    match = _DOI_RE.fullmatch(value.strip().rstrip('.,;:!?)"]}'))
    return match.group(1).rstrip('.,;:!?)"]}') if match else ""


def _plain_arxiv(value: str, *, allow_bare: bool = False) -> str:
    candidate = value.strip()
    match = _ARXIV_RE.fullmatch(candidate)
    if match is not None:
        return match.group(1)
    parts = urlsplit(candidate)
    if parts.scheme.casefold() in {"http", "https"} and (parts.hostname or "").casefold() in {
        "arxiv.org",
        "www.arxiv.org",
    }:
        match = _ARXIV_PATH_RE.fullmatch(parts.path.strip("/"))
        if match is not None:
            return match.group(1)
    if allow_bare:
        match = _ARXIV_BARE_RE.fullmatch(candidate)
    return match.group(1) if match else ""


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    seen: set[tuple[str, str]] = set()
    result: list[Link] = []
    for value in values:
        key = (value.url, value.relation)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _openreview_note_url(base: str, forum_id: str, *, note_id: str = "") -> str:
    url = f"{base}/forum?id={quote(forum_id, safe='')}"
    if note_id:
        url += f"&noteId={quote(note_id, safe='')}"
    return canonicalize_url(url)


def _note_modified_ms(note: Mapping[str, Any], source: str) -> int:
    result = _first_millis(note, ("tmdate", "mdate", "tcdate", "cdate"), source)
    if result is None:
        raise ValueError(f"{source}: note is missing a valid modification timestamp")
    return result


def _first_millis(value: Mapping[str, Any], fields: Sequence[str], source: str) -> int | None:
    for field in fields:
        if value.get(field) is not None:
            return _required_millis(value.get(field), f"note.{field}", source)
    return None


def _required_millis(value: Any, field: str, source: str) -> int:
    result = _optional_millis(value, field, source)
    if result is None:
        raise ValueError(f"{source}: {field} is required")
    return result


def _optional_millis(value: Any, field: str, source: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source}: invalid {field} {value!r}")
    result = value
    if result < 0 or result > _MAX_TIMESTAMP_MS:
        raise ValueError(f"{source}: invalid {field} {value!r}")
    return result


def _millis_iso(value: int) -> str:
    return _isoformat(datetime.fromtimestamp(value / 1_000, tz=UTC))


def _to_millis(value: datetime) -> int:
    return int(_as_utc(value).timestamp() * 1_000)


def _state_after(state: Mapping[str, Any], source: str) -> str:
    if "after" not in state:
        return ""
    return _required_bounded_text(
        state.get("after"), "checkpoint after", source, _MAX_CURSOR_LENGTH
    )


def _state_count(state: Mapping[str, Any], key: str, source: str) -> int | None:
    if key not in state:
        return None
    return _nonnegative_int(state.get(key), f"checkpoint {key}", source)


def _state_timestamp_ms(state: Mapping[str, Any], key: str, source: str) -> int | None:
    if key not in state:
        return None
    return _required_millis(state.get(key), f"checkpoint {key}", source)


def _state_cursor_hashes(value: Any, source: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not _is_sequence(value) or len(value) > _MAX_STATE_CURSOR_HASHES:
        raise ValueError(f"{source}: invalid checkpoint seen_after_hashes")
    result: list[str] = []
    for item in value:
        text = _required_bounded_text(item, "cursor hash", source, 64)
        if not re.fullmatch(r"[0-9a-f]{64}", text):
            raise ValueError(f"{source}: invalid checkpoint cursor hash")
        result.append(text)
    if len(result) != len(set(result)):
        raise ValueError(f"{source}: duplicate checkpoint cursor hash")
    return tuple(result)


def _optional_count(value: Any, field: str, source: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}: invalid {field} {value!r}")
    return value


def _positive_int(value: Any, field: str, source: str) -> int:
    result = _nonnegative_int(value, field, source)
    if result < 1:
        raise ValueError(f"{source}: {field} must be positive")
    return result


def _nonnegative_int(value: Any, field: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: invalid {field} {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: invalid {field} {value!r}") from None
    if result < 0:
        raise ValueError(f"{source}: invalid {field} {value!r}")
    return result


def _optional_timestamp(value: Any, field: str, source: str) -> datetime | None:
    if value is None or value == "":
        return None
    return _required_timestamp(value, field, source)


def _required_timestamp(value: Any, field: str, source: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{source}: invalid {field} {value!r}")
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{source}: invalid {field} {value!r}") from None
    return _as_utc(result)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _web_url(value: Any, source: str) -> str:
    text = _required_text(value, "URL")
    parts = urlsplit(text)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"{source}: URL must be an HTTPS endpoint without credentials/query")
    return text.rstrip("/")


def _required_text(value: Any, field: str) -> str:
    result = _optional_text(value)
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _required_bounded_text(value: Any, field: str, source: str, maximum: int) -> str:
    result = _optional_text(value)
    if not result or len(result) > maximum:
        raise ValueError(f"{source}: invalid {field}")
    return result


def _optional_identifier_text(value: Any, field: str, source: str, maximum: int) -> str:
    if value is None or value == "":
        return ""
    return _required_bounded_text(value, field, source, maximum)


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


__all__ = ["OpenReviewSourceAdapter"]
