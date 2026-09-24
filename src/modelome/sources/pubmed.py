from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]

_SHARD_RE = re.compile(
    r"^pubmed(?P<cycle>\d{2})n(?P<sequence>\d{4,})\.xml\.gz$"
)
_SIDECAR_RE = re.compile(
    r"^pubmed(?P<cycle>\d{2})n(?P<sequence>\d{4,})\.xml\.gz\.md5$"
)
_PUBMED_PAYLOAD_LIKE_RE = re.compile(r"^pubmed.*\.xml\.gz(?:\.md5)?$", re.IGNORECASE)
_LISTING_TIMESTAMP_RE = re.compile(
    r"(?P<modified>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)"
    r"\s+(?P<size>\S+)"
)
_BSD_CHECKSUM_RE = re.compile(
    r"^MD5\s*\((?P<filename>[^)]+)\)\s*=\s*(?P<digest>[0-9a-fA-F]{32})$",
    re.IGNORECASE,
)
_COREUTILS_CHECKSUM_RE = re.compile(
    r"^(?P<digest>[0-9a-fA-F]{32})\s+[* ]?(?P<filename>\S+)$"
)

_SCAN_KEYS = frozenset(
    {
        "cursor",
        "raw_items_seen",
        "scan_baseline_fingerprint",
        "scan_cycle",
        "scan_high_sequence",
        "scan_manifest_fingerprint",
        "scan_mode",
        "scan_start_sequence",
        "scan_total",
        "started_at",
    }
)
_COMPLETE_KEYS = frozenset(
    {
        "baseline_cycle",
        "baseline_file_count",
        "baseline_last_sequence",
        "baseline_manifest_fingerprint",
        "completed_at",
        "last_applied_sequence",
        "update_file_count",
        "update_prefix_fingerprint",
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _IndexEntry:
    name: str
    href: str
    modified: str | None
    size: str | None


@dataclass(frozen=True, slots=True)
class _Shard:
    stream: str
    cycle: str
    sequence: int
    filename: str
    payload_url: str
    payload_modified_at: str
    payload_size: str
    checksum_filename: str
    checksum_url: str
    checksum_modified_at: str
    checksum_size: str


class _IndexParser(HTMLParser):
    """Collect link rows without depending on one Apache HTML template."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[_IndexEntry] = []
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []
        self._pending: tuple[str, str] | None = None
        self._tail: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() != "a":
            return
        self._flush_pending()
        values = {key.casefold(): value for key, value in attrs}
        self._anchor_href = values.get("href")
        self._anchor_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "a" or self._anchor_href is None:
            return
        self._pending = (self._anchor_href, "".join(self._anchor_text).strip())
        self._anchor_href = None
        self._anchor_text = []
        self._tail = []

    def handle_data(self, data: str) -> None:
        if self._anchor_href is not None:
            self._anchor_text.append(data)
        elif self._pending is not None:
            self._tail.append(data)

    def close(self) -> None:
        super().close()
        self._flush_pending()

    def _flush_pending(self) -> None:
        if self._pending is None:
            return
        href, label = self._pending
        match = _LISTING_TIMESTAMP_RE.search(" ".join(self._tail))
        self.entries.append(
            _IndexEntry(
                name=label or PurePosixPath(urlsplit(href).path).name,
                href=href,
                modified=match.group("modified") if match else None,
                size=match.group("size") if match else None,
            )
        )
        self._pending = None
        self._tail = []


class PubMedBulkSourceAdapter:
    """Enumerate PubMed baseline and update XML shards in load order.

    This is a control-plane adapter. It downloads the two small HTTPS directory
    indexes and the MD5 sidecar for each emitted shard, but it deliberately does
    not download the large compressed XML payload. A downstream shard processor
    must verify the checksum, stream-parse the XML, replace citations by PMID,
    and apply ``DeleteCitation`` elements in shard sequence order.
    """

    def __init__(
        self,
        *,
        name: str = "pubmed-bulk",
        baseline_url: str = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/",
        update_url: str = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/",
        page_size: int = 25,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.baseline_url = _directory_url(baseline_url, self.name, "baseline URL")
        self.update_url = _directory_url(update_url, self.name, "update URL")
        self.page_size = int(page_size)
        if self.page_size < 1:
            raise ValueError(f"{self.name}: page size must be positive")
        self.client = client or HttpClient()
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "pubmed-bulk-manifest-v1",
                "baseline_url": self.baseline_url,
                "update_url": self.update_url,
                "page_size": self.page_size,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        now = _isoformat(self.clock())
        baselines, baseline_issues = self._read_index(self.baseline_url, "baseline")
        updates, update_issues = self._read_index(self.update_url, "update")
        issues = [*baseline_issues, *update_issues]
        issues.extend(_validate_application_sequence(self.name, baselines, updates))
        manifest_count = len(baselines) + len(updates)

        if issues:
            retry_state = _stable_state(state)
            return SourcePage(
                records=(),
                next_state=retry_state,
                complete=False,
                upstream_count=manifest_count,
                issues=tuple(issues),
                retry_state=retry_state,
            )

        cycle = baselines[0].cycle
        baseline_last = baselines[-1].sequence
        current_high = updates[-1].sequence if updates else baseline_last
        baseline_fingerprint = _manifest_fingerprint(baselines)

        if _has_scan_state(state):
            plan, scan_state, scan_issues = self._resume_plan(
                state,
                baselines=baselines,
                updates=updates,
                cycle=cycle,
                baseline_fingerprint=baseline_fingerprint,
                current_high=current_high,
            )
            if scan_issues:
                retry_state = _stable_state(state)
                return SourcePage(
                    records=(),
                    next_state=retry_state,
                    complete=False,
                    upstream_count=len(plan),
                    issues=tuple(scan_issues),
                    retry_state=retry_state,
                )
        else:
            plan, scan_state = self._start_plan(
                state,
                baselines=baselines,
                updates=updates,
                cycle=cycle,
                baseline_fingerprint=baseline_fingerprint,
                current_high=current_high,
                now=now,
            )

        scan_total = _required_state_count(scan_state, "scan_total", self.name)
        cursor = _required_state_count(scan_state, "cursor", self.name)
        raw_items_seen = _required_state_count(scan_state, "raw_items_seen", self.name)
        if cursor != raw_items_seen:
            raise ValueError(f"{self.name}: raw_items_seen does not match cursor")
        if scan_total != len(plan):
            raise ValueError(
                f"{self.name}: frozen scan total {scan_total} does not match "
                f"the {len(plan)} manifest entries"
            )
        if cursor > scan_total:
            raise ValueError(f"{self.name}: cursor is beyond the frozen scan total")

        if not plan:
            return SourcePage(
                records=(),
                next_state=_completed_state(
                    cycle=cycle,
                    baselines=baselines,
                    updates=updates,
                    high_sequence=current_high,
                    baseline_fingerprint=baseline_fingerprint,
                    completed_at=now,
                ),
                complete=True,
                upstream_count=0,
            )

        retry_state = dict(scan_state)
        page_shards = plan[cursor : cursor + self.page_size]
        records: list[SourceRecord] = []
        page_issues: list[SourceIssue] = []
        for shard in page_shards:
            try:
                checksum = self._read_checksum(shard)
                records.append(_shard_record(shard, checksum))
            except (TypeError, ValueError) as error:
                page_issues.append(
                    _issue(
                        self.name,
                        f"pubmed:{shard.filename}",
                        "source_normalize",
                        f"{type(error).__name__}: {error}",
                        {
                            "filename": shard.filename,
                            "checksum_url": shard.checksum_url,
                            "sequence": shard.sequence,
                            "stream": shard.stream,
                        },
                    )
                )

        next_cursor = cursor + len(page_shards)
        if not page_shards and next_cursor < scan_total:
            page_issues.append(
                _issue(
                    self.name,
                    "pagination",
                    "source_pagination",
                    "frozen manifest returned an empty page before its declared total",
                    {"cursor": cursor, "scan_total": scan_total},
                )
            )

        complete = next_cursor >= scan_total and not page_issues
        if complete:
            frozen_high = _required_state_count(
                scan_state,
                "scan_high_sequence",
                self.name,
            )
            next_state = _completed_state(
                cycle=cycle,
                baselines=baselines,
                updates=updates,
                high_sequence=frozen_high,
                baseline_fingerprint=baseline_fingerprint,
                completed_at=now,
            )
        else:
            next_state = dict(scan_state)
            next_state["cursor"] = next_cursor
            next_state["raw_items_seen"] = next_cursor

        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=scan_total,
            issues=tuple(page_issues),
            retry_state=retry_state,
        )

    def _read_index(self, url: str, stream: str) -> tuple[list[_Shard], list[SourceIssue]]:
        response: HttpResponse = self.client.get(
            url,
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: {stream} index returned HTTP {response.status}")
        parser = _IndexParser()
        parser.feed(response.text())
        parser.close()
        return _normalize_index(self.name, url, stream, parser.entries)

    def _read_checksum(self, shard: _Shard) -> str:
        response: HttpResponse = self.client.get(
            shard.checksum_url,
            headers={"Accept": "text/plain"},
        )
        if response.status != 200:
            raise ValueError(
                f"{shard.checksum_filename} returned HTTP {response.status}"
            )
        return _parse_checksum(response.text(), shard.filename)

    def _start_plan(
        self,
        state: Mapping[str, Any],
        *,
        baselines: Sequence[_Shard],
        updates: Sequence[_Shard],
        cycle: str,
        baseline_fingerprint: str,
        current_high: int,
        now: str,
    ) -> tuple[list[_Shard], dict[str, Any]]:
        _validate_complete_state_shape(state, self.name)
        previous_cycle = _text(state.get("baseline_cycle"))
        previous_baseline = _text(state.get("baseline_manifest_fingerprint"))
        previous_last = _optional_state_count(state, "last_applied_sequence", self.name)
        previous_update_prefix = _text(state.get("update_prefix_fingerprint"))

        can_increment = bool(
            previous_cycle
            and previous_cycle == cycle
            and previous_baseline == baseline_fingerprint
            and previous_last is not None
            and previous_last >= baselines[-1].sequence
            and previous_last <= current_high
            and previous_update_prefix
            == _update_prefix_fingerprint(updates, previous_last)
        )
        if can_increment:
            mode = "incremental"
            start_sequence = previous_last + 1
            plan = [shard for shard in updates if shard.sequence >= start_sequence]
        else:
            mode = "reload"
            start_sequence = baselines[0].sequence
            plan = [*baselines, *updates]

        scan_state = _stable_state(state)
        scan_state.update(
            {
                "cursor": 0,
                "raw_items_seen": 0,
                "scan_baseline_fingerprint": baseline_fingerprint,
                "scan_cycle": cycle,
                "scan_high_sequence": current_high,
                "scan_manifest_fingerprint": _manifest_fingerprint(plan),
                "scan_mode": mode,
                "scan_start_sequence": start_sequence,
                "scan_total": len(plan),
                "started_at": now,
            }
        )
        return plan, scan_state

    def _resume_plan(
        self,
        state: Mapping[str, Any],
        *,
        baselines: Sequence[_Shard],
        updates: Sequence[_Shard],
        cycle: str,
        baseline_fingerprint: str,
        current_high: int,
    ) -> tuple[list[_Shard], dict[str, Any], list[SourceIssue]]:
        scan_state = dict(state)
        required_text = {
            key: _required_state_text(state, key, self.name)
            for key in (
                "scan_baseline_fingerprint",
                "scan_cycle",
                "scan_manifest_fingerprint",
                "scan_mode",
                "started_at",
            )
        }
        mode = required_text["scan_mode"]
        if mode not in {"reload", "incremental"}:
            raise ValueError(f"{self.name}: invalid scan_mode checkpoint")
        high = _required_state_count(state, "scan_high_sequence", self.name)
        start = _required_state_count(state, "scan_start_sequence", self.name)
        _required_state_count(state, "scan_total", self.name)
        _required_state_count(state, "cursor", self.name)
        _required_state_count(state, "raw_items_seen", self.name)

        if mode == "reload":
            plan = [
                shard
                for shard in (*baselines, *updates)
                if shard.sequence <= high
            ]
        else:
            plan = [
                shard
                for shard in updates
                if start <= shard.sequence <= high
            ]

        drift: list[str] = []
        if required_text["scan_cycle"] != cycle:
            drift.append(
                f"production cycle changed from {required_text['scan_cycle']} to {cycle}"
            )
        if required_text["scan_baseline_fingerprint"] != baseline_fingerprint:
            drift.append("annual baseline changed during the frozen scan")
        if high > current_high:
            drift.append(
                f"manifest high sequence regressed from {high} to {current_high}"
            )
        if required_text["scan_manifest_fingerprint"] != _manifest_fingerprint(plan):
            drift.append("a frozen manifest entry changed or disappeared")
        expected_total = _required_state_count(state, "scan_total", self.name)
        if expected_total != len(plan):
            drift.append(
                f"frozen manifest count changed from {expected_total} to {len(plan)}"
            )

        issues = [
            _issue(
                self.name,
                "manifest-drift",
                "source_manifest",
                message,
                {
                    "current_cycle": cycle,
                    "current_high_sequence": current_high,
                    "frozen_cycle": required_text["scan_cycle"],
                    "frozen_high_sequence": high,
                },
            )
            for message in drift
        ]
        return plan, scan_state, issues


def _normalize_index(
    source: str,
    base_url: str,
    stream: str,
    entries: Sequence[_IndexEntry],
) -> tuple[list[_Shard], list[SourceIssue]]:
    payloads: dict[str, _IndexEntry] = {}
    sidecars: dict[str, _IndexEntry] = {}
    issues: list[SourceIssue] = []

    for entry in entries:
        candidate = entry.name.strip()
        payload_match = _SHARD_RE.fullmatch(candidate)
        sidecar_match = _SIDECAR_RE.fullmatch(candidate)
        if payload_match:
            target = payloads
            payload_name = candidate
        elif sidecar_match:
            target = sidecars
            payload_name = candidate.removesuffix(".md5")
        elif _PUBMED_PAYLOAD_LIKE_RE.fullmatch(candidate):
            issues.append(
                _issue(
                    source,
                    f"{stream}:{candidate}",
                    "source_manifest",
                    "malformed PubMed shard filename",
                    {"entry": candidate, "href": entry.href, "stream": stream},
                )
            )
            continue
        else:
            continue

        if not _safe_listing_href(entry.href, candidate):
            issues.append(
                _issue(
                    source,
                    f"{stream}:{candidate}",
                    "source_manifest",
                    "unsafe or mismatched directory-listing href",
                    {"entry": candidate, "href": entry.href, "stream": stream},
                )
            )
            continue
        if entry.modified is None or entry.size is None:
            issues.append(
                _issue(
                    source,
                    f"{stream}:{candidate}",
                    "source_manifest",
                    "shard listing is missing its modified date or size",
                    {"entry": candidate, "href": entry.href, "stream": stream},
                )
            )
            continue
        if payload_name in target:
            issues.append(
                _issue(
                    source,
                    f"{stream}:{candidate}",
                    "source_manifest",
                    "duplicate shard listing entry",
                    {"entry": candidate, "stream": stream},
                )
            )
            continue
        target[payload_name] = entry

    for filename in sorted(payloads.keys() - sidecars.keys()):
        issues.append(
            _issue(
                source,
                f"{stream}:{filename}",
                "source_manifest",
                "PubMed XML shard has no MD5 sidecar",
                {"filename": filename, "stream": stream},
            )
        )
    for filename in sorted(sidecars.keys() - payloads.keys()):
        issues.append(
            _issue(
                source,
                f"{stream}:{filename}.md5",
                "source_manifest",
                "PubMed MD5 sidecar has no XML shard",
                {"filename": filename, "stream": stream},
            )
        )

    shards: list[_Shard] = []
    for filename in payloads.keys() & sidecars.keys():
        payload = payloads[filename]
        sidecar = sidecars[filename]
        match = _SHARD_RE.fullmatch(filename)
        if match is None:  # The dictionary is populated only after a full match.
            continue
        shards.append(
            _Shard(
                stream=stream,
                cycle=match.group("cycle"),
                sequence=int(match.group("sequence")),
                filename=filename,
                payload_url=urljoin(base_url, payload.href),
                payload_modified_at=_listing_time(payload.modified),
                payload_size=payload.size or "",
                checksum_filename=f"{filename}.md5",
                checksum_url=urljoin(base_url, sidecar.href),
                checksum_modified_at=_listing_time(sidecar.modified),
                checksum_size=sidecar.size or "",
            )
        )
    shards.sort(key=lambda shard: (shard.sequence, shard.filename))
    return shards, issues


def _validate_application_sequence(
    source: str,
    baselines: Sequence[_Shard],
    updates: Sequence[_Shard],
) -> list[SourceIssue]:
    issues: list[SourceIssue] = []
    if not baselines:
        return [
            _issue(
                source,
                "baseline",
                "source_manifest",
                "baseline index contains no complete XML/MD5 shard pairs",
                {"baseline_count": 0, "update_count": len(updates)},
            )
        ]

    cycles = {shard.cycle for shard in (*baselines, *updates)}
    if len(cycles) != 1:
        issues.append(
            _issue(
                source,
                "production-cycle",
                "source_manifest",
                "baseline and update shards do not share one production cycle",
                {"cycles": sorted(cycles)},
            )
        )

    all_sequences: dict[int, list[str]] = {}
    for shard in (*baselines, *updates):
        all_sequences.setdefault(shard.sequence, []).append(shard.filename)
    for sequence, filenames in sorted(all_sequences.items()):
        if len(filenames) > 1:
            issues.append(
                _issue(
                    source,
                    f"sequence:{sequence}",
                    "source_manifest",
                    "duplicate PubMed application sequence",
                    {"sequence": sequence, "filenames": sorted(filenames)},
                )
            )

    expected_baseline = list(range(1, baselines[-1].sequence + 1))
    actual_baseline = [shard.sequence for shard in baselines]
    if actual_baseline != expected_baseline:
        issues.append(
            _issue(
                source,
                "baseline-sequence",
                "source_manifest",
                "baseline shard sequence is not contiguous from one",
                {
                    "actual_count": len(actual_baseline),
                    "first": actual_baseline[0] if actual_baseline else None,
                    "last": actual_baseline[-1] if actual_baseline else None,
                    "missing": _missing_sequences(expected_baseline, actual_baseline),
                },
            )
        )

    if updates:
        expected_updates = list(
            range(baselines[-1].sequence + 1, updates[-1].sequence + 1)
        )
        actual_updates = [shard.sequence for shard in updates]
        if actual_updates != expected_updates:
            issues.append(
                _issue(
                    source,
                    "update-sequence",
                    "source_manifest",
                    "update shard sequence does not continue contiguously after baseline",
                    {
                        "actual_count": len(actual_updates),
                        "baseline_last": baselines[-1].sequence,
                        "first": actual_updates[0],
                        "last": actual_updates[-1],
                        "missing": _missing_sequences(expected_updates, actual_updates),
                    },
                )
            )
    return issues


def _shard_record(shard: _Shard, checksum: str) -> SourceRecord:
    production_year = _production_year(shard.cycle, shard.payload_modified_at)
    if shard.stream == "baseline":
        application = {
            "shard_action": "stage_complete_snapshot_shard",
            "generation_commit_action": "replace_complete_pubmed_snapshot",
            "commit_only_after_all_baseline_shards": True,
            "load_before_updates": True,
        }
    else:
        application = {
            "load_after_complete_baseline": True,
            "mutation_order": "ascending_sequence",
            "citation_action": "insert_or_replace_by_pmid",
            "deletion_element": "DeleteCitation",
            "deletion_action": "delete_by_pmid",
        }
    raw = {
        "application_order": shard.sequence,
        "application_semantics": application,
        "checksum": {
            "algorithm": "md5",
            "modified_at": shard.checksum_modified_at,
            "size": shard.checksum_size,
            "url": shard.checksum_url,
            "value": checksum,
        },
        "filename": shard.filename,
        "manifest_kind": shard.stream,
        "modified_at": shard.payload_modified_at,
        "production_cycle": shard.cycle,
        "production_year": production_year,
        "sequence": shard.sequence,
        "size": shard.payload_size,
        "url": shard.payload_url,
    }
    return SourceRecord(
        source_record_id=f"pubmed:{shard.filename}",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=canonicalize_url(shard.payload_url),
        title=f"PubMed {shard.stream} shard {shard.filename}",
        raw=raw,
        modified_at=shard.payload_modified_at,
        identifiers=(Identifier("pubmed:bulk-file", shard.filename),),
        links=(
            Link(
                shard.payload_url,
                relation="bulk_payload",
                locator="$.url",
                crawl=False,
            ),
            Link(
                shard.checksum_url,
                relation="checksum",
                locator="$.checksum.url",
                crawl=False,
            ),
        ),
    )


def _parse_checksum(value: str, expected_filename: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("MD5 sidecar must contain exactly one nonempty line")
    match = _BSD_CHECKSUM_RE.fullmatch(lines[0]) or _COREUTILS_CHECKSUM_RE.fullmatch(
        lines[0]
    )
    if match is None:
        raise ValueError("MD5 sidecar has an unsupported checksum format")
    filename = match.group("filename").removeprefix("./")
    if filename != expected_filename:
        raise ValueError(
            f"MD5 sidecar names {filename!r}, expected {expected_filename!r}"
        )
    return match.group("digest").casefold()


def _completed_state(
    *,
    cycle: str,
    baselines: Sequence[_Shard],
    updates: Sequence[_Shard],
    high_sequence: int,
    baseline_fingerprint: str,
    completed_at: str,
) -> dict[str, Any]:
    included_updates = [shard for shard in updates if shard.sequence <= high_sequence]
    return {
        "baseline_cycle": cycle,
        "baseline_file_count": len(baselines),
        "baseline_last_sequence": baselines[-1].sequence,
        "baseline_manifest_fingerprint": baseline_fingerprint,
        "completed_at": completed_at,
        "last_applied_sequence": high_sequence,
        "update_file_count": len(included_updates),
        "update_prefix_fingerprint": _manifest_fingerprint(included_updates),
    }


def _manifest_fingerprint(shards: Sequence[_Shard]) -> str:
    return content_hash(
        [
            {
                "checksum_modified_at": shard.checksum_modified_at,
                "checksum_size": shard.checksum_size,
                "checksum_url": shard.checksum_url,
                "filename": shard.filename,
                "modified_at": shard.payload_modified_at,
                "sequence": shard.sequence,
                "size": shard.payload_size,
                "stream": shard.stream,
                "url": shard.payload_url,
            }
            for shard in shards
        ]
    )


def _update_prefix_fingerprint(updates: Sequence[_Shard], high_sequence: int) -> str:
    return _manifest_fingerprint(
        [shard for shard in updates if shard.sequence <= high_sequence]
    )


def _has_scan_state(state: Mapping[str, Any]) -> bool:
    present = _SCAN_KEYS & state.keys()
    if present and present != _SCAN_KEYS:
        missing = ", ".join(sorted(_SCAN_KEYS - present))
        raise ValueError(f"incomplete PubMed frozen-scan checkpoint; missing {missing}")
    return bool(present)


def _validate_complete_state_shape(state: Mapping[str, Any], source: str) -> None:
    present = _COMPLETE_KEYS & state.keys()
    if present and present != _COMPLETE_KEYS:
        missing = ", ".join(sorted(_COMPLETE_KEYS - present))
        raise ValueError(f"{source}: incomplete PubMed completed checkpoint; missing {missing}")


def _stable_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key not in _SCAN_KEYS}


def _required_state_count(state: Mapping[str, Any], key: str, source: str) -> int:
    value = _optional_state_count(state, key, source)
    if value is None:
        raise ValueError(f"{source}: checkpoint is missing {key}")
    return value


def _optional_state_count(
    state: Mapping[str, Any], key: str, source: str
) -> int | None:
    if key not in state:
        return None
    value = state.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{source}: invalid {key} checkpoint")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: invalid {key} checkpoint") from None
    if result < 0:
        raise ValueError(f"{source}: invalid {key} checkpoint")
    return result


def _required_state_text(state: Mapping[str, Any], key: str, source: str) -> str:
    value = _text(state.get(key))
    if not value:
        raise ValueError(f"{source}: checkpoint is missing {key}")
    return value


def _issue(
    source: str,
    item: str,
    stage: str,
    message: str,
    summary: Mapping[str, Any],
) -> SourceIssue:
    return SourceIssue(
        source_record_id=(
            f"{source}:{item}:{content_hash({'message': message, **dict(summary)})[:24]}"
        ),
        stage=stage,
        error=f"{source}: {message}",
        summary=dict(summary),
    )


def _safe_listing_href(href: str, expected_name: str) -> bool:
    parts = urlsplit(href)
    return bool(
        not parts.scheme
        and not parts.netloc
        and not parts.query
        and not parts.fragment
        and parts.path == expected_name
        and PurePosixPath(parts.path).name == expected_name
    )


def _listing_time(value: str | None) -> str:
    if value is None:
        raise ValueError("directory entry has no modified timestamp")
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            parsed = datetime.strptime(value, pattern).replace(tzinfo=UTC)
            return _isoformat(parsed)
        except ValueError:
            continue
    raise ValueError(f"unsupported directory timestamp {value!r}")


def _production_year(cycle: str, modified_at: str) -> int:
    modified_year = datetime.fromisoformat(modified_at.replace("Z", "+00:00")).year
    suffix = int(cycle)
    century = modified_year - (modified_year % 100)
    candidates = (century - 100 + suffix, century + suffix, century + 100 + suffix)
    return min(candidates, key=lambda year: abs(year - modified_year))


def _missing_sequences(expected: Sequence[int], actual: Sequence[int]) -> list[int]:
    actual_set = set(actual)
    missing = [value for value in expected if value not in actual_set]
    return missing[:100]


def _directory_url(value: Any, source: str, field: str) -> str:
    text = _required_text(value, field)
    parts = urlsplit(text)
    if (
        parts.scheme.casefold() not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"{source}: {field} must be an HTTP(S) directory URL")
    return text.rstrip("/") + "/"


def _required_text(value: Any, field: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["PubMedBulkSourceAdapter"]
