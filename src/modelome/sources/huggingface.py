from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

from modelome.http import HttpClient, HttpFailure, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelRelationHint,
    ModelStatus,
    ReleaseHint,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, extract_urls, identifier_from_url

Clock = Callable[[], datetime]
_LINK_RE = re.compile(r'<([^>]+)>\s*((?:;\s*[^,]+)*)')
_REL_RE = re.compile(r'\brel\s*=\s*(?:"([^"]+)"|([^;\s,]+))', re.IGNORECASE)
_WEIGHT_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".flax",
    ".gguf",
    ".ggml",
    ".h5",
    ".keras",
    ".mlmodel",
    ".mlpackage",
    ".msgpack",
    ".onnx",
    ".onnx_data",
    ".pb",
    ".pt",
    ".ptd",
    ".pt2",
    ".pth",
    ".ptl",
    ".pte",
    ".safetensors",
    ".tflite",
    ".torchscript",
)


def _is_weight_file(filename: str, siblings: Sequence[str] = ()) -> bool:
    folded = filename.casefold()
    # Transformers commonly publish sharded checkpoints with a JSON index.
    # Keep recognition specific to known weight-index names so arbitrary JSON
    # metadata does not become a binary artifact reference.
    if folded.endswith(_WEIGHT_SUFFIXES) or folded.endswith(
        (".safetensors.index.json", ".bin.index.json")
    ):
        return True

    # TensorFlow checkpoints consist of a shared-prefix `.index` file and one
    # or more `.data-00000-of-00001` shards. Recognize only paired components
    # so unrelated repository index/data files remain ordinary files.
    folded_siblings = tuple(value.casefold() for value in siblings)
    if folded.endswith(".index"):
        prefix = folded[: -len(".index")]
        return any(
            re.fullmatch(re.escape(prefix) + r"\.data-\d+-of-\d+", value)
            for value in folded_siblings
        )
    match = re.fullmatch(r"(.+)\.data-\d+-of-\d+", folded)
    if match:
        return f"{match.group(1)}.index" in folded_siblings
    return False


def _potential_weight_file(filename: str) -> bool:
    folded = filename.casefold()
    return (
        _is_weight_file(filename)
        or folded.endswith(".index")
        or re.fullmatch(r".+\.data-\d+-of-\d+", folded) is not None
    )


def _safe_repo_filename(value: str) -> bool:
    if not value or value.startswith("/") or "\\" in value:
        return False
    try:
        if len(value.encode("utf-8")) > 1024:
            return False
    except UnicodeEncodeError:
        return False
    if any(ord(char) < 32 for char in value):
        return False
    return all(segment not in {"", ".", ".."} for segment in value.split("/"))


def _safe_weight_file_candidates(values: Sequence[Any]) -> set[str]:
    return {
        value
        for value in values
        if isinstance(value, str) and _safe_repo_filename(value) and _potential_weight_file(value)
    }


def _weight_file_metadata(entry: Mapping[str, Any]) -> dict[str, int | str]:
    """Keep only bounded, stable file identity fields from a tree entry."""

    result: dict[str, int | str] = {}
    lfs = entry.get("lfs")
    raw_lfs = lfs if isinstance(lfs, Mapping) else {}
    size = entry.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        size = raw_lfs.get("size")
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
        result["size_bytes"] = size
    blob_id = entry.get("blobId") or entry.get("blob_id")
    if isinstance(blob_id, str) and re.fullmatch(r"[0-9a-fA-F]{40}", blob_id):
        result["blob_id"] = blob_id.casefold()
    sha256 = raw_lfs.get("sha256")
    if isinstance(sha256, str) and re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        result["lfs_sha256"] = sha256.casefold()
    return result


def _safe_weight_file_metadata(
    value: Any, max_state_bytes: int
) -> tuple[dict[str, dict[str, int | str]], bool]:
    if not isinstance(value, Mapping):
        return {}, False
    result: dict[str, dict[str, int | str]] = {}
    truncated = False
    for filename, details in value.items():
        if (
            not isinstance(filename, str)
            or not _safe_repo_filename(filename)
            or not _potential_weight_file(filename)
            or not isinstance(details, Mapping)
        ):
            continue
        safe_details: dict[str, int | str] = {}
        size = details.get("size_bytes")
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            safe_details["size_bytes"] = size
        blob_id = details.get("blob_id")
        if isinstance(blob_id, str) and re.fullmatch(r"[0-9a-fA-F]{40}", blob_id):
            safe_details["blob_id"] = blob_id.casefold()
        sha256 = details.get("lfs_sha256")
        if isinstance(sha256, str) and re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
            safe_details["lfs_sha256"] = sha256.casefold()
        if safe_details:
            candidate = {**result, filename: safe_details}
            encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            if len(encoded) <= max_state_bytes:
                result[filename] = safe_details
            else:
                truncated = True
    return result, truncated


def _revision_commit_payload(commit: Mapping[str, Any]) -> dict[str, Any]:
    def bounded_text(value: Any, limit: int) -> str | None:
        text = _text(value)
        if not text:
            return None
        return text[:limit]

    authors = []
    for author in _sequence(commit.get("authors")):
        if not isinstance(author, Mapping):
            continue
        projected = {}
        for key in ("username", "name"):
            value = bounded_text(author.get(key), 256)
            if value:
                projected[key] = value
        if projected and len(authors) < 100:
            authors.append(projected)
    return {
        "id": _text(commit.get("id") or commit.get("commit_id")),
        "date": bounded_text(commit.get("date"), 128),
        "created_at": bounded_text(commit.get("created_at"), 128),
        "title": bounded_text(commit.get("title"), 512),
        "message": bounded_text(commit.get("message"), 4096),
        "authors": authors,
    }


def _utcnow() -> datetime:
    return datetime.now(UTC)


class HuggingFaceSourceAdapter:
    """Enumerate public Hugging Face model repositories.

    A first run follows every RFC 8288 ``Link`` cursor. Later runs walk the
    newest-first listing until reaching the previous ``lastModified`` watermark
    minus a configurable overlap. Cursor state also freezes that boundary so an
    interrupted run resumes the same scan. An optional periodic ascending
    ``createdAt`` sweep makes a full cursor pass over the current repository
    listing to recover persistent misses from mutable last-modified ordering.
    The Hub does not document snapshot isolation for cursor pagination, so a
    later sweep may be needed to reconcile repositories created mid-scan.
    """

    # The Hub cursor is provider-issued and page-level checkpoints remain exact;
    # grouping eight already-fetched pages only amortizes local snapshot writes.
    commit_pages = 8

    def __init__(
        self,
        *,
        name: str = "huggingface",
        url: str = "https://huggingface.co/api/models",
        artifact_kind: str | ArtifactKind = ArtifactKind.MODEL_CARD,
        page_size: int = 100,
        overlap_days: int = 2,
        created_at_sweep_interval_days: int = 0,
        max_response_bytes: int = 16 * 1024 * 1024,
        token: str | None = None,
        include_private: bool = False,
        include_revisions: bool = False,
        include_revision_files: bool = False,
        include_checkpoint_file_metadata: bool = False,
        max_revision_tree_pages: int = 20,
        max_revision_weight_file_state_bytes: int = 262_144,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = name
        self.url = url
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.page_size = int(page_size)
        self.overlap_days = int(overlap_days)
        self.created_at_sweep_interval_days = _nonnegative_int(
            created_at_sweep_interval_days, "created_at_sweep_interval_days"
        )
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.token = token
        self.include_private = bool(include_private)
        self.include_revisions = bool(include_revisions)
        self.include_revision_files = bool(include_revision_files)
        if self.include_revision_files and not self.include_revisions:
            raise ValueError("include_revision_files requires include_revisions")
        self.include_checkpoint_file_metadata = bool(include_checkpoint_file_metadata)
        if self.include_checkpoint_file_metadata and self.include_revisions:
            raise ValueError(
                "include_checkpoint_file_metadata cannot be combined with include_revisions"
            )
        self.max_revision_tree_pages = _positive_int(
            max_revision_tree_pages, "max_revision_tree_pages"
        )
        self.max_revision_weight_file_state_bytes = _positive_int(
            max_revision_weight_file_state_bytes,
            "max_revision_weight_file_state_bytes",
        )
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "huggingface-v3",
                "url": self.url,
                "artifact_kind": self.artifact_kind.value,
                "page_size": self.page_size,
                "overlap_days": self.overlap_days,
                "created_at_sweep_interval_days": self.created_at_sweep_interval_days,
                "max_response_bytes": self.max_response_bytes,
                "include_private": self.include_private,
                "include_revisions": self.include_revisions,
                "include_revision_files": self.include_revision_files,
                "include_checkpoint_file_metadata": self.include_checkpoint_file_metadata,
                "max_revision_tree_pages": self.max_revision_tree_pages,
                "max_revision_weight_file_state_bytes": self.max_revision_weight_file_state_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        if self.include_checkpoint_file_metadata and isinstance(
            state.get("checkpoint_metadata_queue"), Sequence
        ):
            return self._fetch_checkpoint_file_metadata_page(state, headers)

        if self.include_revisions and isinstance(state.get("revision_queue"), Sequence):
            return self._fetch_revision_page(state, headers)

        next_url = _text(state.get("next_url"))
        resuming_scan = bool(next_url)
        sweep_in_progress = _text(state.get("coverage_mode")) == "created_at_sweep"
        last_sweep = _parse_timestamp(state.get("created_at_sweep_completed_at"))
        now = _isoformat(self.clock())
        sweep_due = (
            self.created_at_sweep_interval_days > 0
            and (
                last_sweep is None
                or _parse_timestamp(now) is None
                or _parse_timestamp(now)
                >= last_sweep + timedelta(days=self.created_at_sweep_interval_days)
            )
        )
        created_at_sweep = self.created_at_sweep_interval_days > 0 and (
            sweep_in_progress or (not resuming_scan and sweep_due)
        )
        raw_items_seen = _state_count(state, "raw_items_seen") if resuming_scan else 0
        scan_total = _state_count(state, "scan_total") if resuming_scan else None
        scan_count_drifted = state.get("scan_count_drifted") is True if resuming_scan else False
        count_is_complete = not resuming_scan or (
            "raw_items_seen" in state and state.get("raw_count_incomplete") is not True
        )
        prior_watermark = _parse_timestamp(state.get("watermark"))
        if next_url:
            next_url = self._safe_next_url(next_url, self.url)
            cutoff = None if created_at_sweep else _parse_timestamp(state.get("cutoff"))
            scan_high = _parse_timestamp(state.get("scan_high_watermark"))
            response: HttpResponse = self.client.get(next_url, headers=headers)
        else:
            cutoff = (
                None
                if created_at_sweep
                else (
                    prior_watermark - timedelta(days=self.overlap_days)
                    if prior_watermark is not None
                    else None
                )
            )
            scan_high = None
            response = self.client.get(
                self.url,
                params=self._listing_params(created_at_sweep),
                headers=headers,
            )

        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog response exceeds {self.max_response_bytes} bytes"
            )
        payload = response.json()
        if isinstance(payload, Mapping):
            items = payload.get("items") or payload.get("models") or []
        else:
            items = payload
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            raise ValueError(f"{self.name}: expected a JSON array from {response.url}")
        raw_items_seen = (raw_items_seen or 0) + len(items)
        response_total = _integer_header(response.headers, "x-total-count")
        if response_total is not None and count_is_complete:
            if scan_total is not None and response_total != scan_total:
                # Counts can change while cursor pages are being fetched. Keep
                # the largest observed value as a useful truncation guard, but
                # don't let a stale count block an exhausted cursor forever.
                scan_count_drifted = True
            scan_total = max(scan_total or 0, response_total)

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        reached_cutoff = False
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                issues.append(
                    SourceIssue(
                        source_record_id=f"{self.name}:item:{index}",
                        stage="source_normalize",
                        error="TypeError: model result is not a JSON object",
                        summary={"index": index, "value": repr(item)[:1000]},
                    )
                )
                continue
            # An authenticated list can contain repositories visible only to the
            # token. Never persist even their identifiers unless the source was
            # explicitly configured as a private registry.
            if item.get("private") is True and not self.include_private:
                continue
            modified = _parse_timestamp(item.get("lastModified") or item.get("last_modified"))
            if modified is not None and (scan_high is None or modified > scan_high):
                scan_high = modified
            if cutoff is not None and modified is not None and modified < cutoff:
                reached_cutoff = True
                break
            try:
                record = self._record(item)
                if record is not None:
                    records.append(record)
            except (KeyError, TypeError, ValueError) as error:
                raw = dict(item)
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            _text(item.get("id") or item.get("modelId"))
                            or f"{self.name}:malformed:{content_hash(raw)[:32]}"
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index, "raw": raw},
                    )
                )

        raw_link_next = _link_relation(_header(response.headers, "link"), "next")
        link_next = (
            self._safe_next_url(raw_link_next, response.url or self.url)
            if raw_link_next
            else None
        )
        if (
            not reached_cutoff
            and not link_next
            and not scan_count_drifted
            and scan_total is not None
            and raw_items_seen < scan_total
        ):
            raise ValueError(
                f"{self.name}: pagination ended after {raw_items_seen} raw item(s), "
                f"before the known total of {scan_total}"
            )
        complete = reached_cutoff or not link_next
        now = _isoformat(self.clock())
        if complete:
            watermark = (
                prior_watermark
                if created_at_sweep
                else scan_high or prior_watermark or _parse_timestamp(now)
            )
            next_state: dict[str, Any] = {"completed_at": now}
            if watermark is not None:
                next_state["watermark"] = _isoformat(watermark)
            if created_at_sweep:
                next_state["created_at_sweep_completed_at"] = now
            elif state.get("created_at_sweep_completed_at"):
                next_state["created_at_sweep_completed_at"] = state[
                    "created_at_sweep_completed_at"
                ]
        else:
            next_state = {
                "next_url": link_next,
                "started_at": _text(state.get("started_at")) or now,
                "raw_items_seen": raw_items_seen,
            }
            if scan_total is not None:
                next_state["scan_total"] = scan_total
            if scan_count_drifted:
                next_state["scan_count_drifted"] = True
            if not count_is_complete:
                next_state["raw_count_incomplete"] = True
            if prior_watermark is not None:
                next_state["watermark"] = _isoformat(prior_watermark)
            if cutoff is not None:
                next_state["cutoff"] = _isoformat(cutoff)
            if scan_high is not None:
                next_state["scan_high_watermark"] = _isoformat(scan_high)
            if created_at_sweep:
                next_state["coverage_mode"] = "created_at_sweep"
            if state.get("created_at_sweep_completed_at"):
                next_state["created_at_sweep_completed_at"] = state[
                    "created_at_sweep_completed_at"
                ]

        if self.include_checkpoint_file_metadata:
            metadata_queue = [
                dict(record.raw)
                for record in records
                if record.releases
                and _sequence(record.releases[0].metadata.get("weight_files"))
            ]
            if metadata_queue:
                records_without_weights = tuple(
                    record
                    for record in records
                    if not record.releases
                    or not _sequence(record.releases[0].metadata.get("weight_files"))
                )
                next_state = {
                    "checkpoint_metadata_base_state": next_state,
                    "checkpoint_metadata_queue": metadata_queue,
                    "checkpoint_metadata_listing_complete": complete,
                    "checkpoint_metadata_upstream_count": (
                        response_total if response_total is not None else scan_total
                    ),
                }
                return SourcePage(
                    records=records_without_weights,
                    next_state=next_state,
                    complete=False,
                    upstream_count=(
                        response_total if response_total is not None else scan_total
                    ),
                    issues=tuple(issues),
                )

        if self.include_revisions:
            # The Hub's refs and commits endpoints expose exact immutable commit
            # IDs as JSON. Walk them separately so catalog enumeration remains
            # bounded to one HTTP request per checkpoint/page.
            queue = [
                {"model_id": record.source_record_id, "phase": "refs"}
                for record in records
            ]
            if queue:
                next_state = {
                    "base_state": next_state,
                    "revision_queue": queue,
                    "revision_listing_complete": complete,
                }
                complete = False

        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=response_total if response_total is not None else scan_total,
            issues=tuple(issues),
        )

    def _fetch_checkpoint_file_metadata_page(
        self,
        state: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> SourcePage:
        queue = [
            dict(value)
            for value in _sequence(state.get("checkpoint_metadata_queue"))
            if isinstance(value, Mapping)
        ]
        if not queue:
            return SourcePage(
                records=(),
                next_state=dict(state.get("checkpoint_metadata_base_state") or {}),
                complete=state.get("checkpoint_metadata_listing_complete") is True,
                upstream_count=state.get("checkpoint_metadata_upstream_count"),
            )

        item = queue[0]
        repo_id = _hub_repo_id(item.get("id") or item.get("modelId"))
        expected_sha = _text(item.get("sha"))
        if not repo_id or not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
            raise ValueError(f"{self.name}: invalid checkpoint metadata queue identity")
        encoded_id = quote(repo_id, safe="/")
        url = f"https://huggingface.co/api/models/{encoded_id}?blobs=true"
        issues: tuple[SourceIssue, ...] = ()
        metadata: dict[str, dict[str, int | str]] = {}
        metadata_complete = False
        try:
            response: HttpResponse = self.client.get(url, headers=headers)
        except HttpFailure as error:
            if "exceeded" not in str(error) or "bytes" not in str(error):
                raise
            payload = None
            issue_text = (
                f"model detail exceeded the {self.max_response_bytes}-byte metadata limit"
            )
        else:
            if response.status in {401, 403, 404}:
                payload = None
                issue_text = f"model detail is unavailable (HTTP {response.status})"
            elif response.status != 200:
                raise ValueError(f"{self.name}: model detail returned HTTP {response.status}")
            elif len(response.body) > self.max_response_bytes:
                payload = None
                issue_text = (
                    f"model detail exceeded the {self.max_response_bytes}-byte metadata limit"
                )
            else:
                payload = response.json()
                issue_text = ""

        if payload is None:
            issues = (self._checkpoint_metadata_issue(repo_id, issue_text),)
        elif not isinstance(payload, Mapping):
            issues = (
                self._checkpoint_metadata_issue(repo_id, "model detail is not a JSON object"),
            )
        else:
            detail_id = _hub_repo_id(payload.get("id") or payload.get("modelId"))
            detail_sha = _text(payload.get("sha"))
            if detail_id != repo_id:
                issues = (
                    self._checkpoint_metadata_issue(repo_id, "model detail identity changed"),
                )
            elif detail_sha != expected_sha:
                issues = (
                    self._checkpoint_metadata_issue(
                        repo_id,
                        "model detail revision changed during listing",
                        listed_sha=expected_sha,
                        detail_sha=detail_sha or None,
                    ),
                )
            elif not isinstance(payload.get("siblings"), Sequence) or isinstance(
                payload.get("siblings"), (str, bytes, bytearray)
            ):
                issues = (
                    self._checkpoint_metadata_issue(repo_id, "model detail lacks file siblings"),
                )
            else:
                detail_files = {
                    _text(sibling.get("rfilename") or sibling.get("path")): sibling
                    for sibling in _sequence(payload.get("siblings"))
                    if isinstance(sibling, Mapping)
                    and _text(sibling.get("rfilename") or sibling.get("path"))
                }
                listing_files = {
                    _text(sibling.get("rfilename") or sibling.get("path"))
                    for sibling in _sequence(item.get("siblings"))
                    if isinstance(sibling, Mapping)
                    and _text(sibling.get("rfilename") or sibling.get("path"))
                }
                weight_files = {
                    filename
                    for filename in listing_files
                    if _is_weight_file(filename, tuple(listing_files))
                }
                metadata_complete = True
                for filename in weight_files:
                    sibling = detail_files.get(filename)
                    if sibling is None:
                        metadata_complete = False
                        continue
                    details = _weight_file_metadata(sibling)
                    if not details:
                        metadata_complete = False
                        continue
                    metadata[filename] = details
                metadata, truncated = _safe_weight_file_metadata(
                    metadata,
                    self.max_revision_weight_file_state_bytes,
                )
                metadata_complete = metadata_complete and not truncated
                if not metadata_complete:
                    issues = (
                        self._checkpoint_metadata_issue(
                            repo_id,
                            "checkpoint file metadata is incomplete",
                            listed_file_count=len(weight_files),
                            metadata_file_count=len(metadata),
                        ),
                    )

        record = self._record(
            item,
            weight_file_metadata=metadata,
            weight_file_metadata_complete=metadata_complete,
        )
        if record is None:
            records: tuple[SourceRecord, ...] = ()
        else:
            records = (record,)
        remaining = queue[1:]
        base_state = dict(state.get("checkpoint_metadata_base_state") or {})
        if remaining:
            next_state = {
                "checkpoint_metadata_base_state": base_state,
                "checkpoint_metadata_queue": remaining,
                "checkpoint_metadata_listing_complete": (
                    state.get("checkpoint_metadata_listing_complete") is True
                ),
                "checkpoint_metadata_upstream_count": state.get(
                    "checkpoint_metadata_upstream_count"
                ),
            }
            complete = False
        else:
            next_state = base_state
            complete = state.get("checkpoint_metadata_listing_complete") is True
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=complete,
            upstream_count=state.get("checkpoint_metadata_upstream_count"),
            issues=issues,
            advance_on_source_issues=bool(issues),
        )

    def _checkpoint_metadata_issue(
        self,
        repo_id: str,
        message: str,
        **summary: Any,
    ) -> SourceIssue:
        return SourceIssue(
            source_record_id=f"{self.name}:{repo_id}",
            stage="source_normalize",
            error=message,
            summary={
                "model_id": repo_id,
                "file_metadata_status": "incomplete",
                **summary,
            },
        )

    def _listing_params(self, created_at_sweep: bool) -> Mapping[str, Any]:
        return {
            "limit": self.page_size,
            "full": "true",
            "cardData": "true",
            # The Hub API excludes model configuration from `full`;
            # it is a separate opt-in (`config=true` in the REST API).
            "config": "true",
            "sort": "createdAt" if created_at_sweep else "lastModified",
            "direction": 1 if created_at_sweep else -1,
        }

    def _fetch_revision_page(
        self, state: Mapping[str, Any], headers: Mapping[str, str]
    ) -> SourcePage:
        queue = [
            dict(value)
            for value in _sequence(state.get("revision_queue"))
            if isinstance(value, Mapping)
        ]
        if not queue:
            return self._finish_revision_scan(state)
        item = queue[0]
        repo_id = _text(item.get("model_id"))
        if not repo_id:
            raise ValueError(f"{self.name}: revision checkpoint is missing model_id")
        encoded_id = quote(repo_id, safe="/")
        tree_queue = [
            dict(value)
            for value in _sequence(item.get("tree_queue"))
            if isinstance(value, Mapping)
        ]
        if self.include_revision_files and tree_queue:
            task = tree_queue[0]
            commit = task.get("commit")
            if not isinstance(commit, Mapping):
                raise ValueError(f"{self.name}: revision tree task is missing commit metadata")
            sha = _text(commit.get("id") or commit.get("commit_id"))
            if not sha:
                raise ValueError(f"{self.name}: revision tree task is missing commit ID")
            next_url = _text(task.get("next_url"))
            url = self._safe_next_url(next_url, self.url) if next_url else (
                f"https://huggingface.co/api/models/{encoded_id}/tree/"
                f"{quote(sha, safe='')}?recursive=true&expand=false"
            )
            response = self.client.get(url, headers=headers)
            filenames = {
                *_safe_weight_file_candidates(_sequence(task.get("weight_candidates"))),
            }
            candidate_bytes = sum(len(filename.encode("utf-8")) for filename in filenames)
            candidates_truncated = task.get("weight_candidates_truncated") is True
            file_metadata, metadata_state_truncated = _safe_weight_file_metadata(
                task.get("weight_file_metadata"),
                self.max_revision_weight_file_state_bytes,
            )
            metadata_truncated = (
                task.get("weight_file_metadata_truncated") is True or metadata_state_truncated
            )
            page_count = (_state_count(task, "page_count") or 0) + 1
            inaccessible = response.status in {401, 403, 404}
            if inaccessible:
                following = None
            else:
                payload = self._revision_json(response)
                if not isinstance(payload, Sequence) or isinstance(
                    payload, (str, bytes, bytearray)
                ):
                    raise ValueError(f"{self.name}: revision tree must be a JSON array")
                for entry in payload:
                    if not isinstance(entry, Mapping) or entry.get("type", "file") != "file":
                        continue
                    path = _text(entry.get("path") or entry.get("rfilename"))
                    if _potential_weight_file(path):
                        if not _safe_repo_filename(path):
                            # The path looks like a checkpoint but cannot be
                            # represented safely in checkpoint state/URLs.
                            # Do not claim the file list is exhaustive.
                            candidates_truncated = True
                            continue
                        if path not in filenames:
                            path_bytes = len(path.encode("utf-8"))
                            if (
                                candidate_bytes + path_bytes
                                <= self.max_revision_weight_file_state_bytes
                            ):
                                filenames.add(path)
                                candidate_bytes += path_bytes
                            else:
                                candidates_truncated = True
                        if path in filenames and path not in file_metadata:
                            details = _weight_file_metadata(entry)
                            if details:
                                candidate_metadata = {**file_metadata, path: details}
                                encoded_metadata = json.dumps(
                                    candidate_metadata,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                                if (
                                    len(encoded_metadata)
                                    <= self.max_revision_weight_file_state_bytes
                                ):
                                    file_metadata[path] = details
                                else:
                                    metadata_truncated = True
                following = _link_relation(_header(response.headers, "link"), "next")
            complete = (
                inaccessible
                or following is None
                or page_count >= self.max_revision_tree_pages
            )
            if complete:
                weight_files = tuple(
                    sorted(
                        filename
                        for filename in filenames
                        if _is_weight_file(filename, tuple(filenames))
                    )
                )
                record = self._revision_record(
                    repo_id,
                    commit,
                    weight_files=weight_files,
                    weight_files_complete=(
                        following is None and not inaccessible and not candidates_truncated
                    ),
                    weight_file_metadata=file_metadata,
                    weight_file_metadata_truncated=metadata_truncated,
                    git_refs=_sequence(task.get("git_refs")),
                )
                tree_queue.pop(0)
                item["tree_queue"] = tree_queue
                if not tree_queue and item.get("commits_exhausted") is True:
                    queue.pop(0)
                return self._revision_page_result(state, queue, (record,))
            task["next_url"] = self._safe_next_url(following, response.url or url)
            task["page_count"] = page_count
            task["weight_candidates"] = sorted(filenames)
            if file_metadata:
                task["weight_file_metadata"] = file_metadata
            if metadata_truncated:
                task["weight_file_metadata_truncated"] = True
            if candidates_truncated:
                task["weight_candidates_truncated"] = True
            item["tree_queue"] = [task, *tree_queue[1:]]
            return self._revision_page_result(state, queue, ())

        phase = _text(item.get("phase"))
        if phase == "refs":
            url = f"https://huggingface.co/api/models/{encoded_id}/refs"
            response = self.client.get(url, headers=headers)
            payload = self._revision_json(response)
            refs: list[dict[str, Any]] = []
            target_ref_indexes: dict[str, int] = {}
            seen_ref_names: set[str] = set()
            if isinstance(payload, Mapping):
                for group in ("branches", "tags", "converts"):
                    for ref in _sequence(payload.get(group)):
                        if isinstance(ref, Mapping):
                            value = _text(ref.get("ref")) or _text(ref.get("name"))
                            target_commit = _text(
                                ref.get("target_commit") or ref.get("targetCommit")
                            )
                            if not value:
                                continue
                            if target_commit:
                                if target_commit in target_ref_indexes:
                                    refs[target_ref_indexes[target_commit]]["aliases"].append(value)
                                    continue
                                target_ref_indexes[target_commit] = len(refs)
                            elif value in seen_ref_names:
                                continue
                            seen_ref_names.add(value)
                            refs.append(
                                {
                                    "ref": value,
                                    "target_commit": target_commit or None,
                                    "aliases": [value],
                                }
                            )
            item["refs"] = refs
            item["ref_index"] = 0
            item["phase"] = "commits"
            if not refs:
                queue.pop(0)
        elif phase == "commits":
            refs = []
            for value in _sequence(item.get("refs")):
                if isinstance(value, Mapping):
                    ref_value = _text(value.get("ref"))
                    if ref_value:
                        refs.append(
                            {
                                "ref": ref_value,
                                "target_commit": _text(value.get("target_commit")) or None,
                                "aliases": [
                                    _text(alias)
                                    for alias in _sequence(value.get("aliases"))
                                    if _text(alias)
                                ]
                                or [ref_value],
                            }
                        )
                elif _text(value):
                    # Accept older checkpoint states created before ref target
                    # metadata was retained.
                    refs.append(
                        {"ref": _text(value), "target_commit": None, "aliases": [_text(value)]}
                    )
            index = _state_count(item, "ref_index") or 0
            if index >= len(refs):
                queue.pop(0)
                return self._revision_page_result(state, queue, ())
            ref_entry = refs[index]
            ref = ref_entry["ref"]
            target_ref_aliases: dict[str, list[str]] = {}
            for candidate_ref in refs:
                target = _text(candidate_ref.get("target_commit"))
                if target:
                    target_ref_aliases[target] = list(candidate_ref["aliases"])
            next_url = _text(item.get("next_url"))
            url = self._safe_next_url(next_url, self.url) if next_url else (
                f"https://huggingface.co/api/models/{encoded_id}/commits/{quote(ref, safe='')}"
            )
            response = self.client.get(url, headers=headers)
            payload = self._revision_json(response)
            if not isinstance(payload, Sequence) or isinstance(
                payload, (str, bytes, bytearray)
            ):
                raise ValueError(f"{self.name}: revision commits must be a JSON array")
            commits = payload
            valid_commits = tuple(
                commit
                for commit in commits
                if isinstance(commit, Mapping)
                and _text(commit.get("id") or commit.get("commit_id"))
            )
            if self.include_revision_files:
                tree_queue.extend(
                    {
                        "commit": _revision_commit_payload(commit),
                        "weight_candidates": [],
                        "git_refs": target_ref_aliases.get(
                            _text(commit.get("id") or commit.get("commit_id")), []
                        ),
                    }
                    for commit in valid_commits
                )
                item["tree_queue"] = tree_queue
            following = _link_relation(_header(response.headers, "link"), "next")
            if following:
                item["next_url"] = self._safe_next_url(following, response.url or url)
            else:
                item.pop("next_url", None)
                item["ref_index"] = index + 1
                if index + 1 >= len(refs):
                    if self.include_revision_files and tree_queue:
                        item["commits_exhausted"] = True
                    else:
                        queue.pop(0)
            if self.include_revision_files:
                return self._revision_page_result(state, queue, ())
            records = tuple(
                self._revision_record(
                    repo_id,
                    commit,
                    git_refs=target_ref_aliases.get(
                        _text(commit.get("id") or commit.get("commit_id")), []
                    ),
                )
                for commit in valid_commits
            )
            return self._revision_page_result(state, queue, records)
        else:
            raise ValueError(f"{self.name}: invalid revision checkpoint phase {phase!r}")
        return self._revision_page_result(state, queue, ())

    def _revision_json(self, response: HttpResponse) -> Any:
        if response.status != 200:
            raise ValueError(f"{self.name}: revision endpoint returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: revision response exceeds {self.max_response_bytes} bytes"
            )
        return response.json()

    def _revision_record(
        self,
        repo_id: str,
        commit: Mapping[str, Any],
        *,
        weight_files: Sequence[str] = (),
        weight_files_complete: bool | None = None,
        weight_file_metadata: Mapping[str, Mapping[str, int | str]] | None = None,
        weight_file_metadata_truncated: bool = False,
        git_refs: Sequence[Any] = (),
    ) -> SourceRecord:
        sha = _text(commit.get("id") or commit.get("commit_id"))
        model = ModelHint(
            local_id=f"{repo_id}#model",
            name=repo_id,
            identifiers=(Identifier("huggingface:model", repo_id),),
            status=ModelStatus.RELEASED,
            locator="$.repo_id",
        )
        released_at = _text(commit.get("date") or commit.get("created_at")) or None
        revision_id = Identifier("huggingface:revision", f"{repo_id}@{sha}")
        release_metadata: dict[str, Any] = {
            "title": _text(commit.get("title")) or None,
            "message": _text(commit.get("message")) or None,
            "authors": [
                _text(author.get("username") or author.get("name"))
                for author in _sequence(commit.get("authors"))
                if isinstance(author, Mapping)
                and _text(author.get("username") or author.get("name"))
            ],
        }
        model_url = canonicalize_url(f"https://huggingface.co/{quote(repo_id, safe='/')}")
        links = [Link(model_url, relation="model_page", locator="$.repo_id")]
        if weight_files_complete is not None:
            release_metadata["weight_files"] = sorted(set(weight_files))
            release_metadata["weight_files_complete"] = weight_files_complete
            if weight_file_metadata:
                release_metadata["weight_file_metadata"] = {
                    filename: dict(weight_file_metadata[filename])
                    for filename in sorted(weight_file_metadata)
                    if filename in set(weight_files)
                }
            if weight_file_metadata_truncated:
                release_metadata["weight_file_metadata_truncated"] = True
        if refs := sorted({_text(value) for value in git_refs if _text(value)}):
            release_metadata["git_refs"] = refs
        if weight_files_complete is not None:
            links.extend(
                Link(
                    canonicalize_url(
                        f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/"
                        f"{quote(sha, safe='')}/{quote(filename, safe='/')}"
                    ),
                    relation="weights",
                    locator="$.tree",
                    crawl=False,
                )
                for filename in sorted(set(weight_files))
            )
        release = ReleaseHint(
            local_id=f"{repo_id}#release:{sha}",
            model_local_id=model.local_id,
            revision=sha,
            identifiers=(revision_id,),
            released_at=released_at,
            metadata=release_metadata,
            locator="$.id",
        )
        return SourceRecord(
            source_record_id=f"{repo_id}@{sha}",
            kind=self.artifact_kind,
            canonical_url=model_url,
            title=f"{repo_id} {sha[:12]}",
            raw=dict(commit),
            published_at=released_at,
            identifiers=(Identifier("huggingface:model", repo_id), revision_id),
            links=tuple(links),
            models=(model,),
            releases=(release,),
        )

    def _revision_page_result(
        self, state: Mapping[str, Any], queue: list[dict[str, Any]], records: Sequence[SourceRecord]
    ) -> SourcePage:
        next_state = dict(state)
        next_state["revision_queue"] = queue
        if not queue:
            return self._finish_revision_scan(next_state, records)
        return SourcePage(records=tuple(records), next_state=next_state, complete=False)

    def _finish_revision_scan(
        self, state: Mapping[str, Any], records: Sequence[SourceRecord] = ()
    ) -> SourcePage:
        next_state = dict(state.get("base_state") or {})
        was_complete = state.get("revision_listing_complete") is True
        return SourcePage(records=tuple(records), next_state=next_state, complete=was_complete)

    def _safe_next_url(self, value: str, base_url: str) -> str:
        candidate = canonicalize_url(urljoin(base_url, value))
        if _origin(candidate) != _origin(self.url):
            raise ValueError(f"{self.name}: pagination URL changed origin")
        return candidate

    def _record(
        self,
        item: Mapping[str, Any],
        *,
        weight_file_metadata: Mapping[str, Mapping[str, int | str]] | None = None,
        weight_file_metadata_complete: bool | None = None,
    ) -> SourceRecord | None:
        repo_id = _text(item.get("id")) or _text(item.get("modelId"))
        if not repo_id:
            raise ValueError(f"{self.name}: model result is missing id")
        encoded_id = quote(repo_id, safe="/")
        model_url = canonicalize_url(f"https://huggingface.co/{encoded_id}")
        api_url = canonicalize_url(f"https://huggingface.co/api/models/{encoded_id}")

        identifiers: list[Identifier] = [Identifier("huggingface:model", repo_id)]
        links: list[Link] = [
            Link(model_url, relation="model_page", locator="$.id"),
            Link(api_url, relation="metadata", locator="$.id"),
        ]

        card_data = item.get("cardData") or item.get("card_data")
        config = item.get("config")
        for locator, value in (("$.cardData", card_data), ("$.config", config)):
            for url in extract_urls(value):
                links.append(
                    Link(
                        url,
                        relation=_declared_reference_relation(url),
                        locator=locator,
                    )
                )
                if identifier := identifier_from_url(url):
                    identifiers.append(identifier)

        if isinstance(card_data, Mapping):
            card_data_locator = "$.cardData" if "cardData" in item else "$.card_data"
            dataset_values = card_data.get("datasets")
            is_single_dataset = isinstance(dataset_values, str)
            dataset_values = (dataset_values,) if is_single_dataset else _sequence(dataset_values)
            for index, value in enumerate(dataset_values):
                dataset_id = _hub_repo_id(value)
                if not dataset_id:
                    continue
                dataset_url = canonicalize_url(
                    "https://huggingface.co/datasets/" + quote(dataset_id, safe="/")
                )
                links.append(
                    Link(
                        dataset_url,
                        relation="dataset",
                        locator=(
                            f"{card_data_locator}.datasets"
                            if is_single_dataset
                            else f"{card_data_locator}.datasets[{index}]"
                        ),
                    )
                )
                identifiers.append(Identifier("huggingface:dataset", dataset_id))

        commit_sha = _text(item.get("sha"))
        revision = commit_sha or "main"
        sibling_filenames = [
            filename
            for sibling in _sequence(item.get("siblings"))
            if isinstance(sibling, Mapping)
            if (filename := _text(sibling.get("rfilename") or sibling.get("path")))
        ]
        for sibling in _sequence(item.get("siblings")):
            if not isinstance(sibling, Mapping):
                continue
            filename = _text(sibling.get("rfilename")) or _text(sibling.get("path"))
            if not filename:
                continue
            if filename.casefold() == "readme.md":
                links.append(
                    Link(
                        canonicalize_url(
                            f"https://huggingface.co/{encoded_id}/raw/"
                            f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
                        ),
                        relation="model_card_source",
                        locator="$.siblings",
                    )
                )
            elif filename.casefold() == "config.json":
                links.append(
                    Link(
                        canonicalize_url(
                            f"https://huggingface.co/{encoded_id}/raw/"
                            f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
                        ),
                        relation="model_config",
                        locator="$.siblings",
                    )
                )
            elif _is_weight_file(filename, sibling_filenames):
                # The list response is the source-declared artifact inventory.
                # Keep a revision-pinned reference to each checkpoint, but do
                # not enqueue or download binary bytes during model discovery.
                links.append(
                    Link(
                        canonicalize_url(
                            f"https://huggingface.co/{encoded_id}/resolve/"
                            f"{quote(revision, safe='')}/{quote(filename, safe='/')}"
                        ),
                        relation="weights",
                        locator="$.siblings",
                        crawl=False,
                    )
                )

        tags = tuple(_text(value) for value in _sequence(item.get("tags")) if _text(value))
        for tag in tags:
            if identifier := _identifier_from_tag(tag):
                identifiers.append(identifier)

        aliases = tuple(
            value
            for value in (_text(item.get("modelId")), _text(item.get("id")))
            if value and value != repo_id
        )
        model = ModelHint(
            local_id=f"{repo_id}#model",
            name=repo_id,
            identifiers=(Identifier("huggingface:model", repo_id),),
            aliases=aliases,
            status=ModelStatus.RELEASED,
            locator="$.id",
        )
        relations = tuple(self._base_model_relations(item, card_data, model.local_id, repo_id))
        release_identifiers = (
            (Identifier("huggingface:revision", f"{repo_id}@{commit_sha}"),)
            if commit_sha
            else ()
        )
        release_metadata: dict[str, Any] = {
            "library_name": _text(item.get("library_name")) or None,
            "pipeline_tag": _text(item.get("pipeline_tag")) or None,
            "weight_files": sorted(
                filename
                for filename in sibling_filenames
                if _is_weight_file(filename, sibling_filenames)
            ),
            "weight_files_complete": True,
        }
        if weight_file_metadata is not None:
            safe_metadata, truncated = _safe_weight_file_metadata(
                weight_file_metadata,
                self.max_revision_weight_file_state_bytes,
            )
            release_metadata["weight_file_metadata"] = safe_metadata
            release_metadata["weight_file_metadata_complete"] = bool(
                weight_file_metadata_complete and not truncated
            )
            if truncated:
                release_metadata["weight_file_metadata_truncated"] = True
        release = ReleaseHint(
            local_id=f"{repo_id}#release:{revision}",
            model_local_id=model.local_id,
            revision=revision,
            identifiers=release_identifiers,
            released_at=_text(item.get("lastModified") or item.get("last_modified")) or None,
            metadata=release_metadata,
            locator="$.sha" if commit_sha else "$.id",
        )

        text_parts = []
        if pipeline := _text(item.get("pipeline_tag")):
            text_parts.append(f"pipeline: {pipeline}")
        if tags:
            text_parts.append("tags: " + ", ".join(tags))
        if card_data is not None:
            text_parts.append(json.dumps(card_data, ensure_ascii=False, sort_keys=True))

        return SourceRecord(
            source_record_id=repo_id,
            kind=self.artifact_kind,
            canonical_url=model_url,
            title=repo_id,
            raw=dict(item),
            text="\n".join(text_parts),
            published_at=_text(item.get("createdAt") or item.get("created_at")) or None,
            modified_at=_text(item.get("lastModified") or item.get("last_modified")) or None,
            identifiers=_unique_identifiers(identifiers),
            links=_unique_links(links),
            models=(model,),
            model_relations=relations,
            releases=(release,),
        )

    def _base_model_relations(
        self,
        item: Mapping[str, Any],
        card_data: Any,
        subject_local_id: str,
        repo_id: str,
    ) -> Iterable[ModelRelationHint]:
        relation_kinds = {"adapter", "merge", "quantized", "finetune"}
        relation = "base_model"
        if isinstance(card_data, Mapping):
            declared_relation = _text(card_data.get("base_model_relation")).casefold()
            if declared_relation in relation_kinds:
                relation = declared_relation
        candidates: list[tuple[str, str, str]] = []
        candidates.extend(
            (name, locator, relation)
            for name, locator in _named_values(item.get("baseModels"), "$.baseModels")
        )
        candidates.extend(
            (name, locator, relation)
            for name, locator in _named_values(item.get("base_models"), "$.base_models")
        )
        if isinstance(card_data, Mapping):
            candidates.extend(
                (name, locator, relation)
                for name, locator in _named_values(
                    card_data.get("base_model"), "$.cardData.base_model"
                )
            )
            candidates.extend(
                (name, locator, relation)
                for name, locator in _named_values(
                    card_data.get("base_models"), "$.cardData.base_models"
                )
            )
            # `new_version` is an explicit link to a distinct model repo.
            # Require an exact Hub model identifier before creating the edge.
            for candidate, _ in _named_values(
                card_data.get("new_version"), "$.cardData.new_version"
            ):
                name, identifiers = _model_identity(candidate)
                if name and identifiers:
                    candidates.append((name, "$.cardData.new_version", "new_version"))
        for tag in _sequence(item.get("tags")):
            tag_text = _text(tag)
            if tag_text.casefold().startswith("base_model:"):
                candidate = tag_text.split(":", maxsplit=1)[1].strip()
                relation = "base_model"
                possible_relation, separator, remainder = candidate.partition(":")
                if separator and possible_relation.casefold() in relation_kinds:
                    relation = possible_relation.casefold()
                    candidate = remainder.strip()
                candidates.append((candidate, "$.tags", relation))

        seen: set[tuple[str, str]] = set()
        for index, (candidate, locator, relation) in enumerate(candidates):
            name, identifiers = _model_identity(candidate)
            key = (relation, name)
            if not name or name == repo_id or key in seen:
                continue
            seen.add(key)
            target = ModelHint(
                local_id=f"{repo_id}#base-model-{index}",
                name=name,
                identifiers=identifiers,
                status=ModelStatus.DOCUMENTED,
                locator=locator,
            )
            yield ModelRelationHint(
                subject_local_id=subject_local_id,
                predicate=relation,
                target=target,
                locator=locator,
            )


class HuggingFaceDatasetCheckpointSourceAdapter(HuggingFaceSourceAdapter):
    """Enumerate public dataset repos and admit only explicit checkpoint files.

    The Hub's global dataset listing does not reliably include repo siblings.
    This adapter checkpoints the listing cursor and a bounded queue of repo IDs,
    then fetches one repo detail response per page. It verifies the detail SHA
    against the listing SHA before admitting any file identity.
    """

    coverage_limitation = (
        "Covers checkpoint-looking files in public dataset repositories returned "
        "by the unfiltered Hub dataset listing. Each repository is a candidate "
        "model bundle only; dataset contents can include incidental checkpoints. "
        "Private repos and files outside recognized weight formats are excluded."
    )

    def __init__(
        self,
        *,
        name: str = "huggingface-dataset-checkpoints",
        url: str = "https://huggingface.co/api/datasets",
        page_size: int = 10,
        created_at_sweep_interval_days: int = 30,
        max_response_bytes: int = 16 * 1024 * 1024,
        max_checkpoint_files: int = 10_000,
        token: str | None = None,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.max_checkpoint_files = _positive_int(
            max_checkpoint_files, "max_checkpoint_files"
        )
        super().__init__(
            name=name,
            url=url,
            artifact_kind=ArtifactKind.WEIGHTS,
            page_size=page_size,
            overlap_days=2,
            created_at_sweep_interval_days=created_at_sweep_interval_days,
            max_response_bytes=max_response_bytes,
            token=token,
            include_private=False,
            include_revisions=False,
            client=client,
            clock=clock,
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "huggingface-dataset-checkpoints-v1",
                "base_signature": self.checkpoint_signature,
                "max_checkpoint_files": max_checkpoint_files,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        queue = _sequence(state.get("dataset_detail_queue"))
        if not queue:
            base_page = super().fetch_page(state)
            queue = [
                dict(record.raw)
                for record in base_page.records
                if record.raw.get("dataset_detail_pending") is True
            ]
            if not queue:
                return base_page
            next_state = {
                "dataset_base_state": dict(base_page.next_state),
                "dataset_detail_queue": queue,
                "dataset_listing_complete": base_page.complete,
                "dataset_upstream_count": base_page.upstream_count,
            }
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=False,
                upstream_count=base_page.upstream_count,
                issues=base_page.issues,
            )

        repo = queue[0]
        if not isinstance(repo, Mapping):
            raise ValueError(f"{self.name}: invalid dataset detail checkpoint")
        repo_id = _hub_repo_id(repo.get("id"))
        expected_sha = _text(repo.get("sha"))
        if not repo_id or not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
            raise ValueError(f"{self.name}: invalid dataset detail checkpoint identity")
        detail_url = f"https://huggingface.co/api/datasets/{quote(repo_id, safe='/')}"
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        issues: tuple[SourceIssue, ...] = ()
        records: tuple[SourceRecord, ...] = ()
        try:
            response: HttpResponse = self.client.get(detail_url, headers=headers)
        except HttpFailure as error:
            if "exceeded" not in str(error) or "bytes" not in str(error):
                raise
            issues = (
                self._detail_issue(
                    repo_id,
                    f"dataset detail exceeded the {self.max_response_bytes}-byte metadata limit",
                ),
            )
            payload = None
        else:
            if response.status != 200:
                if response.status in {401, 403, 404}:
                    issues = (
                        self._detail_issue(
                            repo_id,
                            f"dataset detail is unavailable (HTTP {response.status})",
                        ),
                    )
                    payload = None
                else:
                    raise ValueError(
                        f"{self.name}: dataset detail returned HTTP {response.status}"
                    )
            elif len(response.body) > self.max_response_bytes:
                issues = (
                    self._detail_issue(
                        repo_id,
                        "dataset detail exceeded the "
                        f"{self.max_response_bytes}-byte metadata limit",
                    ),
                )
                payload = None
            else:
                payload = response.json()
        if payload is None:
            pass
        elif not isinstance(payload, Mapping):
            issues = (self._detail_issue(repo_id, "dataset detail is not a JSON object"),)
        else:
            detail_id = _hub_repo_id(payload.get("id"))
            detail_sha = _text(payload.get("sha"))
            if detail_id != repo_id:
                issues = (self._detail_issue(repo_id, "dataset detail identity changed"),)
            elif detail_sha != expected_sha:
                issues = (
                    SourceIssue(
                        source_record_id=f"{repo_id}@{expected_sha}",
                        stage="source_normalize",
                        error="dataset detail revision changed during listing",
                        summary={
                            "listed_sha": expected_sha,
                            "detail_sha": detail_sha or None,
                            "file_inventory_status": "revision_drift_incomplete",
                        },
                    ),
                )
            elif not isinstance(payload.get("siblings"), Sequence) or isinstance(
                payload.get("siblings"), (str, bytes, bytearray)
            ):
                issues = (self._detail_issue(repo_id, "dataset detail lacks file siblings"),)
            else:
                record = self._checkpoint_record(payload)
                if record is not None:
                    records = (record,)
                    if record.raw.get("weight_files_complete") is False:
                        omitted_files = record.raw.get("omitted_weight_file_count", 0)
                        issues = (
                            SourceIssue(
                                source_record_id=f"{repo_id}@{expected_sha}",
                                stage="source_normalize",
                                error="dataset checkpoint file inventory exceeded configured limit",
                                summary={
                                    "dataset_id": repo_id,
                                    "file_inventory_status": "truncated",
                                    "retained_weight_file_count": len(
                                        record.raw.get("weight_files", [])
                                    ),
                                    "omitted_weight_file_count": omitted_files,
                                    "max_checkpoint_files": self.max_checkpoint_files,
                                },
                            ),
                        )

        remaining = [dict(value) for value in queue[1:] if isinstance(value, Mapping)]
        base_state = dict(state.get("dataset_base_state") or {})
        if remaining:
            next_state = {
                "dataset_base_state": base_state,
                "dataset_detail_queue": remaining,
                "dataset_listing_complete": state.get("dataset_listing_complete") is True,
                "dataset_upstream_count": state.get("dataset_upstream_count"),
            }
            complete = False
        else:
            next_state = base_state
            complete = state.get("dataset_listing_complete") is True
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=complete,
            upstream_count=state.get("dataset_upstream_count"),
            issues=issues,
            advance_on_source_issues=bool(issues),
        )

    def _detail_issue(self, repo_id: str, message: str) -> SourceIssue:
        return SourceIssue(
            source_record_id=f"{self.name}:{repo_id}",
            stage="source_normalize",
            error=message,
            summary={"dataset_id": repo_id, "file_inventory_status": "incomplete"},
        )

    def _listing_params(self, created_at_sweep: bool) -> Mapping[str, Any]:
        # Keep the global list unfiltered. It provides the SHA needed to verify
        # the later detail call, but live list responses omit `siblings`.
        return {
            "limit": self.page_size,
            "full": "true",
            "sort": "createdAt" if created_at_sweep else "lastModified",
            "direction": 1 if created_at_sweep else -1,
        }

    def _record(self, item: Mapping[str, Any]) -> SourceRecord:
        repo_id = _hub_repo_id(item.get("id"))
        if not repo_id:
            raise ValueError(f"{self.name}: dataset result is missing a valid owner/repo id")
        raw_revision = _text(item.get("sha"))
        if not re.fullmatch(r"[0-9a-f]{40}", raw_revision):
            raise ValueError(f"{self.name}: dataset result is missing a full commit SHA")
        tags = [
            _text(value)[:256]
            for value in _sequence(item.get("tags"))[:100]
            if _text(value)
        ]
        metadata = {
            "id": repo_id,
            "sha": raw_revision,
            "dataset_detail_pending": True,
            "gated": item.get("gated") is True,
            "tags": tags,
            "createdAt": _text(item.get("createdAt") or item.get("created_at"))[:128],
            "lastModified": _text(item.get("lastModified") or item.get("last_modified"))[:128],
        }
        return SourceRecord(
            source_record_id=f"{repo_id}@{raw_revision}:detail-queue",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(
                f"https://huggingface.co/datasets/{quote(repo_id, safe='/')}"
            ),
            title=f"{repo_id} dataset file inventory queue",
            raw=metadata,
        )

    def _checkpoint_record(self, item: Mapping[str, Any]) -> SourceRecord | None:
        repo_id = _hub_repo_id(item.get("id"))
        if not repo_id:
            raise ValueError(f"{self.name}: dataset detail is missing a valid owner/repo id")
        raw_revision = _text(item.get("sha"))
        if not re.fullmatch(r"[0-9a-f]{40}", raw_revision):
            raise ValueError(f"{self.name}: dataset detail is missing a full commit SHA")
        siblings = _sequence(item.get("siblings"))
        filenames = {
            _text(sibling.get("rfilename"))
            for sibling in siblings
            if isinstance(sibling, Mapping) and _text(sibling.get("rfilename"))
        }
        candidate_weight_files = sorted(
            filename
            for filename in _safe_weight_file_candidates(tuple(filenames))
            if _is_weight_file(filename, tuple(filenames))
        )
        if not candidate_weight_files:
            return None
        complete = len(candidate_weight_files) <= self.max_checkpoint_files
        weight_files = candidate_weight_files[: self.max_checkpoint_files]
        encoded_id = quote(repo_id, safe="/")
        dataset_url = canonicalize_url(f"https://huggingface.co/datasets/{encoded_id}")
        model_local_id = f"{repo_id}#dataset-checkpoint-bundle"
        checkpoint_id = f"{repo_id}@{raw_revision}"
        links = tuple(
            Link(
                canonicalize_url(
                    f"https://huggingface.co/datasets/{encoded_id}/resolve/"
                    f"{raw_revision}/{quote(filename, safe='/')}"
                ),
                relation="weights",
                locator="$.siblings",
                crawl=False,
            )
            for filename in weight_files
        )
        tags = tuple(_text(value) for value in _sequence(item.get("tags")) if _text(value))
        raw = {
            "id": repo_id,
            "sha": raw_revision,
            "repo_type": "dataset",
            "gated": item.get("gated") is True,
            "weight_files": weight_files,
            "weight_files_complete": complete,
            "omitted_weight_file_count": len(candidate_weight_files) - len(weight_files),
            "tags": list(tags),
        }
        return SourceRecord(
            source_record_id=checkpoint_id,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=dataset_url,
            title=f"{repo_id} checkpoint bundle",
            raw=raw,
            text="tags: " + ", ".join(tags) if tags else "",
            published_at=_text(item.get("createdAt") or item.get("created_at")) or None,
            modified_at=_text(item.get("lastModified") or item.get("last_modified")) or None,
            identifiers=(Identifier("huggingface:dataset", repo_id),),
            links=links,
            models=(
                ModelHint(
                    local_id=model_local_id,
                    name=f"{repo_id} checkpoint bundle",
                    identifiers=(
                        Identifier("huggingface:dataset-checkpoint-candidate", repo_id),
                    ),
                    status=ModelStatus.CANDIDATE,
                    locator="$.siblings",
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"{repo_id}#dataset-release:{raw_revision}",
                    model_local_id=model_local_id,
                    revision=raw_revision,
                    identifiers=(
                        Identifier("huggingface:dataset-revision", checkpoint_id),
                    ),
                    released_at=_text(item.get("lastModified") or item.get("last_modified"))
                    or None,
                    metadata={
                        "repo_type": "dataset",
                        "weight_files": weight_files,
                        "weight_files_complete": complete,
                    },
                    locator="$.sha",
                ),
            ),
        )


def _named_values(value: Any, locator: str) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(value.strip(), locator)] if value.strip() else []
    if isinstance(value, Mapping):
        candidate = next(
            (_text(value.get(key)) for key in ("id", "name", "model") if _text(value.get(key))),
            "",
        )
        return [(candidate, locator)] if candidate else []
    return [
        pair
        for item in _sequence(value)
        for pair in _named_values(item, locator)
    ]


def _model_identity(value: str) -> tuple[str, tuple[Identifier, ...]]:
    value = value.strip()
    if not value:
        return "", ()
    if (
        (identifier := identifier_from_url(value))
        and identifier.namespace == "huggingface:model"
    ):
        return identifier.value, (identifier,)
    # Relation metadata can contain arbitrary URLs as well as repo IDs. Only
    # treat a literal owner/repo pair as a Hub model identity; otherwise an
    # external URL (or a dataset/Space URL) would be promoted to a fake model.
    if _hub_repo_id(value):
        return value, (Identifier("huggingface:model", value),)
    if "://" in value or value.startswith("/"):
        return "", ()
    return value, ()


def _identifier_from_tag(tag: str) -> Identifier | None:
    prefix, separator, value = tag.partition(":")
    if not separator or not value.strip():
        return None
    namespace = prefix.casefold().strip()
    if namespace == "arxiv":
        return Identifier("arxiv", re.sub(r"v\d+$", "", value.strip()))
    if namespace == "doi":
        return Identifier("doi", value.strip().casefold())
    return None


def _hub_repo_id(value: Any) -> str | None:
    """Accept only an explicit owner/repo identifier for a Hub dataset."""

    candidate = _text(value)
    parts = candidate.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        return None
    if any(
        character.isspace()
        or ord(character) < 0x20
        or character in "\\?#"
        for character in candidate
    ):
        return None
    return candidate


def _declared_reference_relation(url: str) -> str:
    """Type a direct Hub-card URL only when its canonical URL proves the type."""

    identifier = identifier_from_url(url)
    if identifier is not None and identifier.namespace in {"arxiv", "doi"}:
        return "paper_reference"
    if identifier is not None and identifier.namespace == "github:repository":
        return "code_reference"
    return "metadata_reference"


def _link_relation(header: str | None, relation: str) -> str | None:
    if not header:
        return None
    wanted = relation.casefold()
    for match in _LINK_RE.finditer(header):
        rel_match = _REL_RE.search(match.group(2))
        relations = (rel_match.group(1) or rel_match.group(2) or "") if rel_match else ""
        if wanted in relations.casefold().split():
            return match.group(1)
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == wanted), None)


def _integer_header(headers: Mapping[str, str], name: str) -> int | None:
    value = _header(headers, name)
    try:
        result = int(value) if value is not None else None
    except ValueError:
        return None
    return result if result is not None and result >= 0 else None


def _state_count(state: Mapping[str, Any], key: str) -> int | None:
    if key not in state:
        return None
    value = state.get(key)
    if isinstance(value, bool):
        raise ValueError(f"invalid Hugging Face checkpoint {key}: {value!r}")
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid Hugging Face checkpoint {key}: {value!r}") from error
    if count < 0:
        raise ValueError(f"invalid Hugging Face checkpoint {key}: {value!r}")
    return count


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _unique_identifiers(values: Iterable[Identifier]) -> tuple[Identifier, ...]:
    return tuple(dict.fromkeys(values))


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    seen: set[tuple[str, str]] = set()
    result = []
    for value in values:
        key = (value.url, value.relation)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    scheme = parts.scheme.casefold()
    port = parts.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (parts.hostname or "").casefold(), port


__all__ = ["HuggingFaceSourceAdapter"]
