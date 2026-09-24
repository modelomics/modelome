from __future__ import annotations

import io
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ARXIV_ID = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})$", re.IGNORECASE
)
_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_METADATA_COLUMNS = (
    "paper_id",
    "title",
    "authors",
    "abstract",
    "categories",
    "primary_category",
    "license",
    "doi",
    "first_version_date",
    "latest_version_date",
    "oai_datestamp",
    "oai_sets",
    "arxiv_abs_url",
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _HttpRangeReader(io.RawIOBase):
    """Bounded random access to an immutable public object over HTTP ranges.

    ``ParquetFile`` needs a seekable file, but a large static archive must not be
    copied locally merely to inspect one row group. The reader verifies every
    response's Content-Range and refuses unbounded reads; this makes the byte
    boundary part of the source's correctness and resource policy.
    """

    def __init__(
        self,
        *,
        client: HttpClient | Any,
        url: str,
        file_size: int,
        max_range_bytes: int,
    ) -> None:
        super().__init__()
        if file_size < 1:
            raise ValueError("range reader file_size must be positive")
        if max_range_bytes < 1:
            raise ValueError("range reader max_range_bytes must be positive")
        self.client = client
        self.url = _web_url(url, "range reader URL")
        self.file_size = int(file_size)
        self.max_range_bytes = int(max_range_bytes)
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self.file_size + offset
        else:
            raise ValueError("unsupported range-reader seek origin")
        if position < 0:
            raise ValueError("range-reader seek before start of file")
        self._position = min(position, self.file_size)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            raise ValueError("range reader refuses an unbounded read")
        if size == 0 or self._position >= self.file_size:
            return b""
        requested = min(size, self.file_size - self._position)
        chunks: list[bytes] = []
        while requested:
            length = min(requested, self.max_range_bytes)
            start = self._position
            end = start + length - 1
            response: HttpResponse = self.client.get(
                self.url,
                headers={
                    "Accept": "application/vnd.apache.parquet",
                    "Range": f"bytes={start}-{end}",
                },
            )
            _verify_range_response(response, start, end, self.file_size)
            chunks.append(response.body)
            self._position += length
            requested -= length
        return b"".join(chunks)

    def readinto(self, buffer: bytearray | memoryview) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)


class ArxivCompleteSnapshotSourceAdapter:
    """Page through the user-provided arXiv metadata snapshot by Parquet row group.

    The source consumes only the CC0 bibliographic metadata layer—not the corpus'
    source, PDF, or TeX payloads—and each page makes bounded HTTP range requests.
    It is a historical backfill for the live arXiv source, so it contributes to the
    same artifact namespace but never performs deletion inference.
    """

    artifact_source: str

    # Each provider page contains up to 10,000 metadata records. Keeping two
    # pages in one local commit halves full-store snapshot writes while retaining
    # a bounded 20,000-record recovery window and normal range pagination.
    commit_pages = 2

    def __init__(
        self,
        *,
        name: str = "arxiv-complete-snapshot",
        artifact_source: str = "arxiv",
        dataset_id: str = "secemp9/arxiv-complete",
        metadata_url: str = "https://huggingface.co/api/datasets/secemp9/arxiv-complete",
        data_path: str = "metadata/train-00000-of-00001.parquet",
        license: str = "mixed-arxiv-author-licenses",
        page_size: int = 10_000,
        max_range_bytes: int = 32 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.artifact_source = _required_text(artifact_source, "artifact source")
        self.dataset_id = _required_text(dataset_id, "dataset ID")
        self.metadata_url = _web_url(metadata_url, "metadata URL")
        self.data_path = _data_path(data_path)
        self.license = _required_text(license, "dataset license")
        if page_size < 1:
            raise ValueError("page_size must be positive")
        self.page_size = int(page_size)
        if max_range_bytes < 64 * 1024:
            raise ValueError("max_range_bytes must be at least 65536")
        self.max_range_bytes = int(max_range_bytes)
        self.client = client or HttpClient(max_response_bytes=self.max_range_bytes)
        self.clock = clock
        self._cached_row_group_key: tuple[str, str, int] | None = None
        self._cached_row_group: pa.Table | None = None
        self.checkpoint_signature = content_hash(
            {
                "adapter": "arxiv-complete-snapshot-v1",
                "artifact_source": self.artifact_source,
                "dataset_id": self.dataset_id,
                "metadata_url": self.metadata_url,
                "data_path": self.data_path,
                "license": self.license,
                "columns": _METADATA_COLUMNS,
                "page_size": self.page_size,
                "max_range_bytes": self.max_range_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision = _commit(state.get("snapshot_revision"))
        completed_revision = _commit(state.get("completed_snapshot_revision"))
        row_group_index = _nonnegative_int(state.get("row_group_index", 0), "row group")
        row_offset = _nonnegative_int(state.get("row_offset", 0), "row offset")
        row_group_count = _nonnegative_int(state.get("row_group_count", 0), "row group count")
        row_count = _nonnegative_int(state.get("row_count", 0), "row count")
        file_size = _nonnegative_int(state.get("file_size", 0), "file size")
        checked_at = _isoformat(self.clock())

        # A frozen in-progress revision is resumed directly. Once it finishes,
        # refresh the small manifest to detect a newer immutable snapshot.
        if not revision or completed_revision:
            metadata_response: HttpResponse = self.client.get(
                self.metadata_url, headers={"Accept": "application/json"}
            )
            metadata = metadata_response.json()
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{self.name}: dataset metadata is not an object")
            revision = _snapshot_revision(metadata, self.dataset_id, self.data_path, self.name)
            if revision == completed_revision:
                return SourcePage(
                    records=(),
                    next_state={
                        "snapshot_revision": revision,
                        "completed_snapshot_revision": revision,
                        "checked_at": checked_at,
                    },
                    complete=True,
                    authoritative_snapshot=False,
                )
            row_group_index = 0
            row_group_count = 0
            row_count = 0
            file_size = 0

        data_url = _resolve_url(self.dataset_id, revision, self.data_path)
        cache_key = (revision, data_url, row_group_index)
        if (
            self._cached_row_group_key == cache_key
            and self._cached_row_group is not None
            and row_group_count
            and row_count
            and file_size
        ):
            table = self._cached_row_group
            actual_group_count = row_group_count
            actual_row_count = row_count
        else:
            if not file_size:
                file_size = _remote_file_size(self.client, data_url)
            reader = _HttpRangeReader(
                client=self.client,
                url=data_url,
                file_size=file_size,
                max_range_bytes=self.max_range_bytes,
            )
            try:
                parquet = pq.ParquetFile(reader)
                actual_group_count = parquet.metadata.num_row_groups
                actual_row_count = parquet.metadata.num_rows
                columns = set(parquet.schema_arrow.names)
                missing = sorted(set(_METADATA_COLUMNS) - columns)
                if missing:
                    raise ValueError(
                        f"{self.name}: metadata Parquet is missing required column(s): "
                        + ", ".join(missing)
                    )
                if row_group_count and row_group_count != actual_group_count:
                    raise ValueError(f"{self.name}: immutable snapshot row-group count changed")
                if row_count and row_count != actual_row_count:
                    raise ValueError(f"{self.name}: immutable snapshot row count changed")
                if row_group_index >= actual_group_count:
                    raise ValueError(f"{self.name}: row-group checkpoint is outside snapshot")
                table = parquet.read_row_group(
                    row_group_index, columns=list(_METADATA_COLUMNS)
                )
                self._cached_row_group_key = cache_key
                self._cached_row_group = table
            except (pa.ArrowException, OSError, ValueError) as error:
                raise ValueError(f"{self.name}: unreadable metadata Parquet row group") from error
            finally:
                reader.close()

        if row_offset >= table.num_rows:
            raise ValueError(f"{self.name}: row offset is outside its row group")
        rows = table.slice(row_offset, self.page_size).to_pylist()

        records, issues = _records(
            rows,
            dataset_id=self.dataset_id,
            revision=revision,
            data_path=self.data_path,
            dataset_license=self.license,
            row_group_index=row_group_index,
        )
        next_offset = row_offset + len(rows)
        if next_offset == table.num_rows:
            next_row_group = row_group_index + 1
            next_offset = 0
        else:
            next_row_group = row_group_index
        complete = next_row_group == actual_group_count
        next_state: dict[str, Any] = {
            "snapshot_revision": revision,
            "data_path": self.data_path,
            "data_url": data_url,
            "file_size": file_size,
            "row_group_index": next_row_group,
            "row_offset": next_offset,
            "row_group_count": actual_group_count,
            "row_count": actual_row_count,
            "checked_at": checked_at,
        }
        if complete:
            next_state["completed_snapshot_revision"] = revision
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=actual_row_count,
            # This snapshot predates live arXiv deltas and does not encode every
            # deletion marker. It therefore must not tombstone the shared arXiv
            # artifact namespace on completion.
            authoritative_snapshot=False,
            issues=tuple(issues),
            advance_on_source_issues=True,
        )


def _remote_file_size(client: HttpClient | Any, url: str) -> int:
    response: HttpResponse = client.get(
        url,
        headers={"Accept": "application/vnd.apache.parquet", "Range": "bytes=0-0"},
    )
    return _verify_range_response(response, 0, 0, expected_file_size=None)


def _verify_range_response(
    response: HttpResponse,
    start: int,
    end: int,
    expected_file_size: int | None,
) -> int:
    if response.status != 206:
        raise ValueError(f"range request returned HTTP {response.status}, expected 206")
    content_range = _required_text(response.headers.get("content-range"), "Content-Range")
    match = _CONTENT_RANGE.fullmatch(content_range)
    if match is None:
        raise ValueError("range response has an invalid Content-Range")
    actual_start, actual_end, file_size = (int(value) for value in match.groups())
    if actual_start != start or actual_end != end:
        raise ValueError("range response does not match requested byte bounds")
    if expected_file_size is not None and file_size != expected_file_size:
        raise ValueError("range response file size changed during read")
    if len(response.body) != end - start + 1:
        raise ValueError("range response body length does not match Content-Range")
    return file_size


def _records(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    revision: str,
    data_path: str,
    dataset_license: str,
    row_group_index: int,
) -> tuple[list[SourceRecord], list[SourceIssue]]:
    records: list[SourceRecord] = []
    issues: list[SourceIssue] = []
    seen: set[str] = set()
    for row_index, row in enumerate(rows):
        raw_paper_id = _text(row.get("paper_id"))
        try:
            paper_id = _arxiv_id(raw_paper_id)
            if paper_id in seen:
                raise ValueError("duplicate paper ID within Parquet row group")
            seen.add(paper_id)
            records.append(
                _record(
                    row,
                    paper_id=paper_id,
                    dataset_id=dataset_id,
                    revision=revision,
                    data_path=data_path,
                    dataset_license=dataset_license,
                    row_group_index=row_group_index,
                )
            )
        except (TypeError, ValueError) as error:
            issues.append(
                SourceIssue(
                    source_record_id=raw_paper_id or f"row-group-{row_group_index}:{row_index}",
                    stage="source_normalize",
                    error=f"{type(error).__name__}: {error}",
                    summary={
                        "row_group": row_group_index,
                        "row_index": row_index,
                        "paper_id": raw_paper_id or None,
                    },
                )
            )
    return records, issues


def _record(
    row: Mapping[str, Any],
    *,
    paper_id: str,
    dataset_id: str,
    revision: str,
    data_path: str,
    dataset_license: str,
    row_group_index: int,
) -> SourceRecord:
    canonical_url = _arxiv_url(paper_id)
    supplied_url = _optional_arxiv_url(row.get("arxiv_abs_url"), paper_id)
    if supplied_url is not None:
        canonical_url = supplied_url
    title = _text(row.get("title")) or f"arXiv {paper_id}"
    abstract = _text(row.get("abstract"))
    paper_license = _optional_url(row.get("license"))
    identifiers = [Identifier("arxiv", paper_id)]
    links = [Link(canonical_url, relation="landing_page", locator="arxiv_abs_url", crawl=False)]
    for doi in _dois(row.get("doi")):
        identifiers.append(Identifier("doi", doi))
        links.append(
            Link(
                canonicalize_url(f"https://doi.org/{quote(doi, safe='/.:_-()')}"),
                relation="published_as",
                locator="doi",
                crawl=False,
            )
        )
    if paper_license is not None:
        links.append(Link(paper_license, relation="license", locator="license", crawl=False))
    categories = _text(row.get("categories")).split()
    raw = {
        "dataset": dataset_id,
        "snapshot_revision": revision,
        "data_path": data_path,
        "row_group": row_group_index,
        "dataset_license": dataset_license,
        "paper_license": paper_license,
        "authors": _text(row.get("authors")),
        "categories": categories,
        "primary_category": _text(row.get("primary_category")) or None,
        "doi": _text(row.get("doi")) or None,
        "oai_datestamp": _text(row.get("oai_datestamp")) or None,
        "oai_sets": _string_list(row.get("oai_sets")),
        "latest_version_date": _timestamp(row.get("latest_version_date")),
    }
    return SourceRecord(
        source_record_id=paper_id,
        kind=ArtifactKind.PAPER,
        canonical_url=canonical_url,
        title=title,
        raw=raw,
        text="\n\n".join(value for value in (title, abstract) if value),
        published_at=_timestamp(row.get("first_version_date")),
        modified_at=_text(row.get("oai_datestamp")) or None,
        identifiers=tuple(identifiers),
        links=tuple(links),
    )


def _snapshot_revision(
    metadata: Mapping[str, Any], dataset_id: str, data_path: str, source: str
) -> str:
    returned_id = _text(metadata.get("id"))
    if returned_id and returned_id != dataset_id:
        raise ValueError(f"{source}: metadata dataset ID does not match configuration")
    if metadata.get("private") is True or metadata.get("gated") is True:
        raise ValueError(f"{source}: configured snapshot must remain public and ungated")
    revision = _commit(metadata.get("sha"))
    if revision is None:
        raise ValueError(f"{source}: metadata is missing a valid immutable revision")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, Sequence) or isinstance(siblings, (str, bytes, bytearray)):
        raise ValueError(f"{source}: metadata siblings are invalid")
    paths = {
        _text(item.get("rfilename"))
        for item in siblings
        if isinstance(item, Mapping) and _text(item.get("rfilename"))
    }
    if data_path not in paths:
        raise ValueError(f"{source}: metadata does not declare configured Parquet path")
    return revision


def _resolve_url(dataset_id: str, revision: str, data_path: str) -> str:
    return canonicalize_url(
        "https://huggingface.co/datasets/"
        f"{quote(dataset_id, safe='/')}/resolve/{revision}/{quote(data_path, safe='/')}"
    )


def _arxiv_id(value: str) -> str:
    result = value.removesuffix(".pdf")
    result = re.sub(r"v[1-9]\d*$", "", result, flags=re.IGNORECASE)
    if _ARXIV_ID.fullmatch(result) is None:
        raise ValueError(f"invalid arXiv paper ID {value!r}")
    return result


def _arxiv_url(paper_id: str) -> str:
    return canonicalize_url(f"https://arxiv.org/abs/{quote(paper_id, safe='/._-')}")


def _optional_arxiv_url(value: Any, paper_id: str) -> str | None:
    candidate = _optional_url(value)
    if candidate is None:
        return None
    parts = urlsplit(candidate)
    if parts.hostname not in {"arxiv.org", "www.arxiv.org"}:
        return None
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) != 2 or segments[0] != "abs":
        return None
    try:
        if _arxiv_id(segments[1]) != paper_id:
            return None
    except ValueError:
        return None
    return candidate


def _dois(value: Any) -> tuple[str, ...]:
    found: list[str] = []
    for match in _DOI.finditer(_text(value)):
        doi = match.group(0).rstrip(".,;:").casefold()
        if doi not in found:
            found.append(doi)
    return tuple(found)


def _timestamp(value: Any) -> str | None:
    if isinstance(value, datetime):
        timestamp = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return _isoformat(timestamp)
    return _text(value) or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in (_text(entry) for entry in value) if item]


def _data_path(value: Any) -> str:
    path = _required_text(value, "dataset data path")
    if path.startswith("/") or ".." in path.split("/"):
        raise ValueError("dataset data path must be a relative file path")
    return path


def _web_url(value: Any, field: str) -> str:
    canonical = canonicalize_url(_required_text(value, field))
    parts = urlsplit(canonical)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"{field} must be an absolute HTTP(S) URL")
    return canonical


def _optional_url(value: Any) -> str | None:
    try:
        return _web_url(value, "optional URL")
    except ValueError:
        return None


def _commit(value: Any) -> str | None:
    result = _text(value)
    return result if _COMMIT.fullmatch(result) else None


def _nonnegative_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} checkpoint must be an integer") from None
    if result < 0:
        raise ValueError(f"{field} checkpoint must be non-negative")
    return result


def _required_text(value: Any, field: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
