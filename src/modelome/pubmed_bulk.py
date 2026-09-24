from __future__ import annotations

import gzip
import hashlib
import hmac
import io
import os
import re
import socket
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from html.entities import html5 as _HTML_ENTITIES
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from modelome.bulk import BulkShardPlan
from modelome.lake import (
    LakeRecord,
    ParquetLandingZone,
    ShardApplicationOrder,
    ShardReceipt,
    canonical_control_sha256,
)
from modelome.models import SourceRecord
from modelome.normalize import canonicalize_url, identifier_from_url

_SHARD_RE = re.compile(
    r"^pubmed(?P<cycle>\d{2})n(?P<sequence>\d{4,})\.xml\.gz$"
)
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_ENTITY_RE = re.compile(br"<!ENTITY\s", re.IGNORECASE)
_ENTITY_REFERENCE_RE = re.compile(br"&(?P<name>[A-Za-z][A-Za-z0-9_.-]*);")
_XML_PREDEFINED_ENTITIES = frozenset({b"amp", b"apos", b"gt", b"lt", b"quot"})
_MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}
_DEFAULT_REPOSITORY_HOSTS = frozenset(
    {
        "bitbucket.org",
        "codeberg.org",
        "github.com",
        "gitlab.com",
        "huggingface.co",
        "sourceforge.net",
    }
)
_RETRYABLE_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
_SAFE_CROSS_ORIGIN_HEADERS = frozenset(
    {"accept", "accept-encoding", "accept-language", "user-agent"}
)


class StreamingResponse(Protocol):
    status: int
    headers: Mapping[str, str]
    url: str

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> StreamingResponse: ...

    def __exit__(self, *args: Any) -> None: ...


class StreamingTransport(Protocol):
    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> AbstractContextManager[StreamingResponse]: ...


@dataclass(frozen=True, slots=True)
class PubMedBulkReceipt:
    source: str
    dataset: str
    release: str
    shard: str
    manifest_kind: str
    sequence: int
    published_md5: str
    control_sha256: str
    upstream_sha256: str
    compressed_bytes: int
    uncompressed_bytes: int | None
    row_count: int
    upsert_count: int | None
    delete_count: int | None
    path: Path
    already_committed: bool


@dataclass(frozen=True, slots=True)
class _ControlShard:
    url: str
    filename: str
    manifest_kind: str
    cycle: str
    production_year: int
    sequence: int
    published_md5: str
    control_sha256: str

    @property
    def release(self) -> str:
        if self.manifest_kind == "baseline":
            return f"{self.production_year}-baseline"
        return f"{self.production_year}-update-{self.sequence:08d}"

    @property
    def lake_order(self) -> ShardApplicationOrder:
        if self.manifest_kind == "baseline":
            return ShardApplicationOrder.snapshot(self.sequence - 1)
        return ShardApplicationOrder.single()

    def bulk_context(self, operation_index: int) -> dict[str, Any]:
        return {
            "application_order": self.sequence,
            "filename": self.filename,
            "manifest_kind": self.manifest_kind,
            "operation_index": operation_index,
            "production_cycle": self.cycle,
            "production_year": self.production_year,
            "sequence": self.sequence,
        }


@dataclass(slots=True)
class _ParseStats:
    uncompressed_bytes: int = 0
    row_count: int = 0
    upsert_count: int = 0
    delete_count: int = 0


@dataclass(frozen=True, slots=True)
class _Download:
    path: Path
    md5: str
    sha256: str
    byte_count: int


class HttpsStreamingTransport:
    """Open a retryable HTTPS response without buffering its body."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        attempts: int = 4,
        user_agent: str = "modelome/0.1",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("transport timeout must be positive")
        if attempts < 1:
            raise ValueError("transport attempts must be positive")
        self.timeout = float(timeout)
        self.attempts = int(attempts)
        self.user_agent = user_agent.strip() or "modelome/0.1"
        self._sleep = sleep

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> AbstractContextManager[StreamingResponse]:
        opener = build_opener(_ValidatedRedirectHandler(redirect_validator))
        request_headers = {"User-Agent": self.user_agent, **dict(headers)}
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            request = Request(url, headers=request_headers, method="GET")
            try:
                return opener.open(request, timeout=self.timeout)  # type: ignore[return-value]
            except HTTPError as error:
                last_error = error
                if error.code not in _RETRYABLE_HTTP:
                    break
            except (TimeoutError, URLError) as error:
                last_error = error
            if attempt + 1 < self.attempts:
                self._sleep(min(2**attempt, 30))
        raise RuntimeError(f"streaming GET failed: {last_error}") from last_error


class _ValidatedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, validator: Callable[[str], None]) -> None:
        super().__init__()
        self.validator = validator

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> Request | None:
        self.validator(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None or _origin(req.full_url) == _origin(newurl):
            return redirected
        for name, _value in redirected.header_items():
            if name.casefold() not in _SAFE_CROSS_ORIGIN_HEADERS:
                redirected.remove_header(name)
        return redirected


class PubMedBulkLoader:
    """Verify, stream-parse, and atomically land one PubMed XML shard."""

    def __init__(
        self,
        lake: ParquetLandingZone,
        *,
        transport: StreamingTransport | None = None,
        source: str = "pubmed",
        dataset: str = "citations",
        allowed_hosts: Sequence[str] = ("ftp.ncbi.nlm.nih.gov",),
        repository_hosts: Sequence[str] = tuple(sorted(_DEFAULT_REPOSITORY_HOSTS)),
        max_compressed_bytes: int = 256 * 1024 * 1024,
        max_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024,
        max_records: int = 250_000,
        max_text_chars_per_record: int = 5_000_000,
        max_urls_per_record: int = 1_000,
        max_list_items_per_record: int = 25_000,
        download_chunk_bytes: int = 1024 * 1024,
        parquet_batch_rows: int = 10_000,
    ) -> None:
        if not isinstance(lake, ParquetLandingZone):
            raise TypeError("lake must be a ParquetLandingZone")
        self.lake = lake
        self.transport = transport or HttpsStreamingTransport()
        self.source = _required_text(source, "source")
        self.dataset = _required_text(dataset, "dataset")
        self.allowed_hosts = _host_set(allowed_hosts, "allowed_hosts")
        self.repository_hosts = _host_set(repository_hosts, "repository_hosts")
        self.max_compressed_bytes = _positive(
            max_compressed_bytes, "max_compressed_bytes"
        )
        self.max_uncompressed_bytes = _positive(
            max_uncompressed_bytes, "max_uncompressed_bytes"
        )
        self.max_records = _positive(max_records, "max_records")
        self.max_text_chars_per_record = _positive(
            max_text_chars_per_record, "max_text_chars_per_record"
        )
        self.max_urls_per_record = _positive(
            max_urls_per_record, "max_urls_per_record"
        )
        self.max_list_items_per_record = _positive(
            max_list_items_per_record, "max_list_items_per_record"
        )
        self.download_chunk_bytes = _positive(
            download_chunk_bytes, "download_chunk_bytes"
        )
        self.parquet_batch_rows = _positive(parquet_batch_rows, "parquet_batch_rows")

    def load(self, control_record: SourceRecord) -> PubMedBulkReceipt:
        shard = _control_shard(control_record)
        self._validate_url(shard.url, expected_filename=shard.filename)
        cached = self.lake.lookup_committed_shard(
            source=self.source,
            dataset=self.dataset,
            release=shard.release,
            shard=shard.filename,
            control_sha256=shard.control_sha256,
            upstream_url=shard.url,
            application_order=shard.lake_order,
        )
        if cached is not None:
            return _bulk_receipt(cached, shard, download=None, stats=None)
        self.lake.initialize()
        with tempfile.TemporaryDirectory(
            prefix="pubmed-download-",
            dir=self.lake.staging_root,
        ) as temporary:
            path = Path(temporary) / "payload.xml.gz"
            download = self._download(shard, path)
            stats = _ParseStats()
            receipt = self.lake.commit_shard(
                source=self.source,
                dataset=self.dataset,
                release=shard.release,
                shard=shard.filename,
                control_sha256=shard.control_sha256,
                upstream_sha256=download.sha256,
                upstream_url=shard.url,
                upstream_bytes=download.byte_count,
                application_order=shard.lake_order,
                records=self._records(download.path, shard, stats),
                batch_rows=self.parquet_batch_rows,
            )
        parsed = not receipt.already_committed
        return _bulk_receipt(
            receipt,
            shard,
            download,
            stats if parsed else None,
        )

    def plan_shard(self, control_record: SourceRecord) -> BulkShardPlan | None:
        """Describe one official payload control without transferring its XML."""

        if "manifest_kind" not in control_record.raw:
            return None
        shard = _control_shard(control_record)
        return BulkShardPlan(
            source=self.source,
            dataset=self.dataset,
            release=shard.release,
            shard=shard.filename,
            control_sha256=shard.control_sha256,
        )

    def shard_order(self, control_record: SourceRecord) -> tuple[Any, ...]:
        """Put a complete baseline before its ordered daily update files."""

        shard = _control_shard(control_record)
        return (
            shard.production_year,
            0 if shard.manifest_kind == "baseline" else 1,
            shard.sequence,
            shard.filename,
        )

    def _download(self, shard: _ControlShard, destination: Path) -> _Download:
        md5 = hashlib.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        byte_count = 0
        headers = {
            "Accept": "application/gzip,application/octet-stream",
            "Accept-Encoding": "identity",
        }
        with self.transport.open(
            shard.url,
            headers=headers,
            redirect_validator=lambda url: self._validate_url(
                url, expected_filename=shard.filename
            ),
        ) as response:
            status = int(getattr(response, "status", 0))
            if status != 200:
                raise ValueError(f"PubMed payload returned HTTP {status}")
            final_url = str(getattr(response, "url", ""))
            self._validate_url(final_url, expected_filename=shard.filename)
            response_headers = {
                str(key).casefold(): str(value)
                for key, value in response.headers.items()
            }
            encoding = response_headers.get("content-encoding", "").strip().casefold()
            if encoding not in {"", "identity"}:
                raise ValueError(
                    "PubMed payload must be transferred without HTTP content encoding"
                )
            content_length = _content_length(response_headers.get("content-length"))
            if content_length is not None and content_length > self.max_compressed_bytes:
                raise ValueError(
                    f"PubMed compressed payload exceeds {self.max_compressed_bytes} bytes"
                )

            with destination.open("xb") as output:
                while True:
                    block = response.read(self.download_chunk_bytes)
                    if not isinstance(block, bytes):
                        raise TypeError("streaming transport returned a non-bytes block")
                    if not block:
                        break
                    byte_count += len(block)
                    if byte_count > self.max_compressed_bytes:
                        raise ValueError(
                            "PubMed compressed payload exceeds "
                            f"{self.max_compressed_bytes} bytes"
                        )
                    md5.update(block)
                    sha256.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())

        if content_length is not None and content_length != byte_count:
            raise ValueError(
                f"PubMed payload length mismatch: expected {content_length}, "
                f"received {byte_count}"
            )
        actual_md5 = md5.hexdigest()
        if not hmac.compare_digest(actual_md5, shard.published_md5):
            raise ValueError(
                f"PubMed MD5 mismatch for {shard.filename}: "
                f"expected {shard.published_md5}, received {actual_md5}"
            )
        return _Download(
            path=destination,
            md5=actual_md5,
            sha256=sha256.hexdigest(),
            byte_count=byte_count,
        )

    def _records(
        self,
        path: Path,
        shard: _ControlShard,
        stats: _ParseStats,
    ) -> Iterator[LakeRecord]:
        with (
            path.open("rb") as compressed,
            gzip.GzipFile(fileobj=compressed, mode="rb") as decompressed,
        ):
            bounded = _BoundedXmlReader(
                decompressed,
                limit=self.max_uncompressed_bytes,
            )
            try:
                yield from self._parse(bounded, shard, stats)
            except (EOFError, gzip.BadGzipFile) as error:
                raise ValueError(
                    f"invalid gzip stream for {shard.filename}: {error}"
                ) from error
            finally:
                stats.uncompressed_bytes = bounded.byte_count

    def _parse(
        self,
        stream: _BoundedXmlReader,
        shard: _ControlShard,
        stats: _ParseStats,
    ) -> Iterator[LakeRecord]:
        active_tag: str | None = None
        active_depth = 0
        active_chars = 0
        root: ET.Element | None = None
        try:
            events = ET.iterparse(stream, events=("start", "end"))
            for event, element in events:
                tag = _local_name(element.tag)
                if root is None and event == "start":
                    root = element
                    if tag != "PubmedArticleSet":
                        raise ValueError(
                            f"PubMed XML root must be PubmedArticleSet, found {tag!r}"
                        )

                if event == "start":
                    if active_tag is None and tag in {
                        "PubmedArticle",
                        "PubmedBookArticle",
                        "DeleteCitation",
                    }:
                        active_tag = tag
                        active_depth = 1
                        active_chars = 0
                    elif active_tag is not None:
                        active_depth += 1
                    continue

                if active_tag is None:
                    continue
                active_chars += len(element.text or "") + len(element.tail or "")
                if active_chars > self.max_text_chars_per_record:
                    raise ValueError(
                        "PubMed record text exceeds "
                        f"{self.max_text_chars_per_record} characters"
                    )
                active_depth -= 1
                if active_depth:
                    continue
                if tag != active_tag:
                    raise ValueError("malformed PubMed XML record nesting")

                if active_tag == "DeleteCitation":
                    if shard.manifest_kind != "update":
                        raise ValueError("DeleteCitation is not valid in a baseline shard")
                    pmids = [
                        _pmid_value(item)
                        for item in _descendants(element, "PMID")
                    ]
                    if not pmids:
                        raise ValueError("DeleteCitation contains no PMID")
                    _limit_items(
                        pmids,
                        self.max_list_items_per_record,
                        "deleted PMIDs",
                    )
                    for pmid, version in pmids:
                        stats.row_count += 1
                        stats.delete_count += 1
                        self._check_record_count(stats.row_count)
                        payload = {
                            "bulk": shard.bulk_context(stats.row_count),
                            "pmid": pmid,
                        }
                        if version:
                            payload["pmid_version"] = version
                        yield LakeRecord(
                            source_record_id=f"pmid:{pmid}",
                            payload=payload,
                            operation="delete",
                        )
                else:
                    stats.row_count += 1
                    stats.upsert_count += 1
                    self._check_record_count(stats.row_count)
                    payload = _article_payload(
                        element,
                        shard=shard,
                        operation_index=stats.row_count,
                        repository_hosts=self.repository_hosts,
                        max_urls=self.max_urls_per_record,
                        max_items=self.max_list_items_per_record,
                    )
                    yield LakeRecord(
                        source_record_id=f"pmid:{payload['pmid']}",
                        payload=payload,
                    )

                element.clear()
                if root is not None:
                    root.clear()
                active_tag = None
                active_chars = 0
        except ET.ParseError as error:
            raise ValueError(f"invalid PubMed XML: {error}") from error

        if root is None:
            raise ValueError("PubMed XML is empty")
        if active_tag is not None:
            raise ValueError(f"unterminated PubMed XML record {active_tag}")

    def _check_record_count(self, count: int) -> None:
        if count > self.max_records:
            raise ValueError(f"PubMed shard exceeds {self.max_records} records")

    def _validate_url(self, url: str, *, expected_filename: str) -> None:
        parts = urlsplit(url)
        host = (parts.hostname or "").casefold().rstrip(".")
        if (
            parts.scheme.casefold() != "https"
            or host not in self.allowed_hosts
            or parts.username is not None
            or parts.password is not None
            or parts.port not in {None, 443}
            or parts.query
            or parts.fragment
        ):
            raise ValueError("PubMed payload URL is outside the allowed HTTPS origins")
        decoded_path = unquote(parts.path)
        segments = decoded_path.split("/")
        if (
            decoded_path != parts.path
            or "\\" in decoded_path
            or "\x00" in decoded_path
            or any(segment in {".", ".."} for segment in segments)
        ):
            raise ValueError("PubMed payload URL contains an unsafe path")
        filename = PurePosixPath(decoded_path).name
        if filename != expected_filename or _SHARD_RE.fullmatch(filename) is None:
            raise ValueError("PubMed payload URL does not match the manifest filename")


class _BoundedXmlReader(io.RawIOBase):
    def __init__(self, stream: BinaryIO, *, limit: int) -> None:
        self.stream = stream
        self.limit = limit
        self.byte_count = 0
        self._scan_tail = b""
        self._entity_tail = b""
        self._output = bytearray()
        self._eof = False

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        requested = 64 * 1024 if size < 0 else size
        while len(self._output) < requested and not self._eof:
            self._fill()
        result = bytes(self._output[:requested])
        del self._output[:requested]
        return result

    def _fill(self) -> None:
        remaining = self.limit - self.byte_count
        data = self.stream.read(min(64 * 1024, remaining + 1))
        if not isinstance(data, bytes):
            raise TypeError("gzip reader returned a non-bytes block")
        self.byte_count += len(data)
        if self.byte_count > self.limit:
            raise ValueError(
                f"PubMed uncompressed payload exceeds {self.limit} bytes"
            )
        scan = self._scan_tail + data
        if _ENTITY_RE.search(scan):
            raise ValueError("PubMed XML must not declare custom entities")
        self._scan_tail = scan[-16:]

        if not data:
            self._eof = True
            self._output.extend(_normalize_entity_references(self._entity_tail))
            self._entity_tail = b""
            return
        pending = self._entity_tail + data
        last_ampersand = pending.rfind(b"&")
        if last_ampersand >= 0 and b";" not in pending[last_ampersand:]:
            tail = pending[last_ampersand:]
            if len(tail) > 256:
                raise ValueError("PubMed XML entity reference exceeds 256 bytes")
            pending, self._entity_tail = pending[:last_ampersand], tail
        else:
            self._entity_tail = b""
        self._output.extend(_normalize_entity_references(pending))


def _control_shard(record: SourceRecord) -> _ControlShard:
    if not isinstance(record, SourceRecord):
        raise TypeError("control_record must be a SourceRecord")
    raw = record.raw
    if not isinstance(raw, Mapping):
        raise TypeError("PubMed control record raw value must be a mapping")
    filename = _required_text(raw.get("filename"), "PubMed filename")
    match = _SHARD_RE.fullmatch(filename)
    if match is None:
        raise ValueError("PubMed control record has an invalid filename")
    if record.source_record_id != f"pubmed:{filename}":
        raise ValueError("PubMed control source_record_id does not match its filename")
    url = _required_text(raw.get("url"), "PubMed payload URL")
    if canonicalize_url(url) != canonicalize_url(record.canonical_url):
        raise ValueError("PubMed control canonical URL does not match its payload URL")
    manifest_kind = _required_text(
        raw.get("manifest_kind"), "PubMed manifest kind"
    ).casefold()
    if manifest_kind not in {"baseline", "update"}:
        raise ValueError("PubMed manifest kind must be 'baseline' or 'update'")
    cycle = _required_text(raw.get("production_cycle"), "PubMed production cycle")
    if cycle != match.group("cycle"):
        raise ValueError("PubMed production cycle does not match its filename")
    sequence = _integer(raw.get("sequence"), "PubMed sequence", minimum=1)
    if sequence != int(match.group("sequence")):
        raise ValueError("PubMed sequence does not match its filename")
    application_order = _integer(
        raw.get("application_order"), "PubMed application order", minimum=1
    )
    if application_order != sequence:
        raise ValueError("PubMed application order must equal its shard sequence")
    production_year = _integer(
        raw.get("production_year"), "PubMed production year", minimum=1900
    )
    if production_year > 9999:
        raise ValueError("PubMed production year is invalid")
    checksum = raw.get("checksum")
    if not isinstance(checksum, Mapping):
        raise ValueError("PubMed control record has no checksum object")
    if _required_text(checksum.get("algorithm"), "checksum algorithm").casefold() != "md5":
        raise ValueError("PubMed control checksum algorithm must be MD5")
    published_md5 = _required_text(checksum.get("value"), "published MD5").casefold()
    if _MD5_RE.fullmatch(published_md5) is None:
        raise ValueError("PubMed control record has an invalid published MD5")
    return _ControlShard(
        url=url,
        filename=filename,
        manifest_kind=manifest_kind,
        cycle=cycle,
        production_year=production_year,
        sequence=sequence,
        published_md5=published_md5,
        control_sha256=canonical_control_sha256(
            {
                "canonical_url": canonicalize_url(record.canonical_url),
                "raw": dict(raw),
                "source_record_id": record.source_record_id,
            }
        ),
    )


def _normalize_entity_references(value: bytes) -> bytes:
    """Resolve safe standard entities without loading PubMed's external DTD.

    The PubMed DTD imports the W3C HTML/MathML entity sets. ElementTree does not
    retrieve external DTDs, so recognized names are converted to numeric Unicode
    references. Unknown names remain literal evidence text instead of causing an
    entire bulk shard to be lost. Inline entity declarations are rejected by the
    bounded reader before this function is called.
    """

    def replacement(match: re.Match[bytes]) -> bytes:
        raw_name = match.group("name")
        if raw_name in _XML_PREDEFINED_ENTITIES:
            return match.group(0)
        name = raw_name.decode("ascii")
        resolved = _HTML_ENTITIES.get(f"{name};")
        if resolved is None:
            return b"&#38;" + raw_name + b";"
        return "".join(f"&#{ord(character)};" for character in resolved).encode()

    return _ENTITY_REFERENCE_RE.sub(replacement, value)


def _article_payload(
    element: ET.Element,
    *,
    shard: _ControlShard,
    operation_index: int,
    repository_hosts: frozenset[str],
    max_urls: int,
    max_items: int,
) -> dict[str, Any]:
    pmid_element = _first_descendant(element, "PMID")
    if pmid_element is None:
        raise ValueError("PubMed article contains no PMID")
    pmid, pmid_version = _pmid_value(pmid_element)
    title_element = _first_descendant(element, "ArticleTitle")
    if title_element is None:
        title_element = _first_descendant(element, "BookTitle")
    title = _element_text(title_element)

    abstract_sections = []
    for item in _descendants(element, "AbstractText"):
        section = {"text": _element_text(item)}
        if label := _clean_text(item.attrib.get("Label")):
            section["label"] = label
        if category := _clean_text(item.attrib.get("NlmCategory")):
            section["category"] = category
        if section["text"] or len(section) > 1:
            abstract_sections.append(section)
    _limit_items(abstract_sections, max_items, "abstract sections")
    abstract = "\n\n".join(
        str(section["text"]) for section in abstract_sections if section["text"]
    )

    article_ids: list[dict[str, str]] = []
    for item in _descendants(element, "ArticleId"):
        value = _element_text(item)
        namespace = _clean_text(item.attrib.get("IdType")).casefold()
        if namespace == "doi":
            value = _normalize_doi(value)
        elif namespace == "pmc":
            value = value.upper()
        if value and namespace:
            article_ids.append({"namespace": namespace, "value": value})
    for item in _descendants(element, "ELocationID"):
        value = _element_text(item)
        namespace = _clean_text(item.attrib.get("EIdType")).casefold()
        if namespace == "doi":
            value = _normalize_doi(value)
        candidate = {"namespace": namespace, "value": value}
        if value and namespace and candidate not in article_ids:
            article_ids.append(candidate)
    article_ids = _unique_mappings(article_ids)
    _limit_items(article_ids, max_items, "article identifiers")
    doi = _first_identifier(article_ids, "doi")
    pmc = _first_identifier(article_ids, "pmc")

    authors = _authors(element, max_items)
    cited_urls = _urls(element, max_urls)
    repository_urls = [
        url
        for url in cited_urls
        if _is_repository_url(url, repository_hosts)
    ]
    publication = _publication_metadata(element, max_items)

    payload: dict[str, Any] = {
        "abstract": abstract,
        "abstract_sections": abstract_sections,
        "article_kind": _local_name(element.tag),
        "authors": authors,
        "bulk": shard.bulk_context(operation_index),
        "cited_urls": cited_urls,
        "identifiers": article_ids,
        "pmid": pmid,
        "publication": publication,
        "repository_urls": repository_urls,
        "title": title,
    }
    if pmid_version:
        payload["pmid_version"] = pmid_version
    if doi:
        payload["doi"] = _normalize_doi(doi)
    if pmc:
        payload["pmc"] = pmc.upper()
    return payload


def _publication_metadata(element: ET.Element, max_items: int) -> dict[str, Any]:
    citation = _first_descendant(element, "MedlineCitation")
    article = _first_descendant(element, "Article")
    journal = _first_descendant(element, "Journal")
    journal_issue = _first_descendant(element, "JournalIssue")
    journal_info = _first_descendant(element, "MedlineJournalInfo")
    publication: dict[str, Any] = {}
    if citation is not None:
        publication.update(
            _nonempty(
                {
                    "citation_status": citation.attrib.get("Status"),
                    "indexing_method": citation.attrib.get("IndexingMethod"),
                    "owner": citation.attrib.get("Owner"),
                }
            )
        )
    if article is not None:
        publication.update(_nonempty({"publication_model": article.attrib.get("PubModel")}))
    if journal_issue is not None:
        publication.update(
            _nonempty(
                {
                    "cited_medium": journal_issue.attrib.get("CitedMedium"),
                    "issue": _child_text(journal_issue, "Issue"),
                    "volume": _child_text(journal_issue, "Volume"),
                }
            )
        )
        if date := _date_metadata(_first_child(journal_issue, "PubDate")):
            publication["publication_date"] = date
    if journal is not None:
        issn = _first_child(journal, "ISSN")
        publication.update(
            _nonempty(
                {
                    "issn": _element_text(issn),
                    "issn_type": issn.attrib.get("IssnType") if issn is not None else None,
                    "journal": _child_text(journal, "Title"),
                    "journal_abbreviation": _child_text(journal, "ISOAbbreviation"),
                }
            )
        )
    if journal_info is not None:
        publication.update(
            _nonempty(
                {
                    "country": _child_text(journal_info, "Country"),
                    "issn_linking": _child_text(journal_info, "ISSNLinking"),
                    "medline_ta": _child_text(journal_info, "MedlineTA"),
                    "nlm_unique_id": _child_text(journal_info, "NlmUniqueID"),
                }
            )
        )

    publication.update(
        _nonempty(
            {
                "completed_date": _date_metadata(
                    _first_descendant(element, "DateCompleted")
                ),
                "pagination": _element_text(_first_descendant(element, "MedlinePgn")),
                "publication_status": _element_text(
                    _first_descendant(element, "PublicationStatus")
                ),
                "revised_date": _date_metadata(
                    _first_descendant(element, "DateRevised")
                ),
            }
        )
    )
    languages = _unique_texts(
        _element_text(item) for item in _descendants(element, "Language")
    )
    publication_types = _unique_texts(
        _element_text(item) for item in _descendants(element, "PublicationType")
    )
    history = []
    for item in _descendants(element, "PubMedPubDate"):
        date = _date_metadata(item)
        if date:
            date["status"] = _clean_text(item.attrib.get("PubStatus"))
            history.append(date)
    article_dates = [
        date
        for item in _descendants(element, "ArticleDate")
        if (date := _date_metadata(item))
    ]
    for value, label in (
        (article_dates, "article dates"),
        (history, "publication history"),
        (languages, "languages"),
        (publication_types, "publication types"),
    ):
        _limit_items(value, max_items, label)
    if article_dates:
        publication["article_dates"] = article_dates
    if history:
        publication["history"] = history
    if languages:
        publication["languages"] = languages
    if publication_types:
        publication["publication_types"] = publication_types
    return publication


def _authors(element: ET.Element, max_items: int) -> list[dict[str, Any]]:
    result = []
    for author in _descendants(element, "Author"):
        value: dict[str, Any] = _nonempty(
            {
                "collective_name": _child_text(author, "CollectiveName"),
                "fore_name": _child_text(author, "ForeName"),
                "initials": _child_text(author, "Initials"),
                "last_name": _child_text(author, "LastName"),
                "suffix": _child_text(author, "Suffix"),
                "valid": author.attrib.get("ValidYN"),
            }
        )
        affiliations = _unique_texts(
            _element_text(item) for item in _descendants(author, "Affiliation")
        )
        _limit_items(affiliations, max_items, "author affiliations")
        identifiers = []
        for item in _descendants(author, "Identifier"):
            identifier = _nonempty(
                {
                    "namespace": item.attrib.get("Source"),
                    "value": _element_text(item),
                }
            )
            if identifier:
                identifiers.append(identifier)
        _limit_items(identifiers, max_items, "author identifiers")
        if affiliations:
            value["affiliations"] = affiliations
        if identifiers:
            value["identifiers"] = _unique_mappings(identifiers)
        if value:
            result.append(value)
        _limit_items(result, max_items, "authors")
    return result


def _urls(element: ET.Element, max_items: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in element.iter():
        values = [item.text or "", item.tail or "", *item.attrib.values()]
        for value in values:
            for match in _URL_RE.finditer(value):
                url = canonicalize_url(match.group(0))
                if url in seen:
                    continue
                seen.add(url)
                result.append(url)
                if len(result) > max_items:
                    raise ValueError(
                        f"PubMed record contains more than {max_items} cited URLs"
                    )
    return result


def _is_repository_url(url: str, repository_hosts: frozenset[str]) -> bool:
    identifier = identifier_from_url(url)
    if identifier is not None and identifier.namespace in {
        "github:repository",
        "huggingface:model",
    }:
        return True
    host = (urlsplit(url).hostname or "").casefold().removeprefix("www.")
    return host in repository_hosts


def _pmid_value(element: ET.Element) -> tuple[str, str | None]:
    pmid = _element_text(element)
    if not pmid or not pmid.isascii() or not pmid.isdecimal() or int(pmid) < 1:
        raise ValueError(f"invalid PubMed PMID {pmid!r}")
    version = _clean_text(element.attrib.get("Version")) or None
    if version is not None and (
        not version.isascii() or not version.isdecimal() or int(version) < 1
    ):
        raise ValueError(f"invalid PubMed PMID version {version!r}")
    return str(int(pmid)), str(int(version)) if version is not None else None


def _date_metadata(element: ET.Element | None) -> dict[str, str]:
    if element is None:
        return {}
    result = _nonempty(
        {
            "day": _child_text(element, "Day"),
            "medline_date": _child_text(element, "MedlineDate"),
            "minute": _child_text(element, "Minute"),
            "month": _child_text(element, "Month"),
            "season": _child_text(element, "Season"),
            "second": _child_text(element, "Second"),
            "year": _child_text(element, "Year"),
        }
    )
    if hour := _child_text(element, "Hour"):
        result["hour"] = hour
    display = _date_display(result)
    if display:
        result["display"] = display
    return result


def _date_display(value: Mapping[str, str]) -> str:
    if medline := value.get("medline_date"):
        return medline
    year = value.get("year", "")
    if not year:
        return ""
    month = value.get("month", "")
    normalized_month = _MONTHS.get(month[:3].casefold(), month.zfill(2) if month.isdigit() else "")
    day = value.get("day", "")
    if normalized_month and day.isdigit():
        return f"{year}-{normalized_month}-{day.zfill(2)}"
    if normalized_month:
        return f"{year}-{normalized_month}"
    return year


def _first_identifier(identifiers: Sequence[Mapping[str, str]], namespace: str) -> str:
    for item in identifiers:
        if item.get("namespace", "").casefold() == namespace:
            return item.get("value", "")
    return ""


def _normalize_doi(value: str) -> str:
    result = unquote(value).strip()
    lowered = result.casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        if lowered.startswith(prefix):
            result = result[len(prefix) :]
            break
    return result.strip().casefold()


def _descendants(element: ET.Element, name: str) -> Iterator[ET.Element]:
    for item in element.iter():
        if _local_name(item.tag) == name:
            yield item


def _first_descendant(element: ET.Element, name: str) -> ET.Element | None:
    return next(_descendants(element, name), None)


def _first_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if _local_name(child.tag) == name:
            return child
    return None


def _child_text(element: ET.Element, name: str) -> str:
    return _element_text(_first_child(element, name))


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return _clean_text("".join(element.itertext()))


def _clean_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", value).strip() if isinstance(value, str) else ""


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


def _nonempty(values: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, str):
            if cleaned := _clean_text(value):
                result[key] = cleaned
        elif value is not None and value != {} and value != []:
            result[key] = value
    return result


def _unique_texts(values: Iterator[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _unique_mappings(values: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    result = []
    for value in values:
        item = {str(key): str(child) for key, child in value.items() if str(child)}
        key = tuple(sorted(item.items()))
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _limit_items(values: Sequence[Any], maximum: int, label: str) -> None:
    if len(values) > maximum:
        raise ValueError(f"PubMed record contains more than {maximum} {label}")


def _bulk_receipt(
    receipt: ShardReceipt,
    shard: _ControlShard,
    download: _Download | None,
    stats: _ParseStats | None,
) -> PubMedBulkReceipt:
    compressed_bytes = (
        download.byte_count if download is not None else receipt.upstream_bytes
    )
    if compressed_bytes is None:
        raise ValueError("committed PubMed shard has no upstream byte count")
    return PubMedBulkReceipt(
        source=receipt.source,
        dataset=receipt.dataset,
        release=receipt.release,
        shard=receipt.shard,
        manifest_kind=shard.manifest_kind,
        sequence=shard.sequence,
        published_md5=download.md5 if download is not None else shard.published_md5,
        control_sha256=receipt.control_sha256,
        upstream_sha256=receipt.upstream_sha256,
        compressed_bytes=compressed_bytes,
        uncompressed_bytes=stats.uncompressed_bytes if stats is not None else None,
        row_count=receipt.row_count,
        upsert_count=stats.upsert_count if stats is not None else None,
        delete_count=stats.delete_count if stats is not None else None,
        path=receipt.path,
        already_committed=receipt.already_committed,
    )


def _host_set(values: Sequence[str], label: str) -> frozenset[str]:
    result = frozenset(
        value.strip().casefold().rstrip(".")
        for value in values
        if isinstance(value, str) and value.strip()
    )
    if not result:
        raise ValueError(f"{label} must contain at least one hostname")
    for host in result:
        if (
            "/" in host
            or ":" in host
            or host == "localhost"
            or host.endswith(".localhost")
        ):
            raise ValueError(f"{label} contains an invalid hostname")
        try:
            address = socket.inet_pton(socket.AF_INET, host)
        except OSError:
            try:
                address = socket.inet_pton(socket.AF_INET6, host)
            except OSError:
                continue
        if address:
            raise ValueError(f"{label} must contain DNS hostnames, not IP addresses")
    return result


def _content_length(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    value = value.strip()
    if not value.isascii() or not value.isdecimal():
        raise ValueError("PubMed payload has an invalid Content-Length")
    result = int(value)
    if result < 0:
        raise ValueError("PubMed payload has an invalid Content-Length")
    return result


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be an integer") from None
    if result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _positive(value: Any, label: str) -> int:
    return _integer(value, label, minimum=1)


def _required_text(value: Any, label: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    return (
        parts.scheme.casefold(),
        (parts.hostname or "").casefold(),
        parts.port or (443 if parts.scheme.casefold() == "https" else None),
    )


__all__ = [
    "HttpsStreamingTransport",
    "PubMedBulkLoader",
    "PubMedBulkReceipt",
    "StreamingResponse",
    "StreamingTransport",
]
