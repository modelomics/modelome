from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient, HttpFailure, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash, extract_urls, identifier_from_url

Clock = Callable[[], datetime]

_OAI_NAMESPACE = "http://www.openarchives.org/OAI/2.0/"
_OAI_IDENTIFIER_PREFIX = "oai:pubmedcentral.nih.gov:"
_PMCID_RE = re.compile(r"PMC([1-9]\d*)", re.IGNORECASE)
_NUMERIC_ID_RE = re.compile(r"[1-9]\d*")
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_ORCID_RE = re.compile(r"(?:https?://orcid\.org/)?(\d{4}-\d{4}-\d{4}-[\dX]{4})", re.I)
_PMC_FTP_URL_RE = re.compile(
    r"ftp://ftp\.ncbi\.nlm\.nih\.gov/pub/pmc/[^\s<>\"']+", re.IGNORECASE
)
_MODEL_RESOURCE_CONTEXT_RE = re.compile(
    r"\b(?:trained|pre[- ]?trained|downloadable|released)\s+"
    r"(?:deep[- ]learning\s+)?models?\b|"
    r"\b(?:model\s+)?(?:weights?|checkpoints?|parameters?)\b|"
    r"\bweights?\s+for\s+(?:the\s+)?(?:trained\s+)?models?\b",
    re.IGNORECASE,
)
_DATE_TYPES = ("epub", "electronic", "ppub", "print", "collection")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class PmcRepositoryIdentity:
    """Harvest boundaries and deletion semantics declared by PMC's OAI service."""

    repository_name: str
    base_url: str
    protocol_version: str
    earliest_datestamp: str
    deleted_record: str
    granularity: str


class PmcSourceAdapter:
    """Enumerate reusable PMC full text without a subject or venue filter.

    ``pmc-open`` is an upstream rights collection, not a scientific-content
    selection. The metadata format is fixed to full JATS so a scan cannot
    silently degrade into citation-only records.
    """

    metadata_prefix = "pmc"
    set_spec = "pmc-open"

    def __init__(
        self,
        *,
        name: str = "pmc",
        url: str = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/",
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        initial_lookback_days: int = 7,
        overlap_days: int = 2,
        max_response_bytes: int = 64 * 1024 * 1024,
        max_record_bytes: int = 32 * 1024 * 1024,
        max_records_per_page: int = 100,
        max_elements_per_record: int = 250_000,
        max_text_chars_per_record: int = 10_000_000,
        max_authors_per_record: int = 10_000,
        max_external_urls_per_record: int = 100_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name)
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.initial_lookback_days = _nonnegative_config(
            initial_lookback_days, "initial_lookback_days"
        )
        self.overlap_days = _nonnegative_config(overlap_days, "overlap_days")
        self.max_response_bytes = _positive_config(max_response_bytes, "max_response_bytes")
        self.max_record_bytes = _positive_config(max_record_bytes, "max_record_bytes")
        self.max_records_per_page = _positive_config(
            max_records_per_page, "max_records_per_page"
        )
        self.max_elements_per_record = _positive_config(
            max_elements_per_record, "max_elements_per_record"
        )
        self.max_text_chars_per_record = _positive_config(
            max_text_chars_per_record, "max_text_chars_per_record"
        )
        self.max_authors_per_record = _positive_config(
            max_authors_per_record, "max_authors_per_record"
        )
        self.max_external_urls_per_record = _positive_config(
            max_external_urls_per_record, "max_external_urls_per_record"
        )
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "pmc-open-oai-jats-v1",
                "url": canonicalize_url(self.url),
                "artifact_kind": self.artifact_kind.value,
                "metadata_prefix": self.metadata_prefix,
                "set_spec": self.set_spec,
                "initial_lookback_days": self.initial_lookback_days,
                "overlap_days": self.overlap_days,
                "max_response_bytes": self.max_response_bytes,
                "max_record_bytes": self.max_record_bytes,
                "max_records_per_page": self.max_records_per_page,
                "max_elements_per_record": self.max_elements_per_record,
                "max_text_chars_per_record": self.max_text_chars_per_record,
                "max_authors_per_record": self.max_authors_per_record,
                "max_external_urls_per_record": self.max_external_urls_per_record,
            }
        )

    def identify(self) -> PmcRepositoryIdentity:
        """Read and validate the repository's declared OAI-PMH contract."""

        response = self._get({"verb": "Identify"})
        root = _parse_oai(response, self.name)
        errors = _oai_errors(root)
        if errors:
            detail = "; ".join(f"{code}: {message}" for code, message in errors)
            raise ValueError(f"{self.name}: OAI-PMH Identify error: {detail}")
        identify = _single_child(root, _OAI_NAMESPACE, "Identify", self.name)
        repository_name = _required_oai_child_text(identify, "repositoryName", self.name)
        base_url = _web_url(
            _required_oai_child_text(identify, "baseURL", self.name), self.name
        )
        if canonicalize_url(base_url) != canonicalize_url(self.url):
            raise ValueError(
                f"{self.name}: OAI-PMH Identify baseURL does not match configured URL"
            )
        protocol_version = _required_oai_child_text(
            identify, "protocolVersion", self.name
        )
        if protocol_version != "2.0":
            raise ValueError(
                f"{self.name}: unsupported OAI-PMH protocol version {protocol_version!r}"
            )
        earliest_datestamp = _normalize_datestamp(
            _required_oai_child_text(identify, "earliestDatestamp", self.name),
            self.name,
        )
        deleted_record = _required_oai_child_text(identify, "deletedRecord", self.name)
        if deleted_record not in {"no", "persistent", "transient"}:
            raise ValueError(
                f"{self.name}: invalid OAI-PMH deletedRecord policy {deleted_record!r}"
            )
        granularity = _required_oai_child_text(identify, "granularity", self.name)
        if granularity not in {"YYYY-MM-DD", "YYYY-MM-DDThh:mm:ssZ"}:
            raise ValueError(
                f"{self.name}: unsupported OAI-PMH datestamp granularity {granularity!r}"
            )
        return PmcRepositoryIdentity(
            repository_name=repository_name,
            base_url=base_url,
            protocol_version=protocol_version,
            earliest_datestamp=earliest_datestamp,
            deleted_record=deleted_record,
            granularity=granularity,
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        now = _as_utc(self.clock())
        closed_through = now.date() - timedelta(days=1)
        prior_watermark = _optional_date(state.get("watermark"), "watermark", self.name)
        window_start, window_end, frozen = self._window(
            state,
            prior_watermark=prior_watermark,
            closed_through=closed_through,
        )
        token = _state_token(state, frozen=frozen, source=self.name)
        raw_items_seen = _state_count(state, "raw_items_seen", self.name) or 0
        if token is None and raw_items_seen:
            raise ValueError(f"{self.name}: raw_items_seen requires a resumption token")
        scan_total = _state_count(state, "scan_total", self.name)
        token_hashes = _state_token_hashes(state, self.name)
        if token is not None and (token_hash := content_hash(token)) not in token_hashes:
            token_hashes.append(token_hash)

        if window_start > window_end:
            next_state: dict[str, Any] = {"completed_at": _isoformat(now)}
            if prior_watermark is not None:
                next_state["watermark"] = prior_watermark.isoformat()
            return SourcePage(
                records=(), next_state=next_state, complete=True, upstream_count=0
            )

        retry_state = self._page_state(
            state,
            token=token,
            token_hashes=token_hashes,
            raw_items_seen=raw_items_seen,
            scan_total=scan_total,
            window_start=window_start,
            window_end=window_end,
            prior_watermark=prior_watermark,
            now=now,
        )
        params: dict[str, str] = {"verb": "ListRecords"}
        if token is None:
            params.update(
                {
                    "metadataPrefix": self.metadata_prefix,
                    "set": self.set_spec,
                    "from": window_start.isoformat(),
                    "until": window_end.isoformat(),
                }
            )
        else:
            # OAI-PMH requires the opaque token to be the only request argument.
            params["resumptionToken"] = token

        root = _parse_oai(self._get(params), self.name)
        errors = _oai_errors(root)
        if errors:
            return self._error_page(
                errors,
                token=token,
                retry_state=retry_state,
                window_start=window_start,
                window_end=window_end,
                prior_watermark=prior_watermark,
                now=now,
            )

        list_records = _single_child(root, _OAI_NAMESPACE, "ListRecords", self.name)
        record_elements = _children(list_records, _OAI_NAMESPACE, "record")
        if len(record_elements) > self.max_records_per_page:
            raise ValueError(
                f"{self.name}: OAI-PMH page exceeds "
                f"{self.max_records_per_page} records"
            )
        token_element = _optional_single_child(
            list_records, _OAI_NAMESPACE, "resumptionToken", self.name
        )
        next_token = _node_text(token_element) or None
        response_total = _optional_nonnegative_attribute(
            token_element, "completeListSize", self.name
        )
        response_cursor = _optional_nonnegative_attribute(
            token_element, "cursor", self.name
        )

        issues: list[SourceIssue] = []
        if response_cursor is not None and response_cursor != raw_items_seen:
            issues.append(
                self._pagination_issue(
                    token,
                    window_start,
                    window_end,
                    (
                        f"response cursor {response_cursor} does not match "
                        f"the expected offset {raw_items_seen}"
                    ),
                    raw_items_seen,
                    scan_total,
                    len(record_elements),
                )
            )
        if response_total is not None:
            if scan_total is not None and response_total != scan_total:
                issues.append(
                    self._pagination_issue(
                        token,
                        window_start,
                        window_end,
                        f"completeListSize changed from {scan_total} to {response_total}",
                        raw_items_seen,
                        scan_total,
                        len(record_elements),
                    )
                )
            scan_total = response_total

        records: list[SourceRecord] = []
        page_ids: set[str] = set()
        for index, element in enumerate(record_elements):
            try:
                record = self._record(element)
                if record.source_record_id in page_ids:
                    raise ValueError(
                        f"{self.name}: duplicate record {record.source_record_id!r} "
                        "on one page"
                    )
                page_ids.add(record.source_record_id)
                records.append(record)
            except (KeyError, TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=_malformed_record_id(
                            element, index, self.name
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={
                            "index": index,
                            "raw_xml": ET.tostring(
                                element, encoding="unicode"
                            )[:12_000],
                        },
                    )
                )

        next_raw_items_seen = raw_items_seen + len(record_elements)
        if next_token and not record_elements:
            issues.append(
                self._pagination_issue(
                    token,
                    window_start,
                    window_end,
                    "pagination returned a resumption token with no records",
                    raw_items_seen,
                    scan_total,
                    0,
                )
            )
        if next_token and content_hash(next_token) in token_hashes:
            issues.append(
                self._pagination_issue(
                    token,
                    window_start,
                    window_end,
                    "resumption token repeated; pagination made no forward progress",
                    raw_items_seen,
                    scan_total,
                    len(record_elements),
                )
            )
        if scan_total is not None and next_raw_items_seen > scan_total:
            issues.append(
                self._pagination_issue(
                    token,
                    window_start,
                    window_end,
                    (
                        f"page ended at offset {next_raw_items_seen}, beyond "
                        f"completeListSize {scan_total}"
                    ),
                    raw_items_seen,
                    scan_total,
                    len(record_elements),
                )
            )
        if next_token and scan_total is not None and next_raw_items_seen >= scan_total:
            issues.append(
                self._pagination_issue(
                    token,
                    window_start,
                    window_end,
                    "response supplied another token after reaching completeListSize",
                    raw_items_seen,
                    scan_total,
                    len(record_elements),
                )
            )
        if not next_token and scan_total is not None and next_raw_items_seen < scan_total:
            issues.append(
                self._pagination_issue(
                    token,
                    window_start,
                    window_end,
                    (
                        f"pagination ended after {next_raw_items_seen} raw record(s), "
                        f"before completeListSize {scan_total}"
                    ),
                    raw_items_seen,
                    scan_total,
                    len(record_elements),
                )
            )

        if issues:
            next_state = dict(retry_state)
            complete = False
        elif next_token:
            next_state = self._page_state(
                state,
                token=next_token,
                token_hashes=(*token_hashes, content_hash(next_token)),
                raw_items_seen=next_raw_items_seen,
                scan_total=scan_total,
                window_start=window_start,
                window_end=window_end,
                prior_watermark=prior_watermark,
                now=now,
                token_element=token_element,
            )
            complete = False
        else:
            next_state = {
                "watermark": window_end.isoformat(),
                "completed_at": _isoformat(now),
            }
            complete = True

        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=scan_total,
            issues=tuple(issues),
            retry_state=retry_state,
        )

    def _get(self, params: Mapping[str, str]) -> HttpResponse:
        try:
            response: HttpResponse = self.client.get(
                self.url,
                params=params,
                headers={
                    "Accept": "application/xml, text/xml;q=0.9",
                    "Accept-Encoding": "gzip, deflate",
                },
            )
        except HttpFailure as error:
            # PMC returns valid OAI noRecordsMatch documents with HTTP 404 and
            # some argument errors with HTTP 400. Preserve those bounded bodies
            # so OAI error semantics, rather than the transport status, decide
            # whether a frozen scan is complete or must fail.
            response = _oai_http_error_response(
                error, self.url, self.max_response_bytes, self.name
            )
        if response.status not in {200, 400, 404}:
            raise ValueError(f"{self.name}: OAI-PMH returned HTTP {response.status}")
        body = _decoded_body(response, self.max_response_bytes, self.name)
        return HttpResponse(response.status, response.headers, body, response.url)

    def _window(
        self,
        state: Mapping[str, Any],
        *,
        prior_watermark: date | None,
        closed_through: date,
    ) -> tuple[date, date, bool]:
        raw_start = state.get("window_start")
        raw_end = state.get("window_end")
        if raw_start is not None or raw_end is not None:
            if raw_start is None or raw_end is None:
                raise ValueError(f"{self.name}: frozen window requires both boundaries")
            window_start = _required_date(raw_start, "window_start", self.name)
            window_end = _required_date(raw_end, "window_end", self.name)
            if window_start > window_end:
                raise ValueError(f"{self.name}: window_start must not follow window_end")
            if window_end > closed_through:
                raise ValueError(f"{self.name}: window_end must be a closed UTC day")
            return window_start, window_end, True

        if "resumption_token" in state:
            raise ValueError(
                f"{self.name}: resumption-token checkpoint is missing its frozen window"
            )
        if prior_watermark is None:
            lookback = max(self.initial_lookback_days, 1)
            window_start = closed_through - timedelta(days=lookback - 1)
        else:
            if prior_watermark > closed_through:
                raise ValueError(
                    f"{self.name}: watermark is later than the last closed UTC day"
                )
            window_start = prior_watermark + timedelta(days=1 - self.overlap_days)
        return window_start, closed_through, False

    def _page_state(
        self,
        state: Mapping[str, Any],
        *,
        token: str | None,
        token_hashes: Sequence[str],
        raw_items_seen: int,
        scan_total: int | None,
        window_start: date,
        window_end: date,
        prior_watermark: date | None,
        now: datetime,
        token_element: ET.Element | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "started_at": _text(state.get("started_at")) or _isoformat(now),
            "raw_items_seen": raw_items_seen,
        }
        if token is not None:
            result["resumption_token"] = token
        if token_hashes:
            result["seen_token_hashes"] = list(dict.fromkeys(token_hashes))
        if scan_total is not None:
            result["scan_total"] = scan_total
        if prior_watermark is not None:
            result["watermark"] = prior_watermark.isoformat()
        if token_element is not None:
            expiration = _text(token_element.get("expirationDate"))
            if expiration:
                result["token_expires_at"] = _normalize_datestamp(
                    expiration, self.name
                )
        return result

    def _error_page(
        self,
        errors: Sequence[tuple[str, str]],
        *,
        token: str | None,
        retry_state: Mapping[str, Any],
        window_start: date,
        window_end: date,
        prior_watermark: date | None,
        now: datetime,
    ) -> SourcePage:
        if len(errors) == 1 and errors[0][0] == "noRecordsMatch" and token is None:
            return SourcePage(
                records=(),
                next_state={
                    "watermark": window_end.isoformat(),
                    "completed_at": _isoformat(now),
                },
                complete=True,
                upstream_count=0,
                retry_state=retry_state,
            )
        if len(errors) == 1 and errors[0][0] == "badResumptionToken" and token is not None:
            restart_state = self._page_state(
                {},
                token=None,
                token_hashes=(),
                raw_items_seen=0,
                scan_total=None,
                window_start=window_start,
                window_end=window_end,
                prior_watermark=prior_watermark,
                now=now,
            )
            message = errors[0][1] or "upstream rejected the resumption token"
            issue = self._pagination_issue(
                token,
                window_start,
                window_end,
                f"badResumptionToken: {message}; restarting the frozen window",
                0,
                None,
                0,
            )
            return SourcePage(
                records=(),
                next_state=restart_state,
                complete=False,
                issues=(issue,),
                retry_state=restart_state,
            )
        detail = "; ".join(f"{code}: {message}" for code, message in errors)
        raise ValueError(f"{self.name}: OAI-PMH error: {detail}")

    def _pagination_issue(
        self,
        token: str | None,
        window_start: date,
        window_end: date,
        message: str,
        raw_items_seen: int,
        scan_total: int | None,
        received: int,
    ) -> SourceIssue:
        summary = {
            "token_hash": content_hash(token) if token else None,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "raw_items_seen": raw_items_seen,
            "complete_list_size": scan_total,
            "received_count": received,
        }
        return SourceIssue(
            source_record_id=(
                f"{self.name}:pagination:"
                f"{content_hash({**summary, 'message': message})[:32]}"
            ),
            stage="source_pagination",
            error=f"{self.name}: {message}",
            summary=summary,
        )

    def _record(self, element: ET.Element) -> SourceRecord:
        header = _single_child(element, _OAI_NAMESPACE, "header", self.name)
        oai_identifier = _required_node_text(
            _single_child(header, _OAI_NAMESPACE, "identifier", self.name),
            "OAI identifier",
            self.name,
        )
        pmcid = _pmcid_from_oai(oai_identifier, self.name)
        datestamp = _normalize_datestamp(
            _required_node_text(
                _single_child(header, _OAI_NAMESPACE, "datestamp", self.name),
                "OAI datestamp",
                self.name,
            ),
            self.name,
        )
        status = _text(header.get("status")).casefold()
        set_specs = [
            _node_text(item) for item in _children(header, _OAI_NAMESPACE, "setSpec")
        ]
        set_specs = [item for item in set_specs if item]
        header_raw = {
            "identifier": oai_identifier,
            "datestamp": datestamp,
            "status": status or None,
            "set_specs": set_specs,
        }
        canonical_url = _pmc_url(pmcid)

        if status:
            if status != "deleted":
                raise ValueError(f"{self.name}: unsupported OAI header status {status!r}")
            metadata = _optional_single_child(
                element, _OAI_NAMESPACE, "metadata", self.name
            )
            if metadata is not None:
                raise ValueError(
                    f"{self.name}: deleted record unexpectedly contains metadata"
                )
            return SourceRecord(
                source_record_id=pmcid,
                kind=self.artifact_kind,
                canonical_url=canonical_url,
                title=f"[deleted PMC record] {pmcid}",
                raw={"oai_header": header_raw, "deleted": True},
                modified_at=datestamp,
                identifiers=(Identifier("pmcid", pmcid),),
                links=(
                    Link(
                        canonical_url,
                        relation="landing_page",
                        locator="header.identifier",
                        crawl=False,
                    ),
                ),
                deleted=True,
            )

        if self.set_spec not in set_specs:
            raise ValueError(
                f"{self.name}: active record is not declared in {self.set_spec!r}"
            )
        metadata = _single_child(element, _OAI_NAMESPACE, "metadata", self.name)
        metadata_children = list(metadata)
        if len(metadata_children) != 1:
            raise ValueError(
                f"{self.name}: record metadata must contain one JATS payload element"
            )
        article = metadata_children[0]
        jats_namespace, local_name = _split_tag(article.tag)
        if local_name != "article":
            raise ValueError(
                f"{self.name}: expected JATS article metadata, received {local_name!r}"
            )
        jats_xml_bytes = ET.tostring(article, encoding="utf-8")
        if len(jats_xml_bytes) > self.max_record_bytes:
            raise ValueError(
                f"{self.name}: JATS record exceeds {self.max_record_bytes} bytes"
            )
        element_count = sum(1 for _ in article.iter())
        if element_count > self.max_elements_per_record:
            raise ValueError(
                f"{self.name}: JATS record exceeds "
                f"{self.max_elements_per_record} elements"
            )

        front = _required_single_child_local(article, "front", self.name)
        article_meta = _required_single_child_local(front, "article-meta", self.name)
        raw_identifiers = _article_identifiers(article_meta)
        metadata_pmcids = [
            _pmcid(value, self.name)
            for namespace, value in raw_identifiers
            if namespace in {"pmc", "pmcid", "pmcaid"}
        ]
        if metadata_pmcids and any(value != pmcid for value in metadata_pmcids):
            raise ValueError(
                f"{self.name}: header PMCID {pmcid!r} does not match JATS PMCID"
            )

        title = _element_text(_first_descendant(article_meta, "article-title")) or pmcid
        abstracts = [
            _element_text(item) for item in _children_local(article_meta, "abstract")
        ]
        abstract = "\n\n".join(item for item in abstracts if item)
        body_element = _first_child_local(article, "body")
        body_text = _element_text(body_element)
        text = "\n\n".join(item for item in (title, abstract, body_text) if item)
        if len(text) > self.max_text_chars_per_record:
            raise ValueError(
                f"{self.name}: extracted record text exceeds "
                f"{self.max_text_chars_per_record} characters"
            )

        authors = _authors(article_meta, self.max_authors_per_record, self.name)
        journal = _journal_metadata(article)
        dates = _article_dates(article_meta)
        licenses = _license_metadata(article_meta)
        external_urls = _external_urls(
            article, self.max_external_urls_per_record, self.name
        )
        identifiers = [Identifier("pmcid", pmcid)]
        pmid = _first_identifier(raw_identifiers, "pmid")
        if pmid:
            if not _NUMERIC_ID_RE.fullmatch(pmid):
                raise ValueError(f"{self.name}: invalid PMID {pmid!r}")
            identifiers.append(Identifier("pmid", pmid))
        doi = _normalize_doi(_first_identifier(raw_identifiers, "doi"))
        if doi:
            identifiers.append(Identifier("doi", doi))
        for namespace, value in raw_identifiers:
            if namespace == "pmcid-ver" and value:
                identifiers.append(Identifier("pmcid:version", value.upper()))

        links = [
            Link(
                canonical_url,
                relation="landing_page",
                locator="header.identifier",
                crawl=False,
            ),
            Link(
                canonical_url,
                relation="open_full_text",
                locator="metadata.article",
                crawl=False,
            ),
        ]
        if pmid:
            links.append(
                Link(
                    canonicalize_url(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                    relation="indexed_as",
                    locator="metadata.article-meta.article-id[pmid]",
                    crawl=False,
                )
            )
        if doi:
            links.append(
                Link(
                    canonicalize_url(
                        f"https://doi.org/{quote(doi, safe='/():._-')}"
                    ),
                    relation="published_as",
                    locator="metadata.article-meta.article-id[doi]",
                    crawl=False,
                )
            )
        license_urls = {
            url
            for license_item in licenses
            for url in license_item.get("urls", [])
        }
        for item in external_urls:
            url = item["url"]
            if url in license_urls:
                relation = "license"
                crawl = False
            elif item.get("resource_type") == "model_artifact":
                relation = "model_artifact"
                crawl = False
            elif _is_repository_url(url):
                relation = "implementation"
                crawl = True
            else:
                relation = "references"
                crawl = False
            links.append(
                Link(url, relation=relation, locator=item["locator"], crawl=crawl)
            )

        published_at = _preferred_publication_date(dates)
        jats = {
            "namespace": jats_namespace or None,
            "article_type": _text(article.get("article-type")) or None,
            "language": _xml_language(article),
            "identifiers": [
                {"namespace": namespace, "value": value}
                for namespace, value in raw_identifiers
            ],
            "title": title,
            "abstract": abstract,
            "abstracts": abstracts,
            "body_text": body_text,
            "authors": authors,
            "journal": journal,
            "dates": dates,
            "licenses": licenses,
            "external_urls": external_urls,
            "element_count": element_count,
        }
        return SourceRecord(
            source_record_id=pmcid,
            kind=self.artifact_kind,
            canonical_url=canonical_url,
            title=title,
            raw={
                "oai_header": header_raw,
                "jats": jats,
                "jats_xml": jats_xml_bytes.decode("utf-8"),
            },
            text=text,
            published_at=published_at,
            modified_at=datestamp,
            identifiers=_unique_identifiers(identifiers),
            links=_unique_links(links),
        )


def _decoded_body(response: HttpResponse, limit: int, source: str) -> bytes:
    body = response.body
    if not isinstance(body, bytes):
        raise ValueError(f"{source}: OAI-PMH response body must be bytes")
    encoding = _text(response.headers.get("content-encoding")).casefold()
    if encoding in {"", "identity"}:
        decoded = body
    elif encoding == "gzip":
        decoded = _decompress(body, limit, 16 + zlib.MAX_WBITS, source)
    elif encoding == "deflate":
        try:
            decoded = _decompress(body, limit, zlib.MAX_WBITS, source)
        except ValueError:
            decoded = _decompress(body, limit, -zlib.MAX_WBITS, source)
    else:
        raise ValueError(f"{source}: unsupported HTTP content encoding {encoding!r}")
    if len(decoded) > limit:
        raise ValueError(f"{source}: OAI-PMH response exceeds {limit} decoded bytes")
    return decoded


def _oai_http_error_response(
    error: HttpFailure, url: str, limit: int, source: str
) -> HttpResponse:
    cause = error.__cause__
    if not isinstance(cause, HTTPError) or cause.code not in {400, 404}:
        raise error
    try:
        body = cause.read(limit + 1)
        headers = {
            key.casefold(): value for key, value in cause.headers.items()
        }
    finally:
        cause.close()
    if len(body) > limit:
        raise ValueError(
            f"{source}: OAI-PMH HTTP error response exceeds {limit} bytes"
        )
    return HttpResponse(status=cause.code, headers=headers, body=body, url=url)


def _decompress(body: bytes, limit: int, window_bits: int, source: str) -> bytes:
    decoder = zlib.decompressobj(window_bits)
    try:
        result = decoder.decompress(body, limit + 1)
        if len(result) > limit or decoder.unconsumed_tail:
            raise ValueError(
                f"{source}: OAI-PMH response exceeds {limit} decoded bytes"
            )
        result += decoder.flush(limit + 1 - len(result))
    except zlib.error:
        raise ValueError(f"{source}: invalid compressed OAI-PMH response") from None
    if len(result) > limit:
        raise ValueError(f"{source}: OAI-PMH response exceeds {limit} decoded bytes")
    if not decoder.eof:
        raise ValueError(f"{source}: truncated compressed OAI-PMH response")
    return result


def _parse_oai(response: HttpResponse, source: str) -> ET.Element:
    body = response.body
    if re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", body, flags=re.IGNORECASE):
        raise ValueError(f"{source}: unsafe XML declaration in {response.url}")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise ValueError(f"{source}: malformed XML from {response.url}: {error}") from None
    namespace, local_name = _split_tag(root.tag)
    if namespace != _OAI_NAMESPACE or local_name != "OAI-PMH":
        raise ValueError(f"{source}: response root is not an OAI-PMH 2.0 document")
    return root


def _oai_errors(root: ET.Element) -> tuple[tuple[str, str], ...]:
    return tuple(
        (_text(element.get("code")) or "unknown", _node_text(element))
        for element in _children(root, _OAI_NAMESPACE, "error")
    )


def _article_identifiers(article_meta: ET.Element) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for element in _children_local(article_meta, "article-id"):
        namespace = _text(element.get("pub-id-type")).casefold()
        value = _element_text(element)
        if namespace and value:
            candidate = (namespace, value)
            if candidate not in values:
                values.append(candidate)
    return tuple(values)


def _authors(
    article_meta: ET.Element, limit: int, source: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for contrib in _descendants(article_meta, "contrib"):
        contrib_type = _text(contrib.get("contrib-type")).casefold()
        if contrib_type and contrib_type != "author":
            continue
        name = _first_child_local(contrib, "name")
        string_name = _first_child_local(contrib, "string-name")
        collab = _first_child_local(contrib, "collab")
        given = _child_text_local(name, "given-names")
        surname = _child_text_local(name, "surname")
        prefix = _child_text_local(name, "prefix")
        suffix = _child_text_local(name, "suffix")
        display = _element_text(string_name) or _element_text(collab)
        if not display:
            display = " ".join(
                item for item in (prefix, given, surname, suffix) if item
            )
        item: dict[str, Any] = {
            "name": display or None,
            "given_names": given or None,
            "surname": surname or None,
            "prefix": prefix or None,
            "suffix": suffix or None,
            "collaboration": _element_text(collab) or None,
            "corresponding": _text(contrib.get("corresp")).casefold()
            in {"yes", "true", "1"},
            "affiliation_refs": [
                value
                for xref in _descendants(contrib, "xref")
                if _text(xref.get("ref-type")).casefold() == "aff"
                for value in _text(xref.get("rid")).split()
                if value
            ],
        }
        for contrib_id in _descendants(contrib, "contrib-id"):
            if (
                _text(contrib_id.get("contrib-id-type")).casefold() == "orcid"
                and (match := _ORCID_RE.fullmatch(_element_text(contrib_id)))
            ):
                item["orcid"] = match.group(1).upper()
                break
        result.append(
            {
                key: value
                for key, value in item.items()
                if value is not None and value != []
            }
        )
        if len(result) > limit:
            raise ValueError(f"{source}: JATS record exceeds {limit} authors")
    return result


def _journal_metadata(article: ET.Element) -> dict[str, Any]:
    journal_meta = _first_descendant(article, "journal-meta")
    article_meta = _first_descendant(article, "article-meta")
    if journal_meta is None:
        return {}
    journal_ids = [
        {
            "type": _text(item.get("journal-id-type")) or None,
            "value": _element_text(item),
        }
        for item in _children_local(journal_meta, "journal-id")
        if _element_text(item)
    ]
    issns = [
        {
            "publication_type": _text(item.get("pub-type")) or None,
            "value": _element_text(item),
        }
        for item in _children_local(journal_meta, "issn")
        if _element_text(item)
    ]
    publisher = _first_descendant(journal_meta, "publisher")
    result: dict[str, Any] = {
        "title": _element_text(_first_descendant(journal_meta, "journal-title")) or None,
        "identifiers": journal_ids,
        "issns": issns,
        "publisher_name": _child_text_local(publisher, "publisher-name") or None,
        "publisher_location": _child_text_local(publisher, "publisher-loc") or None,
        "volume": _child_text_local(article_meta, "volume") or None,
        "issue": _child_text_local(article_meta, "issue") or None,
    }
    return {
        key: value
        for key, value in result.items()
        if value is not None and value != []
    }


def _article_dates(article_meta: ET.Element) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in _children_local(article_meta, "pub-date"):
        value = _jats_date(item)
        if value:
            result.append(
                {
                    "kind": _text(item.get("pub-type")) or "publication",
                    "value": value,
                }
            )
    for event in _descendants(article_meta, "event"):
        value = _jats_date(_first_child_local(event, "date"))
        if value:
            result.append(
                {
                    "kind": _text(event.get("event-type")) or "event",
                    "value": value,
                }
            )
    for history_date in _descendants(article_meta, "history"):
        for item in _children_local(history_date, "date"):
            value = _jats_date(item)
            if value:
                result.append(
                    {
                        "kind": _text(item.get("date-type")) or "history",
                        "value": value,
                    }
                )
    unique = dict.fromkeys((item["kind"], item["value"]) for item in result)
    return [
        {"kind": kind, "value": value}
        for kind, value in unique
    ]


def _jats_date(element: ET.Element | None) -> str:
    if element is None:
        return ""
    iso_value = _text(element.get("iso-8601-date"))
    if iso_value:
        return _normalize_jats_date(iso_value)
    year = _child_text_local(element, "year")
    month = _child_text_local(element, "month")
    day = _child_text_local(element, "day")
    if not re.fullmatch(r"\d{4}", year):
        return ""
    if not month:
        return year
    if not month.isdigit() or not 1 <= int(month) <= 12:
        return year
    month_value = f"{int(month):02d}"
    if not day:
        return f"{year}-{month_value}"
    if not day.isdigit():
        return f"{year}-{month_value}"
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return f"{year}-{month_value}"


def _normalize_jats_date(value: str) -> str:
    candidate = value.replace(" ", "T", 1)
    try:
        if len(candidate) == 10:
            return date.fromisoformat(candidate).isoformat()
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None:
        return parsed.isoformat()
    return _isoformat(parsed)


def _preferred_publication_date(dates: Sequence[Mapping[str, str]]) -> str | None:
    by_kind = {item.get("kind", "").casefold(): item.get("value", "") for item in dates}
    for kind in _DATE_TYPES:
        if value := by_kind.get(kind):
            return value
    return _text(dates[0].get("value")) if dates else None


def _license_metadata(article_meta: ET.Element) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for element in _descendants(article_meta, "license"):
        urls: list[str] = []
        for node in element.iter():
            for value in (*node.attrib.values(), _element_text(node)):
                urls.extend(extract_urls(value))
        result.append(
            {
                "license_type": _text(element.get("license-type")) or None,
                "text": _element_text(element),
                "urls": list(dict.fromkeys(urls)),
            }
        )
    return result


def _external_urls(
    article: ET.Element, limit: int, source: str
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    parents = {child: parent for parent in article.iter() for child in parent}
    for index, element in enumerate(article.iter()):
        local_name = _split_tag(element.tag)[1]
        context = _link_context(element, parents)
        candidates: list[tuple[str, str]] = []
        for attribute, value in element.attrib.items():
            candidates.extend(
                (url, f"metadata.article[{index}].{_split_tag(attribute)[1]}")
                for url in _external_candidate_urls(value)
            )
        if element.text:
            candidates.extend(
                (url, f"metadata.article[{index}].text")
                for url in _external_candidate_urls(element.text)
            )
        if element.tail:
            candidates.extend(
                (url, f"metadata.article[{index}].tail")
                for url in _external_candidate_urls(element.tail)
            )
        for url, locator in candidates:
            canonical = _safe_url(url)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            item = {"url": canonical, "locator": locator, "element": local_name}
            if (
                _is_zenodo_record(canonical)
                and _MODEL_RESOURCE_CONTEXT_RE.search(context)
            ) or _is_pmc_supplementary_model_artifact(
                element, canonical, context, parents
            ):
                item["resource_type"] = "model_artifact"
            result.append(item)
            if len(result) > limit:
                raise ValueError(f"{source}: JATS record exceeds {limit} external URLs")
    return result


def _link_context(element: ET.Element, parents: Mapping[ET.Element, ET.Element]) -> str:
    """Return nearby JATS prose for a link, without classifying whole articles."""

    current = element
    while current in parents:
        current = parents[current]
        if _split_tag(current.tag)[1] in {"p", "named-content", "caption"}:
            return _element_text(current)
    return _element_text(element)


def _is_zenodo_record(url: str) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").casefold()
    path = parts.path.casefold()
    return (
        host == "zenodo.org" and re.match(r"^/(?:records?|record)/\d+(?:/|$)", path) is not None
    ) or (host in {"doi.org", "dx.doi.org"} and path.startswith("/10.5281/zenodo."))


def _external_candidate_urls(value: str) -> list[str]:
    urls = extract_urls(value)
    urls.extend(
        canonicalize_url(match.group(0))
        for match in _PMC_FTP_URL_RE.finditer(value)
    )
    return list(dict.fromkeys(urls))


def _is_pmc_supplementary_model_artifact(
    element: ET.Element,
    url: str,
    context: str,
    parents: Mapping[ET.Element, ET.Element],
) -> bool:
    """Recognize model files directly declared as PMC supplementary material."""

    parts = urlsplit(url)
    if (parts.hostname or "").casefold() not in {
        "ftp.ncbi.nlm.nih.gov",
        "cdn.ncbi.nlm.nih.gov",
    }:
        return False
    current = element
    supplementary = False
    while True:
        if _split_tag(current.tag)[1] in {
            "supplementary-material",
            "inline-supplementary-material",
        }:
            supplementary = True
            break
        parent = parents.get(current)
        if parent is None:
            break
        current = parent
    return supplementary and _MODEL_RESOURCE_CONTEXT_RE.search(context) is not None


def _is_repository_url(url: str) -> bool:
    identifier = identifier_from_url(url)
    if identifier is not None and identifier.namespace in {
        "github:repository",
        "huggingface:model",
    }:
        return True
    return urlsplit(url).path.casefold().endswith(".git")


def _pmcid_from_oai(value: Any, source: str) -> str:
    text = _text(value)
    if not text.casefold().startswith(_OAI_IDENTIFIER_PREFIX):
        raise ValueError(f"{source}: invalid PMC OAI identifier {value!r}")
    numeric = text[len(_OAI_IDENTIFIER_PREFIX) :]
    if not _NUMERIC_ID_RE.fullmatch(numeric):
        raise ValueError(f"{source}: invalid PMC OAI identifier {value!r}")
    return f"PMC{numeric}"


def _pmcid(value: Any, source: str) -> str:
    text = _text(value)
    if _NUMERIC_ID_RE.fullmatch(text):
        return f"PMC{text}"
    match = _PMCID_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"{source}: invalid PMCID {value!r}")
    return f"PMC{match.group(1)}"


def _pmc_url(pmcid: str) -> str:
    return canonicalize_url(
        f"https://pmc.ncbi.nlm.nih.gov/articles/{quote(pmcid, safe='')}/"
    )


def _normalize_doi(value: Any) -> str:
    text = _text(value)
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    match = _DOI_RE.fullmatch(text.rstrip(".,;:"))
    return match.group(0).casefold() if match else ""


def _first_identifier(
    identifiers: Sequence[tuple[str, str]], namespace: str
) -> str:
    return next((value for kind, value in identifiers if kind == namespace), "")


def _xml_language(article: ET.Element) -> str | None:
    return _text(article.get("{http://www.w3.org/XML/1998/namespace}lang")) or None


def _element_text(element: ET.Element | None) -> str:
    return "" if element is None else " ".join("".join(element.itertext()).split())


def _first_descendant(parent: ET.Element | None, name: str) -> ET.Element | None:
    if parent is None:
        return None
    return next((item for item in parent.iter() if _split_tag(item.tag)[1] == name), None)


def _required_single_child_local(
    parent: ET.Element, name: str, source: str
) -> ET.Element:
    matches = _children_local(parent, name)
    if len(matches) != 1:
        raise ValueError(
            f"{source}: expected exactly one JATS {name} element, found {len(matches)}"
        )
    return matches[0]


def _descendants(parent: ET.Element, name: str) -> Iterable[ET.Element]:
    return (item for item in parent.iter() if _split_tag(item.tag)[1] == name)


def _children_local(parent: ET.Element | None, name: str) -> list[ET.Element]:
    if parent is None:
        return []
    return [item for item in parent if _split_tag(item.tag)[1] == name]


def _first_child_local(parent: ET.Element | None, name: str) -> ET.Element | None:
    return next(iter(_children_local(parent, name)), None)


def _child_text_local(parent: ET.Element | None, name: str) -> str:
    return _element_text(_first_child_local(parent, name))


def _required_oai_child_text(element: ET.Element, name: str, source: str) -> str:
    return _required_node_text(
        _single_child(element, _OAI_NAMESPACE, name, source),
        f"OAI-PMH {name}",
        source,
    )


def _malformed_record_id(element: ET.Element, index: int, source: str) -> str:
    try:
        header = _single_child(element, _OAI_NAMESPACE, "header", source)
        identifier = _single_child(header, _OAI_NAMESPACE, "identifier", source)
        return _pmcid_from_oai(_node_text(identifier), source)
    except ValueError:
        xml = ET.tostring(element, encoding="unicode")
        return f"{source}:malformed:{content_hash(xml)[:32]}:{index}"


def _state_token(state: Mapping[str, Any], *, frozen: bool, source: str) -> str | None:
    if "resumption_token" not in state:
        return None
    if not frozen:
        raise ValueError(f"{source}: resumption token requires a frozen window")
    token = _required_text(state.get("resumption_token"), "resumption_token")
    if len(token) > 16_384:
        raise ValueError(f"{source}: resumption token is unreasonably large")
    return token


def _state_token_hashes(state: Mapping[str, Any], source: str) -> list[str]:
    value = state.get("seen_token_hashes", ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{source}: invalid seen_token_hashes checkpoint")
    result: list[str] = []
    for item in value:
        text = _text(item)
        if not re.fullmatch(r"[0-9a-f]{64}", text):
            raise ValueError(f"{source}: invalid token hash checkpoint")
        if text not in result:
            result.append(text)
    return result


def _state_count(state: Mapping[str, Any], key: str, source: str) -> int | None:
    if key not in state:
        return None
    return _nonnegative_integer(state.get(key), f"{source}: invalid {key} checkpoint")


def _optional_nonnegative_attribute(
    element: ET.Element | None, name: str, source: str
) -> int | None:
    if element is None or element.get(name) is None:
        return None
    return _nonnegative_integer(
        element.get(name), f"{source}: invalid resumptionToken {name}"
    )


def _nonnegative_integer(value: Any, message: str) -> int:
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if result < 0:
        raise ValueError(message)
    return result


def _nonnegative_config(value: Any, field: str) -> int:
    return _nonnegative_integer(value, f"{field} must be a nonnegative integer")


def _positive_config(value: Any, field: str) -> int:
    result = _nonnegative_config(value, field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _optional_date(value: Any, field: str, source: str) -> date | None:
    if value is None or _text(value) == "":
        return None
    return _required_date(value, field, source)


def _required_date(value: Any, field: str, source: str) -> date:
    if isinstance(value, datetime):
        return _as_utc(value).date()
    if isinstance(value, date):
        return value
    text = _text(value)
    try:
        if len(text) == 10:
            return date.fromisoformat(text)
        return _as_utc(datetime.fromisoformat(text.replace("Z", "+00:00"))).date()
    except ValueError:
        raise ValueError(f"{source}: invalid {field}: {value!r}") from None


def _normalize_datestamp(value: str, source: str) -> str:
    try:
        if len(value) == 10:
            return date.fromisoformat(value).isoformat()
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{source}: invalid OAI datestamp {value!r}") from None
    return _isoformat(parsed)


def _single_child(parent: ET.Element, namespace: str, name: str, source: str) -> ET.Element:
    matches = _children(parent, namespace, name)
    if len(matches) != 1:
        raise ValueError(
            f"{source}: expected exactly one {name} element, found {len(matches)}"
        )
    return matches[0]


def _optional_single_child(
    parent: ET.Element, namespace: str, name: str, source: str
) -> ET.Element | None:
    matches = _children(parent, namespace, name)
    if len(matches) > 1:
        raise ValueError(
            f"{source}: expected at most one {name} element, found {len(matches)}"
        )
    return matches[0] if matches else None


def _children(parent: ET.Element, namespace: str, name: str) -> list[ET.Element]:
    return [child for child in parent if child.tag == f"{{{namespace}}}{name}"]


def _split_tag(tag: str) -> tuple[str, str]:
    if tag.startswith("{") and "}" in tag:
        namespace, local_name = tag[1:].split("}", 1)
        return namespace, local_name
    return "", tag


def _node_text(element: ET.Element | None) -> str:
    return "" if element is None else "".join(element.itertext()).strip()


def _required_node_text(element: ET.Element, field: str, source: str) -> str:
    return _required_text(_node_text(element), f"{source}: {field}")


def _safe_url(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    canonical = canonicalize_url(text)
    parts = urlsplit(canonical)
    if parts.scheme in {"http", "https"} and parts.hostname:
        return canonical
    if (
        parts.scheme == "ftp"
        and (parts.hostname or "").casefold() == "ftp.ncbi.nlm.nih.gov"
        and parts.path.casefold().startswith("/pub/pmc/")
    ):
        return canonical
    return ""


def _web_url(value: Any, source: str) -> str:
    url = _text(value)
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"{source}: API URL must be an HTTP(S) URL")
    return url


def _unique_identifiers(values: Sequence[Identifier]) -> tuple[Identifier, ...]:
    return tuple(dict.fromkeys(values))


def _unique_links(values: Sequence[Link]) -> tuple[Link, ...]:
    seen: set[tuple[str, str]] = set()
    result: list[Link] = []
    for value in values:
        key = (value.url, value.relation)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = ["PmcRepositoryIdentity", "PmcSourceAdapter"]
