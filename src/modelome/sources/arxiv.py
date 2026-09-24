from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import (
    canonicalize_url,
    content_hash,
    extract_url_mentions,
    infer_url_relation,
)

Clock = Callable[[], datetime]

_OAI_NAMESPACE = "http://www.openarchives.org/OAI/2.0/"
_ARXIV_RAW_NAMESPACE = "http://arxiv.org/OAI/arXivRaw/"
_VERSION_RE = re.compile(r"v[1-9]\d*")
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_OAI_IDENTIFIER_PREFIX = "oai:arxiv.org:"


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ArxivRepositoryIdentity:
    """Harvest boundaries and deletion semantics declared by arXiv's OAI endpoint."""

    repository_name: str
    base_url: str
    protocol_version: str
    earliest_datestamp: str
    deleted_record: str
    granularity: str


class ArxivSourceAdapter:
    """Enumerate arXiv's complete, all-subject OAI-PMH change stream."""

    def __init__(
        self,
        *,
        name: str = "arxiv",
        url: str = "https://oaipmh.arxiv.org/oai",
        artifact_kind: str | ArtifactKind = ArtifactKind.PAPER,
        initial_lookback_days: int = 7,
        overlap_days: int = 2,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name)
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.initial_lookback_days = int(initial_lookback_days)
        self.overlap_days = int(overlap_days)
        if self.initial_lookback_days < 0:
            raise ValueError(f"{self.name}: initial lookback days must be nonnegative")
        if self.overlap_days < 0:
            raise ValueError(f"{self.name}: overlap days must be nonnegative")
        self.client = client or HttpClient()
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "arxiv-oai-arxivraw-v2-complete-bootstrap",
                "url": self.url,
                "artifact_kind": self.artifact_kind.value,
                "initial_lookback_days": self.initial_lookback_days,
                "bootstrap": "identify-earliest-datestamp",
                "overlap_days": self.overlap_days,
            }
        )

    def identify(self) -> ArxivRepositoryIdentity:
        """Read the repository-declared lower harvest boundary.

        The value is discovered from OAI-PMH rather than copied into configuration,
        so a clean installation can construct a complete historical scan without a
        model-name seed or a hand-maintained start date.
        """

        response: HttpResponse = self.client.get(
            self.url,
            params={"verb": "Identify"},
            headers={"Accept": "application/xml, text/xml;q=0.9"},
        )
        root = _parse_oai(response.body, self.name, response.url)
        errors = _oai_errors(root)
        if errors:
            detail = "; ".join(f"{code}: {message}" for code, message in errors)
            raise ValueError(f"{self.name}: OAI-PMH Identify error: {detail}")
        identify = _single_child(root, _OAI_NAMESPACE, "Identify", self.name)
        repository_name = _required_oai_child_text(
            identify, "repositoryName", self.name
        )
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
                f"{self.name}: unsupported OAI-PMH protocol version "
                f"{protocol_version!r}"
            )
        earliest_datestamp = _normalize_datestamp(
            _required_oai_child_text(identify, "earliestDatestamp", self.name),
            self.name,
        )
        deleted_record = _required_oai_child_text(
            identify, "deletedRecord", self.name
        )
        if deleted_record not in {"no", "persistent", "transient"}:
            raise ValueError(
                f"{self.name}: invalid OAI-PMH deletedRecord policy "
                f"{deleted_record!r}"
            )
        granularity = _required_oai_child_text(identify, "granularity", self.name)
        if granularity not in {"YYYY-MM-DD", "YYYY-MM-DDThh:mm:ssZ"}:
            raise ValueError(
                f"{self.name}: unsupported OAI-PMH datestamp granularity "
                f"{granularity!r}"
            )
        return ArxivRepositoryIdentity(
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
        bootstrap_start = None
        if (
            prior_watermark is None
            and state.get("window_start") is None
            and state.get("window_end") is None
            and "resumption_token" not in state
        ):
            # A clean live stream must begin at the repository's declared
            # lower bound so persistent deletion headers predating the normal
            # incremental lookback are observed as well.
            identity = self.identify()
            bootstrap_start = date.fromisoformat(identity.earliest_datestamp[:10])
        window_start, window_end, frozen = self._window(
            state,
            prior_watermark=prior_watermark,
            closed_through=closed_through,
            bootstrap_start=bootstrap_start,
        )
        token = _state_token(state, frozen=frozen, source=self.name)
        raw_items_seen = _state_count(state, "raw_items_seen", self.name) or 0
        if token is None and raw_items_seen:
            raise ValueError(f"{self.name}: raw_items_seen requires a resumption token")
        scan_total = _state_count(state, "scan_total", self.name)
        token_hashes = _state_token_hashes(state, self.name)
        if token is not None:
            token_hash = content_hash(token)
            if token_hash not in token_hashes:
                token_hashes.append(token_hash)

        if window_start > window_end:
            next_state: dict[str, Any] = {"completed_at": _isoformat(now)}
            if prior_watermark is not None:
                next_state["watermark"] = prior_watermark.isoformat()
            return SourcePage(records=(), next_state=next_state, complete=True, upstream_count=0)

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
                    "metadataPrefix": "arXivRaw",
                    "from": window_start.isoformat(),
                    "until": window_end.isoformat(),
                }
            )
        else:
            # OAI-PMH requires resumptionToken to be the only argument besides verb.
            params["resumptionToken"] = token

        response: HttpResponse = self.client.get(
            self.url,
            params=params,
            headers={"Accept": "application/xml, text/xml;q=0.9"},
        )
        root = _parse_oai(response.body, self.name, response.url)
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
        token_element = _optional_single_child(
            list_records,
            _OAI_NAMESPACE,
            "resumptionToken",
            self.name,
        )
        next_token = _node_text(token_element) or None
        response_total = _optional_nonnegative_attribute(
            token_element,
            "completeListSize",
            self.name,
        )
        response_cursor = _optional_nonnegative_attribute(
            token_element,
            "cursor",
            self.name,
        )

        issues: list[SourceIssue] = []
        pagination_fault = False
        if response_cursor is not None and response_cursor != raw_items_seen:
            pagination_fault = True
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
                pagination_fault = True
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
                        f"{self.name}: duplicate record {record.source_record_id!r} on one page"
                    )
                page_ids.add(record.source_record_id)
                records.append(record)
            except (KeyError, TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=_malformed_record_id(element, index, self.name),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={
                            "index": index,
                            "raw_xml": ET.tostring(element, encoding="unicode")[:12000],
                        },
                    )
                )

        next_raw_items_seen = raw_items_seen + len(record_elements)
        if next_token and not record_elements:
            pagination_fault = True
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
            pagination_fault = True
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
            pagination_fault = True
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
            pagination_fault = True
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
            pagination_fault = True
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

        if pagination_fault:
            next_state = dict(retry_state)
            complete = False
        elif next_token:
            next_hashes = [*token_hashes, content_hash(next_token)]
            next_state = self._page_state(
                state,
                token=next_token,
                token_hashes=next_hashes,
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

    def _window(
        self,
        state: Mapping[str, Any],
        *,
        prior_watermark: date | None,
        closed_through: date,
        bootstrap_start: date | None = None,
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
            if bootstrap_start is not None:
                window_start = bootstrap_start
            else:
                lookback = max(self.initial_lookback_days, 1)
                window_start = closed_through - timedelta(days=lookback - 1)
        else:
            if prior_watermark > closed_through:
                raise ValueError(f"{self.name}: watermark is later than the last closed UTC day")
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
                result["token_expires_at"] = expiration
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
        arxiv_id = _arxiv_id(oai_identifier, self.name)
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
        canonical_url = _arxiv_url("abs", arxiv_id)

        if status:
            if status != "deleted":
                raise ValueError(f"{self.name}: unsupported OAI header status {status!r}")
            metadata = _optional_single_child(element, _OAI_NAMESPACE, "metadata", self.name)
            if metadata is not None:
                raise ValueError(f"{self.name}: deleted record unexpectedly contains metadata")
            return SourceRecord(
                source_record_id=arxiv_id,
                kind=self.artifact_kind,
                canonical_url=canonical_url,
                title=f"[deleted arXiv record] {arxiv_id}",
                raw={
                    "oai_header": header_raw,
                    "deleted": True,
                    "record_xml": ET.tostring(element, encoding="unicode"),
                },
                modified_at=datestamp,
                identifiers=(Identifier("arxiv", arxiv_id),),
                links=(Link(canonical_url, relation="landing_page", locator="header.identifier"),),
                deleted=True,
            )

        metadata = _single_child(element, _OAI_NAMESPACE, "metadata", self.name)
        metadata_children = list(metadata)
        if len(metadata_children) != 1:
            raise ValueError(f"{self.name}: record metadata must contain one payload element")
        raw_element = metadata_children[0]
        namespace, local_name = _split_tag(raw_element.tag)
        if namespace != _ARXIV_RAW_NAMESPACE or local_name != "arXivRaw":
            raise ValueError(
                f"{self.name}: expected arXivRaw metadata, received "
                f"{{{namespace}}}{local_name}"
            )
        raw_fields, versions = _arxiv_raw_fields(raw_element, self.name)
        metadata_id = _arxiv_id(_required_field(raw_fields, "id", self.name), self.name)
        if metadata_id != arxiv_id:
            raise ValueError(
                f"{self.name}: header ID {arxiv_id!r} does not match metadata ID {metadata_id!r}"
            )
        if not versions:
            raise ValueError(f"{self.name}: arXivRaw record has no versions")

        title = _collapse_space(_text(raw_fields.get("title"))) or arxiv_id
        abstract = _collapse_space(_text(raw_fields.get("abstract")))
        identifiers = [Identifier("arxiv", arxiv_id)]
        links = [Link(canonical_url, relation="landing_page", locator="header.identifier")]
        for index, version in enumerate(versions):
            version_id = f"{arxiv_id}{version['version']}"
            identifiers.append(Identifier("arxiv:version", version_id))
            links.extend(
                (
                    Link(
                        _arxiv_url("abs", version_id),
                        relation="version",
                        locator=f"metadata.version[{index}]",
                    ),
                    Link(
                        _arxiv_url("pdf", version_id),
                        relation="full_text",
                        locator=f"metadata.version[{index}]",
                    ),
                    Link(
                        _arxiv_url("src", version_id),
                        relation="source_archive",
                        locator=f"metadata.version[{index}]",
                    ),
                )
            )
        for doi in _dois(raw_fields.get("doi")):
            identifiers.append(Identifier("doi", doi))
            links.append(
                Link(
                    canonicalize_url(f"https://doi.org/{quote(doi, safe='/.:_-()')}"),
                    relation="published_as",
                    locator="metadata.doi",
                )
            )
        if license_url := _optional_web_url(raw_fields.get("license")):
            links.append(Link(license_url, relation="license", locator="metadata.license"))
        comments = _text(raw_fields.get("comments"))
        for url, span in extract_url_mentions(comments):
            if _optional_web_url(url):
                links.append(
                    Link(
                        url,
                        relation=infer_url_relation(comments, span),
                        locator=f"metadata.comments:{span}",
                    )
                )
        for url, span in extract_url_mentions(abstract):
            if _optional_web_url(url):
                links.append(
                    Link(
                        url=url,
                        relation=infer_url_relation(abstract, span),
                        locator=f"metadata.abstract:{span}",
                    )
                )

        raw_payload = dict(raw_fields)
        raw_payload["versions"] = versions
        categories = _text(raw_fields.get("categories")).split()
        raw_payload["categories_list"] = categories
        return SourceRecord(
            source_record_id=arxiv_id,
            kind=self.artifact_kind,
            canonical_url=canonical_url,
            title=title,
            raw={
                "oai_header": header_raw,
                "arxiv_raw": raw_payload,
                "record_xml": ET.tostring(element, encoding="unicode"),
            },
            text="\n\n".join(item for item in (title, abstract) if item),
            published_at=versions[0]["date_normalized"],
            modified_at=datestamp,
            identifiers=_unique_identifiers(identifiers),
            links=_unique_links(links),
        )


def _parse_oai(body: bytes, source: str, response_url: str) -> ET.Element:
    if re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", body, flags=re.IGNORECASE):
        raise ValueError(f"{source}: unsafe XML declaration in {response_url}")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise ValueError(f"{source}: malformed XML from {response_url}: {error}") from None
    namespace, local_name = _split_tag(root.tag)
    if namespace != _OAI_NAMESPACE or local_name != "OAI-PMH":
        raise ValueError(f"{source}: response root is not an OAI-PMH 2.0 document")
    return root


def _oai_errors(root: ET.Element) -> tuple[tuple[str, str], ...]:
    return tuple(
        (_text(element.get("code")) or "unknown", _node_text(element))
        for element in _children(root, _OAI_NAMESPACE, "error")
    )


def _required_oai_child_text(element: ET.Element, name: str, source: str) -> str:
    return _required_node_text(
        _single_child(element, _OAI_NAMESPACE, name, source),
        f"OAI-PMH {name}",
        source,
    )


def _arxiv_raw_fields(
    element: ET.Element,
    source: str,
) -> tuple[dict[str, Any], list[dict[str, str | None]]]:
    fields: dict[str, Any] = {}
    versions: list[dict[str, str | None]] = []
    seen_versions: set[str] = set()
    for child in element:
        namespace, name = _split_tag(child.tag)
        if namespace != _ARXIV_RAW_NAMESPACE:
            raise ValueError(f"{source}: unexpected namespace in arXivRaw payload: {namespace!r}")
        if name == "version":
            version = _text(child.get("version"))
            if not _VERSION_RE.fullmatch(version):
                raise ValueError(f"{source}: invalid arXiv version {version!r}")
            if version in seen_versions:
                raise ValueError(f"{source}: duplicate arXiv version {version!r}")
            seen_versions.add(version)
            version_date = _required_node_text(
                _single_child(child, _ARXIV_RAW_NAMESPACE, "date", source),
                f"date for {version}",
                source,
            )
            versions.append(
                {
                    "version": version,
                    "date": version_date,
                    "date_normalized": _normalize_version_date(version_date, source),
                    "size": _optional_child_text(child, _ARXIV_RAW_NAMESPACE, "size", source),
                    "source_type": _optional_child_text(
                        child,
                        _ARXIV_RAW_NAMESPACE,
                        "source_type",
                        source,
                    ),
                }
            )
            continue
        value = "".join(child.itertext())
        if name in fields:
            current = fields[name]
            fields[name] = [*current, value] if isinstance(current, list) else [current, value]
        else:
            fields[name] = value
    return fields, versions


def _arxiv_id(value: Any, source: str) -> str:
    result = _text(value)
    if result.casefold().startswith(_OAI_IDENTIFIER_PREFIX):
        result = result[len(_OAI_IDENTIFIER_PREFIX) :]
    if result.casefold().startswith("arxiv:"):
        result = result[6:]
    result = result.removesuffix(".pdf")
    result = re.sub(r"v[1-9]\d*$", "", result, flags=re.IGNORECASE)
    if (
        not result
        or any(character.isspace() for character in result)
        or result.startswith("/")
        or result.endswith("/")
        or ".." in result
        or any(character in result for character in "?#%\\")
        or not re.fullmatch(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?", result)
    ):
        raise ValueError(f"{source}: invalid arXiv identifier {value!r}")
    return result


def _arxiv_url(route: str, arxiv_id: str) -> str:
    encoded = quote(arxiv_id, safe="/._-")
    return canonicalize_url(f"https://arxiv.org/{route}/{encoded}")


def _dois(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in values:
        for match in _DOI_RE.finditer(_text(item)):
            doi = match.group(0).rstrip(".,;:").casefold()
            if doi not in result:
                result.append(doi)
    return tuple(result)


def _normalize_version_date(value: str, source: str) -> str:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: invalid arXiv version date {value!r}") from None
    if parsed is None:
        raise ValueError(f"{source}: invalid arXiv version date {value!r}")
    return _isoformat(parsed)


def _normalize_datestamp(value: str, source: str) -> str:
    try:
        if len(value) == 10:
            return date.fromisoformat(value).isoformat()
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{source}: invalid OAI datestamp {value!r}") from None
    return _isoformat(parsed)


def _malformed_record_id(element: ET.Element, index: int, source: str) -> str:
    try:
        header = _single_child(element, _OAI_NAMESPACE, "header", source)
        identifier = _single_child(header, _OAI_NAMESPACE, "identifier", source)
        return _arxiv_id(_node_text(identifier), source)
    except ValueError:
        xml = ET.tostring(element, encoding="unicode")
        return f"{source}:malformed:{content_hash(xml)[:32]}:{index}"


def _state_token(state: Mapping[str, Any], *, frozen: bool, source: str) -> str | None:
    if "resumption_token" not in state:
        return None
    if not frozen:
        raise ValueError(f"{source}: resumption token requires a frozen window")
    token = _required_text(state.get("resumption_token"), "resumption_token")
    if len(token) > 16384:
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
    element: ET.Element | None,
    name: str,
    source: str,
) -> int | None:
    if element is None or element.get(name) is None:
        return None
    return _nonnegative_integer(
        element.get(name),
        f"{source}: invalid resumptionToken {name}",
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


def _single_child(parent: ET.Element, namespace: str, name: str, source: str) -> ET.Element:
    matches = _children(parent, namespace, name)
    if len(matches) != 1:
        raise ValueError(f"{source}: expected exactly one {name} element, found {len(matches)}")
    return matches[0]


def _optional_single_child(
    parent: ET.Element,
    namespace: str,
    name: str,
    source: str,
) -> ET.Element | None:
    matches = _children(parent, namespace, name)
    if len(matches) > 1:
        raise ValueError(f"{source}: expected at most one {name} element, found {len(matches)}")
    return matches[0] if matches else None


def _optional_child_text(
    parent: ET.Element,
    namespace: str,
    name: str,
    source: str,
) -> str | None:
    child = _optional_single_child(parent, namespace, name, source)
    return _node_text(child) or None


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


def _required_field(fields: Mapping[str, Any], field: str, source: str) -> Any:
    value = fields.get(field)
    if not _text(value):
        raise ValueError(f"{source}: arXivRaw record is missing {field}")
    return value


def _optional_web_url(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    canonical = canonicalize_url(text)
    parts = urlsplit(canonical)
    return canonical if parts.scheme in {"http", "https"} and parts.hostname else ""


def _web_url(value: Any, source: str) -> str:
    url = _optional_web_url(value)
    if not url:
        raise ValueError(f"{source}: API URL must be an HTTP(S) URL")
    return url.rstrip("/")


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


def _collapse_space(value: str) -> str:
    return " ".join(value.split())


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


__all__ = ["ArxivRepositoryIdentity", "ArxivSourceAdapter"]
