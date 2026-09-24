"""Public Zenodo ``Model``-resource catalog ingestion.

Zenodo's public records API exposes a cross-domain ``resource_type.type:model``
collection.  That category includes ML models as well as non-ML research and
creative models, so this adapter preserves source-native catalog, version, DOI,
relation, and file evidence without declaring every record to be a neural model.
The normal conservative extractor can subsequently make neural-model *candidate*
claims only where the source text supports them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, quote, urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

Clock = Callable[[], datetime]
_RECORD_ID = re.compile(r"^[1-9][0-9]*$")
_DOI = re.compile(r"^10\.[0-9]{4,9}/\S+$", re.IGNORECASE)
_MAX_PAGE_SIZE = 1_000
# Zenodo documents a lower search-page ceiling for anonymous requests. This
# adapter does not authenticate, so larger configured values are only a desired
# upper bound and must be clamped to the public API's documented limit.
_ANONYMOUS_MAX_PAGE_SIZE = 25
_CHECKPOINT_FILENAME = re.compile(
    r"(?:^|[._-])(?:checkpoint|ckpt|model|model_weights|pretrain(?:ed)?|weights)(?:[._-]|$)",
    re.IGNORECASE,
)
_MODEL_WEIGHT_SUFFIX = re.compile(
    r"\.(?:bin|ckpt|h5|hdf5|model|onnx|pt|pth|safetensors)$", re.IGNORECASE
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ZenodoModelRecordsSourceAdapter:
    """Enumerate public Zenodo records explicitly classified as ``Model``.

    The source starts at the oldest record and follows only Zenodo's same-origin
    next-page URLs.  Its resource-type label is retained as catalog evidence,
    rather than promoted to a model declaration: Zenodo uses ``Model`` for such
    things as scientific ontologies and 3D objects in addition to ML systems.
    """

    coverage_limitation = (
        "Covers public Zenodo records returned by the exact "
        "resource_type.type:model query, including all published versions. "
        "Zenodo's cross-domain Model type is not "
        "itself a neural-model assertion, so the adapter preserves catalog and "
        "artifact evidence without source-declaring a model. Files are referenced "
        "but never downloaded; the public list does not supply deletion semantics "
        "or establish coverage of records categorized under another resource type."
    )

    def __init__(
        self,
        *,
        name: str = "zenodo-model-records",
        url: str = "https://zenodo.org/api/records",
        query: str = "resource_type.type:model",
        sort: str = "oldest",
        page_size: int = 100,
        max_response_bytes: int = 16 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name, "catalog URL")
        self.query = _required_text(query, "catalog query")
        self.sort = _required_text(sort, "catalog sort")
        self.page_size = _positive_int(page_size, "page_size", self.name)
        if self.page_size > _MAX_PAGE_SIZE:
            raise ValueError(
                f"{self.name}: page_size must be at most {_MAX_PAGE_SIZE:,}"
            )
        self.max_response_bytes = _positive_int(
            max_response_bytes, "max_response_bytes", self.name
        )
        self.request_page_size = min(self.page_size, _ANONYMOUS_MAX_PAGE_SIZE)
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "zenodo-model-records-v1",
                "url": self.url,
                "query": self.query,
                "sort": self.sort,
                "all_versions": True,
                "page_size": self.page_size,
                "max_response_bytes": self.max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        next_url = _text(state.get("next_url"))
        if next_url:
            page_number = _state_positive_int(state.get("page_number"), "page_number")
            raw_items_seen = _state_nonnegative_int(
                state.get("raw_items_seen"), "raw_items_seen"
            )
            scan_total = _state_nonnegative_int(state.get("scan_total"), "scan_total")
            request_url = self._next_url(next_url, expected_page=page_number)
            response: HttpResponse = self.client.get(
                request_url,
                headers={"Accept": "application/json"},
            )
        else:
            page_number = 1
            raw_items_seen = 0
            scan_total = None
            request_url = self.url
            response = self.client.get(
                self.url,
                params={
                    "q": self.query,
                    "sort": self.sort,
                    "page": page_number,
                    "size": self.request_page_size,
                    "all_versions": "true",
                },
                headers={"Accept": "application/json"},
            )

        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog response exceeds {self.max_response_bytes} bytes"
            )
        items, response_total, raw_next = self._items(response)
        if scan_total is None:
            scan_total = response_total
        elif scan_total != response_total:
            raise ValueError(
                f"{self.name}: provider total changed from {scan_total} to "
                f"{response_total}; restart the scan"
            )
        if len(items) > self.request_page_size:
            raise ValueError(
                f"{self.name}: catalog returned {len(items)} rows, exceeding "
                f"anonymous API page size {self.request_page_size}"
            )
        if not items and raw_items_seen < scan_total:
            raise ValueError(
                f"{self.name}: catalog returned an empty page before provider total"
            )

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, item in enumerate(items):
            try:
                records.append(self._record(item, index))
            except (TypeError, ValueError) as error:
                raw = dict(item) if isinstance(item, Mapping) else {"value": repr(item)[:1_000]}
                identifier = _text(item.get("id")) if isinstance(item, Mapping) else ""
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            identifier
                            or f"{self.name}:malformed:{content_hash(raw)[:32]}"
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index, "raw": raw},
                    )
                )
        raw_items_seen += len(items)
        if raw_items_seen > scan_total:
            raise ValueError(
                f"{self.name}: observed {raw_items_seen} rows beyond provider total "
                f"{scan_total}"
            )

        next_state: dict[str, Any]
        if raw_next:
            next_page = page_number + 1
            safe_next = self._next_url(raw_next, expected_page=next_page)
            if safe_next == request_url:
                raise ValueError(f"{self.name}: pagination URL did not advance")
            if raw_items_seen >= scan_total:
                raise ValueError(
                    f"{self.name}: catalog supplied a next page after its provider total"
                )
            next_state = {
                "next_url": safe_next,
                "page_number": next_page,
                "raw_items_seen": raw_items_seen,
                "scan_total": scan_total,
                "started_at": _text(state.get("started_at")) or _isoformat(self.clock()),
            }
            complete = False
        else:
            if raw_items_seen != scan_total:
                raise ValueError(
                    f"{self.name}: pagination ended after {raw_items_seen} rows, "
                    f"before provider total {scan_total}"
                )
            next_state = {
                "completed_at": _isoformat(self.clock()),
                "observed_record_count": raw_items_seen,
                "provider_total": scan_total,
                "query": self.query,
                "sort": self.sort,
            }
            complete = True

        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=scan_total if complete else None,
            authoritative_snapshot=False,
            issues=tuple(issues),
        )

    def _items(self, response: HttpResponse) -> tuple[tuple[Any, ...], int, str | None]:
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: response must be a JSON object")
        hits = payload.get("hits")
        if not isinstance(hits, Mapping):
            raise ValueError(f"{self.name}: response.hits must be an object")
        items = hits.get("hits")
        if not _is_sequence(items):
            raise ValueError(f"{self.name}: response.hits.hits must be an array")
        total = _nonnegative_int(hits.get("total"), "response.hits.total", self.name)
        links = payload.get("links")
        if not isinstance(links, Mapping):
            raise ValueError(f"{self.name}: response.links must be an object")
        next_url = _optional_url_text(links.get("next"), self.name, "response.links.next")
        return tuple(items), total, next_url

    def _next_url(self, value: str, *, expected_page: int) -> str:
        result = canonicalize_url(urljoin(self.url, value))
        expected = urlsplit(self.url)
        actual = urlsplit(result)
        if (
            actual.scheme != expected.scheme
            or actual.hostname != expected.hostname
            or actual.port != expected.port
            or actual.username is not None
            or actual.password is not None
            or actual.path.rstrip("/") != expected.path.rstrip("/")
        ):
            raise ValueError(f"{self.name}: pagination URL is outside the catalog endpoint")
        actual_pairs = parse_qsl(actual.query, keep_blank_values=True)
        expected_pairs = {
            "q": self.query,
            "sort": self.sort,
            "size": str(self.request_page_size),
            "page": str(expected_page),
            "all_versions": "true",
        }
        if len(actual_pairs) != len(expected_pairs) or dict(actual_pairs) != expected_pairs:
            raise ValueError(
                f"{self.name}: pagination URL changed the configured catalog scope"
            )
        if len({key for key, _ in actual_pairs}) != len(actual_pairs):
            raise ValueError(f"{self.name}: pagination URL contains duplicate parameters")
        return result

    def _record(self, item: Any, index: int) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError(f"catalog item {index} is not an object")
        record_id = _record_id(item.get("id"), self.name, f"catalog item {index} id")
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"catalog item {record_id} has no metadata object")
        resource_type = metadata.get("resource_type")
        if (
            not isinstance(resource_type, Mapping)
            or _text(resource_type.get("type")).casefold() != "model"
        ):
            raise ValueError(f"catalog item {record_id} is not typed as a Zenodo Model")
        links = item.get("links")
        if not isinstance(links, Mapping):
            raise ValueError(f"catalog item {record_id} has no links object")
        page_url = _required_url(
            links.get("self_html"), self.name, f"catalog item {record_id} page URL"
        )
        metadata_url = _required_url(
            links.get("self"), self.name, f"catalog item {record_id} metadata URL"
        )
        title = _text(metadata.get("title")) or _text(item.get("title")) or record_id
        description = _markup_text(metadata.get("description"))
        record_identifiers: list[Identifier] = [Identifier("zenodo:record", record_id)]
        links_out = [
            Link(page_url, relation="catalog_page", locator="$.links.self_html"),
            Link(metadata_url, relation="metadata", locator="$.links.self"),
        ]
        for locator, raw_doi, relation in (
            ("$.doi_url", item.get("doi_url"), "doi"),
            ("$.links.doi", links.get("doi"), "doi"),
            ("$.conceptdoi", item.get("conceptdoi"), "concept_doi"),
        ):
            doi = _doi(raw_doi)
            if doi is None:
                continue
            if relation == "doi":
                record_identifiers.append(Identifier("doi", doi))
            links_out.append(Link(_doi_url(doi), relation=relation, locator=locator))
        # Zenodo exposes both human-facing and API links for the concept parent,
        # latest version, and version list. Keep each source-provided locator so
        # downstream crawlers can follow the version graph without guessing URLs.
        for key, relation in (
            ("parent", "version_parent_metadata"),
            ("parent_html", "version_parent"),
            ("latest", "latest_version_metadata"),
            ("latest_html", "latest_version"),
            ("versions", "versions"),
        ):
            if url := _optional_web_url(links.get(key)):
                links_out.append(Link(url, relation=relation, locator=f"$.links.{key}"))
        for related_index, related in enumerate(_sequence(metadata.get("related_identifiers"))):
            if not isinstance(related, Mapping):
                continue
            related_identifier = related.get("identifier")
            relation = _related_identifier_relation(
                related.get("relation"), related.get("resource_type")
            )
            # A DOI explicitly declared IsIdenticalTo is a source-asserted
            # identity bridge. Other related DOIs describe versions,
            # publications, datasets, or citations and must remain links only.
            if relation == "identical_to" and (doi := _doi(related_identifier)):
                record_identifiers.append(Identifier("doi", doi))
            if url := _related_url(related_identifier):
                links_out.append(
                    Link(
                        url,
                        relation=relation,
                        locator=f"$.metadata.related_identifiers[{related_index}].identifier",
                    )
                )
        for file_index, file in enumerate(_sequence(item.get("files"))):
            if not isinstance(file, Mapping):
                continue
            file_links = file.get("links")
            if not isinstance(file_links, Mapping):
                continue
            if url := _optional_web_url(file_links.get("self")):
                # A filename can provide candidate-level checkpoint evidence,
                # but it never changes the Zenodo resource type or declares the
                # containing record to be a neural model. Require both an
                # explicit checkpoint/weights token and a model-weight format.
                key = _text(file.get("key"))
                if _CHECKPOINT_FILENAME.search(key) and _MODEL_WEIGHT_SUFFIX.search(key):
                    links_out.append(
                        Link(
                            url,
                            relation="checkpoint",
                            locator=f"$.files[{file_index}].key",
                            crawl=False,
                        )
                    )
                links_out.append(
                    Link(
                        url,
                        relation="artifact_file",
                        locator=f"$.files[{file_index}].links.self",
                        crawl=False,
                    )
                )
        return SourceRecord(
            source_record_id=f"record:{record_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=page_url,
            title=title,
            raw=dict(item),
            text="\n\n".join(value for value in (title, description) if value),
            published_at=(
                _text(metadata.get("publication_date"))
                or _text(item.get("created"))
                or None
            ),
            modified_at=_text(item.get("modified")) or _text(item.get("updated")) or None,
            identifiers=tuple(dict.fromkeys(record_identifiers)),
            links=_unique_links(links_out),
        )


class _MarkupTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if value := data.strip():
            self.parts.append(value)


def _markup_text(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parser = _MarkupTextParser()
    parser.feed(text)
    parser.close()
    return " ".join(parser.parts)


def _related_relation(value: Any) -> str:
    resource_type = _text(value).casefold()
    if resource_type in {"publication", "image", "poster", "presentation", "lesson"}:
        return "related_publication"
    if resource_type == "software":
        return "related_code"
    if resource_type == "dataset":
        return "related_dataset"
    return "related_resource"


def _related_identifier_relation(value: Any, resource_type: Any) -> str:
    """Preserve the relation asserted by the record, with type fallback.

    Zenodo's related identifier schema has a controlled ``relation`` value
    (for example ``IsSupplementTo`` or ``IsNewVersionOf``). Resource type is
    useful evidence too, but alone it loses whether the record cites, extends,
    or is a version of the linked resource.
    """
    relation = re.sub(r"[^a-z]", "", _text(value).casefold())
    relation_names = {
        "isnewversionof": "version_of",
        "ispreviousversionof": "previous_version_of",
        "isvariantformof": "variant_of",
        "isidenticalto": "identical_to",
        "isderivedfrom": "derived_from",
        "issupplementto": "supplements",
        "issupplementedby": "supplemented_by",
        "ispartof": "part_of",
        "haspart": "has_part",
        "iscitedby": "cited_by",
        "cites": "cites",
        "isreferencedby": "referenced_by",
        "references": "references",
        "isdocumentedby": "documented_by",
        "iscompiledby": "compiled_by",
        "isrequiredby": "required_by",
        "continues": "continues",
        "iscontinuedby": "continued_by",
    }
    return relation_names.get(relation, _related_relation(resource_type))


def _related_url(value: Any) -> str | None:
    if url := _optional_web_url(value):
        return url
    if doi := _doi(value):
        return _doi_url(doi)
    return None


def _doi(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    parsed = urlsplit(text)
    if parsed.scheme.casefold() in {"http", "https"} and parsed.hostname:
        host = parsed.hostname.casefold()
        if host == "doi.org" or host.endswith(".doi.org"):
            text = parsed.path.lstrip("/")
    if text.casefold().startswith("doi:"):
        text = text[4:].strip()
    if not _DOI.fullmatch(text):
        return None
    return text.casefold()


def _doi_url(doi: str) -> str:
    return canonicalize_url(f"https://doi.org/{quote(doi, safe='/():._-')}")


def _record_id(value: Any, source: str, label: str) -> str:
    text = _record_id_text(value)
    if not _RECORD_ID.fullmatch(text):
        raise ValueError(f"{source}: {label} must be a positive Zenodo record ID")
    return text


def _record_id_text(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    return _text(value)


def _required_url(value: Any, source: str, label: str) -> str:
    result = _optional_web_url(value)
    if result is None:
        raise ValueError(f"{source}: {label} must be an absolute HTTP(S) URL")
    return result


def _optional_web_url(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    result = canonicalize_url(text)
    parsed = urlsplit(result)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return result


def _optional_url_text(value: Any, source: str, label: str) -> str | None:
    text = _text(value)
    if not text:
        return None
    if _optional_web_url(text) is None:
        raise ValueError(f"{source}: {label} must be an absolute HTTP(S) URL")
    return text


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda link: (link.url, link.relation, link.locator or "", link.crawl),
        )
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _sequence(value: Any) -> Sequence[Any]:
    return value if _is_sequence(value) else ()


def _nonnegative_int(value: Any, label: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}: {label} must be a nonnegative integer")
    return value


def _state_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"checkpoint {label} must be a nonnegative integer")
    return value


def _state_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"checkpoint {label} must be a positive integer")
    return value


def _positive_int(value: Any, label: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {label} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{source}: {label} must be a positive integer")
    return result


def _required_text(value: Any, label: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{label} is required")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _web_url(value: Any, source: str, label: str) -> str:
    result = _optional_web_url(value)
    if result is None:
        raise ValueError(f"{source}: {label} must be an absolute HTTP(S) URL")
    return result


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
