"""Harvest Zenodo OAI-PMH for records with neural-model checkpoint evidence.

Zenodo recommends OAI-PMH for bulk metadata access. This adapter scans its
complete DCAT stream, which includes file distributions, and emits candidate
hints only where record metadata describes neural models and a distribution
names a model/checkpoint/weights file in a recognized serialization format. The
candidate is not a documented or released model assertion.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import unquote, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, normalize_name

Clock = Callable[[], datetime]
_OAI = "http://www.openarchives.org/OAI/2.0/"
_RECORD_ID = re.compile(r"^oai:zenodo\.org:([1-9][0-9]*)$")
_DOI = re.compile(r"10\.5281/zenodo\.([1-9][0-9]*)", re.IGNORECASE)
_FILE_MARKER = re.compile(
    r"(?:^|[._\-/ ])(?:checkpoint|ckpt|model[_ -]?weights|weights|model)(?:[._\-/ ]|$)",
    re.I,
)
_FILE_SUFFIX = re.compile(
    r"\.(?:pth\.tar|pt\.tar|tar\.gz|safetensors|msgpack|mlmodel|tflite|pt2|keras|gguf|ggml|"
    r"ckpt|hdf5|onnx|h5|pth|pt|bin)$",
    re.I,
)
_MODEL_SUFFIX = re.compile(r"\.(?:gguf|ggml|keras|mlmodel|pt2|tflite)$", re.I)
_TRAILING_FILE_MARKER = re.compile(
    r"(?:[._ -]+)(?:checkpoint|ckpt|model[_ -]?weights|weights|model)$", re.I
)
_NEURAL_SCOPE = re.compile(
    r"\b(?:deep[ -]learning|neural[ -](?:network|model|architecture|operator)|"
    r"artificial[ -]neural[ -](?:network|model))\b",
    re.I,
)
_TOKEN_TTL_SECONDS = 105
_MAX_TOKEN_LENGTH = 8_192
_MAX_PAGE_RECORDS = 50


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ZenodoOaiModelCandidatesSourceAdapter:
    """Bounded OAI-PMH traversal yielding candidate-only checkpoint evidence."""

    coverage_limitation = (
        "Scans the public Zenodo OAI-PMH DCAT stream, one provider page per call. "
        "Only records with neural-model wording and an explicitly named model, "
        "checkpoint, or weights file in a recognized model serialization format "
        "are emitted. "
        "This is candidate evidence, not a model declaration; restricted files are "
        "not downloaded. Zenodo resumption tokens expire after approximately two "
        "minutes, so persisted page checkpoints must be resumed promptly."
    )

    def __init__(
        self,
        *,
        name: str = "zenodo-oai-model-candidates",
        url: str = "https://zenodo.org/oai2d",
        max_response_bytes: int = 16 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not name.strip() or not _https_url(url):
            raise ValueError("source name must be non-empty and OAI URL must be Zenodo HTTPS")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.name = name
        self.url = canonicalize_url(url)
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "zenodo-oai-dcat-model-candidates-v1",
                "url": self.url,
                "metadata_prefix": "dcat",
                "max_response_bytes": max_response_bytes,
                "admission": "neural model wording plus named model/checkpoint/weights file",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        token = state.get("resumption_token")
        if token is not None:
            if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_LENGTH:
                raise ValueError(f"{self.name}: invalid OAI-PMH resumption token")
            issued_at = _parse_time(state.get("token_issued_at"), self.name)
            age = (self.clock().astimezone(UTC) - issued_at).total_seconds()
            if age >= _TOKEN_TTL_SECONDS:
                raise ValueError(
                    f"{self.name}: resumption token is near Zenodo's two-minute expiry; "
                    "restart the harvest from an empty checkpoint"
                )
            params = {"verb": "ListRecords", "resumptionToken": token}
        else:
            if state:
                raise ValueError(f"{self.name}: unexpected checkpoint without a resumption token")
            params = {"verb": "ListRecords", "metadataPrefix": "dcat"}

        response: HttpResponse = self.client.get(
            self.url,
            params=params,
            headers={"Accept": "application/xml, text/xml;q=0.9"},
        )
        response_received_at = self.clock()
        if response.status != 200:
            raise ValueError(f"{self.name}: OAI-PMH returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: OAI-PMH response exceeds {self.max_response_bytes} bytes"
            )
        root = _parse_xml(response.body, self.name)
        errors = [node for node in root if _local(node.tag) == "error"]
        if errors:
            detail = "; ".join(
                f"{node.get('code', 'unknown')}: {''.join(node.itertext()).strip()}"
                for node in errors
            )
            raise ValueError(f"{self.name}: OAI-PMH ListRecords error: {detail}")
        listing = next((node for node in root if _local(node.tag) == "ListRecords"), None)
        if listing is None:
            raise ValueError(f"{self.name}: response is missing OAI-PMH ListRecords")
        raw_records = [node for node in listing if _local(node.tag) == "record"]
        if len(raw_records) > _MAX_PAGE_RECORDS:
            raise ValueError(
                f"{self.name}: OAI-PMH returned {len(raw_records)} records; "
                f"documented maximum is {_MAX_PAGE_RECORDS}"
            )
        records = tuple(
            candidate
            for raw_record in raw_records
            if (candidate := self._candidate(raw_record)) is not None
        )
        token_node = next(
            (node for node in listing if _local(node.tag) == "resumptionToken"), None
        )
        next_token = (token_node.text or "").strip() if token_node is not None else ""
        pages_seen = _counter(state.get("pages_seen", 0), "pages_seen", self.name) + 1
        records_seen = _counter(state.get("records_seen", 0), "records_seen", self.name)
        records_seen += len(raw_records)
        candidates_seen = _counter(
            state.get("candidates_seen", 0), "candidates_seen", self.name
        ) + len(records)
        if next_token:
            if next_token == token:
                raise ValueError(f"{self.name}: OAI-PMH repeated its resumption token")
            next_state = {
                "resumption_token": next_token,
                # The token is already aging while we parse potentially large
                # pages. Timestamp it at response receipt so processing time
                # cannot make the checkpoint look younger than it is.
                "token_issued_at": _isoformat(response_received_at),
                "pages_seen": pages_seen,
                "records_seen": records_seen,
                "candidates_seen": candidates_seen,
            }
            complete = False
        else:
            next_state = {
                "completed_at": _isoformat(self.clock()),
                "pages_seen": pages_seen,
                "records_seen": records_seen,
                "candidates_seen": candidates_seen,
            }
            complete = True
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=complete,
            upstream_count=None,
            authoritative_snapshot=False,
        )

    def _candidate(self, raw_record: ET.Element) -> SourceRecord | None:
        header = next((node for node in raw_record if _local(node.tag) == "header"), None)
        if header is None or header.get("status") == "deleted":
            return None
        identifier = _child_text(header, "identifier")
        match = _RECORD_ID.fullmatch(identifier)
        if match is None:
            raise ValueError(f"{self.name}: invalid Zenodo OAI identifier {identifier!r}")
        record_id = match.group(1)
        metadata = next((node for node in raw_record if _local(node.tag) == "metadata"), None)
        if metadata is None:
            return None

        strings: dict[str, list[str]] = {}
        for element in metadata.iter():
            key = _local(element.tag).casefold()
            if key in {"title", "description", "keyword", "subject", "type", "format"} and (
                value := _element_value(element)
            ):
                strings.setdefault(key, []).append(value)
        title_values = strings.get("title", [])
        title = title_values[0] if title_values else f"Zenodo record {record_id}"
        context = "\n".join(
            value
            for key in ("title", "description", "keyword", "subject", "type", "format")
            for value in strings.get(key, [])
        )
        if _NEURAL_SCOPE.search(context) is None:
            return None

        matching_files: list[tuple[str, str]] = []
        all_file_links: list[tuple[str, str]] = []
        for distribution in metadata.iter():
            if not _is_distribution(distribution):
                continue
            label_parts: list[str] = []
            urls: list[str] = []
            about_url = _attribute_value(distribution, "about")
            if about_url and _zenodo_distribution_file_url(about_url):
                urls.append(canonicalize_url(about_url))
            for element in distribution.iter():
                key = _local(element.tag).casefold()
                if key in {"title", "name", "format", "mediatype"} and (
                    value := _element_value(element)
                ):
                    label_parts.append(value)
                if (
                    key in {"downloadurl", "accessurl"}
                    and (value := _element_value(element))
                    and _zenodo_file_url(value)
                ):
                    urls.append(canonicalize_url(value))
            urls = list(dict.fromkeys(urls))
            label = " ".join(label_parts)
            file_evidence = (label, *(urlsplit(url).path for url in urls))
            if any(
                _is_model_file_evidence(value)
                for value in file_evidence
            ):
                matching_files.extend((url, label or url) for url in urls)
            all_file_links.extend((url, label or url) for url in urls)
        # DCAT permits the distribution object itself to be expressed as an RDF
        # resource link, without an inline dcat:Distribution description.
        for element in metadata.iter():
            if _local(element.tag).casefold() != "distribution":
                continue
            value = _attribute_value(element, "resource")
            if not value or not _zenodo_distribution_file_url(value):
                continue
            url = canonicalize_url(value)
            label = urlsplit(url).path.rsplit("/", 1)[-1]
            all_file_links.append((url, label))
            if _is_model_file_evidence(label):
                matching_files.append((url, label))
        all_file_links = list(dict.fromkeys(all_file_links))
        matching_files = list(dict.fromkeys(matching_files))
        if not matching_files:
            return None

        context_key = f" {normalize_name(context)} "
        models: list[ModelHint] = []
        candidate_ids_by_url: dict[str, list[str]] = {}
        for index, (url, _) in enumerate(matching_files):
            filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
            stem = _FILE_SUFFIX.sub("", filename)
            candidate_name = _TRAILING_FILE_MARKER.sub("", stem).replace("_", " ").strip()
            candidate_handle = candidate_name
            normalized = normalize_name(candidate_name)
            name_tokens = normalized.split()
            context_match = re.search(
                r"(?<!\w)" + r"[\W_]+".join(re.escape(token) for token in name_tokens) + r"(?!\w)",
                context,
                re.IGNORECASE,
            )
            if (
                not normalized
                or len(normalized) < 3
                or f" {normalized} " not in context_key
                or normalized in {"model", "weights", "checkpoint", "data"}
                or context_match is None
            ):
                continue
            candidate_name = context_match.group(0)
            # Context matching uses a punctuation-folded form, but local
            # candidate IDs must preserve the exact source filename stem.
            # Otherwise handles such as ``Alpha-Net`` and ``Alpha.Net`` can
            # collapse into one candidate despite naming separate files.
            local_id = f"file-candidate:{content_hash(candidate_handle)[:16]}"
            if local_id not in {model.local_id for model in models}:
                models.append(
                    ModelHint(
                        local_id=local_id,
                        name=candidate_name,
                        status=ModelStatus.CANDIDATE,
                        confidence=0.72,
                        locator=f"$.dcat:Distribution[{index}]",
                    )
                )
            candidate_ids_by_url.setdefault(url, []).append(local_id)
        if not models:
            return None

        links = [
            Link(url, relation="artifact_file", locator="$.dcat:Distribution", crawl=False)
            for url, _ in all_file_links
        ]
        links.extend(
            Link(
                url,
                relation="checkpoint",
                locator="$.dcat:Distribution",
                crawl=False,
                model_local_ids=tuple(candidate_ids_by_url.get(url, ())),
            )
            for url, _ in matching_files
        )
        doi = next(
            (
                value
                for element in metadata.iter()
                if _local(element.tag).casefold() == "identifier"
                and (value := _element_value(element))
                and _DOI.search(value)
            ),
            None,
        )
        identifiers = [Identifier("zenodo:record", record_id)]
        if doi_match := (_DOI.search(doi) if doi else None):
            normalized_doi = f"10.5281/zenodo.{doi_match.group(1)}"
            identifiers.append(Identifier("doi", normalized_doi.casefold()))
            page_url = f"https://doi.org/{normalized_doi}"
        else:
            page_url = f"https://zenodo.org/records/{record_id}"
        links.insert(0, Link(page_url, relation="catalog_page", locator="$.oai.identifier"))
        return SourceRecord(
            source_record_id=f"record:{record_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=page_url,
            title=title,
            raw={
                "oai_identifier": identifier,
                "oai_datestamp": _child_text(header, "datestamp"),
                "oai_set_specs": [
                    (node.text or "").strip()
                    for node in header
                    if _local(node.tag) == "setSpec" and (node.text or "").strip()
                ],
                "candidate_signal": (
                    "neural metadata plus named model/checkpoint/weights distribution"
                ),
                "checkpoint_files": [label for _, label in matching_files],
            },
            text="\n\n".join((title, context)),
            identifiers=tuple(identifiers),
            links=tuple(sorted(set(links), key=lambda link: (link.url, link.relation))),
            models=tuple(models),
        )


def _parse_xml(body: bytes, source: str) -> ET.Element:
    try:
        return ET.fromstring(body)
    except ET.ParseError as error:
        raise ValueError(f"{source}: response is not valid XML") from error


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def _child_text(parent: ET.Element, name: str) -> str:
    child = next((node for node in parent if _local(node.tag) == name), None)
    return (child.text or "").strip() if child is not None else ""


def _element_value(element: ET.Element) -> str:
    text = " ".join(part.strip() for part in element.itertext() if part.strip())
    if text:
        return text
    return next(
        (value.strip() for key, value in element.attrib.items() if _local(key) == "resource"),
        "",
    )


def _attribute_value(element: ET.Element, name: str) -> str:
    return next(
        (value.strip() for key, value in element.attrib.items() if _local(key) == name),
        "",
    )


def _is_distribution(element: ET.Element) -> bool:
    if _local(element.tag) == "Distribution":
        return True
    if _local(element.tag) != "Description":
        return False
    return any(
        _local(child.tag) == "type"
        and _element_value(child).rstrip("/#").endswith(("#Distribution", "/Distribution"))
        for child in element
    )


def _zenodo_distribution_file_url(value: str) -> bool:
    parts = urlsplit(value)
    path = unquote(parts.path)
    return (
        _zenodo_file_url(value)
        and "/files/" in path
        and _FILE_SUFFIX.search(path) is not None
    )


def _is_model_file_evidence(value: str) -> bool:
    return _FILE_SUFFIX.search(value) is not None and (
        _FILE_MARKER.search(value) is not None or _MODEL_SUFFIX.search(value) is not None
    )


def _zenodo_file_url(value: str) -> bool:
    parts = urlsplit(value)
    return (
        parts.scheme == "https"
        and parts.hostname is not None
        and (parts.hostname == "zenodo.org" or parts.hostname.endswith(".zenodo.org"))
        and parts.username is None
        and parts.password is None
    )


def _https_url(value: str) -> bool:
    parts = urlsplit(value)
    return (
        parts.scheme == "https"
        and parts.hostname == "zenodo.org"
        and parts.path.rstrip("/") == "/oai2d"
        and parts.username is None
        and parts.password is None
    )


def _counter(value: Any, label: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}: checkpoint {label} must be a nonnegative integer")
    return value


def _parse_time(value: Any, source: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{source}: token checkpoint is missing token_issued_at")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{source}: token_issued_at is invalid") from error
    if result.tzinfo is None:
        raise ValueError(f"{source}: token_issued_at must include a timezone")
    return result.astimezone(UTC)


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
