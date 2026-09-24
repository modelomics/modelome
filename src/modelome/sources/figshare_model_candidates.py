"""Bounded Figshare OAI traversal for explicitly described model-weight files."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

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
_OAI_ID = re.compile(r"^oai:figshare\.com:article/([1-9][0-9]*)$")
_DOI = re.compile(r"^10\.\S+$", re.IGNORECASE)
_MODEL_SCOPE = re.compile(
    r"\b(?:deep learning|neural network|neural model|trained model|model weights?)\b",
    re.I,
)
_WEIGHT_CUE = re.compile(r"\b(?:weight|weights|checkpoint|checkpoints)\b", re.I)
_GENERIC_WEIGHT_CONTEXT = re.compile(
    r"\bpre[- ]?trained\s+(?:model\s+)?weights?\b|\bmodel\s+weights?\b|"
    r"\bweights?\s*\(",
    re.I,
)
_WEIGHT_FILE_SUFFIXES = (".ckpt", ".pt", ".pt2", ".pth", ".safetensors", ".bin", ".h5", ".hdf5")
_NON_WEIGHT_FILENAME = re.compile(
    r"\b(?:dataset|example|vocabulary|vocab|tokenizer|readme|metadata|prediction|generated|docking)\b",
    re.I,
)
_ASSET_CLAUSE = re.compile(
    r"(?:\bFiles?:\s*)?\d+[.,)]\s*([a-z0-9][a-z0-9_.-]*)\s*:\s*(.*?)(?="
    r"\s+\d+[.,)]\s*[a-z0-9][a-z0-9_.-]*\s*:|$)",
    re.I | re.S,
)
_FILE_URL = re.compile(r"^/files/([1-9][0-9]*)$")
_MAX_TOKEN_LENGTH = 8192
_TOKEN_TTL = timedelta(minutes=60)
_TOKEN_SAFETY = timedelta(minutes=5)
_MAX_RECORDS_PER_PAGE = 50
_MAX_ARTICLE_VERSIONS = 50


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        clean = " ".join(data.split())
        if clean:
            self.parts.append(clean)


class FigshareModelCandidatesSourceAdapter:
    """Traverse one half-open Figshare OAI date window, emitting cautious candidates.

    Figshare has no Model article type. Current OAI candidate signals trigger full
    bounded version expansion. An OAI current-version number above one also triggers
    prior-version checks, covering candidate cues removed from the latest article.
    Each candidate still needs neural/deep-learning wording and either a named
    weight/checkpoint clause or explicit pretrained-weight context paired with a
    serialization filename in its own version metadata. For the current version,
    OAI and REST URLs must match; historical version detail establishes its own
    exact article/version/file relation. Candidate status preserves uncertainty.
    """

    coverage_limitation = (
        "Covers public Figshare records with publication datestamps in the configured "
        "half-open date window [from_date, until_date). OAI exposes each article's "
        "latest version only; records whose latest metadata signals a model candidate "
        "are expanded through Figshare's public version-list and version-detail API. "
        "For other articles whose OAI current-version number is greater than one, "
        "prior versions are checked for historical-only candidates. Both paths are "
        "bounded to 50 versions per article. Articles whose version number is absent "
        "or malformed cannot be expanded through the historical-only path. Candidate "
        "files must be named as model weights or checkpoints in that version's "
        "description and match its exact public file list. A generic pretrained-weight "
        "statement can pair only with checkpoint-like serialization file suffixes. "
        "Files are never downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "figshare-model-candidates",
        oai_url: str = "https://api.figshare.com/v2/oai",
        api_url: str = "https://api.figshare.com/v2/articles",
        from_date: str,
        until_date: str,
        max_response_bytes: int = 8 * 1024 * 1024,
        max_versions_per_article: int = _MAX_ARTICLE_VERSIONS,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not name.strip() or not _https_url(oai_url) or not _https_url(api_url):
            raise ValueError("name and HTTPS Figshare endpoints are required")
        self.from_date = _date_argument(from_date, "from_date")
        self.until_date = _date_argument(until_date, "until_date")
        if self.from_date >= self.until_date:
            raise ValueError("until_date must be later than from_date")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if (
            isinstance(max_versions_per_article, bool)
            or not isinstance(max_versions_per_article, int)
            or not 1 <= max_versions_per_article <= _MAX_ARTICLE_VERSIONS
        ):
            raise ValueError(
                f"max_versions_per_article must be between 1 and {_MAX_ARTICLE_VERSIONS}"
            )
        self.name = name
        self.oai_url = canonicalize_url(oai_url)
        self.api_url = canonicalize_url(api_url).rstrip("/")
        self.max_response_bytes = max_response_bytes
        self.max_versions_per_article = max_versions_per_article
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "figshare-model-candidates-oai-mets-v2",
                "oai_url": self.oai_url,
                "api_url": self.api_url,
                "metadata_prefix": "mets",
                "from": self.from_date,
                "until": self.until_date,
                "max_response_bytes": max_response_bytes,
                "max_versions_per_article": max_versions_per_article,
                "admission": (
                    "neural metadata plus named or generic pretrained-weight file evidence"
                ),
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if state and (state.get("from") != self.from_date or state.get("until") != self.until_date):
            raise ValueError(f"{self.name}: checkpoint does not match configured date window")
        token = state.get("resumption_token")
        if token is not None:
            if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_LENGTH:
                raise ValueError(f"{self.name}: invalid OAI-PMH resumption token")
            expires = _parse_time(state.get("token_expires_at"), self.name)
            if expires - self.clock().astimezone(UTC) <= _TOKEN_SAFETY:
                raise ValueError(
                    f"{self.name}: resumption token is near Figshare's 60-minute expiry; "
                    "restart this date window from an empty checkpoint"
                )
            params = {"verb": "ListRecords", "resumptionToken": token}
        else:
            if state:
                raise ValueError(f"{self.name}: checkpoint has no resumption token")
            params = {
                "verb": "ListRecords",
                "metadataPrefix": "mets",
                "from": self.from_date,
                "until": self.until_date,
            }
        response: HttpResponse = self.client.get(
            self.oai_url,
            params=params,
            headers={"Accept": "application/xml, text/xml;q=0.9"},
        )
        received_at = self.clock().astimezone(UTC)
        self._check_response(response, "OAI-PMH")
        root = _parse_xml(response.body, self.name)
        error_nodes = [node for node in root if _local(node.tag) == "error"]
        if error_nodes:
            if len(error_nodes) == 1 and error_nodes[0].get("code") == "noRecordsMatch":
                return self._page((), state, 0, received_at)
            detail = "; ".join(
                f"{node.get('code', 'unknown')}: {''.join(node.itertext()).strip()}"
                for node in error_nodes
            )
            raise ValueError(f"{self.name}: OAI-PMH error: {detail}")
        listing = next((node for node in root if _local(node.tag) == "ListRecords"), None)
        if listing is None:
            raise ValueError(f"{self.name}: response is missing OAI-PMH ListRecords")
        raw_records = [node for node in listing if _local(node.tag) == "record"]
        if len(raw_records) > _MAX_RECORDS_PER_PAGE:
            raise ValueError(f"{self.name}: OAI-PMH returned more than 50 records")
        candidates: list[SourceRecord] = []
        for raw_record in raw_records:
            evidence = _oai_record(raw_record)
            if evidence is None:
                continue
            record_id, title, description, page_url, file_urls, version = evidence
            current_candidate_signal = _has_candidate_signal(title, description)
            if not current_candidate_signal and not _has_prior_versions(version):
                continue
            versions = self._versions(record_id)
            if version and version not in {str(item) for item in versions}:
                raise ValueError(
                    f"{self.name}: OAI version {version} is missing from article "
                    f"{record_id} version list"
                )
            for article_version in versions:
                if not current_candidate_signal and str(article_version) == version:
                    # The current OAI metadata was already checked; only prior
                    # versions can contain a historical-only candidate here.
                    continue
                article = self._article_version(record_id, article_version)
                version_title = _text(article.get("title")) or title
                version_description = _plain_text(_text(article.get("description")) or description)
                if not _has_candidate_signal(version_title, version_description):
                    continue
                named_files = _candidate_file_stems(version_description, article)
                version_urls = (
                    file_urls if str(article_version) == version else _rest_file_urls(article)
                )
                candidate = _candidate_record(
                    article,
                    record_id=record_id,
                    oai_title=version_title,
                    oai_description=version_description,
                    page_url=page_url,
                    oai_file_urls=version_urls,
                    oai_version=str(article_version),
                    named_files=named_files,
                    source=self.name,
                    api_url=self.api_url,
                )
                if candidate is not None:
                    candidates.append(candidate)
        token_node = next((node for node in listing if _local(node.tag) == "resumptionToken"), None)
        next_token = (token_node.text or "").strip() if token_node is not None else ""
        return self._page(
            tuple(candidates), state, len(raw_records), received_at, next_token, token_node
        )

    def _page(
        self,
        records: tuple[SourceRecord, ...],
        state: Mapping[str, Any],
        record_count: int,
        received_at: datetime,
        next_token: str = "",
        token_node: ET.Element | None = None,
    ) -> SourcePage:
        prior_pages = _counter(state.get("pages_seen", 0), "pages_seen", self.name)
        prior_records = _counter(state.get("records_seen", 0), "records_seen", self.name)
        prior_candidates = _counter(state.get("candidates_seen", 0), "candidates_seen", self.name)
        pages_seen = prior_pages + 1
        records_seen = prior_records + record_count
        candidates_seen = prior_candidates + len(records)
        if next_token:
            if next_token == state.get("resumption_token"):
                raise ValueError(f"{self.name}: OAI-PMH repeated its resumption token")
            expires = _token_expiry(token_node, received_at, self.name)
            next_state = {
                "from": self.from_date,
                "until": self.until_date,
                "resumption_token": next_token,
                "token_expires_at": expires.isoformat().replace("+00:00", "Z"),
                "pages_seen": pages_seen,
                "records_seen": records_seen,
                "candidates_seen": candidates_seen,
            }
            complete = False
        else:
            next_state = {
                "from": self.from_date,
                "until": self.until_date,
                "completed_at": self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z"),
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

    def _versions(self, record_id: str) -> tuple[int, ...]:
        response: HttpResponse = self.client.get(
            f"{self.api_url}/{record_id}/versions",
            headers={"Accept": "application/json"},
        )
        self._check_response(response, "article versions")
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"{self.name}: article {record_id} has no version list")
        if len(payload) > self.max_versions_per_article:
            raise ValueError(
                f"{self.name}: article {record_id} exceeds the configured version limit"
            )
        versions: list[int] = []
        for item in payload:
            if not isinstance(item, Mapping):
                raise ValueError(f"{self.name}: malformed version entry for article {record_id}")
            version = item.get("version")
            if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                raise ValueError(f"{self.name}: invalid version number for article {record_id}")
            url = item.get("url")
            expected = f"{self.api_url}/{record_id}/versions/{version}"
            if url is not None and canonicalize_url(_text(url)) != canonicalize_url(expected):
                raise ValueError(f"{self.name}: unexpected version URL for article {record_id}")
            versions.append(version)
        if len(set(versions)) != len(versions):
            raise ValueError(f"{self.name}: repeated version number for article {record_id}")
        return tuple(sorted(versions))

    def _article_version(self, record_id: str, version: int) -> Mapping[str, Any]:
        response: HttpResponse = self.client.get(
            f"{self.api_url}/{record_id}/versions/{version}",
            headers={"Accept": "application/json"},
        )
        self._check_response(response, "article version detail")
        payload = response.json()
        if (
            not isinstance(payload, Mapping)
            or str(payload.get("id", "")) != record_id
            or _text(payload.get("version")) != str(version)
        ):
            raise ValueError(f"{self.name}: article version identity does not match OAI record")
        return payload

    def _check_response(self, response: HttpResponse, endpoint: str) -> None:
        if response.status != 200:
            raise ValueError(f"{self.name}: {endpoint} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: {endpoint} response exceeds byte limit")


def _candidate_record(
    article: Mapping[str, Any],
    *,
    record_id: str,
    oai_title: str,
    oai_description: str,
    page_url: str,
    oai_file_urls: set[str],
    oai_version: str | None,
    named_files: set[str],
    source: str,
    api_url: str,
) -> SourceRecord | None:
    title = _text(article.get("title")) or oai_title
    description = _plain_text(_text(article.get("description")) or oai_description)
    if _MODEL_SCOPE.search(f"{title}\n{description}") is None:
        return None
    version = _text(article.get("version"))
    if oai_version and version and version != oai_version:
        raise ValueError(f"{source}: OAI and REST versions differ for article {record_id}")
    files = article.get("files")
    if not isinstance(files, list):
        raise ValueError(f"{source}: article {record_id} has no file array")
    matches: list[tuple[int, Mapping[str, Any], str]] = []
    for index, item in enumerate(files):
        if not isinstance(item, Mapping):
            continue
        filename = _text(item.get("name"))
        file_url = _file_url(item.get("download_url"))
        if not filename or file_url is None:
            continue
        if _filename_stem(filename) in named_files and file_url in oai_file_urls:
            matches.append((index, item, file_url))
    if not matches:
        return None
    generic_weight_context = (
        not _named_weight_files(description)
        and _GENERIC_WEIGHT_CONTEXT.search(description) is not None
    )
    version_page = _text(article.get("url_public_html")) or _text(article.get("figshare_url"))
    if not version_page:
        version_page = f"{page_url.rstrip('/')}/{version}"
    article_url = _page_url(version_page)
    local_id = f"model:figshare:{record_id}:v{version}"
    model = ModelHint(
        local_id=local_id,
        name=_model_name(title),
        identifiers=(Identifier("figshare:article", record_id),),
        status=ModelStatus.CANDIDATE,
        confidence=0.82,
        locator=(
            "version description weight context and REST serialization filename"
            if generic_weight_context
            else "version description named-file clause and REST file-name match"
        ),
    )
    identifiers = [
        Identifier("figshare:article", record_id),
        Identifier("figshare:article-version", f"{record_id}:v{version}"),
    ]
    doi = _text(article.get("doi"))
    if doi and _DOI.fullmatch(doi):
        identifiers.append(Identifier("doi", doi.casefold()))
    links = [Link(article_url, relation="catalog_page", locator="$.figshare_url")]
    matched_files: list[dict[str, Any]] = []
    for index, item, url in matches:
        matched_files.append(
            {"id": _text(item.get("id")), "name": _text(item.get("name")), "url": url}
        )
        links.append(
            Link(
                url,
                relation="checkpoint",
                locator=f"$.files[{index}].download_url",
                crawl=False,
                model_local_ids=(local_id,),
            )
        )
    return SourceRecord(
        source_record_id=f"article:{record_id}:version:{version}",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=article_url,
        title=title,
        raw={
            "provider": "Figshare",
            "article_id": record_id,
            "article_version": version or oai_version,
            "version_detail_url": f"{api_url}/{record_id}/versions/{version}",
            "defined_type_name": _text(article.get("defined_type_name")),
            "candidate_signal": (
                "model metadata identifies pretrained weights and version lists "
                "checkpoint-like files"
                if generic_weight_context
                else "model/deep-learning metadata names a weight/checkpoint file"
            ),
            "matched_files": matched_files,
            "binary_reachability_checked": False,
        },
        text=f"{title}\n\n{description}",
        published_at=_text(article.get("published_date")) or None,
        modified_at=_text(article.get("modified_date")) or None,
        identifiers=tuple(identifiers),
        links=tuple(links),
        models=(model,),
    )


def _oai_record(
    record: ET.Element,
) -> tuple[str, str, str, str, set[str], str | None] | None:
    header = next((node for node in record if _local(node.tag) == "header"), None)
    if header is None or header.get("status") == "deleted":
        return None
    identifier = next((_text(node.text) for node in header if _local(node.tag) == "identifier"), "")
    match = _OAI_ID.fullmatch(identifier)
    if match is None:
        raise ValueError(f"invalid Figshare OAI identifier {identifier!r}")
    metadata = next((node for node in record if _local(node.tag) == "metadata"), None)
    if metadata is None:
        return None
    title = ""
    description = ""
    page_url = ""
    version = None
    file_urls: set[str] = set()
    for element in metadata.iter():
        local = _local(element.tag)
        value = _element_text(element)
        if local == "title" and value and not title:
            title = value
        elif local == "description" and value and not description:
            description = value
        elif local == "relation" and value and "figshare.com/articles/" in value:
            page_url = value
        elif local == "hasPart" and value:
            if url := _file_url(value):
                file_urls.add(url)
        elif local == "hasVersion" and value:
            version = value
        elif local == "FLocat":
            href = next((v for key, v in element.attrib.items() if _local(key) == "href"), "")
            if url := _file_url(href):
                file_urls.add(url)
    if not title or not description:
        return None
    return match.group(1), title, description, page_url, file_urls, version


def _named_weight_files(description: str) -> set[str]:
    clean = _plain_text(description)
    found: set[str] = set()
    for match in _ASSET_CLAUSE.finditer(clean):
        label = normalize_name(match.group(1)).replace(" ", "")
        clause = match.group(2)
        if label and _WEIGHT_CUE.search(clause) and not re.search(r"\bdataset\b", clause, re.I):
            found.add(label)
    return found


def _has_candidate_signal(title: str, description: str) -> bool:
    return bool(
        _MODEL_SCOPE.search(f"{title}\n{description}")
        and (_named_weight_files(description) or _GENERIC_WEIGHT_CONTEXT.search(description))
    )


def _candidate_file_stems(description: str, article: Mapping[str, Any]) -> set[str]:
    named = _named_weight_files(description)
    if named or _GENERIC_WEIGHT_CONTEXT.search(description) is None:
        return named
    files = article.get("files")
    if not isinstance(files, list):
        return set()
    candidates: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            continue
        filename = _text(item.get("name"))
        lowered = filename.casefold()
        tokens = re.sub(r"[-_.]+", " ", filename)
        if (
            filename
            and lowered.endswith(_WEIGHT_FILE_SUFFIXES)
            and _NON_WEIGHT_FILENAME.search(tokens) is None
        ):
            candidates.add(_filename_stem(filename))
    return candidates


def _has_prior_versions(version: str | None) -> bool:
    return bool(version and version.isdecimal() and int(version) > 1)


def _filename_stem(filename: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", filename.rsplit("/", 1)[-1])
    return normalize_name(stem).replace(" ", "")


def _rest_file_urls(article: Mapping[str, Any]) -> set[str]:
    files = article.get("files")
    if not isinstance(files, list):
        return set()
    return {
        url
        for item in files
        if isinstance(item, Mapping)
        if (url := _file_url(item.get("download_url"))) is not None
    }


def _model_name(title: str) -> str:
    name = re.split(r"\s*[:—–-]\s*", title, maxsplit=1)[0].strip()
    return name or title


def _file_url(value: Any) -> str | None:
    raw = _text(value)
    parsed = urlsplit(raw)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"ndownloader.figshare.com", "ndownloader.figsh.com"}
        or not _FILE_URL.fullmatch(parsed.path)
        or parsed.query
        or parsed.fragment
    ):
        return None
    return canonicalize_url(raw)


def _page_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith("figshare.com")
    ):
        raise ValueError("article URL must be an HTTPS Figshare URL")
    return canonicalize_url(value)


def _date_argument(value: str, name: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must use YYYY-MM-DD") from error
    return parsed.isoformat()


def _token_expiry(node: ET.Element | None, received: datetime, source: str) -> datetime:
    if node is not None and node.get("expirationDate"):
        return _parse_time(node.get("expirationDate"), source)
    return received + _TOKEN_TTL


def _parse_time(value: Any, source: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{source}: invalid token expiration time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{source}: invalid token expiration time") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{source}: token expiration must include a timezone")
    return parsed.astimezone(UTC)


def _counter(value: Any, field: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}: checkpoint {field} must be non-negative")
    return value


def _https_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
    )


def _plain_text(value: str) -> str:
    parser = _TextParser()
    parser.feed(value)
    parser.close()
    return " ".join(parser.parts)


def _element_text(element: ET.Element) -> str:
    return " ".join(" ".join(element.itertext()).split())


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return ""


def _parse_xml(body: bytes, source: str) -> ET.Element:
    try:
        return ET.fromstring(body)
    except ET.ParseError as error:
        raise ValueError(f"{source}: invalid OAI-PMH XML: {error}") from error
