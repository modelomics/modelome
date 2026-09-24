"""Discover Dryad datasets with explicitly described model-weight files."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

from modelome.http import HttpClient
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_SCOPE = re.compile(
    r"\b(?:deep learning|neural network|machine learning) models?\b|\btrained models?\b",
    re.I,
)
_WEIGHT_DESCRIPTION = re.compile(
    r"\b(?:model weights|weights for (?:the |a )?(?:trained )?models?|"
    r"trained (?:neural )?network weights|checkpoint weights)\b",
    re.I,
)
_MODEL_ARCHIVE_DESCRIPTION = re.compile(
    r"\bcontains\s+(?:the\s+)?(?:current\s+)?(?:DL|deep learning)\s+models\b"
    r"(?=.*(?:\.pth|PyTorch|weights))",
    re.I | re.S,
)
_FILE_NAME = re.compile(
    r"(?:weights?|checkpoint|model).{0,80}\.(?:tar\.gz|safetensors|hdf5|onnx|gguf|"
    r"ckpt|pth|pt|bin|h5|tar|zip)$",
    re.I,
)
_FILE_PAGE_SIZE = 100
_MAX_FILE_PAGES = 100
_MAX_PAGE_SIZE = 10
_QUERIES = (
    '"deep learning models"',
    '"neural network weights"',
    '"model checkpoint"',
    '"model weights"',
)


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


class DryadModelCandidatesSourceAdapter:
    """Scan one paginated Dryad phrase search page and its exact file metadata.

    Search results are only candidate discovery. A record is emitted only when
    its own descriptive metadata describes a learned model and at least one
    listed file explicitly describes model weights or a checkpoint.
    """

    coverage_limitation = (
        "Search mode covers a fixed set of model-weight and deep-learning phrases and "
        "misses records that use other searchable wording. When dataset_doi is configured, "
        "the adapter reads only that one known record. Metadata requests are sequential. Dryad "
        "documents that API accounts receive eight times the anonymous request rate, but "
        "does not publish the anonymous numeric quota in the API guide. Files are never "
        "downloaded; provider file download links may require authentication."
    )

    def __init__(
        self,
        *,
        name: str = "dryad-model-weight-candidates",
        base_url: str = "https://datadryad.org/api/v2",
        page_size: int = 1,
        dataset_doi: str | None = None,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not name.strip() or not _is_https(base_url):
            raise ValueError("source name and HTTPS Dryad API URL are required")
        if not 1 <= int(page_size) <= _MAX_PAGE_SIZE:
            raise ValueError(f"page_size must be between 1 and {_MAX_PAGE_SIZE}")
        self.name = name
        self.base_url = canonicalize_url(base_url).rstrip("/")
        self.page_size = int(page_size)
        self.dataset_doi = _validate_doi(dataset_doi) if dataset_doi is not None else None
        self.client = client or HttpClient()
        self.checkpoint_signature = content_hash(
            {
                "adapter": "dryad-model-weight-candidates-v1",
                "base_url": self.base_url,
                "queries": _QUERIES,
                "page_size": self.page_size,
                "dataset_doi": self.dataset_doi,
                "file_page_size": _FILE_PAGE_SIZE,
                "max_file_pages": _MAX_FILE_PAGES,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if self.dataset_doi is not None:
            return self._fetch_fixed_record()
        page = _positive_int(state.get("page", 1), "page")
        query_index = _nonnegative_int(state.get("query_index", 0), "query_index")
        if query_index >= len(_QUERIES):
            raise ValueError(f"{self.name}: query_index is outside configured query set")
        query = _QUERIES[query_index]
        if state and state.get("query", query) != query:
            raise ValueError(f"{self.name}: checkpoint query does not match adapter")
        response = self.client.get(
            f"{self.base_url}/search",
            params={"q": query, "page": page, "per_page": self.page_size},
        )
        payload = _object(response.json(), "search response")
        embedded = _object(payload.get("_embedded"), "search._embedded")
        datasets = embedded.get("stash:datasets")
        if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)):
            raise ValueError(f"{self.name}: search response is missing stash:datasets")
        count = _nonnegative_int(payload.get("count"), "search.count")
        total = _nonnegative_int(payload.get("total"), "search.total")
        if len(datasets) != count or count > self.page_size:
            raise ValueError(f"{self.name}: invalid search page count")
        records: list[SourceRecord] = []
        for item in datasets:
            if not isinstance(item, Mapping):
                raise ValueError(f"{self.name}: search result must be an object")
            candidate = self._dataset(item)
            if candidate is not None:
                records.append(candidate)
        seen = _nonnegative_int(state.get("records_seen", 0), "records_seen") + count
        if seen > total:
            raise ValueError(f"{self.name}: received more search results than total")
        if count == 0 and seen < total:
            raise ValueError(f"{self.name}: search page ended before the declared total")
        query_complete = seen >= total
        complete = query_complete and query_index == len(_QUERIES) - 1
        if query_complete and not complete:
            next_state = {
                "query_index": query_index + 1,
                "query": _QUERIES[query_index + 1],
                "records_seen": 0,
            }
        elif not query_complete:
            next_state = {"query_index": query_index, "query": query, "records_seen": seen}
            next_state["page"] = page + 1
        else:
            next_state = {"query_index": query_index, "query": query, "records_seen": seen}
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=total,
        )

    def _fetch_fixed_record(self) -> SourcePage:
        assert self.dataset_doi is not None
        identifier = f"doi:{self.dataset_doi}"
        record_url = f"{self.base_url}/datasets/{quote(identifier, safe='')}"
        response = self.client.get(record_url)
        if response.status != 200:
            raise ValueError(f"{self.name}: record API returned HTTP {response.status}")
        dataset = _object(response.json(), "record response")
        if dataset.get("identifier") != identifier:
            raise ValueError(f"{self.name}: record response DOI does not match configured DOI")
        record = self._dataset(dataset)
        return SourcePage(
            records=(record,) if record is not None else (),
            next_state={"dataset_doi": self.dataset_doi},
            complete=True,
            upstream_count=1,
        )

    def _dataset(self, dataset: Mapping[str, Any]) -> SourceRecord | None:
        identifier = _text(dataset.get("identifier"), "dataset.identifier")
        if not identifier.startswith("doi:"):
            raise ValueError(f"{self.name}: unsupported dataset identifier")
        doi = identifier[4:]
        doi = _validate_doi(doi)
        title = _text(dataset.get("title"), "dataset.title")
        description = _plain_text(
            "\n".join(
                value
                for value in (
                    _optional_text(dataset.get("abstract")),
                    _optional_text(dataset.get("usageNotes")),
                )
                if value
            )
        )
        if _SCOPE.search(f"{title}\n{description}") is None:
            return None
        links = _object(dataset.get("_links"), "dataset._links")
        version_link = _object(links.get("stash:version"), "dataset stash:version")
        version_url = _same_origin_url(version_link.get("href"), self.base_url)
        files = self._version_files(version_url)
        weighted_files = [
            (item, evidence)
            for item in files
            if isinstance(item, Mapping)
            and (evidence := _weight_file_evidence(item, description)) is not None
        ]
        if not weighted_files:
            return None

        canonical_url = f"https://datadryad.org/dataset/{quote(identifier, safe='')}"
        model = ModelHint(
            local_id="dataset-model",
            name=title,
            status=ModelStatus.CANDIDATE,
            confidence=0.65,
            locator="dataset title/abstract or usage notes and file description",
        )
        file_links: list[Link] = _related_work_links(dataset.get("relatedWorks"))
        releases: list[ReleaseHint] = []
        for file, weight_evidence in weighted_files:
            filename = _text(file.get("path"), "file.path")
            file_desc = _plain_text(_optional_text(file.get("description")) or "")
            file_links_block = _object(file.get("_links"), "file._links")
            self_link = _object(file_links_block.get("self"), "file self link")
            metadata_url = _same_origin_url(self_link.get("href"), self.base_url)
            download_block = file_links_block.get("stash:download")
            if not isinstance(download_block, Mapping):
                raise ValueError(f"{self.name}: matching file has no download link")
            download_url = _same_origin_url(download_block.get("href"), self.base_url)
            file_id = urlsplit(metadata_url).path.rsplit("/", 1)[-1]
            locator = f"Dryad file {filename}: {file_desc or weight_evidence}"[:2_000]
            file_links.append(Link(metadata_url, "file_metadata", locator=locator, crawl=False))
            file_links.append(Link(download_url, "weights", locator=locator, crawl=False))
            release_identifiers = [Identifier("dryad-file", file_id)]
            digest = _optional_text(file.get("digest"))
            digest_type = _optional_text(file.get("digestType"))
            if digest and digest_type:
                release_identifiers.append(Identifier(digest_type.casefold(), digest))
            releases.append(
                ReleaseHint(
                    local_id=f"file-{file_id}",
                    model_local_id=model.local_id,
                    identifiers=tuple(release_identifiers),
                    metadata={
                        "filename": filename,
                        "size": file.get("size"),
                        "mime_type": file.get("mimeType"),
                        "digest": digest,
                        "digest_type": digest_type,
                        "download_url": download_url,
                    },
                    confidence=0.65,
                    locator=locator,
                )
            )
        return SourceRecord(
            source_record_id=f"dryad:{doi}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonical_url,
            title=title,
            raw={"dataset": dict(dataset), "files": [dict(file) for file, _ in weighted_files]},
            text=description,
            identifiers=(Identifier("doi", doi),),
            links=tuple(file_links),
            models=(model,),
            releases=tuple(releases),
        )

    def _version_files(self, version_url: str) -> list[Mapping[str, Any]]:
        endpoint = version_url.rstrip("/") + "/files"
        next_url = endpoint
        total: int | None = None
        files: list[Mapping[str, Any]] = []
        seen_urls: set[str] = set()
        for _page_number in range(_MAX_FILE_PAGES):
            if next_url in seen_urls:
                raise ValueError(f"{self.name}: repeated file pagination link")
            seen_urls.add(next_url)
            response = self.client.get(next_url)
            payload = _object(response.json(), "files response")
            page_total = _nonnegative_int(payload.get("total"), "files.total")
            count = _nonnegative_int(payload.get("count"), "files.count")
            if total is None:
                total = page_total
                if total > _FILE_PAGE_SIZE * _MAX_FILE_PAGES:
                    raise ValueError(
                        f"{self.name}: file listing exceeds configured 10,000-file cap"
                    )
            elif total != page_total:
                raise ValueError(f"{self.name}: file total changed during paginated listing")
            embedded = _object(payload.get("_embedded"), "files._embedded")
            page_files = embedded.get("stash:files")
            if (
                not isinstance(page_files, Sequence)
                or isinstance(page_files, (str, bytes))
                or len(page_files) != count
                or count > _FILE_PAGE_SIZE
            ):
                raise ValueError(f"{self.name}: files response has invalid stash:files")
            if any(not isinstance(item, Mapping) for item in page_files):
                raise ValueError(f"{self.name}: file entries must be objects")
            files.extend(page_files)
            if len(files) > total:
                raise ValueError(f"{self.name}: received more files than declared total")
            if len(files) == total:
                return files
            pagination_links = _object(payload.get("_links"), "files._links")
            next_link = pagination_links.get("next")
            if count == 0 or not isinstance(next_link, Mapping):
                raise ValueError(f"{self.name}: file listing ended before declared total")
            next_url = _same_origin_url(next_link.get("href"), self.base_url)
        raise ValueError(f"{self.name}: file listing exceeded {_MAX_FILE_PAGES} pages")


def _weight_file_evidence(file: Mapping[str, Any], dataset_description: str) -> str | None:
    name = _optional_text(file.get("path")) or ""
    desc = _plain_text(_optional_text(file.get("description")) or "")
    if _WEIGHT_DESCRIPTION.search(desc):
        return "file_description"
    if not desc and _FILE_NAME.search(name):
        return "file_name: explicit weight/checkpoint/model token and model artifact suffix"
    if _basename(name).casefold() == "_models.zip":
        note = _model_archive_note(dataset_description, _basename(name))
        if note:
            return f"dataset_usageNotes_file_section: {note}"
    if _basename(name).casefold() == "_models.zip" and _MODEL_ARCHIVE_DESCRIPTION.search(desc):
        return "file_description_contains_model_archive"
    return None


def _model_archive_note(description: str, filename: str) -> str | None:
    pattern = re.compile(
        rf"\bFile:\s*{re.escape(filename)}\s*Description:\s*(.*?)(?=\bFile:\s*|\Z)",
        re.I | re.S,
    )
    match = pattern.search(description)
    if match is None:
        return None
    detail = match.group(1).strip()
    if _MODEL_ARCHIVE_DESCRIPTION.search(detail):
        return detail[:1_000]
    return None


def _basename(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def _related_work_links(value: Any) -> list[Link]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("dataset.relatedWorks must be an array")
    links: list[Link] = []
    for index, work in enumerate(value):
        if not isinstance(work, Mapping):
            continue
        identifier = _optional_text(work.get("identifier"))
        kind = (_optional_text(work.get("identifierType")) or "").casefold()
        relation = _optional_text(work.get("relationship")) or "related"
        if not identifier:
            continue
        if kind == "doi":
            doi = identifier.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
            if not re.fullmatch(r"10\.\d{4,9}/\S+", doi, re.I):
                continue
            url = f"https://doi.org/{doi}"
        elif kind == "url":
            parts = urlsplit(identifier)
            if parts.scheme != "https" or not parts.netloc or parts.username or parts.password:
                continue
            url = canonicalize_url(identifier)
        else:
            continue
        links.append(
            Link(
                url,
                relation=f"related_work:{relation}",
                locator=f"dataset relatedWorks[{index}]",
                crawl=False,
            )
        )
    return links


def _same_origin_url(value: Any, base_url: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("missing URL")
    url = urljoin(base_url.rstrip("/") + "/", value)
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.netloc != "datadryad.org"
        or parts.username
        or parts.password
    ):
        raise ValueError("URL must be an HTTPS URL on datadryad.org")
    return canonicalize_url(url)


def _is_https(value: str) -> bool:
    parts = urlsplit(value)
    return parts.scheme == "https" and parts.netloc == "datadryad.org"


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: Any, field: str) -> int:
    number = _nonnegative_int(value, field)
    if number < 1:
        raise ValueError(f"{field} must be positive")
    return number


def _validate_doi(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"10\.\d{4,9}/\S+", value, re.I):
        raise ValueError("dataset_doi must be a valid DOI")
    return value


def _plain_text(value: str) -> str:
    parser = _HTMLText()
    parser.feed(value)
    parser.close()
    return " ".join(parser.parts)
