"""Historical exact TensorFlow Hub handles declared by the official site source."""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient, HttpResponse
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

_REPOSITORY = "tensorflow/tfhub.dev"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MODEL_DOCUMENT = re.compile(
    r"^assets/docs/(?P<publisher>[^/]+)/models/(?P<model>.+)/(?P<version>[0-9]+)\.md$"
)
_TITLE = re.compile(
    r"^#\s+Module\s+(?P<handle>\S+)(?:\s+(?P<summary>.*))?$", re.MULTILINE
)
_COMMENT_METADATA = re.compile(r"<!--\s*(?P<key>[a-z][a-z0-9-]*):\s*(?P<value>.*?)\s*-->", re.I)


class TensorFlowHubArchiveSourceAdapter:
    """Enumerate versioned model-card paths from TensorFlow Hub's archived site repo.

    The repository is the first-party static source tree used to publish
    ``tfhub.dev`` documentation. A commit-pinned GitHub archive bounds discovery
    to tracked files and gives each handle an immutable evidence revision. This
    records historical documentation and exact handle strings; it does not claim
    that model bytes remain available after migration or deletion.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only versioned model documentation under assets/docs/*/models in "
        "the public tensorflow/tfhub.dev source archive at one observed commit. "
        "An explicit module heading establishes the handle when publisher and "
        "version agree with the archive path; otherwise the path is the exact "
        "handle candidate. It does not assert current asset availability, "
        "enumerate untracked catalog entries, or download model bytes."
    )

    def __init__(
        self,
        *,
        name: str = "tensorflow-hub-archive",
        max_archive_bytes: int = 128 * 1024 * 1024,
        max_document_bytes: int = 2 * 1024 * 1024,
        max_documents: int = 20_000,
        max_records: int = 20_000,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        self.name = name.strip()
        self.max_archive_bytes = _positive_int(max_archive_bytes, "max_archive_bytes")
        self.max_document_bytes = _positive_int(max_document_bytes, "max_document_bytes")
        self.max_documents = _positive_int(max_documents, "max_documents")
        self.max_records = _positive_int(max_records, "max_records")
        self.client = client or HttpClient(max_response_bytes=self.max_archive_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "tensorflow-hub-archive-v2",
                "repository": _REPOSITORY,
                "branch": "master",
                "max_archive_bytes": self.max_archive_bytes,
                "max_document_bytes": self.max_document_bytes,
                "max_documents": self.max_documents,
                "max_records": self.max_records,
                "admission": (
                    "versioned docs paths; explicit module handle must match "
                    "publisher/version and path slug, allowing a leading "
                    "models/ segment"
                ),
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{_REPOSITORY}"

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/commits/master"

    def archive_url(self, revision: str) -> str:
        return f"{self.repository_url}/archive/{quote(revision, safe='')}.zip"

    def blob_url(self, revision: str, path: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(path, safe='/')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            if etag := _header(commit_response.headers, "etag"):
                next_state["commit_etag"] = etag
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        archive_response = self.client.get(
            self.archive_url(revision), headers={"Accept": "application/zip"}
        )
        if archive_response.status != 200:
            raise ValueError(f"{self.name}: source archive returned HTTP {archive_response.status}")
        if len(archive_response.body) > self.max_archive_bytes:
            raise ValueError(
                f"{self.name}: source archive exceeds {self.max_archive_bytes} bytes"
            )
        documents = _model_documents(
            archive_response.body,
            source=self.name,
            max_document_bytes=self.max_document_bytes,
            max_documents=self.max_documents,
        )
        if len(documents) > self.max_records:
            raise ValueError(f"{self.name}: archive exceeds {self.max_records} model records")
        records = tuple(
            self._record(path, document, revision) for path, document in sorted(documents.items())
        )
        if not records:
            raise ValueError(f"{self.name}: archive contains no versioned model documents")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "archive_url": self.archive_url(revision),
            "archive_sha256": content_hash(archive_response.body),
            "model_count": len(records),
            "document_count": len(documents),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _revision(self) -> tuple[str, HttpResponse]:
        response = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _record(self, path: str, document: str, revision: str) -> SourceRecord:
        match = _MODEL_DOCUMENT.fullmatch(path)
        assert match is not None
        publisher = match.group("publisher")
        model_path = match.group("model")
        version = match.group("version")
        path_handle = f"{publisher}/{model_path}/{version}"
        title_match = _TITLE.search(document)
        title_handle = (
            title_match.group("handle").rstrip(".,;:)")
            if title_match is not None
            else None
        )
        handle = path_handle
        normalized_title_handle = (
            _normalize_explicit_handle(title_handle) if title_handle is not None else None
        )
        if title_handle is not None:
            assert normalized_title_handle is not None
            title_parts = normalized_title_handle.split("/")
            if (
                len(title_parts) < 3
                or title_parts[0] != publisher
                or title_parts[-1] != version
            ):
                raise ValueError(f"{self.name}: {path} has a conflicting module heading")
            title_model_path = "/".join(title_parts[1:-1])
            if title_model_path not in {model_path, f"models/{model_path}"}:
                raise ValueError(f"{self.name}: {path} has a conflicting module heading")
            handle = normalized_title_handle

        model_key = handle.rsplit("/", 1)[0]
        locator = f"{path}: versioned model document path"
        model_local_id = f"model:{content_hash(model_key)[:24]}"
        model_identifier = Identifier("tensorflow-hub:model", model_key)
        handle_identifier = Identifier("tensorflow-hub:handle", handle)
        model_url = f"https://tfhub.dev/{handle}"
        doc_url = self.blob_url(revision, path)
        metadata = _document_metadata(document)
        release_metadata = {
            "handle": handle,
            "identity_source": (
                "explicit_module_heading" if title_handle is not None else "versioned_document_path"
            ),
            "path_handle_candidate": path_handle,
            "declared_title_handle": title_handle,
            "normalized_declared_title_handle": normalized_title_handle,
            "availability": "unverified",
            "archive_path": path,
            "metadata": metadata,
        }
        model = ModelHint(
            local_id=model_local_id,
            name=model_key.removeprefix(f"{publisher}/"),
            aliases=(publisher, model_key),
            identifiers=(model_identifier,),
            status=ModelStatus.DOCUMENTED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{content_hash(handle)[:24]}",
            model_local_id=model_local_id,
            version=version,
            revision=revision,
            identifiers=(handle_identifier,),
            metadata=release_metadata,
            locator=locator,
        )
        links = [
            Link(model_url, relation="model_page", locator=locator, crawl=False,
                 model_local_ids=(model_local_id,)),
            Link(doc_url, relation="model_card", locator=locator, crawl=False,
                 model_local_ids=(model_local_id,)),
            Link(self.repository_url, relation="source_repository", crawl=False),
        ]
        if asset := metadata.get("asset_path"):
            links.append(
                Link(asset, relation="weights", locator=locator, crawl=False,
                     model_local_ids=(model_local_id,))
            )
        summary = (title_match.group("summary") or "") if title_match is not None else ""
        text = "\n".join((handle, summary, *(f"{key}: {value}" for key, value in metadata.items())))
        return SourceRecord(
            source_record_id=f"tensorflow-hub-handle:{content_hash(handle)[:24]}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(model_url),
            title=handle,
            raw={
                "repository": _REPOSITORY,
                "revision": revision,
                "archive_path": path,
                "handle": handle,
                "path_handle_candidate": path_handle,
                "document_sha256": content_hash(document.encode()),
                "metadata": metadata,
                "availability": "unverified",
            },
            text=text,
            identifiers=(model_identifier, handle_identifier),
            links=tuple(links),
            models=(model,),
            releases=(release,),
        )


def _model_documents(
    archive: bytes,
    *,
    source: str,
    max_document_bytes: int,
    max_documents: int,
) -> dict[str, str]:
    try:
        package = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as error:
        raise ValueError(f"{source}: source archive is not a valid zip file") from error
    with package:
        found: dict[str, str] = {}
        roots: set[str] = set()
        for info in package.infolist():
            if info.is_dir():
                continue
            path = _archive_path(info.filename, source)
            parts = path.split("/", 1)
            if len(parts) != 2:
                continue
            roots.add(parts[0])
            relative = parts[1]
            if not _MODEL_DOCUMENT.fullmatch(relative):
                continue
            if len(found) >= max_documents:
                raise ValueError(f"{source}: archive has more than {max_documents} model documents")
            if info.file_size > max_document_bytes:
                raise ValueError(
                    f"{source}: model document {relative} exceeds {max_document_bytes} bytes"
                )
            content = package.read(info)
            if len(content) > max_document_bytes:
                raise ValueError(
                    f"{source}: model document {relative} exceeds {max_document_bytes} bytes"
                )
            if relative in found:
                raise ValueError(f"{source}: duplicate model document {relative}")
            try:
                found[relative] = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{source}: model document {relative} is not UTF-8") from error
        if len(roots) > 1:
            raise ValueError(f"{source}: archive has multiple root directories")
        return found


def _document_metadata(document: str) -> dict[str, str]:
    allowed = {"asset-path", "task", "network-architecture", "format", "fine-tunable", "license"}
    result = {}
    for match in _COMMENT_METADATA.finditer(document):
        key = match.group("key").lower()
        value = match.group("value").strip()
        if key not in allowed or not value:
            continue
        if key == "asset-path":
            if not value.startswith(("https://", "http://")):
                continue
            key = "asset_path"
        result[key] = value
    return result


def _normalize_explicit_handle(value: str) -> str:
    """Remove only the zero-width non-joiner spellings found in archived handles."""
    return re.sub(r"(?i)&zwnj;", "", value).replace("\u200c", "")


def _archive_path(value: str, source: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"{source}: archive contains an unsafe path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{source}: archive contains an unsafe path")
    return value


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return next((value for key, value in headers.items() if key.lower() == name.lower()), None)
