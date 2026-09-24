"""Pinned inventory of model entrypoints declared on the first-party PyTorch Hub listing.

This source reads only the curated ``pytorch/hub`` listing repository. It does
not clone or execute the model repositories named by a ``torch.hub.load`` call.
Each declared repository-ref-entrypoint tuple is a distinct model identity;
TorchVision declarations are excluded because its versioned weight registry is
already covered by :mod:`torchvision_weight_registry`.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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

Clock = Callable[[], datetime]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REF = re.compile(r"^[A-Za-z0-9_.\-/]+$")
_LOAD = re.compile(
    r"torch\.hub\.load\s*\(\s*(['\"])(?P<repository>[^'\"]+)\1\s*,\s*"
    r"(['\"])(?P<entrypoint>[^'\"]+)\3",
    re.DOTALL,
)
_PAGE_FIELD = re.compile(r"^\s*(?P<key>[a-z][a-z0-9_-]*)\s*\|\s*(?P<value>.*?)\s*$")
_MAX_LISTING_FILES = 10_000


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _HubDeclaration:
    repository: str
    ref: str | None
    entrypoint: str
    title: str
    summary: str
    page_path: str
    locator: str
    source_sha256: str

    @property
    def identity(self) -> str:
        ref = f":{self.ref}" if self.ref else ""
        return f"{self.repository}{ref}#{self.entrypoint}"


class TorchHubListingSourceAdapter:
    """Inventory curated PyTorch Hub model pages and literal load examples.

    The Hub page is the evidence for inclusion. Repository refs and callable
    entrypoints come only from literal ``torch.hub.load(repo, entrypoint)``
    declarations in that page. An omitted ref means the repository's default
    branch, not an inferred version. Explicit refs are retained as source-native
    Hub binding releases; no model code or checkpoint is fetched.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers literal GitHub torch.hub.load declarations on PyTorch's curated "
        "pytorch/hub model pages at one observed commit. It excludes pytorch/vision "
        "to avoid duplicating the TorchVision weight registry. It does not cover "
        "unlisted Hub repositories or infer models by crawling them."
    )

    def __init__(
        self,
        *,
        name: str = "torch-hub-listing-extra",
        repository: str = "pytorch/hub",
        branch: str = "master",
        max_archive_bytes: int = 32 * 1024 * 1024,
        max_file_bytes: int = 2 * 1024 * 1024,
        max_files: int = _MAX_LISTING_FILES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.max_archive_bytes = _positive_int(max_archive_bytes, "max_archive_bytes")
        self.max_file_bytes = _positive_int(max_file_bytes, "max_file_bytes")
        self.max_files = _positive_int(max_files, "max_files")
        self.client = client or HttpClient(max_response_bytes=self.max_archive_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "torch-hub-listing-extra-v1",
                "repository": self.repository,
                "branch": self.branch,
                "max_archive_bytes": self.max_archive_bytes,
                "max_file_bytes": self.max_file_bytes,
                "max_files": self.max_files,
                "admission": "literal torch.hub.load declarations on curated model pages",
                "excluded_repository": "pytorch/vision",
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )

    def archive_url(self, revision: str) -> str:
        return f"{self.repository_url}/archive/{quote(revision, safe='')}.zip"

    def blob_url(self, revision: str, path: str) -> str:
        return f"{self.repository_url}/blob/{quote(revision, safe='')}/{quote(path, safe='/')}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
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
            self.archive_url(revision),
            headers={"Accept": "application/zip"},
        )
        if archive_response.status != 200:
            raise ValueError(
                f"{self.name}: listing archive returned HTTP {archive_response.status}"
            )
        if len(archive_response.body) > self.max_archive_bytes:
            raise ValueError(f"{self.name}: listing archive exceeds {self.max_archive_bytes} bytes")
        declarations, file_count = _parse_listing_archive(
            archive_response.body,
            revision=revision,
            source=self.name,
            max_file_bytes=self.max_file_bytes,
            max_files=self.max_files,
        )
        records = tuple(_record(group, revision, self) for group in declarations)
        if not records:
            raise ValueError(f"{self.name}: curated listing contains no eligible Hub declarations")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "archive_url": self.archive_url(revision),
            "archive_sha256": content_hash(archive_response.body),
            "listing_file_count": file_count,
            "model_count": len(records),
            "explicit_ref_count": sum(
                item.ref is not None for group in declarations for item in group
            ),
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
        response: HttpResponse = self.client.get(
            self.commit_url,
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response


def _parse_listing_archive(
    archive: bytes,
    *,
    revision: str,
    source: str,
    max_file_bytes: int,
    max_files: int,
) -> tuple[tuple[tuple[_HubDeclaration, ...], ...], int]:
    try:
        package = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as error:
        raise ValueError(f"{source}: listing archive is not a ZIP file") from error
    pages: list[_HubDeclaration] = []
    count = 0
    roots: set[str] = set()
    with package:
        for info in package.infolist():
            if info.is_dir() or not info.filename.casefold().endswith(".md"):
                continue
            path = _archive_path(info.filename, source)
            if "/" not in path:
                continue
            root, relative = path.split("/", 1)
            roots.add(root)
            count += 1
            if count > max_files:
                raise ValueError(
                    f"{source}: listing archive has more than {max_files} markdown files"
                )
            if info.file_size > max_file_bytes:
                raise ValueError(
                    f"{source}: listing page {relative!r} exceeds {max_file_bytes} bytes"
                )
            if not relative or "/" in relative:
                continue
            text = package.read(info).decode("utf-8", errors="strict")
            fields = _page_fields(text)
            if fields.get("layout") != "hub_detail" or not fields.get("title"):
                continue
            for match in _LOAD.finditer(text):
                repository, ref = _parse_repo_ref(match.group("repository"), source)
                if repository.casefold() == "pytorch/vision":
                    continue
                entrypoint = match.group("entrypoint").strip()
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", entrypoint):
                    continue
                pages.append(
                    _HubDeclaration(
                        repository=repository,
                        ref=ref,
                        entrypoint=entrypoint,
                        title=fields["title"],
                        summary=fields.get("summary", ""),
                        page_path=relative,
                        locator=f"{relative}:torch.hub.load:{entrypoint}",
                        source_sha256=content_hash(text.encode()),
                    )
                )
    if len(roots) != 1:
        raise ValueError(f"{source}: listing archive must contain one repository root")
    if count == 0:
        raise ValueError(f"{source}: listing archive has no markdown pages")
    grouped: dict[str, dict[str, _HubDeclaration]] = {}
    for item in pages:
        grouped.setdefault(item.identity, {})[item.page_path] = item
    groups = tuple(
        tuple(items[path] for path in sorted(items)) for _identity, items in sorted(grouped.items())
    )
    return groups, count


def _page_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = _PAGE_FIELD.match(line)
        if match:
            fields[match.group("key")] = match.group("value").strip()
    return fields


def _parse_repo_ref(value: str, source: str) -> tuple[str, str | None]:
    value = value.strip()
    repository, separator, ref = value.partition(":")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError(f"{source}: Hub load declaration has an invalid GitHub repository")
    if separator and (
        not ref or not _REF.fullmatch(ref) or ref.startswith("/") or ref.endswith("/")
    ):
        raise ValueError(f"{source}: Hub load declaration has an invalid repository ref")
    return repository, ref if separator else None


def _record(
    group: tuple[_HubDeclaration, ...], revision: str, adapter: TorchHubListingSourceAdapter
) -> SourceRecord:
    declaration = group[0]
    identity = declaration.identity
    local_id = f"hub:{identity}"
    model = ModelHint(
        local_id=local_id,
        name=declaration.entrypoint,
        identifiers=(Identifier("pytorch:hub-load", identity),),
        status=ModelStatus.DOCUMENTED,
        locator=declaration.locator,
    )
    ref = declaration.ref
    releases = ()
    if ref is not None:
        releases = (
            ReleaseHint(
                local_id=f"hub-ref:{ref}",
                model_local_id=local_id,
                version=ref,
                identifiers=(Identifier("pytorch:hub-binding", identity),),
                metadata={
                    "repository": declaration.repository,
                    "ref": ref,
                    "evidence": "torch.hub.load declaration",
                },
                locator=declaration.locator,
            ),
        )
    repo_url = f"https://github.com/{declaration.repository}"
    ref_url = (
        f"https://github.com/{declaration.repository}/tree/{quote(ref, safe='/')}"
        if ref
        else repo_url
    )
    listing_url = adapter.blob_url(revision, declaration.page_path)
    links = [
        Link(
            listing_url,
            relation="hub_listing",
            locator=declaration.locator,
            crawl=False,
            model_local_ids=(local_id,),
        ),
        Link(
            ref_url,
            relation="source_repository",
            locator=declaration.locator,
            crawl=False,
            model_local_ids=(local_id,),
        ),
    ]
    for page in group[1:]:
        links.append(
            Link(
                adapter.blob_url(revision, page.page_path),
                relation="hub_listing",
                locator=page.locator,
                crawl=False,
                model_local_ids=(local_id,),
            )
        )
    title = (
        declaration.title
        if declaration.title.casefold() == declaration.entrypoint.casefold()
        else f"{declaration.title} ({declaration.entrypoint})"
    )
    load_declarations = [
        {
            "repository": item.repository,
            "ref": item.ref,
            "entrypoint": item.entrypoint,
            "page": item.page_path,
            "locator": item.locator,
        }
        for item in group
    ]
    return SourceRecord(
        source_record_id=f"hub-load:{content_hash(identity)[:24]}",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url=canonicalize_url(listing_url),
        title=title,
        raw={
            "listing_repository": adapter.repository,
            "listing_revision": revision,
            "listing_path": declaration.page_path,
            "listing_sha256": declaration.source_sha256,
            "declared_repository": declaration.repository,
            "declared_ref": ref,
            "entrypoint": declaration.entrypoint,
            "load_declarations": load_declarations,
        },
        text=f"PyTorch Hub entrypoint: {identity}\n{declaration.summary}".strip(),
        identifiers=(Identifier("pytorch:hub-load", identity),),
        links=tuple(links),
        models=(model,),
        releases=releases,
    )


def _archive_path(value: str, source: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"{source}: listing archive contains an unsafe path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{source}: listing archive contains an unsafe path")
    return value


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _header(headers: Mapping[str, Any], key: str) -> str:
    value = headers.get(key) or headers.get(key.casefold()) or headers.get(key.title())
    return _text(value)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["TorchHubListingSourceAdapter"]
