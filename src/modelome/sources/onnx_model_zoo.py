from __future__ import annotations

import html
import re
from collections.abc import Callable, Mapping
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
from modelome.normalize import canonicalize_url, content_hash, extract_urls, identifier_from_url

Clock = Callable[[], datetime]

# The maintained archival index uses Markdown table rows. A row is admitted only
# when its first cell declares a local ``validated/...`` model path; narrative
# rows and headings are intentionally not converted into model identities.
_LOCAL_MODEL_LINK = re.compile(
    r"\[(?P<name>[^\]\r\n]+)\]\((?P<path>validated/[A-Za-z0-9_.\-/]+)\)",
    re.IGNORECASE,
)
_MARKDOWN_LINK = re.compile(r"\[[^\]\r\n]*\]\((?P<target>[^)\s]+)(?:\s+[^)]*)?\)")
_HEADING = re.compile(r"^#{1,6}\s+(?P<name>.+?)\s*$")
_TAG = re.compile(r"<[^>]+>")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OnnxModelZooSourceAdapter:
    """Import every linked validated entry from ONNX Model Zoo's public index.

    The Zoo is archived rather than a current weight distribution. Each row is
    therefore retained as source-backed historical release evidence with its
    exact repository path, paper link(s), and any declared external artifact
    link(s), without downloading LFS objects or assuming a replacement Hub ID.
    """

    def __init__(
        self,
        *,
        name: str = "onnx-model-zoo",
        url: str = "https://raw.githubusercontent.com/onnx/models/main/README.md",
        repository_url: str = "https://github.com/onnx/models",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 10_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, "catalog URL")
        self.repository_url = _web_url(repository_url, "repository URL")
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "onnx-model-zoo-v2",
                "url": self.url,
                "repository_url": self.repository_url,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "linked validated model paths and same-folder first-party asset links",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        headers = {"Accept": "text/markdown,text/plain"}
        if etag := _text(state.get("etag")):
            headers["If-None-Match"] = etag
        if last_modified := _text(state.get("http_last_modified")):
            headers["If-Modified-Since"] = last_modified
        response: HttpResponse = self.client.get(self.url, headers=headers)
        checked_at = _isoformat(self.clock())
        if response.status == 304:
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(records=(), next_state=next_state, complete=True)
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: catalog exceeds {self.max_response_bytes} bytes")

        document = response.text()
        catalog_hash = content_hash(response.body)
        entries, ignored_table_rows = self._entries(document)
        if not entries:
            raise ValueError(f"{self.name}: catalog contained no linked validated model rows")
        if len(entries) > self.max_entries:
            raise ValueError(f"{self.name}: catalog exceeds {self.max_entries} model entries")
        records = tuple(self._record(entry, catalog_hash=catalog_hash) for entry in entries)
        next_state: dict[str, Any] = {
            "checked_at": checked_at,
            "content_hash": catalog_hash,
            "entry_count": len(entries),
            "ignored_table_rows": ignored_table_rows,
        }
        if etag := _header(response.headers, "etag"):
            next_state["etag"] = etag
        if last_modified := _header(response.headers, "last-modified"):
            next_state["http_last_modified"] = last_modified
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(entries),
            authoritative_snapshot=True,
        )

    def _entries(self, document: str) -> tuple[tuple[_ModelEntry, ...], int]:
        section = "ONNX Model Zoo"
        entries: dict[str, _ModelEntry] = {}
        ignored_table_rows = 0
        for line_number, line in enumerate(document.splitlines(), start=1):
            if heading := _HEADING.match(line):
                section = _plain_text(heading.group("name")) or section
                continue
            if not line.lstrip().startswith("|"):
                continue
            cells = _table_cells(line)
            if len(cells) < 2 or _table_separator(cells):
                continue
            local = _LOCAL_MODEL_LINK.search(cells[0])
            if local is None:
                ignored_table_rows += 1
                continue
            path = _safe_path(local.group("path"))
            name = _plain_text(local.group("name"))
            if not name:
                ignored_table_rows += 1
                continue
            references = tuple(
                dict.fromkeys(url for cell in cells[1:] for url in extract_urls(cell))
            )
            assets = tuple(
                dict.fromkeys(
                    asset
                    for cell in cells[1:]
                    for target in _MARKDOWN_LINK.findall(cell)
                    if (asset := _first_party_asset(target, path, self.repository_url))
                )
            )
            entry = _ModelEntry(
                name=name,
                path=path,
                section=section,
                description=_plain_text(cells[2]) if len(cells) > 2 else "",
                references=references,
                assets=assets,
                locator=f"line:{line_number}",
            )
            existing = entries.get(path)
            if existing is None:
                entries[path] = entry
            elif name != existing.name:
                entries[path] = existing.with_alias(name)
        return tuple(entries.values()), ignored_table_rows

    def _record(self, entry: _ModelEntry, *, catalog_hash: str) -> SourceRecord:
        encoded_path = quote(entry.path, safe="/")
        model_url = canonicalize_url(f"{self.repository_url}/tree/main/{encoded_path}")
        artifact_identifier = Identifier("onnx:model-zoo-artifact", entry.path)
        model_identifier = Identifier("onnx:model-zoo", entry.path)
        model = ModelHint(
            local_id=f"{entry.path}#model",
            name=entry.name,
            identifiers=(model_identifier,),
            aliases=entry.aliases,
            status=ModelStatus.RELEASED,
            locator=entry.locator,
        )
        links = [
            Link(model_url, relation="model_page", locator=entry.locator),
            Link(self.repository_url, relation="source_repository", locator=entry.locator),
        ]
        for index, url in enumerate(entry.references):
            relation = _reference_relation(url)
            links.append(
                Link(
                    url,
                    relation=relation,
                    locator=f"{entry.locator}:reference[{index}]",
                )
            )
        for index, asset_path in enumerate(entry.assets):
            asset_url = canonicalize_url(
                f"{self.repository_url}/blob/main/{quote(asset_path, safe='/')}"
            )
            suffix = asset_path.rsplit("/", 1)[-1].casefold()
            relation = "inference_artifact" if suffix.endswith(".onnx") else "artifact_reference"
            links.append(
                Link(asset_url, relation=relation, locator=f"{entry.locator}:asset[{index}]")
            )
        release = ReleaseHint(
            local_id=f"{entry.path}#release",
            model_local_id=model.local_id,
            identifiers=(model_identifier,),
            metadata={
                "format": "ONNX",
                "catalog_revision_sha256": catalog_hash,
                "historical_archive": True,
                "model_path": entry.path,
            },
            locator=entry.locator,
        )
        text_parts = [entry.name, entry.section]
        if entry.description:
            text_parts.append(entry.description)
        return SourceRecord(
            source_record_id=f"model:{entry.path}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=model_url,
            title=entry.name,
            raw={
                "catalog_url": self.url,
                "repository_url": self.repository_url,
                "catalog_revision_sha256": catalog_hash,
                "model_path": entry.path,
                "section": entry.section,
                "description": entry.description,
                "references": list(entry.references),
                "first_party_assets": list(entry.assets),
                "locator": entry.locator,
                "historical_archive": True,
            },
            text="\n".join(text_parts),
            identifiers=(artifact_identifier,),
            links=tuple(_unique_links(links)),
            models=(model,),
            releases=(release,),
        )


class _ModelEntry:
    __slots__ = (
        "name",
        "path",
        "section",
        "description",
        "references",
        "assets",
        "locator",
        "aliases",
    )

    def __init__(
        self,
        *,
        name: str,
        path: str,
        section: str,
        description: str,
        references: tuple[str, ...],
        assets: tuple[str, ...] = (),
        locator: str,
        aliases: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.path = path
        self.section = section
        self.description = description
        self.references = references
        self.assets = assets
        self.locator = locator
        self.aliases = aliases

    def with_alias(self, alias: str) -> _ModelEntry:
        aliases = self.aliases if alias in self.aliases else (*self.aliases, alias)
        return _ModelEntry(
            name=self.name,
            path=self.path,
            section=self.section,
            description=self.description,
            references=self.references,
            assets=self.assets,
            locator=self.locator,
            aliases=aliases,
        )


def _table_cells(line: str) -> tuple[str, ...]:
    values = line.strip().split("|")
    if values and not values[0].strip():
        values = values[1:]
    if values and not values[-1].strip():
        values = values[:-1]
    return tuple(value.strip() for value in values)


def _table_separator(cells: tuple[str, ...]) -> bool:
    return bool(cells) and all(bool(cell) and set(cell) <= {"-", ":", " "} for cell in cells)


def _safe_path(value: str) -> str:
    path = value.strip().strip("/")
    if not path.casefold().startswith("validated/"):
        raise ValueError("model path is outside the validated catalog")
    unsafe_segment = any(part in {"", ".", ".."} for part in path.split("/"))
    if "/../" in f"/{path}/" or "//" in path or unsafe_segment:
        raise ValueError("model path is unsafe")
    return path


def _first_party_asset(target: str, model_path: str, repository_url: str) -> str | None:
    """Return a repository asset target only when it is inside this model folder."""
    value = html.unescape(target).strip()
    if value.startswith(("https://", "http://")):
        from urllib.parse import urlsplit

        parts = urlsplit(value)
        repo = urlsplit(repository_url)
        if parts.hostname != repo.hostname:
            return None
        path = parts.path
        repo_prefix = "/" + repo.path.strip("/")
        if not path.startswith(repo_prefix + "/"):
            return None
        path = path[len(repo_prefix) :]
        for marker in ("/blob/main/", "/tree/main/", "/raw/main/"):
            if marker in path:
                path = path.split(marker, 1)[1]
                break
        else:
            return None
    else:
        path = value.removeprefix("./").lstrip("/")
        if path.startswith("validated/"):
            pass
        else:
            path = f"{model_path}/{path}"
    try:
        path = _safe_path(path)
    except ValueError:
        return None
    if not path.startswith(f"{model_path}/") or path.endswith("/"):
        return None
    return path


def _plain_text(value: str) -> str:
    return " ".join(html.unescape(_TAG.sub("", value)).split())


def _reference_relation(url: str) -> str:
    identifier = identifier_from_url(url)
    if identifier is not None and identifier.namespace in {"arxiv", "doi"}:
        return "paper_reference"
    if identifier is not None and identifier.namespace == "github:repository":
        return "code_reference"
    return "artifact_reference"


def _unique_links(values: list[Link]) -> tuple[Link, ...]:
    result: list[Link] = []
    seen: set[tuple[str, str, str | None]] = set()
    for value in values:
        key = (value.url, value.relation, value.locator)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _web_url(value: str, field: str) -> str:
    result = canonicalize_url(value)
    if not result.startswith(("https://", "http://")):
        raise ValueError(f"{field} must be an HTTP(S) URL")
    return result


def _required_text(value: Any, field: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == wanted), None)


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
