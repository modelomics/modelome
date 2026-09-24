"""Enumerate direct checkpoints in MindSpore's official ModelZoo file index."""

from __future__ import annotations

import hashlib
import re
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

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
_ROOT_URL = "https://download.mindspore.cn/model_zoo/official/"
_ROOT_PATH = "/model_zoo/official/"
_HOST = "download.mindspore.cn"
_CHECKPOINT_SUFFIXES = (".ckpt", ".mindir", ".air")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _IndexEntry:
    href: str
    label: str


class _IndexParser(HTMLParser):
    def __init__(self, *, max_links: int) -> None:
        super().__init__(convert_charrefs=True)
        self.max_links = max_links
        self.entries: list[_IndexEntry] = []
        self._href: str | None = None
        self._label: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "a":
            attributes = {key.casefold(): value or "" for key, value in attrs}
            self._href = attributes.get("href") or None
            self._label = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._label.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a" and self._href is not None:
            label = " ".join("".join(self._label).split())
            self.entries.append(_IndexEntry(self._href, label))
            if len(self.entries) > self.max_links:
                raise ValueError(f"MindSpore index exceeds {self.max_links} links")
            self._href = None
            self._label = []


def _index_entries(document: str, *, max_links: int) -> tuple[_IndexEntry, ...]:
    parser = _IndexParser(max_links=max_links)
    parser.feed(document)
    return tuple(parser.entries)


def _safe_index_url(base_url: str, href: str) -> str | None:
    candidate = urljoin(base_url, href)
    parsed = urlsplit(candidate)
    path = unquote(parsed.path)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").casefold() != _HOST
        or parsed.username is not None
        or parsed.password is not None
        or not path.startswith(_ROOT_PATH)
        or "\\" in path
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        return None
    return candidate


def _checkpoint_filename(url: str) -> str | None:
    filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    return filename if filename.casefold().endswith(_CHECKPOINT_SUFFIXES) else None


def _isoformat(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise ValueError("clock must return a datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class MindSporeModelZooSourceAdapter:
    """Read direct checkpoint file links from the official MindSpore index.

    It follows only same-host directories under ``/model_zoo/official/`` and
    records only explicit ``.ckpt``, ``.mindir``, or ``.air`` links. It does not
    infer artifact URLs from model names or admit source-code links and archives.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers direct checkpoint links under download.mindspore.cn/model_zoo/official. "
        "It does not cover the deprecated/community trees, external MindSpore Hub, "
        "model repositories, archives without an exposed direct checkpoint, or "
        "models whose index folders are absent from the official file server."
    )

    def __init__(
        self,
        *,
        name: str = "mindspore-modelzoo",
        root_url: str = _ROOT_URL,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_total_bytes: int = 64 * 1024 * 1024,
        max_pages: int = 2_000,
        max_links_per_page: int = 20_000,
        max_artifacts: int = 20_000,
        max_depth: int = 8,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name is required")
        root = urlsplit(root_url)
        if (
            root.scheme != "https"
            or (root.hostname or "").casefold() != _HOST
            or not root.path.startswith(_ROOT_PATH)
            or root.path.rstrip("/") != _ROOT_PATH.rstrip("/")
        ):
            raise ValueError("root_url must be the official MindSpore ModelZoo directory")
        for value, field in (
            (max_response_bytes, "max_response_bytes"),
            (max_total_bytes, "max_total_bytes"),
            (max_pages, "max_pages"),
            (max_links_per_page, "max_links_per_page"),
            (max_artifacts, "max_artifacts"),
            (max_depth, "max_depth"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        self.name = name.strip()
        self.root_url = root_url
        self.max_response_bytes = max_response_bytes
        self.max_total_bytes = max_total_bytes
        self.max_pages = max_pages
        self.max_links_per_page = max_links_per_page
        self.max_artifacts = max_artifacts
        self.max_depth = max_depth
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "mindspore-modelzoo-index-v1",
                "root_url": self.root_url,
                "max_response_bytes": self.max_response_bytes,
                "max_total_bytes": self.max_total_bytes,
                "max_pages": self.max_pages,
                "max_links_per_page": self.max_links_per_page,
                "max_artifacts": self.max_artifacts,
                "max_depth": self.max_depth,
                "admission": "explicit same-host .ckpt, .mindir, or .air links",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        pending: deque[tuple[str, int]] = deque([(self.root_url, 0)])
        visited: set[str] = set()
        bodies: list[tuple[str, bytes]] = []
        artifacts: dict[str, tuple[str, str]] = {}
        total_bytes = 0
        while pending:
            index_url, depth = pending.popleft()
            if index_url in visited:
                continue
            if len(visited) >= self.max_pages:
                raise ValueError(f"{self.name}: index exceeds {self.max_pages} pages")
            visited.add(index_url)
            response: HttpResponse = self.client.get(
                index_url,
                headers={"Accept": "text/html,application/xhtml+xml"},
            )
            if response.status != 200:
                raise ValueError(f"{self.name}: index {index_url} returned HTTP {response.status}")
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: index {index_url} exceeds per-page byte limit")
            total_bytes += len(response.body)
            if total_bytes > self.max_total_bytes:
                raise ValueError(
                    f"{self.name}: crawled indexes exceed {self.max_total_bytes} bytes"
                )
            bodies.append((index_url, response.body))
            for entry in _index_entries(response.text(), max_links=self.max_links_per_page):
                url = _safe_index_url(index_url, entry.href)
                if not url:
                    continue
                filename = _checkpoint_filename(url)
                if filename:
                    artifacts[url] = (filename, index_url)
                    if len(artifacts) > self.max_artifacts:
                        raise ValueError(
                            f"{self.name}: catalog exceeds {self.max_artifacts} checkpoints"
                        )
                elif urlsplit(url).path.endswith("/"):
                    if depth >= self.max_depth:
                        raise ValueError(
                            f"{self.name}: index nesting exceeds configured depth {self.max_depth}"
                        )
                    pending.append((url, depth + 1))

        if not artifacts:
            raise ValueError(f"{self.name}: official ModelZoo index contains no direct checkpoints")
        digest = hashlib.sha256()
        for url, body in sorted(bodies):
            digest.update(url.encode())
            digest.update(b"\0")
            digest.update(body)
        revision = digest.hexdigest()
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = _isoformat(self.clock())
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("checkpoint_count")),
            )
        records = tuple(
            self._record(filename, artifact_url, index_url, revision)
            for artifact_url, (filename, index_url) in sorted(artifacts.items())
        )
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": _isoformat(self.clock()),
                "index_page_count": len(visited),
                "checkpoint_count": len(records),
                "index_sha256": revision,
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        filename: str,
        artifact_url: str,
        index_url: str,
        revision: str,
    ) -> SourceRecord:
        decoded_path = unquote(urlsplit(artifact_url).path)
        relative_path = decoded_path.removeprefix(_ROOT_PATH)
        basename = filename
        for suffix in _CHECKPOINT_SUFFIXES:
            if basename.casefold().endswith(suffix):
                basename = basename[: -len(suffix)]
                break
        local_id = f"model:{relative_path}"
        identifier = Identifier("mindspore:modelzoo-checkpoint", relative_path)
        model = ModelHint(
            local_id=local_id,
            name=basename,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=artifact_url,
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact_url),
            title=filename,
            raw={
                "repository": "mindspore-ai/models",
                "registry_root": self.root_url,
                "index_url": index_url,
                "artifact_url": artifact_url,
                "relative_path": relative_path,
                "snapshot_revision": revision,
            },
            text=f"MindSpore ModelZoo checkpoint: {filename}",
            identifiers=(identifier,),
            links=(
                Link(artifact_url, relation="weights", locator=filename, crawl=False),
                Link(index_url, relation="model_card", locator=filename, crawl=False),
                Link(
                    "https://github.com/mindspore-ai/models",
                    relation="source_repository",
                    crawl=False,
                ),
            ),
            models=(model,),
            releases=(
                ReleaseHint(
                    local_id=f"release:{relative_path}",
                    model_local_id=local_id,
                    revision=revision,
                    identifiers=(Identifier("mindspore:modelzoo-release", relative_path),),
                    metadata={"artifact_url": artifact_url, "index_url": index_url},
                    locator=artifact_url,
                ),
            ),
        )


__all__ = ["MindSporeModelZooSourceAdapter"]
