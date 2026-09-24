"""Pinned extraction of the documented GraphGPS OGB-LSC release archive."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpResponse
from modelome.models import SourcePage
from modelome.normalize import canonicalize_url, content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)

_HEADING = re.compile(r"^#{1,6}[ \t]+(?P<title>.+?)\s*$")
_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<tail>.*)$")
_ANNOUNCEMENT = re.compile(r"You can download our pretrained GPS-deep \(\d+ MB\)\.")
_WGET = re.compile(r"^\s*wget\s+(?P<url>https://\S+)\s*$")


class GraphGPSReleaseAssetSourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Extract the one explicitly documented pretrained GraphGPS release asset.

    The source is the OGB-LSC inference section in the first-party README. The
    adapter admits one `wget` URL only when the section also declares the exact
    pretrained GPS-deep model identity. It does not crawl the Dropbox archive.
    """

    coverage_limitation = (
        "Covers only the single pretrained GPS-deep archive explicitly named in the "
        "GraphGPS OGB-LSC inference section. It does not infer unpublished or local "
        "training outputs, inspect the archive, or download its contents."
    )

    def __init__(self, *, document_path: str = "README.md", **kwargs: Any) -> None:
        self.document_path = document_path
        super().__init__(source_path=document_path, **kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "graphgps-release-asset-v1",
                "repository": self.repository,
                "branch": self.branch,
                "document_path": self.document_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "one first-party GPS-deep wget archive in OGB-LSC inference section",
            }
        )

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
        response: HttpResponse = self.client.get(
            self.raw_url(revision),
            headers={"Accept": "text/markdown,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds {self.max_response_bytes} bytes")
        url = _extract_release_url(response.text(), self.name)
        checkpoint = _Checkpoint(
            handle="GPS-deep",
            url=_zip_asset_url(url, self.name),
            locator=f"{self.document_path}:OGB-LSC inference section",
        )
        record = self._record(checkpoint, revision, response.body)
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": 1,
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=(record,),
            next_state=next_state,
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _model_name(self, handle: str) -> str:
        return handle


def _extract_release_url(document: str, source: str) -> str:
    in_section = False
    section_found = False
    announcement_found = False
    in_fence: tuple[str, int] | None = None
    urls: list[str] = []
    for line in document.splitlines():
        if heading := _HEADING.match(line):
            title = heading.group("title").strip()
            if title == "Inference and submission files for OGB-LSC leaderboard":
                in_section = True
                section_found = True
            elif in_section:
                in_section = False
            continue
        if not in_section:
            continue
        if _ANNOUNCEMENT.search(line):
            announcement_found = True
        if fence := _FENCE.match(line):
            marker = fence.group("fence")
            if in_fence is None:
                in_fence = (marker[0], len(marker))
            elif (
                marker[0] == in_fence[0]
                and len(marker) >= in_fence[1]
                and not fence.group("tail").strip()
            ):
                in_fence = None
            continue
        if in_fence is None:
            continue
        match = _WGET.match(line)
        if match:
            urls.append(match.group("url"))

    if not section_found or not announcement_found:
        raise ValueError(f"{source}: expected the documented pretrained GPS-deep release section")
    if len(urls) != 1:
        raise ValueError(f"{source}: GPS-deep section must declare exactly one wget asset URL")
    return urls[0]


def _zip_asset_url(value: str, source: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or not parsed.path.casefold().endswith(".zip")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{source}: GPS-deep asset must be a direct HTTPS ZIP URL")
    return canonicalize_url(value)
