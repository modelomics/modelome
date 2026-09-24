"""First-party Microsoft VQ-Diffusion checkpoint download manifest."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpResponse
from modelome.models import Link, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _isoformat,
    _text,
)

_WGET = re.compile(r"^\s*wget\s+(?P<url>https://\S+)\s*(?:#.*)?$")
_CAT = re.compile(
    r"^\s*cat\s+(?P<stem>[A-Za-z0-9_.-]+)_\*\s*>\s*"
    r"(?P<filename>[A-Za-z0-9_.-]+\.pth)\s*$"
)
_ASSET = re.compile(r"^[A-Za-z0-9_.-]+$")


class MicrosoftVqDiffusionCheckpointManifestSourceAdapter(
    StaticJsonCheckpointRegistrySourceAdapter
):
    """Enumerate exact VQ-Diffusion checkpoint files and release asset parts.

    The adapter reads the official download script at a pinned commit. It never
    runs the shell script or downloads the binary assets. Multi-part checkpoints
    retain every exact GitHub release URL that the script concatenates.
    """

    coverage_limitation = (
        "Covers only checkpoint files and shard groups explicitly enumerated in "
        "Microsoft VQ-Diffusion's download script. It does not fetch assets or "
        "infer the contents of release files."
    )

    def __init__(
        self,
        *,
        name: str = "microsoft-vq-diffusion-checkpoints",
        repository: str = "microsoft/VQ-Diffusion",
        branch: str = "main",
        source_path: str = "vqdiffusion_download_checkpoints.sh",
        provider_namespace: str = "microsoft:vq-diffusion",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 1_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            repository=repository,
            branch=branch,
            source_path=source_path,
            provider_namespace=provider_namespace,
            max_response_bytes=max_response_bytes,
            max_entries=max_entries,
            **kwargs,
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "microsoft-vq-diffusion-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "provider_namespace": provider_namespace,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "first-party literal wget assets and cat shard mappings",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, _commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            model_count = state.get("model_count")
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=model_count if isinstance(model_count, int) else None,
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: download manifest returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: download manifest exceeds byte limit")
        entries = _parse_manifest(
            response.text(), source=self.name, path=self.source_path, maximum=self.max_entries
        )
        records = tuple(
            self._record_for_assets(entry, revision, response.body)
            for entry in entries
        )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
        }
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record_for_assets(
        self,
        entry: _CheckpointAssets,
        revision: str,
        source: bytes,
    ) -> SourceRecord:
        base = self._record(
            _Checkpoint(entry.handle, entry.urls[0], entry.locator), revision, source
        )
        model_local_id = base.models[0].local_id
        checkpoint_links = tuple(
            Link(
                url=url,
                relation="weights",
                locator=f"{entry.locator}:asset:{index}",
                crawl=False,
                model_local_ids=(model_local_id,),
            )
            for index, url in enumerate(entry.urls, start=1)
        )
        links = tuple(link for link in base.links if link.relation != "weights") + checkpoint_links
        release = replace(
            base.releases[0],
            metadata={
                **base.releases[0].metadata,
                "weight_urls": list(entry.urls),
                "weight_asset_count": len(entry.urls),
            },
        )
        return replace(base, links=links, releases=(release,))


class _CheckpointAssets:
    __slots__ = ("handle", "urls", "locator")

    def __init__(self, handle: str, urls: tuple[str, ...], locator: str) -> None:
        self.handle = handle
        self.urls = urls
        self.locator = locator


def _parse_manifest(
    document: str,
    *,
    source: str,
    path: str,
    maximum: int,
) -> tuple[_CheckpointAssets, ...]:
    downloads: dict[str, str] = {}
    output_groups: list[tuple[str, str, int]] = []
    for line_number, line in enumerate(document.splitlines(), start=1):
        if wget := _WGET.fullmatch(line):
            url = canonicalize_url(wget.group("url"))
            if not _valid_asset_url(url):
                raise ValueError(f"{source}: release asset URL is outside the approved path")
            asset = urlsplit(url).path.rsplit("/", 1)[-1]
            if not _ASSET.fullmatch(asset) or asset in downloads:
                raise ValueError(f"{source}: invalid or duplicate release asset {asset!r}")
            downloads[asset] = url
            continue
        if cat := _CAT.fullmatch(line):
            output_groups.append((cat.group("stem"), cat.group("filename"), line_number))

    mapped_assets: set[str] = set()
    entries: list[_CheckpointAssets] = []
    handles: set[str] = set()
    for stem, filename, line_number in output_groups:
        handle = filename.removesuffix(".pth")
        if handle in handles:
            raise ValueError(f"{source}: duplicate checkpoint output {filename!r}")
        assets = tuple(
            url
            for name, url in downloads.items()
            if name.startswith(stem + "_")
        )
        if not assets:
            raise ValueError(f"{source}: shard group {stem!r} has no literal downloads")
        handles.add(handle)
        mapped_assets.update(urlsplit(url).path.rsplit("/", 1)[-1] for url in assets)
        entries.append(
            _CheckpointAssets(handle, assets, f"{path}:line:{line_number}")
        )

    for asset, url in downloads.items():
        if asset in mapped_assets:
            continue
        if not asset.casefold().endswith(".pth"):
            raise ValueError(f"{source}: downloaded shard {asset!r} has no output mapping")
        handle = asset.removesuffix(".pth")
        if handle in handles:
            raise ValueError(f"{source}: duplicate checkpoint output {asset!r}")
        handles.add(handle)
        entries.append(
            _CheckpointAssets(handle, (url,), f"{path}:asset:{asset}")
        )

    if not entries:
        raise ValueError(f"{source}: no model checkpoint assets found")
    if len(entries) > maximum:
        raise ValueError(f"{source}: checkpoint manifest exceeds {maximum} entries")
    return tuple(entries)


def _valid_asset_url(url: str) -> bool:
    parts = urlsplit(url)
    return (
        parts.scheme == "https"
        and parts.hostname == "github.com"
        and parts.path.startswith("/tzco/storage/releases/download/vqdiffusion/")
        and bool(parts.path.rsplit("/", 1)[-1])
    )


__all__ = ["MicrosoftVqDiffusionCheckpointManifestSourceAdapter"]
