"""Sony AI Woosh checkpoint assets from the immutable v1.0.0 release page."""

from __future__ import annotations

import re
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

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

_REPOSITORY = "SonyResearch/Woosh"
_RELEASE_TAG = "v1.0.0"
_RELEASE_PAGE_PATH = f"/{_REPOSITORY}/releases/expanded_assets/{_RELEASE_TAG}"
_RELEASE_DOWNLOAD_PREFIX = f"/{_REPOSITORY}/releases/download/{_RELEASE_TAG}/"
_MODEL_ASSETS = {
    "TextConditionerA.zip": ("TextConditionerA", "audio-conditioning"),
    "TextConditionerV.zip": ("TextConditionerV", "video-conditioning"),
    "Woosh-AE.zip": ("Woosh-AE", "audio-autoencoder"),
    "Woosh-CLAP.zip": ("Woosh-CLAP", "audio-text-alignment"),
    "Woosh-DFlow.zip": ("Woosh-DFlow", "distilled-text-to-audio-generation"),
    "Woosh-DVFlow-8s.zip": ("Woosh-DVFlow-8s", "distilled-video-to-audio-generation"),
    "Woosh-Flow.zip": ("Woosh-Flow", "text-to-audio-generation"),
    "Woosh-VFlow-8s.zip": ("Woosh-VFlow-8s", "video-to-audio-generation"),
}
_NON_MODEL_ASSETS = {"samples.zip", "reaper-script-demo.mp4"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SonyWooshReleaseAssetsSourceAdapter:
    """Index eight exact model ZIP assets in Sony AI's first Woosh release."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the eight model/component ZIP assets in SonyResearch/Woosh v1.0.0. It "
        "excludes sample media and does not expand ZIP contents or retrieve asset bytes."
    )

    def __init__(
        self,
        *,
        name: str = "sony-woosh-release-assets",
        repository: str = _REPOSITORY,
        release_tag: str = _RELEASE_TAG,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_assets: int = 100,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY or release_tag != _RELEASE_TAG:
            raise ValueError("repository and release_tag must identify the first Woosh release")
        if not name.strip() or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (max_response_bytes, max_assets)
        ):
            raise ValueError("name and positive limits are required")
        self.name, self.repository, self.release_tag = name, repository, release_tag
        self.max_response_bytes, self.max_assets = max_response_bytes, max_assets
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "sony-woosh-release-assets-v1",
                "repository": repository,
                "release_tag": release_tag,
                "max_response_bytes": max_response_bytes,
                "max_assets": max_assets,
                "admission": "exact named model zip assets in pinned first-party release",
            }
        )

    @property
    def release_page_url(self) -> str:
        return f"https://github.com{_RELEASE_PAGE_PATH}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(self.release_page_url, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: release asset page returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: release asset page exceeds response byte limit")
        try:
            document = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: release asset page is not UTF-8") from exc
        assets = _parse_assets(document, maximum=self.max_assets)
        digest = content_hash(response.body)
        if digest == state.get("source_digest"):
            return SourcePage((), dict(state), True, upstream_count=len(assets))
        records = tuple(self._record(filename, checksum, digest) for filename, checksum in assets)
        next_state = {"source_digest": digest, "record_count": len(records)}
        return SourcePage(
            records,
            next_state,
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, filename: str, checksum: str, digest: str) -> SourceRecord:
        model_name, category = _MODEL_ASSETS[filename]
        path = f"{_RELEASE_DOWNLOAD_PREFIX}{filename}"
        url = f"https://github.com{path}"
        model_value = f"{self.release_tag}/{filename}"
        locator = f"SonyResearch/Woosh GitHub release {self.release_tag}: {filename}"
        return SourceRecord(
            source_record_id=f"sony-woosh:{model_value}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"Sony Woosh {model_name} checkpoint bundle",
            identifiers=(Identifier("sony-research:woosh-release-asset", model_value),),
            links=(
                Link(url, relation="weights", crawl=False),
                Link(f"https://github.com/{self.repository}", relation="repository", crawl=False),
                Link(self.release_page_url, relation="release", crawl=False, locator=locator),
            ),
            raw={
                "record_type": "sony_woosh_release_asset",
                "asset_filename": filename,
                "release_tag": self.release_tag,
                "asset_url": url,
                "category": category,
                "asset_sha256": checksum,
                "source_page_sha256": digest,
                "checkpoint_bytes_fetched": False,
            },
            models=(
                ModelHint(
                    local_id=f"model:{model_name.casefold()}",
                    name=f"Sony Woosh {model_name}",
                    aliases=(model_name, filename),
                    identifiers=(Identifier("sony-research:woosh-model", model_value),),
                    status=ModelStatus.RELEASED,
                    locator=locator,
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{model_value}",
                    model_local_id=f"model:{model_name.casefold()}",
                    version=self.release_tag,
                    identifiers=(Identifier("sony-research:woosh-release-asset", model_value),),
                    metadata={
                        "filename": filename,
                        "asset_url": url,
                        "asset_sha256": checksum,
                        "category": category,
                    },
                    locator=locator,
                ),
            ),
        )


def _parse_assets(document: str, *, maximum: int) -> tuple[tuple[str, str], ...]:
    parser = _AssetLinkParser()
    parser.feed(document)
    if len(parser.assets) > maximum:
        raise ValueError(f"Sony Woosh release asset count exceeds {maximum}")
    seen: set[str] = set()
    model_assets: set[str] = set()
    for path, checksum in parser.assets:
        if not path.startswith(_RELEASE_DOWNLOAD_PREFIX):
            continue
        filename = path.removeprefix(_RELEASE_DOWNLOAD_PREFIX)
        if "/" in filename or "?" in filename or "#" in filename:
            raise ValueError("unexpected Sony Woosh release asset URL")
        if filename in seen:
            raise ValueError(f"duplicate Sony Woosh release asset: {filename}")
        seen.add(filename)
        if filename in _MODEL_ASSETS:
            if checksum is None or not _SHA256.fullmatch(checksum):
                raise ValueError(f"Sony Woosh asset is missing a valid SHA-256: {filename}")
            model_assets.add(filename)
        elif filename not in _NON_MODEL_ASSETS:
            raise ValueError(f"unexpected Sony Woosh release asset: {filename}")
    if model_assets != set(_MODEL_ASSETS):
        missing = sorted(set(_MODEL_ASSETS) - model_assets)
        raise ValueError(f"Sony Woosh release is missing model assets: {missing}")
    checksums = {path.rsplit("/", 1)[-1]: checksum for path, checksum in parser.assets}
    return tuple((filename, checksums[filename]) for filename in _MODEL_ASSETS)


class _AssetLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.assets: list[tuple[str, str | None]] = []
        self._row_path: str | None = None
        self._row_sha256: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "li":
            self._row_path = None
            self._row_sha256 = None
            return
        if tag == "a":
            href = attributes.get("href")
            if not isinstance(href, str):
                return
            parsed = urlsplit(href)
            if (parsed.scheme or parsed.netloc) and (
                parsed.scheme != "https" or parsed.netloc != "github.com"
            ):
                return
            self._row_path = parsed.path
        elif tag == "clipboard-copy":
            value = attributes.get("value")
            if isinstance(value, str) and value.startswith("sha256:"):
                self._row_sha256 = value.removeprefix("sha256:")

    def handle_endtag(self, tag: str) -> None:
        if tag == "li" and self._row_path is not None:
            self.assets.append((self._row_path, self._row_sha256))
            self._row_path = None
            self._row_sha256 = None


__all__ = ["SonyWooshReleaseAssetsSourceAdapter"]
