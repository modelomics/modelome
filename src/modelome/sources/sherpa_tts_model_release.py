"""Enumerate sherpa-onnx's first-party finite TTS model release assets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

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

_API_URL = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/tts-models"
_REPOSITORY_URL = "https://github.com/k2-fsa/sherpa-onnx"
_RELEASE_URL = f"{_REPOSITORY_URL}/releases/tag/tts-models"
_RELEASE_TAG = "tts-models"
_ASSETS_API_TEMPLATE = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/{release_id}/assets"
_ASSET_PAGE_SIZE = 100
_NON_MODEL_ARCHIVES = frozenset({"espeak-ng-data.tar.bz2"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SherpaTtsModelReleaseSourceAdapter:
    """Read metadata for model archives in the official ``tts-models`` release.

    The GitHub release asset list is the broad finite inventory, including model
    packages omitted from selected documentation examples. This records archive
    URLs and file metadata without downloading or inspecting archive contents.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers `.tar.bz2` model archives in k2-fsa/sherpa-onnx's `tts-models` "
        "release tag. It excludes auxiliary phonemizer data and other release "
        "tags; asset names are provider identities and do not assert architecture."
    )

    def __init__(
        self,
        *,
        name: str = "sherpa-tts-model-release",
        max_response_bytes: int = 16 * 1024 * 1024,
        max_assets: int = 1000,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        for label, value in (
            ("max_response_bytes", max_response_bytes),
            ("max_assets", max_assets),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_assets = max_assets
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "sherpa-tts-model-release-v2",
                "release_tag": _RELEASE_TAG,
                "max_response_bytes": max_response_bytes,
                "max_assets": max_assets,
                "excluded_auxiliary_archives": sorted(_NON_MODEL_ARCHIVES),
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            _API_URL,
            headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: release endpoint returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: release response exceeds response limit")
        release = response.json()
        assets, release_id, updated_at = self._assets(release)
        embedded = release.get("assets")
        paginated = self._paginated_assets(release_id)
        embedded_signatures = _asset_signatures(embedded, self.name)
        paginated_signatures = _asset_signatures(paginated, self.name)
        inventory_digest = content_hash([list(row) for row in sorted(paginated_signatures)])
        discrepancy = embedded_signatures != paginated_signatures
        embedded_only_count = len(embedded_signatures - paginated_signatures)
        paginated_only_count = len(paginated_signatures - embedded_signatures)
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if (
            _text(state.get("completed_release_id")) == release_id
            and _text(state.get("asset_inventory_sha256")) == inventory_digest
        ):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            next_state["verified_release_asset_count"] = len(paginated)
            next_state["embedded_asset_count"] = len(embedded_signatures)
            next_state["embedded_inventory_matches"] = not discrepancy
            next_state["embedded_only_asset_count"] = embedded_only_count
            next_state["paginated_only_asset_count"] = paginated_only_count
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative(state.get("model_count")),
            )
        response_digest = content_hash(response.body)
        authoritative_release = dict(release)
        authoritative_release["assets"] = paginated
        assets, _, _ = self._assets(authoritative_release)
        records = tuple(
            self._record(asset, release_id, updated_at, inventory_digest) for asset in assets
        )
        next_state: dict[str, Any] = {
            "completed_release_id": release_id,
            "completed_release_updated_at": updated_at,
            "checked_at": checked_at,
            "release_response_sha256": response_digest,
            "asset_inventory_sha256": inventory_digest,
            "model_count": len(records),
            "asset_count": len(assets),
            "verified_release_asset_count": len(paginated),
            "embedded_asset_count": len(embedded_signatures),
            "embedded_inventory_matches": not discrepancy,
            "embedded_only_asset_count": embedded_only_count,
            "paginated_only_asset_count": paginated_only_count,
        }
        if etag := _header(response.headers, "etag"):
            next_state["etag"] = etag
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _assets(self, release: Any) -> tuple[list[dict[str, Any]], str, str]:
        if not isinstance(release, Mapping) or release.get("tag_name") != _RELEASE_TAG:
            raise ValueError(f"{self.name}: response is not the expected TTS release")
        release_id = release.get("id")
        if isinstance(release_id, bool) or not isinstance(release_id, int) or release_id < 1:
            raise ValueError(f"{self.name}: release ID is invalid")
        updated_at = release.get("updated_at")
        if not isinstance(updated_at, str) or not updated_at:
            raise ValueError(f"{self.name}: release update timestamp is missing")
        raw_assets = release.get("assets")
        if not isinstance(raw_assets, Sequence) or isinstance(raw_assets, (str, bytes)):
            raise ValueError(f"{self.name}: release assets must be an array")
        if len(raw_assets) > self.max_assets:
            raise ValueError(f"{self.name}: release exceeds asset limit")
        found: dict[str, dict[str, Any]] = {}
        for raw in raw_assets:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{self.name}: asset entry is not an object")
            filename = raw.get("name")
            if not isinstance(filename, str) or not filename.endswith(".tar.bz2"):
                continue
            if filename in _NON_MODEL_ARCHIVES:
                continue
            if filename in found:
                raise ValueError(f"{self.name}: duplicate release asset {filename!r}")
            url = raw.get("browser_download_url")
            _validate_asset_url(url, filename, self.name)
            size = raw.get("size")
            asset_id = raw.get("id")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"{self.name}: invalid size for {filename!r}")
            if isinstance(asset_id, bool) or not isinstance(asset_id, int) or asset_id < 1:
                raise ValueError(f"{self.name}: invalid ID for {filename!r}")
            found[filename] = {
                "filename": filename,
                "model_name": filename.removesuffix(".tar.bz2"),
                "url": canonicalize_url(url),
                "size_bytes": size,
                "asset_id": asset_id,
                "content_type": _text(raw.get("content_type")) or None,
                "created_at": _text(raw.get("created_at")) or None,
                "updated_at": _text(raw.get("updated_at")) or None,
            }
        if not found:
            raise ValueError(f"{self.name}: release contains no TTS model archives")
        return [found[key] for key in sorted(found)], str(release_id), updated_at

    def _paginated_assets(self, release_id: str) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        # An additional request after a full final page certifies that there are no
        # further assets, including when the total is an exact multiple of 100.
        max_pages = self.max_assets // _ASSET_PAGE_SIZE + 2
        for page_number in range(1, max_pages + 1):
            url = (
                f"{_ASSETS_API_TEMPLATE.format(release_id=release_id)}"
                f"?per_page={_ASSET_PAGE_SIZE}&page={page_number}"
            )
            response: HttpResponse = self.client.get(
                url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: release assets endpoint returned HTTP {response.status}"
                )
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: release assets response exceeds response limit")
            batch = response.json()
            if not isinstance(batch, list) or len(batch) > _ASSET_PAGE_SIZE:
                raise ValueError(
                    f"{self.name}: paginated release assets must be an array of at most 100"
                )
            if len(assets) + len(batch) > self.max_assets:
                raise ValueError(f"{self.name}: paginated release exceeds asset limit")
            _asset_signatures(batch, self.name)
            assets.extend(batch)
            if len(batch) < _ASSET_PAGE_SIZE:
                return assets
        raise ValueError(f"{self.name}: release asset pagination exceeded page limit")

    def _record(
        self, asset: Mapping[str, Any], release_id: str, updated_at: str, release_digest: str
    ) -> SourceRecord:
        model_name = str(asset["model_name"])
        filename = str(asset["filename"])
        local_id = f"model:{model_name}"
        identifier = Identifier("sherpa-onnx:tts-package", model_name)
        model = ModelHint(
            local_id=local_id,
            name=model_name,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=filename,
        )
        release = ReleaseHint(
            local_id=f"release:{model_name}",
            model_local_id=local_id,
            revision=release_id,
            identifiers=(Identifier("sherpa-onnx:tts-release-asset", filename),),
            metadata={
                "release_tag": _RELEASE_TAG,
                "release_id": release_id,
                "release_updated_at": updated_at,
                "filename": filename,
                "weight_url": asset["url"],
                "size_bytes": asset["size_bytes"],
                "asset_id": asset["asset_id"],
                "content_type": asset["content_type"],
                "asset_created_at": asset["created_at"],
                "asset_updated_at": asset["updated_at"],
            },
            locator=filename,
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(str(asset["url"])),
            title=model_name,
            raw={
                "repository": "k2-fsa/sherpa-onnx",
                "release_sha256": release_digest,
                **dict(asset),
            },
            text=f"sherpa-onnx text-to-speech model package {model_name}.",
            identifiers=(identifier,),
            links=(
                Link(_RELEASE_URL, "model_card", crawl=False, model_local_ids=(local_id,)),
                Link(
                    _REPOSITORY_URL,
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
                Link(str(asset["url"]), "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _validate_asset_url(url: Any, filename: str, source: str) -> None:
    if not isinstance(url, str):
        raise ValueError(f"{source}: asset download URL is missing")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.split("/")
        != [
            "",
            "k2-fsa",
            "sherpa-onnx",
            "releases",
            "download",
            "tts-models",
            filename,
        ]
    ):
        raise ValueError(f"{source}: invalid TTS release asset URL for {filename!r}")


def _asset_signatures(raw_assets: Any, source: str) -> set[tuple[Any, ...]]:
    if not isinstance(raw_assets, list):
        raise ValueError(f"{source}: release assets must be an array")
    signatures: set[tuple[Any, ...]] = set()
    for asset in raw_assets:
        if not isinstance(asset, Mapping):
            raise ValueError(f"{source}: asset entry is not an object")
        asset_id = asset.get("id")
        if isinstance(asset_id, bool) or not isinstance(asset_id, int) or asset_id < 1:
            raise ValueError(f"{source}: release asset ID is invalid")
        signatures.add(
            (asset_id, asset.get("name"), asset.get("size"), asset.get("browser_download_url"))
        )
    if len(signatures) != len(raw_assets):
        raise ValueError(f"{source}: duplicate or conflicting release asset entries")
    if len({signature[0] for signature in signatures}) != len(signatures):
        raise ValueError(f"{source}: duplicate release asset IDs")
    return signatures


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _nonnegative(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return next(
        (value for key, value in headers.items() if key.casefold() == name.casefold()), None
    )


__all__ = ["SherpaTtsModelReleaseSourceAdapter"]
