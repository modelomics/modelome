"""Enumerate Sherpa-ONNX's first-party ASR model release archives."""

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

_API_URL = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/asr-models"
_ASSETS_API = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/{release_id}/assets"
_REPOSITORY_URL = "https://github.com/k2-fsa/sherpa-onnx"
_RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models"
_PAGE_SIZE = 100
_NON_MODEL_ARCHIVES = frozenset(
    {"spoken-language-identification-test-wavs.tar.bz2", "librknnrt-android.tar.bz2"}
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SherpaAsrModelReleaseSourceAdapter:
    """Read release metadata for the finite Sherpa-ONNX ASR archive inventory."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers `.tar.bz2` checkpoint packages in the official asr-models release. "
        "It excludes the spoken-language test audio and RKNN runtime archives, and "
        "does not download or inspect package contents."
    )

    def __init__(
        self,
        *,
        name: str = "sherpa-asr-model-release",
        max_response_bytes: int = 16 * 1024 * 1024,
        max_assets: int = 1200,
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
                "adapter": "sherpa-asr-model-release-v1",
                "release_tag": "asr-models",
                "max_response_bytes": max_response_bytes,
                "max_assets": max_assets,
                "excluded_archives": sorted(_NON_MODEL_ARCHIVES),
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
        release_id, updated_at, embedded = _release_metadata(release, self.name)
        assets = self._paginated_assets(release_id)
        inventory = _asset_signatures(assets, self.name)
        embedded_inventory = _asset_signatures(embedded, self.name)
        inventory_digest = content_hash([list(row) for row in sorted(inventory)])
        discrepancies = {
            "embedded_asset_count": len(embedded_inventory),
            "verified_release_asset_count": len(inventory),
            "embedded_inventory_matches": embedded_inventory == inventory,
            "embedded_only_asset_count": len(embedded_inventory - inventory),
            "paginated_only_asset_count": len(inventory - embedded_inventory),
        }
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if (
            _text(state.get("completed_release_id")) == release_id
            and _text(state.get("asset_inventory_sha256")) == inventory_digest
        ):
            next_state = dict(state)
            next_state.update(discrepancies)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative(state.get("model_count")),
            )

        model_assets = [asset for asset in assets if _is_model_archive(asset)]
        if not model_assets:
            raise ValueError(f"{self.name}: release contains no ASR model archives")
        records = tuple(
            self._record(asset, release_id, updated_at, inventory_digest) for asset in model_assets
        )
        next_state: dict[str, Any] = {
            "completed_release_id": release_id,
            "completed_release_updated_at": updated_at,
            "checked_at": checked_at,
            "asset_inventory_sha256": inventory_digest,
            "model_count": len(records),
            "asset_count": len(assets),
            **discrepancies,
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

    def _paginated_assets(self, release_id: str) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        max_pages = self.max_assets // _PAGE_SIZE + 2
        for page_number in range(1, max_pages + 1):
            url = (
                f"{_ASSETS_API.format(release_id=release_id)}"
                f"?per_page={_PAGE_SIZE}&page={page_number}"
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
            if not isinstance(batch, list) or len(batch) > _PAGE_SIZE:
                raise ValueError(
                    f"{self.name}: paginated release assets must be an array of at most 100"
                )
            _asset_signatures(batch, self.name)
            if len(assets) + len(batch) > self.max_assets:
                raise ValueError(f"{self.name}: paginated release exceeds asset limit")
            assets.extend(batch)
            if len(batch) < _PAGE_SIZE:
                _asset_signatures(assets, self.name)
                return assets
        raise ValueError(f"{self.name}: release asset pagination exceeded page limit")

    def _record(
        self, asset: Mapping[str, Any], release_id: str, updated_at: str, inventory_digest: str
    ) -> SourceRecord:
        filename = str(asset["name"])
        model_name = filename.removesuffix(".tar.bz2")
        url = str(asset["browser_download_url"])
        local_id = f"model:{model_name}"
        identifier = Identifier("sherpa-onnx:asr-package", model_name)
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
            identifiers=(Identifier("sherpa-onnx:asr-release-asset", filename),),
            metadata={
                "release_tag": "asr-models",
                "release_id": release_id,
                "release_updated_at": updated_at,
                "filename": filename,
                "weight_url": url,
                "size_bytes": asset["size"],
                "asset_id": asset["id"],
                "content_type": asset.get("content_type"),
                "asset_created_at": asset.get("created_at"),
                "asset_updated_at": asset.get("updated_at"),
            },
            locator=filename,
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(url),
            title=model_name,
            raw={
                "repository": "k2-fsa/sherpa-onnx",
                "release_sha256": inventory_digest,
                "task": "automatic-speech-recognition",
                "filename": filename,
                "asset_id": asset["id"],
                "size_bytes": asset["size"],
                "weight_url": url,
            },
            text=f"sherpa-onnx automatic speech recognition model package {model_name}.",
            identifiers=(identifier,),
            links=(
                Link(_RELEASE_URL, "model_card", crawl=False, model_local_ids=(local_id,)),
                Link(
                    _REPOSITORY_URL,
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
                Link(url, "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _release_metadata(release: Any, source: str) -> tuple[str, str, Any]:
    if not isinstance(release, Mapping) or release.get("tag_name") != "asr-models":
        raise ValueError(f"{source}: response is not the expected ASR release")
    release_id = release.get("id")
    if isinstance(release_id, bool) or not isinstance(release_id, int) or release_id < 1:
        raise ValueError(f"{source}: release ID is invalid")
    updated_at = release.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at:
        raise ValueError(f"{source}: release update timestamp is missing")
    embedded = release.get("assets")
    if not isinstance(embedded, Sequence) or isinstance(embedded, (str, bytes)):
        raise ValueError(f"{source}: release assets must be an array")
    return str(release_id), updated_at, embedded


def _asset_signatures(raw_assets: Any, source: str) -> set[tuple[Any, ...]]:
    if not isinstance(raw_assets, Sequence) or isinstance(raw_assets, (str, bytes)):
        raise ValueError(f"{source}: release assets must be an array")
    signatures: set[tuple[Any, ...]] = set()
    for asset in raw_assets:
        if not isinstance(asset, Mapping):
            raise ValueError(f"{source}: asset entry is not an object")
        asset_id = asset.get("id")
        if isinstance(asset_id, bool) or not isinstance(asset_id, int) or asset_id < 1:
            raise ValueError(f"{source}: release asset ID is invalid")
        name = asset.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{source}: release asset name is missing")
        size = asset.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{source}: invalid size for {name!r}")
        url = asset.get("browser_download_url")
        _validate_asset_url(url, name, source)
        signatures.add((asset_id, name, size, url))
    if len(signatures) != len(raw_assets) or len({row[0] for row in signatures}) != len(signatures):
        raise ValueError(f"{source}: duplicate release asset IDs or entries")
    return signatures


def _is_model_archive(asset: Mapping[str, Any]) -> bool:
    return asset["name"].endswith(".tar.bz2") and asset["name"] not in _NON_MODEL_ARCHIVES


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
        != ["", "k2-fsa", "sherpa-onnx", "releases", "download", "asr-models", filename]
    ):
        raise ValueError(f"{source}: invalid ASR release asset URL for {filename!r}")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _nonnegative(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return next(
        (value for key, value in headers.items() if key.casefold() == name.casefold()), None
    )


__all__ = ["SherpaAsrModelReleaseSourceAdapter"]
