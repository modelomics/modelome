"""Enumerate the official RT-1-X JAX checkpoint objects in public GCS."""

from __future__ import annotations

import re
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

_BUCKET = "gdm-robotics-open-x-embodiment"
_PREFIX = "open_x_embodiment_and_rt_x_oss/rt_1_x_jax/"
_OBJECT_PATH = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OpenXRT1XCheckpointSourceAdapter:
    """Read only object names under the exact RT-1-X prefix in official docs."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only public GCS objects under the exact rt_1_x_jax prefix published "
        "by google-deepmind/open_x_embodiment. It does not enumerate datasets, other "
        "RT-X variants, third-party conversions, or RT-2, and does not download bytes."
    )

    def __init__(
        self,
        *,
        name: str = "open-x-rt1x-checkpoint",
        max_response_bytes: int = 8 * 1024 * 1024,
        max_files: int = 5000,
        max_pages: int = 50,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        for value, label in (
            (max_response_bytes, "max_response_bytes"),
            (max_files, "max_files"),
            (max_pages, "max_pages"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_files = max_files
        self.max_pages = max_pages
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "open-x-rt1x-gcs-checkpoint-v1",
                "bucket": _BUCKET,
                "prefix": _PREFIX,
                "max_response_bytes": max_response_bytes,
                "max_files": max_files,
                "max_pages": max_pages,
            }
        )

    @property
    def repository_url(self) -> str:
        return "https://github.com/google-deepmind/open_x_embodiment"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        files, page_bodies = self._list_objects()
        if not files:
            raise ValueError(f"{self.name}: public GCS prefix contains no objects")
        checked_at = self.clock()
        if checked_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        manifest_body = b"\n".join(page_bodies)
        record = self._record(files, manifest_body)
        return SourcePage(
            records=(record,),
            next_state={
                "checked_at": checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "object_count": len(files),
                "manifest_sha256": content_hash(manifest_body),
                "gcs_prefix": _PREFIX,
            },
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _list_objects(self) -> tuple[tuple[dict[str, Any], ...], tuple[bytes, ...]]:
        token: str | None = None
        seen_tokens: set[str] = set()
        files: dict[str, dict[str, Any]] = {}
        bodies: list[bytes] = []
        endpoint = f"https://storage.googleapis.com/storage/v1/b/{_BUCKET}/o"
        for _ in range(self.max_pages):
            params: dict[str, str] = {
                "prefix": _PREFIX,
                "fields": "items(name,size,generation,md5Hash),nextPageToken",
            }
            if token:
                params["pageToken"] = token
            response: HttpResponse = self.client.get(
                endpoint,
                params=params,
                headers={"Accept": "application/json"},
            )
            if response.status != 200:
                raise ValueError(f"{self.name}: GCS object listing returned HTTP {response.status}")
            if len(response.body) > self.max_response_bytes:
                raise ValueError(
                    f"{self.name}: GCS listing exceeds {self.max_response_bytes} bytes"
                )
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError(f"{self.name}: GCS listing response is not an object")
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise ValueError(f"{self.name}: GCS listing items are not a list")
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name.startswith(_PREFIX):
                    continue
                if name.endswith("/"):
                    continue
                relative_name = name.removeprefix(_PREFIX)
                if not _OBJECT_PATH.fullmatch(relative_name):
                    raise ValueError(
                        f"{self.name}: invalid GCS object name under checkpoint prefix"
                    )
                generation = _optional_decimal(item.get("generation"))
                if generation is None:
                    raise ValueError(f"{self.name}: GCS object {name} has no exact generation")
                files[name] = {
                    "name": name,
                    "size": _optional_decimal(item.get("size")),
                    "generation": generation,
                    "md5_hash": _optional_text(item.get("md5Hash")),
                }
                if len(files) > self.max_files:
                    raise ValueError(f"{self.name}: GCS prefix exceeds {self.max_files} objects")
            bodies.append(response.body)
            next_token = payload.get("nextPageToken")
            if next_token is None:
                return tuple(files[key] for key in sorted(files)), tuple(bodies)
            if not isinstance(next_token, str) or not next_token.strip():
                raise ValueError(f"{self.name}: GCS listing returned an invalid page token")
            if next_token in seen_tokens:
                raise ValueError(f"{self.name}: GCS listing repeated a page token")
            seen_tokens.add(next_token)
            token = next_token
        raise ValueError(f"{self.name}: GCS listing exceeds {self.max_pages} pages")

    def _record(
        self,
        files: tuple[dict[str, Any], ...],
        manifest_body: bytes,
    ) -> SourceRecord:
        local_id = "model:rt-1-x-jax"
        identifier = Identifier("open-x-embodiment:model", "RT-1-X-JAX")
        model_url = "https://github.com/google-deepmind/open_x_embodiment/blob/main/README.md"
        model = ModelHint(
            local_id=local_id,
            name="RT-1-X JAX",
            identifiers=(identifier,),
            aliases=("RT-1-X", "RT-1-X JAX"),
            status=ModelStatus.RELEASED,
            locator=_PREFIX,
        )
        manifest = []
        links = []
        for item in files:
            path = item["name"]
            url = f"https://storage.googleapis.com/{_BUCKET}/{quote(path, safe='/')}"
            generation = item["generation"]
            url = f"{url}?generation={generation}"
            links.append(
                Link(
                    url,
                    relation="model_artifact",
                    locator=path,
                    crawl=False,
                    model_local_ids=(local_id,),
                )
            )
            manifest.append(
                {
                    "object_name": path,
                    "url": url,
                    "generation": item.get("generation"),
                    "md5_hash": item.get("md5_hash"),
                    "size_bytes": item.get("size"),
                }
            )
        metadata = {
            "repository": "google-deepmind/open_x_embodiment",
            "gcs_bucket": _BUCKET,
            "gcs_prefix": _PREFIX,
            "objects": manifest,
            "manifest_sha256": content_hash(manifest_body),
        }
        release = ReleaseHint(
            local_id="release:rt-1-x-jax",
            model_local_id=local_id,
            identifiers=(Identifier("open-x-embodiment:checkpoint", "rt_1_x_jax"),),
            metadata=metadata,
            locator=_PREFIX,
        )
        return SourceRecord(
            source_record_id="open-x-embodiment:rt-1-x-jax",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(model_url),
            title="Open X-Embodiment RT-1-X JAX checkpoint",
            raw=metadata,
            text="RT-1-X JAX checkpoint; public GCS object prefix " + _PREFIX,
            identifiers=(identifier,),
            links=(
                Link(model_url, relation="model_catalog", crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
                *links,
            ),
            models=(model,),
            releases=(release,),
        )


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_decimal(value: Any) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return str(value)
    if isinstance(value, str) and value.isdecimal():
        return value
    return None
