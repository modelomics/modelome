"""List the exact GraphCast checkpoints in Google DeepMind's public GCS prefix.

The first-party WeatherNext GraphCast notebook uses anonymous access to list
``graphcast/params/`` in ``dm_graphcast`` and selects one of those checkpoints.
This adapter follows that metadata-only listing, emitting model weight URLs but
never downloading objects or listing other bucket prefixes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import content_hash

_BUCKET = "dm_graphcast"
_PREFIX = "graphcast/params/"
_API = "https://storage.googleapis.com/storage/v1/b/dm_graphcast/o"
_BUCKET_PAGE = "https://console.cloud.google.com/storage/browser/dm_graphcast/graphcast/params"
_DOCUMENTATION = (
    "https://github.com/google-deepmind/weathernext/blob/main/docs/weathernext1_graph/README.md"
)
_NOTEBOOK = (
    "https://github.com/google-deepmind/weathernext/blob/main/docs/"
    "weathernext1_graph/graphcast_demo.ipynb"
)


class GoogleGraphCastCheckpointInventorySourceAdapter:
    """Enumerate GraphCast checkpoint objects from the documented GCS prefix."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only .npz objects whose names identify one of the three GraphCast "
        "families documented by Google DeepMind, under graphcast/params/ in the "
        "public dm_graphcast bucket. It does not list any other prefix or fetch bytes."
    )

    def __init__(
        self,
        *,
        name: str = "google-graphcast-checkpoint-inventory",
        page_size: int = 1000,
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not name:
            raise ValueError("source name must not be empty")
        if isinstance(page_size, bool) or page_size <= 0 or page_size > 1000:
            raise ValueError("page_size must be between 1 and 1000")
        if isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.name = name
        self.page_size = page_size
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "google-graphcast-checkpoint-inventory-v1",
                "bucket": _BUCKET,
                "prefix": _PREFIX,
                "page_size": page_size,
                "max_response_bytes": max_response_bytes,
                "admission": "documented GraphCast-family .npz objects",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        params: dict[str, str] = {
            "prefix": _PREFIX,
            "maxResults": str(self.page_size),
            "fields": "items(name,generation,updated,size),nextPageToken",
        }
        page_token = state.get("page_token")
        if page_token is not None:
            if not isinstance(page_token, str) or not page_token:
                raise ValueError(f"{self.name}: invalid page token")
            params["pageToken"] = page_token
        response: HttpResponse = self.client.get(
            _API,
            params=params,
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: GCS returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: GCS response exceeds configured byte limit")
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: GCS listing response must be an object")
        items = payload.get("items", [])
        if not isinstance(items, list) or len(items) > self.page_size:
            raise ValueError(f"{self.name}: GCS listing items are invalid or oversized")
        records = tuple(
            record for item in items if (record := _record(self.name, item)) is not None
        )
        next_token = payload.get("nextPageToken")
        if next_token is not None and (not isinstance(next_token, str) or not next_token):
            raise ValueError(f"{self.name}: GCS returned an invalid next-page token")
        next_state: dict[str, Any] = {
            "prefix": _PREFIX,
            "bucket": _BUCKET,
        }
        if next_token is not None:
            next_state["page_token"] = next_token
        else:
            next_state["completed"] = True
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=next_token is None,
            upstream_count=len(items),
        )


def _record(source: str, value: Any) -> SourceRecord | None:
    if not isinstance(value, Mapping):
        raise ValueError("GCS listing item must be an object")
    name = value.get("name")
    if not isinstance(name, str) or not name.startswith(_PREFIX):
        return None
    filename = name.removeprefix(_PREFIX)
    family = _family(filename)
    if family is None:
        return None
    generation = value.get("generation")
    if not isinstance(generation, str) or not generation.isdigit():
        raise ValueError(f"GCS object {name!r} lacks a valid generation")
    object_url = (
        "https://storage.googleapis.com/download/storage/v1/b/"
        f"{_BUCKET}/o/{quote(name, safe='')}?alt=media&generation={generation}"
    )
    updated = value.get("updated")
    size = value.get("size")
    metadata: dict[str, str | int] = {"object_name": name, "generation": generation}
    if isinstance(updated, str):
        metadata["updated"] = updated
    if isinstance(size, str) and size.isdigit():
        metadata["size_bytes"] = int(size)
    local_id = filename
    return SourceRecord(
        source_record_id=f"{source}:{filename}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=object_url,
        title=f"{family} checkpoint",
        raw={
            "bucket": _BUCKET,
            "object_name": name,
            "generation": generation,
            "documentation_url": _DOCUMENTATION,
            "notebook_url": _NOTEBOOK,
        },
        identifiers=(Identifier("google:graphcast-checkpoint", filename),),
        links=(
            Link(
                object_url,
                relation="weights",
                locator="GCS.objects.name",
                crawl=False,
                model_local_ids=(local_id,),
            ),
            Link(
                _BUCKET_PAGE,
                relation="checkpoint_inventory",
                locator="GCS prefix",
                crawl=False,
                model_local_ids=(local_id,),
            ),
            Link(
                _DOCUMENTATION,
                relation="documentation",
                locator="pretrained models",
                crawl=False,
                model_local_ids=(local_id,),
            ),
        ),
        models=(
            ModelHint(
                local_id=local_id,
                name=family,
                identifiers=(Identifier("google:graphcast-model", family),),
                locator="Google DeepMind WeatherNext Graph README",
            ),
        ),
        releases=(
            ReleaseHint(
                local_id=f"{local_id}@{generation}",
                model_local_id=local_id,
                revision=generation,
                identifiers=(Identifier("google:graphcast-checkpoint", filename),),
                metadata=metadata,
                locator="GCS.objects.name and generation",
            ),
        ),
    )


def _family(filename: str) -> str | None:
    if not filename.endswith(".npz"):
        return None
    if filename.startswith("GraphCast_operational "):
        return "GraphCast_operational"
    if filename.startswith("GraphCast_small "):
        return "GraphCast_small"
    if filename.startswith("GraphCast "):
        return "GraphCast"
    return None
