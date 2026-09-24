"""List source-declared GenCast checkpoint objects in Google DeepMind's GCS prefix."""

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
_PREFIX = "gencast/params/"
_API = "https://storage.googleapis.com/storage/v1/b/dm_graphcast/o"
_BUCKET_PAGE = "https://console.cloud.google.com/storage/browser/dm_graphcast/gencast/params"
_DOCUMENTATION = (
    "https://github.com/google-deepmind/weathernext/blob/main/docs/weathernext1_gen/README.md"
)
_NOTEBOOK = (
    "https://github.com/google-deepmind/weathernext/blob/main/docs/"
    "weathernext1_gen/gencast_mini_demo.ipynb"
)
_MODELS = {
    "GenCast 0p25deg <2019",
    "GenCast 0p25deg Operational <2022",
    "GenCast 1p0deg <2019",
    "GenCast 1p0deg Mini <2019",
}


class GoogleGenCastCheckpointInventorySourceAdapter:
    """Enumerate only the four GenCast variants listed in its first-party README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only exact .npz objects under gencast/params/ whose basenames are "
        "one of the four GenCast variants declared in Google DeepMind's first-party "
        "README. The first-party mini demo uses an anonymous GCS client to list this "
        "prefix. It does not list other prefixes, infer new variants, or fetch object bytes."
    )

    def __init__(
        self,
        *,
        name: str = "google-gencast-checkpoint-inventory",
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
                "adapter": "google-gencast-checkpoint-inventory-v1",
                "bucket": _BUCKET,
                "prefix": _PREFIX,
                "models": sorted(_MODELS),
                "page_size": page_size,
                "max_response_bytes": max_response_bytes,
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
        next_state: dict[str, Any] = {"prefix": _PREFIX, "bucket": _BUCKET}
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
    if not filename.endswith(".npz") or filename.removesuffix(".npz") not in _MODELS:
        return None
    generation = value.get("generation")
    if not isinstance(generation, str) or not generation.isdigit():
        raise ValueError(f"GCS object {name!r} lacks a valid generation")
    object_url = (
        "https://storage.googleapis.com/download/storage/v1/b/"
        f"{_BUCKET}/o/{quote(name, safe='')}?alt=media&generation={generation}"
    )
    model_local_id = filename.removesuffix(".npz")
    updated = value.get("updated")
    size = value.get("size")
    metadata: dict[str, str | int] = {"object_name": name, "generation": generation}
    if isinstance(updated, str):
        metadata["updated"] = updated
    if isinstance(size, str) and size.isdigit():
        metadata["size_bytes"] = int(size)
    return SourceRecord(
        source_record_id=f"{source}:{filename}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=object_url,
        title=f"{model_local_id} checkpoint",
        raw={
            "bucket": _BUCKET,
            "object_name": name,
            "generation": generation,
            "documentation_url": _DOCUMENTATION,
            "notebook_url": _NOTEBOOK,
        },
        identifiers=(Identifier("google:gencast-checkpoint", filename),),
        links=(
            Link(
                object_url,
                relation="weights",
                locator="GCS.objects.name and generation",
                crawl=False,
                model_local_ids=(model_local_id,),
            ),
            Link(
                _BUCKET_PAGE,
                relation="checkpoint_inventory",
                locator="GCS gencast/params prefix",
                crawl=False,
                model_local_ids=(model_local_id,),
            ),
            Link(
                _DOCUMENTATION,
                relation="documentation",
                locator="four pretrained GenCast variants",
                crawl=False,
                model_local_ids=(model_local_id,),
            ),
        ),
        models=(
            ModelHint(
                local_id=model_local_id,
                name=model_local_id,
                identifiers=(Identifier("google:gencast-model", model_local_id),),
                locator="Google DeepMind WeatherNext 1 Gen README",
            ),
        ),
        releases=(
            ReleaseHint(
                local_id=f"{model_local_id}@{generation}",
                model_local_id=model_local_id,
                revision=generation,
                identifiers=(Identifier("google:gencast-checkpoint", filename),),
                metadata=metadata,
                locator="GCS.objects.name and generation",
            ),
        ),
    )
