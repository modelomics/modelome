"""Metadata-only index of Asteroid's official Zenodo model community."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
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

_COMMUNITY = "asteroid-models"
_API_ROOT = "https://zenodo.org/api/records"
_COMMUNITY_URL = "https://zenodo.org/communities/asteroid-models"
_REPOSITORY_URL = "https://github.com/asteroid-team/asteroid"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AsteroidZenodoModelsAdapter:
    """Enumerate checkpoint files in Asteroid's curated Zenodo community.

    Zenodo record metadata is read only; model file content is never requested.
    The model community also contains non-checkpoint records, which are excluded.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers records in Asteroid's Zenodo community that expose exactly one .pth "
        "checkpoint file. Dataset archives and records with other file layouts are "
        "excluded; checkpoint bytes are never downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "asteroid-zenodo-models",
        client: Any | None = None,
        clock: Any = _utcnow,
        page_size: int = 25,
        max_records: int = 500,
        max_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        for label, value in (
            ("page_size", page_size),
            ("max_records", max_records),
            ("max_response_bytes", max_response_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.page_size = page_size
        self.max_records = max_records
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "asteroid-zenodo-model-community-v1",
                "community": _COMMUNITY,
                "page_size": page_size,
                "max_records": max_records,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        entries: list[Mapping[str, Any]] = []
        ids: set[str] = set()
        total: int | None = None
        digest_material: list[bytes] = []
        page = 1
        while True:
            query = f"communities={_COMMUNITY}&page={page}&size={self.page_size}&sort=newest"
            url = f"{_API_ROOT}?{query}"
            response = self.client.get(url, headers={"Accept": "application/json"})
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: Zenodo community API returned HTTP {response.status}"
                )
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: Zenodo metadata exceeds response limit")
            payload = response.json()
            hits = payload.get("hits") if isinstance(payload, Mapping) else None
            if not isinstance(hits, Mapping) or not isinstance(hits.get("hits"), list):
                raise ValueError(f"{self.name}: Zenodo response has no hits list")
            current_total = hits.get("total")
            if isinstance(current_total, Mapping):
                current_total = current_total.get("value")
            if (
                isinstance(current_total, bool)
                or not isinstance(current_total, int)
                or current_total < 0
            ):
                raise ValueError(f"{self.name}: Zenodo response has invalid total")
            if total is None:
                total = current_total
                if total > self.max_records:
                    raise ValueError(
                        f"{self.name}: community exceeds {self.max_records} record limit"
                    )
            elif current_total != total:
                raise ValueError(f"{self.name}: community total changed during pagination")
            batch = hits["hits"]
            if len(batch) > self.page_size or (not batch and len(entries) < total):
                raise ValueError(f"{self.name}: invalid or incomplete Zenodo page")
            digest_material.append(response.body)
            for entry in batch:
                if not isinstance(entry, Mapping):
                    raise ValueError(f"{self.name}: Zenodo hit is not an object")
                record_id = str(entry.get("id", ""))
                if not record_id.isdigit() or record_id in ids:
                    raise ValueError(f"{self.name}: invalid or duplicate record id")
                ids.add(record_id)
                communities = _record_communities(entry)
                if communities is not None and _COMMUNITY not in communities:
                    raise ValueError(f"{self.name}: record is outside the Asteroid community")
                entries.append(entry)
            if len(entries) == total:
                break
            if not batch or page >= (self.max_records // self.page_size) + 2:
                raise ValueError(f"{self.name}: Zenodo pagination did not reach terminal page")
            page += 1

        if total is None or len(entries) != total:
            raise ValueError(f"{self.name}: incomplete Zenodo community snapshot")
        digest = content_hash(b"".join(digest_material))
        records = tuple(
            record for entry in entries if (record := self._model_record(entry, digest)) is not None
        )
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        return SourcePage(
            records=records,
            next_state={
                "checked_at": checked_at,
                "community": _COMMUNITY,
                "community_record_count": total,
                "model_count": len(records),
                "metadata_sha256": digest,
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _model_record(self, entry: Mapping[str, Any], digest: str) -> SourceRecord | None:
        record_id = str(entry["id"])
        metadata = entry.get("metadata")
        files = entry.get("files")
        if not isinstance(metadata, Mapping) or not isinstance(files, list):
            raise ValueError(f"{self.name}: record {record_id} has invalid metadata/files")
        checkpoints = [
            item
            for item in files
            if isinstance(item, Mapping) and str(item.get("key", "")).endswith(".pth")
        ]
        if not checkpoints:
            return None
        if len(checkpoints) != 1:
            raise ValueError(f"{self.name}: record {record_id} has multiple .pth files")
        item = checkpoints[0]
        filename = str(item.get("key", ""))
        links = item.get("links")
        weight_url = links.get("self") if isinstance(links, Mapping) else None
        _validate_file_url(weight_url, record_id, filename, self.name)
        title = metadata.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{self.name}: record {record_id} has no title")
        creators = metadata.get("creators", [])
        creator_names = (
            [
                c.get("name")
                for c in creators
                if isinstance(c, Mapping) and isinstance(c.get("name"), str)
            ]
            if isinstance(creators, list)
            else []
        )
        model_id = f"model:zenodo:{record_id}"
        doi = metadata.get("doi")
        identifiers = [Identifier("asteroid:zenodo-record", record_id)]
        if isinstance(doi, str) and doi:
            identifiers.append(Identifier("doi", doi))
        model = ModelHint(
            local_id=model_id,
            name=title.strip(),
            aliases=tuple(creator_names),
            identifiers=tuple(identifiers),
            status=ModelStatus.RELEASED,
            locator=filename,
        )
        checksum = item.get("checksum")
        size = item.get("size")
        record_url = f"https://zenodo.org/records/{record_id}"
        release = ReleaseHint(
            local_id=f"release:zenodo:{record_id}",
            model_local_id=model_id,
            identifiers=(Identifier("asteroid:zenodo-file", f"{record_id}/{filename}"),),
            metadata={
                "record_id": record_id,
                "record_doi": doi,
                "filename": filename,
                "weight_url": weight_url,
                "checksum": checksum,
                "size_bytes": size,
                "metadata_sha256": digest,
                "upstream_record_url": record_url,
            },
            locator=filename,
        )
        return SourceRecord(
            source_record_id=f"zenodo:{record_id}:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(weight_url),
            title=title.strip(),
            raw={
                "record_id": record_id,
                "filename": filename,
                "weight_url": weight_url,
                "checksum": checksum,
                "size_bytes": size,
            },
            text=f"Asteroid community checkpoint {title.strip()} by {', '.join(creator_names)}.",
            identifiers=tuple(identifiers),
            links=(
                Link(record_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(_COMMUNITY_URL, "source_index", crawl=False, model_local_ids=(model_id,)),
                Link(
                    _REPOSITORY_URL,
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _record_communities(entry: Mapping[str, Any]) -> set[str] | None:
    communities = entry.get("communities")
    if isinstance(communities, Mapping):
        entries = communities.get("entries")
        if isinstance(entries, Mapping):
            return {str(slug) for slug in entries}
    return None


def _validate_file_url(url: Any, record_id: str, filename: str, source: str) -> None:
    if not isinstance(url, str):
        raise ValueError(f"{source}: checkpoint content URL is missing")
    parsed = urlsplit(url)
    expected_path = f"/api/records/{record_id}/files/{filename}/content"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "zenodo.org"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != expected_path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{source}: invalid Zenodo checkpoint content URL")


__all__ = ["AsteroidZenodoModelsAdapter"]
