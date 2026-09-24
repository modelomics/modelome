"""FLOWER VLA's author-maintained collection and exact Hub weight files."""

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

_COLLECTION = "mbreuss/flower-vla"
_TITLE = "FLOWER VLA"
_OWNER = "mbreuss"
_REPO = re.compile(r"^mbreuss/flower_[A-Za-z0-9_-]+$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_PATH = re.compile(r"^[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*$")
_WEIGHT_NAMES = {"model.safetensors", "360000_model_weights.pt"}


class FlowerVLACollectionAdapter:
    """Resolve collection member model ids to exact pinned checkpoint files."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public model repos in the FLOWER VLA collection when each contains "
        "one recognized root weight file (model.safetensors or 360000_model_weights.pt). "
        "Excludes dependencies and unlisted files; does not download weights."
    )

    def __init__(
        self,
        *,
        name: str = "flower-vla-collection",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 24,
        client: HttpClient | Any | None = None,
        clock: Any = lambda: datetime.now(UTC),
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        for value, label in (
            (max_response_bytes, "max_response_bytes"),
            (max_entries, "max_entries"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "flower-vla-collection-v1",
                "collection": _COLLECTION,
                "collection_title": _TITLE,
                "owner": _OWNER,
                "admission": "public collection model with one recognized root weight file",
                "weight_names": sorted(_WEIGHT_NAMES),
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
            }
        )

    @property
    def collection_url(self) -> str:
        return f"https://huggingface.co/collections/{_COLLECTION}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        response = self.client.get(
            f"https://huggingface.co/api/collections/{_COLLECTION}",
            headers={"Accept": "application/json"},
        )
        self._require(response, "first-party collection API")
        payload = response.json()
        if not isinstance(payload, Mapping) or payload.get("title") != _TITLE:
            raise ValueError(f"{self.name}: collection identity changed")
        if payload.get("private") is not False:
            raise ValueError(f"{self.name}: collection is not public")
        owner = payload.get("owner")
        if not isinstance(owner, Mapping) or owner.get("name") != _OWNER:
            raise ValueError(f"{self.name}: collection owner changed")
        items = payload.get("items")
        if not isinstance(items, list) or not items or len(items) > self.max_entries:
            raise ValueError(f"{self.name}: invalid collection item count")
        repos: dict[str, Mapping[str, Any]] = {}
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"{self.name}: malformed collection item")
            if item.get("type") != "model":
                continue  # The collection includes its paper too.
            repo = item.get("id")
            if (
                item.get("repoType") != "model"
                or item.get("private") is not False
                or item.get("author") != _OWNER
                or not isinstance(repo, str)
                or not _REPO.fullmatch(repo)
            ):
                raise ValueError(f"{self.name}: unexpected or non-public collection model")
            if repo in repos:
                raise ValueError(f"{self.name}: duplicate model {repo}")
            repos[repo] = item
        if not repos:
            raise ValueError(f"{self.name}: collection has no models")

        records: list[SourceRecord] = []
        listings: list[bytes] = []
        for repo, item in repos.items():
            info_response = self.client.get(
                f"https://huggingface.co/api/models/{repo}",
                headers={"Accept": "application/json"},
            )
            self._require(info_response, f"Hub metadata for {repo}")
            info = info_response.json()
            revision = info.get("sha", "") if isinstance(info, Mapping) else ""
            if not isinstance(revision, str) or not _SHA.fullmatch(revision):
                raise ValueError(f"{self.name}: missing full revision for {repo}")
            tree_response = self.client.get(
                f"https://huggingface.co/api/models/{repo}/tree/{revision}",
                params={"recursive": "true", "expand": "true"},
                headers={"Accept": "application/json"},
            )
            self._require(tree_response, f"Hub file listing for {repo}")
            tree = tree_response.json()
            if not isinstance(tree, list):
                raise ValueError(f"{self.name}: malformed file listing for {repo}")
            listings.append(tree_response.body)
            weights = [
                row for row in tree
                if isinstance(row, Mapping)
                and row.get("type") == "file"
                and row.get("path") in _WEIGHT_NAMES
            ]
            if len(weights) != 1:
                raise ValueError(f"{self.name}: expected one recognized root weight in {repo}")
            weight = weights[0]
            path, oid, size = weight.get("path"), weight.get("oid"), weight.get("size")
            if not isinstance(path, str) or not _PATH.fullmatch(path):
                raise ValueError(f"{self.name}: invalid weight path for {repo}")
            if not isinstance(oid, str) or not oid:
                raise ValueError(f"{self.name}: missing object id for {repo}")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"{self.name}: invalid size for {repo}")
            records.append(self._record(repo, item, revision, path, oid, size, response.body))

        checked_at = self.clock()
        if checked_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return SourcePage(
            records=tuple(records),
            next_state={
                "checked_at": checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "collection_sha256": content_hash(response.body),
                "listing_sha256": content_hash(b"\n".join(listings)),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self, repo: str, item: Mapping[str, Any], revision: str, path: str,
        oid: str, size: int, collection_body: bytes,
    ) -> SourceRecord:
        name = repo.rsplit("/", 1)[-1]
        local_id = f"model:{name.casefold()}"
        identifier = Identifier("flower:model", repo)
        artifact = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(path, safe='/')}"
        note = item.get("note")
        note = note.get("text", "") if isinstance(note, Mapping) else ""
        metadata = {
            "collection": _COLLECTION,
            "hub_repository": repo,
            "revision": revision,
            "weight_path": path,
            "weight_oid": oid,
            "weight_size_bytes": size,
            "note": note,
            "collection_sha256": content_hash(collection_body),
        }
        return SourceRecord(
            source_record_id=f"flower:{name.casefold()}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(artifact),
            title=f"FLOWER {name.removeprefix('flower_').replace('_', ' ').title()}",
            raw=metadata,
            text=f"{name}; {note}",
            identifiers=(identifier,),
            links=(
                Link(artifact, relation="model_artifact", locator=path, crawl=False,
                     model_local_ids=(local_id,)),
                Link(f"https://huggingface.co/{repo}", relation="model_card", crawl=False),
                Link(self.collection_url, relation="model_collection", crawl=False),
            ),
            models=(ModelHint(
                local_id=local_id, name=name, identifiers=(identifier,), aliases=(repo,),
                status=ModelStatus.RELEASED, locator=f"{_COLLECTION}:item:{repo}",
            ),),
            releases=(ReleaseHint(
                local_id=f"release:{name.casefold()}", model_local_id=local_id,
                revision=revision,
                identifiers=(Identifier("flower:checkpoint", f"{repo}@{revision}:{path}"),),
                metadata=metadata, locator=f"{_COLLECTION}:item:{repo}",
            ),),
        )

    def _require(self, response: HttpResponse, label: str) -> None:
        if response.status != 200:
            raise ValueError(f"{self.name}: {label} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: {label} exceeds {self.max_response_bytes} bytes")
