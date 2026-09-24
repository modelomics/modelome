"""Ai2's first-party MolmoBot policy collection and exact Hub weight files."""

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

_COLLECTION = "allenai/molmobot-models"
_COLLECTION_TITLE = "MolmoBot-Models"
_HF_ORG = "allenai"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID = re.compile(r"^allenai/MolmoBot-[A-Za-z0-9.-]+$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MolmoBotPolicyCollectionAdapter:
    """Enumerate Ai2 collection members and resolve their exact weight files."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public models in Ai2's MolmoBot-Models collection when each repository "
        "publishes one unambiguous root model.pt or model.safetensors file. Auxiliary "
        "metadata and non-collection repositories are not treated as model weights."
    )

    def __init__(
        self,
        *,
        name: str = "molmobot-policy-collection",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 32,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
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
                "adapter": "molmobot-policy-collection-v1",
                "collection": _COLLECTION,
                "collection_title": _COLLECTION_TITLE,
                "hf_organization": _HF_ORG,
                "admission": (
                    "public collection model with one root model.pt or model.safetensors"
                ),
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
            }
        )

    @property
    def collection_url(self) -> str:
        return f"https://huggingface.co/collections/{_COLLECTION}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        collection_response: HttpResponse = self.client.get(
            f"https://huggingface.co/api/collections/{_COLLECTION}",
            headers={"Accept": "application/json"},
        )
        self._require_response(collection_response, "first-party collection API")
        payload = collection_response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: collection response is not an object")
        if payload.get("title") != _COLLECTION_TITLE or payload.get("private") is not False:
            raise ValueError(f"{self.name}: collection identity or visibility changed")
        owner = payload.get("owner")
        if not isinstance(owner, Mapping) or owner.get("name") != _HF_ORG:
            raise ValueError(f"{self.name}: collection is not owned by Ai2")
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError(f"{self.name}: collection has no model inventory")
        if len(items) > self.max_entries:
            raise ValueError(f"{self.name}: collection exceeds {self.max_entries} entries")

        checked_items: dict[str, Mapping[str, Any]] = {}
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"{self.name}: malformed collection item")
            repo_id = item.get("id")
            if (
                item.get("type") != "model"
                or item.get("repoType") != "model"
                or item.get("private") is not False
                or item.get("author") != _HF_ORG
                or not isinstance(repo_id, str)
                or not _REPO_ID.fullmatch(repo_id)
            ):
                raise ValueError(f"{self.name}: collection has a non-public or unexpected item")
            if repo_id in checked_items:
                raise ValueError(f"{self.name}: duplicate collection model {repo_id}")
            checked_items[repo_id] = item

        records: list[SourceRecord] = []
        listing_hashes: list[bytes] = []
        for repo_id, item in checked_items.items():
            info_response: HttpResponse = self.client.get(
                f"https://huggingface.co/api/models/{repo_id}",
                headers={"Accept": "application/json"},
            )
            self._require_response(info_response, f"Hub model metadata {repo_id}")
            info = info_response.json()
            revision = info.get("sha", "") if isinstance(info, Mapping) else ""
            if not isinstance(revision, str) or not _SHA.fullmatch(revision):
                raise ValueError(f"{self.name}: no full revision SHA for {repo_id}")
            tree_response: HttpResponse = self.client.get(
                f"https://huggingface.co/api/models/{repo_id}/tree/{revision}",
                params={"recursive": "false", "expand": "false"},
                headers={"Accept": "application/json"},
            )
            self._require_response(tree_response, f"Hub file listing {repo_id}")
            tree = tree_response.json()
            if not isinstance(tree, list):
                raise ValueError(f"{self.name}: Hub file listing is not a list for {repo_id}")
            listing_hashes.append(tree_response.body)
            weights = [
                entry
                for entry in tree
                if isinstance(entry, Mapping)
                and entry.get("type") == "file"
                and entry.get("path") in {"model.pt", "model.safetensors"}
            ]
            if len(weights) != 1:
                raise ValueError(f"{self.name}: expected one root weight file for {repo_id}")
            weight = weights[0]
            path = weight["path"]
            if not _SAFE_PATH.fullmatch(path):
                raise ValueError(f"{self.name}: unsafe weight path in {repo_id}")
            oid = weight.get("oid")
            size_bytes = weight.get("size")
            if not isinstance(oid, str) or not oid.strip():
                raise ValueError(f"{self.name}: weight has no object id for {repo_id}")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
                raise ValueError(f"{self.name}: weight has invalid size for {repo_id}")
            lfs = weight.get("lfs") if isinstance(weight.get("lfs"), Mapping) else {}
            records.append(
                self._record(
                    repo_id,
                    item,
                    revision,
                    path,
                    oid,
                    size_bytes,
                    lfs,
                    collection_response.body,
                )
            )

        checked_at = self.clock()
        if checked_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return SourcePage(
            records=tuple(records),
            next_state={
                "checked_at": checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "collection_sha256": content_hash(collection_response.body),
                "listing_sha256": content_hash(b"\n".join(listing_hashes)),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        repo_id: str,
        item: Mapping[str, Any],
        revision: str,
        path: str,
        oid: str,
        size_bytes: int,
        lfs: Mapping[str, Any],
        collection_body: bytes,
    ) -> SourceRecord:
        model_name = repo_id.rsplit("/", 1)[-1]
        local_id = f"model:{model_name.casefold()}"
        identifier = Identifier("molmobot:model", repo_id)
        file_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{quote(path, safe='/')}"
        note = item.get("note")
        note_text = note.get("text", "") if isinstance(note, Mapping) else ""
        metadata = {
            "collection": _COLLECTION,
            "hub_repository": repo_id,
            "revision": revision,
            "weight_path": path,
            "weight_oid": oid,
            "weight_size_bytes": size_bytes,
            "lfs_oid": lfs.get("oid"),
            "note": note_text,
            "last_modified": item.get("lastModified"),
            "pipeline_tag": item.get("pipeline_tag"),
            "parameter_count": item.get("numParameters"),
            "collection_sha256": content_hash(collection_body),
        }
        model = ModelHint(
            local_id=local_id,
            name=model_name,
            identifiers=(identifier,),
            aliases=(repo_id,),
            status=ModelStatus.RELEASED,
            locator=f"{_COLLECTION}:item:{repo_id}",
        )
        release = ReleaseHint(
            local_id=f"release:{model_name.casefold()}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("molmobot:checkpoint", f"{repo_id}@{revision}:{path}"),),
            metadata=metadata,
            locator=f"{_COLLECTION}:item:{repo_id}",
        )
        return SourceRecord(
            source_record_id=f"molmobot:{model_name.casefold()}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(file_url),
            title=f"MolmoBot {model_name.removeprefix('MolmoBot-')}",
            raw=metadata,
            text=f"{model_name}; {note_text}",
            identifiers=(identifier,),
            links=(
                Link(
                    file_url,
                    relation="model_artifact",
                    locator=path,
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
                Link(f"https://huggingface.co/{repo_id}", relation="model_card", crawl=False),
                Link(self.collection_url, relation="model_collection", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )

    def _require_response(self, response: HttpResponse, label: str) -> None:
        if response.status != 200:
            raise ValueError(f"{self.name}: {label} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: {label} exceeds {self.max_response_bytes} bytes")
