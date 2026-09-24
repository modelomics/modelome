"""Microsoft CogACT's official size variants and exact checkpoint files."""

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

_REPOSITORY = "microsoft/CogACT"
_DOCUMENT = "README.md"
_HF_ORG = "CogACT"
_VARIANTS = ("Small", "Base", "Large")
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CogACTCheckpointRegistryAdapter:
    """Join the first-party three-size release list to exact Hub checkpoint files."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the three CogACT size variants named by the Microsoft repository and "
        "their exact published `.pt` checkpoints. It does not enumerate fine-tuned "
        "community repositories or download weight data."
    )

    def __init__(
        self,
        *,
        name: str = "cogact-checkpoint-registry",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 8,
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
        if max_entries < len(_VARIANTS):
            raise ValueError(
                f"max_entries must allow all {len(_VARIANTS)} official CogACT variants"
            )
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "cogact-checkpoint-registry-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "hf_organization": _HF_ORG,
                "variants": _VARIANTS,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "three README-declared sizes joined to matching Hub checkpoint file",
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{_REPOSITORY}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        document_url = f"https://raw.githubusercontent.com/{_REPOSITORY}/main/{_DOCUMENT}"
        document_response: HttpResponse = self.client.get(
            document_url, headers={"Accept": "text/markdown,text/plain"}
        )
        self._require_response(document_response, "first-party README")
        document = document_response.text()
        if not _declares_all_variants(document):
            raise ValueError(f"{self.name}: README no longer declares all three size variants")

        records: list[SourceRecord] = []
        listing_hashes: list[bytes] = []
        for size in _VARIANTS:
            model_name = f"CogACT-{size}"
            repo_id = f"{_HF_ORG}/{model_name}"
            info_response: HttpResponse = self.client.get(
                f"https://huggingface.co/api/models/{repo_id}",
                headers={"Accept": "application/json"},
            )
            self._require_response(info_response, f"Hub model metadata {repo_id}")
            info = info_response.json()
            revision = info.get("sha", "") if isinstance(info, Mapping) else ""
            if not isinstance(revision, str) or not _SHA.fullmatch(revision):
                raise ValueError(
                    f"{self.name}: Hub metadata did not return a full commit SHA for {repo_id}"
                )

            response: HttpResponse = self.client.get(
                f"https://huggingface.co/api/models/{repo_id}/tree/{revision}/checkpoints",
                params={"recursive": "false", "expand": "false"},
                headers={"Accept": "application/json"},
            )
            self._require_response(response, f"checkpoint listing {repo_id}")
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"{self.name}: checkpoint listing is not a list for {repo_id}")
            listing_hashes.append(response.body)
            expected_path = f"checkpoints/{model_name}.pt"
            matches = [
                item
                for item in payload
                if isinstance(item, Mapping)
                and item.get("type") == "file"
                and item.get("path") == expected_path
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{self.name}: expected exactly one published file {expected_path}"
                )
            item = matches[0]
            oid = item.get("oid")
            size_bytes = item.get("size")
            if not isinstance(oid, str) or not oid.strip():
                raise ValueError(f"{self.name}: checkpoint file has no object id: {expected_path}")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
                raise ValueError(f"{self.name}: checkpoint file has invalid size: {expected_path}")
            lfs = item.get("lfs") if isinstance(item.get("lfs"), Mapping) else {}
            records.append(
                self._record(
                    size,
                    repo_id,
                    revision,
                    expected_path,
                    oid,
                    size_bytes,
                    lfs,
                    document_response.body,
                    document_url,
                )
            )

        checked_at = self.clock()
        if checked_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return SourcePage(
            records=tuple(records),
            next_state={
                "checked_at": checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "document_sha256": content_hash(document_response.body),
                "checkpoint_listing_sha256": content_hash(b"\n".join(listing_hashes)),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        size: str,
        repo_id: str,
        revision: str,
        path: str,
        oid: str,
        size_bytes: int,
        lfs: Mapping[str, Any],
        document: bytes,
        document_url: str,
    ) -> SourceRecord:
        model_name = f"CogACT-{size}"
        model_local_id = f"model:{size.casefold()}"
        identifier = Identifier("cogact:model", model_name)
        file_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{quote(path, safe='/')}"
        metadata = {
            "repository": _REPOSITORY,
            "document_path": _DOCUMENT,
            "hub_repository": repo_id,
            "revision": revision,
            "checkpoint_path": path,
            "checkpoint_oid": oid,
            "checkpoint_size_bytes": size_bytes,
            "lfs_oid": lfs.get("oid"),
            "action_model_size": size,
            "source_document_sha256": content_hash(document),
        }
        model = ModelHint(
            local_id=model_local_id,
            name=model_name,
            identifiers=(identifier,),
            aliases=(f"CogACT {size}",),
            status=ModelStatus.RELEASED,
            locator=f"README.md:CogACT-{size}",
        )
        release = ReleaseHint(
            local_id=f"release:{size.casefold()}",
            model_local_id=model_local_id,
            revision=revision,
            identifiers=(Identifier("cogact:checkpoint", f"{repo_id}@{revision}:{path}"),),
            metadata=metadata,
            locator=f"README.md:CogACT-{size}",
        )
        return SourceRecord(
            source_record_id=f"cogact:{size.casefold()}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(file_url),
            title=f"CogACT {size}",
            raw=metadata,
            text=f"{model_name}; official CogACT action model size {size}",
            identifiers=(identifier,),
            links=(
                Link(
                    file_url,
                    relation="model_artifact",
                    locator=path,
                    crawl=False,
                    model_local_ids=(model_local_id,),
                ),
                Link(f"https://huggingface.co/{repo_id}", relation="model_card", crawl=False),
                Link(
                    document_url,
                    relation="model_catalog",
                    locator=f"README.md:CogACT-{size}",
                    crawl=False,
                ),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )

    def _require_response(self, response: HttpResponse, label: str) -> None:
        if response.status != 200:
            raise ValueError(f"{self.name}: {label} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: {label} exceeds {self.max_response_bytes} bytes")


def _declares_all_variants(document: str) -> bool:
    if not re.search(r"release\s+three\s+CogACT\s+models", document, flags=re.IGNORECASE):
        return False
    return all(re.search(rf"CogACT-{size}\b", document) for size in _VARIANTS)
