"""Pinned inventory of pretrained models in the official Gensim-data list."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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

Clock = Callable[[], datetime]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_HANDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_REPOSITORY = "RaRe-Technologies/gensim-data"
_MANIFEST = "list.json"
_DOWNLOAD_BASE = "https://github.com/RaRe-Technologies/gensim-data/releases/download"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GensimDownloaderModelRegistrySourceAdapter:
    """Enumerate single-file pretrained models declared under list.json/models.

    The first-party manifest separates model handles from corpora. The adapter
    uses those literal handle identities and admits only one-part gzip model
    weights with a matching MD5 and first-party reader-code release path. The
    download URL follows Gensim's downloader implementation and is not fetched.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers single-file model weights in Gensim-data's official `models` "
        "manifest section. Corpora, multipart entries, malformed/incomplete rows, "
        "and weight-byte downloads are excluded."
    )

    def __init__(
        self,
        *,
        name: str = "gensim-downloader-models",
        repository: str = _REPOSITORY,
        branch: str = "master",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 10_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not name.strip() or repository != _REPOSITORY or not branch.strip():
            raise ValueError("name, official repository, and branch are required")
        if max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("response and entry limits must be positive")
        self.name, self.repository, self.branch = name, repository, branch
        self.max_response_bytes, self.max_entries = max_response_bytes, max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "gensim-downloader-models-v1",
            "repository": repository,
            "branch": branch,
            "manifest": _MANIFEST,
            "admission": "models section, one gzip file, matching MD5 and reader path",
            "max_response_bytes": max_response_bytes,
            "max_entries": max_entries,
        })

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        commit_payload = commit.json()
        revision = commit_payload.get("sha") if isinstance(commit_payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage((), {**state, "checked_at": checked}, True,
                              upstream_count=state.get("model_count"))

        manifest_url = (
            f"https://raw.githubusercontent.com/{self.repository}/{revision}/{_MANIFEST}"
        )
        response = self.client.get(manifest_url, headers={"Accept": "application/json"})
        if response.status != 200:
            raise ValueError(f"{self.name}: manifest returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: manifest exceeds response limit")
        try:
            payload = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{self.name}: invalid JSON manifest") from exc
        models = payload.get("models") if isinstance(payload, Mapping) else None
        if not isinstance(models, Mapping):
            raise ValueError(f"{self.name}: manifest has no models object")
        if len(models) > self.max_entries:
            raise ValueError(f"{self.name}: model count exceeds configured limit")
        rows = _model_rows(models, self.name)
        if not rows:
            raise ValueError(f"{self.name}: no supported model weight rows found")
        records = tuple(self._record(handle, row, revision) for handle, row in rows)
        return SourcePage(
            records,
            {"completed_revision": revision, "checked_at": checked,
             "manifest_sha256": content_hash(response.body), "model_count": len(records)},
            True, upstream_count=len(records), authoritative_snapshot=True,
        )

    def _record(self, handle: str, row: Mapping[str, Any], revision: str) -> SourceRecord:
        filename = f"{handle}.gz"
        weight_url = f"{_DOWNLOAD_BASE}/{quote(handle, safe='')}/{quote(filename, safe='')}"
        manifest_url = (
            f"{self.repository_url}/blob/{revision}/{_MANIFEST}"
        )
        model_id = f"model:{handle}"
        namespace = "gensim:model"
        description = row.get("description")
        model = ModelHint(
            model_id,
            handle,
            identifiers=(Identifier(namespace, handle),),
            aliases=(handle,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}", model_id, version=handle,
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={
                "repository": self.repository,
                "manifest_revision": revision,
                "checkpoint_handle": handle,
                "checkpoint_filename": filename,
                "weight_url": weight_url,
                "file_size": row["file_size"],
                "md5": row["checksum"],
                "parts": 1,
            },
        )
        return SourceRecord(
            source_record_id=f"model:{handle}", kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(weight_url), title=f"Gensim {handle}",
            raw={"repository": self.repository, "manifest_revision": revision,
                 "manifest_path": _MANIFEST, "handle": handle,
                 "file_name": filename, "weight_url": weight_url,
                 "file_size": row["file_size"], "md5": row["checksum"],
                 "description": description},
            text=str(description or f"Gensim-data pretrained model: {handle}"),
            identifiers=(Identifier(namespace, handle),),
            links=(
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(manifest_url, "model_card", crawl=False,
                     model_local_ids=(model_id,)),
                Link(self.repository_url, "source_implementation", crawl=False,
                     model_local_ids=(model_id,)),
            ),
            models=(model,), releases=(release,),
        )


def _model_rows(
    models: Mapping[str, Any], source: str
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for handle, row in sorted(models.items()):
        if not isinstance(handle, str) or not _HANDLE.fullmatch(handle):
            raise ValueError(f"{source}: invalid source-native model handle")
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}: model {handle} is not an object")
        reader_code = row.get("reader_code")
        parts = row.get("parts")
        file_size = row.get("file_size")
        if (
            row.get("file_name") != f"{handle}.gz"
            or parts != 1
            or not isinstance(file_size, int)
            or isinstance(file_size, bool)
            or file_size <= 0
            or not isinstance(row.get("checksum"), str)
            or not _MD5.fullmatch(row["checksum"])
            or reader_code != f"{_DOWNLOAD_BASE}/{handle}/__init__.py"
        ):
            continue
        rows.append((handle, row))
    return tuple(rows)


__all__ = ["GensimDownloaderModelRegistrySourceAdapter"]
