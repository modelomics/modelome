"""First-party GPT4All desktop/Python model-download manifest."""

from __future__ import annotations

import json
import re
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
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import content_hash

_REPOSITORY = "nomic-ai/gpt4all"
_BRANCH = "main"
_MANIFEST_PATH = "gpt4all-chat/metadata/models3.json"
_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,254}$")
_MD5 = re.compile(r"^[0-9a-fA-F]{32}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


class Gpt4AllModelCatalogSourceAdapter:
    """Enumerate exact downloadable model files from GPT4All's official manifest.

    The source is the same `models3.json` manifest that GPT4All documents for
    its Python `list_models()` helper. Each admitted row has an exact filename
    and provider-declared GPT4All-hosted HTTPS download URL. Rows that point to
    Hugging Face are omitted because the upstream hub is already separately
    indexed. The adapter records URLs as non-crawled artifact references and
    never downloads model bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Enumerates GPT4All's first-party `models3.json` download manifest. It "
        "excludes Hugging Face-hosted rows already covered by hub catalogs and "
        "does not enumerate arbitrary Ollama models. It "
        "does not fetch model artifacts. Rows without exact filenames and "
        "provider-declared download URLs are quarantined."
    )

    def __init__(
        self,
        *,
        name: str = "gpt4all-model-catalog",
        repository: str = _REPOSITORY,
        branch: str = _BRANCH,
        source_path: str = _MANIFEST_PATH,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 10_000,
        client: HttpClient | Any | None = None,
        clock=_utcnow,
    ) -> None:
        if not name.strip() or repository != _REPOSITORY or branch != _BRANCH:
            raise ValueError("name and official GPT4All repository/main branch are required")
        if source_path != _MANIFEST_PATH:
            raise ValueError("source_path must name GPT4All's official models3.json")
        if max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("response and entry limits must be positive")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.source_path = source_path
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "gpt4all-model-catalog-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "admission": "filename-and-first-party-manifest-url",
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{self.branch}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit_response.status != 200:
            raise ValueError(
                f"{self.name}: commit endpoint returned HTTP {commit_response.status}"
            )
        commit_payload = commit_response.json()
        commit = (
            commit_payload.get("sha")
            if isinstance(commit_payload, Mapping)
            else None
        )
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"{self.name}: invalid repository revision")

        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if commit == state.get("completed_revision"):
            return SourcePage(
                records=(),
                next_state={**state, "checked_at": checked},
                complete=True,
                upstream_count=state.get("model_count"),
            )

        manifest_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{commit}/{self.source_path}"
        )
        response = self.client.get(manifest_url, headers={"Accept": "application/json"})
        if response.status != 200:
            raise ValueError(f"{self.name}: manifest returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: manifest exceeds response limit")
        try:
            payload = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{self.name}: manifest is not valid JSON") from exc
        if not isinstance(payload, list):
            raise ValueError(f"{self.name}: manifest root must be an array")
        if len(payload) > self.max_entries:
            raise ValueError(f"{self.name}: model count exceeds configured limit")

        source_url = f"{self.repository_url}/blob/{commit}/{self.source_path}"
        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        seen_filenames: set[str] = set()
        for index, row in enumerate(payload):
            if not isinstance(row, Mapping):
                issues.append(
                    SourceIssue(str(index), "parse", "manifest row is not an object")
                )
                continue
            filename = _text(row.get("filename"))
            name = _text(row.get("name"))
            download_url = _download_url(row.get("url"))
            if not filename or not _FILENAME.fullmatch(filename) or not name or not download_url:
                issues.append(
                    SourceIssue(
                        _text(row.get("order")) or str(index),
                        "parse",
                        (
                            "row lacks a safe exact filename, display name, or "
                            "supported HTTPS download URL"
                        ),
                        summary={
                            "name": name,
                            "filename": filename,
                            "order": _text(row.get("order")),
                        },
                    )
                )
                continue
            if filename in seen_filenames:
                raise ValueError(f"{self.name}: duplicate model filename {filename!r}")
            seen_filenames.add(filename)
            self._validate_checksums(row, filename)
            records.append(
                self._record(
                    row=row,
                    filename=filename,
                    name=name,
                    download_url=download_url,
                    source_url=source_url,
                    revision=commit,
                )
            )
        if not records:
            raise ValueError(f"{self.name}: no downloadable model rows found")
        return SourcePage(
            records=tuple(records),
            next_state={
                "completed_revision": commit,
                "checked_at": checked,
                "manifest_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
            issues=tuple(issues),
            advance_on_source_issues=bool(issues),
        )

    @staticmethod
    def _validate_checksums(row: Mapping[str, Any], filename: str) -> None:
        for key, pattern in (("md5sum", _MD5), ("sha256sum", _SHA256)):
            value = row.get(key)
            if value is None:
                continue
            if not isinstance(value, str) or not pattern.fullmatch(value):
                raise ValueError(f"invalid {key} for model file {filename!r}")

    @staticmethod
    def _record(
        *,
        row: Mapping[str, Any],
        filename: str,
        name: str,
        download_url: str,
        source_url: str,
        revision: str,
    ) -> SourceRecord:
        namespace = "gpt4all:model-file"
        local_id = f"model-file:{filename}"
        identifier = Identifier(namespace, filename)
        model = ModelHint(
            local_id=local_id,
            name=name,
            identifiers=(identifier,),
            aliases=(filename,),
            status=ModelStatus.RELEASED,
            locator="manifest:filename",
        )
        release = ReleaseHint(
            local_id=f"{local_id}#release",
            model_local_id=local_id,
            version=filename,
            revision=revision,
            identifiers=(Identifier(f"{namespace}:release", filename),),
            metadata={
                "filename": filename,
                "download_url": download_url,
                "md5sum": row.get("md5sum"),
                "sha256sum": row.get("sha256sum"),
                "filesize": row.get("filesize"),
                "quantization": row.get("quant"),
                "parameters": row.get("parameters"),
                "minimum_gpt4all_version": row.get("requires"),
                "removed_in_gpt4all_version": row.get("removedIn"),
                "manifest_revision": revision,
            },
            locator="manifest:filename",
        )
        raw = dict(row)
        raw["gpt4all_manifest_revision"] = revision
        return SourceRecord(
            source_record_id=filename,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=download_url,
            title=filename,
            raw=raw,
            identifiers=(identifier,),
            links=(
                Link(
                    download_url,
                    relation="model_artifact",
                    locator="manifest:url",
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
                Link(
                    source_url,
                    relation="catalog_record",
                    locator="manifest:source",
                    crawl=False,
                    model_local_ids=(local_id,),
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _download_url(value: Any) -> str | None:
    url = _text(value)
    if not url:
        return None
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not (host == "gpt4all.io" or host.endswith(".gpt4all.io"))
    ):
        return None
    return url


__all__ = ["Gpt4AllModelCatalogSourceAdapter"]
