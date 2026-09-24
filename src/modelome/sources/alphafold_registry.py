"""Enumerate the current official AlphaFold parameter archive URL."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

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
_SOURCE_URL = re.compile(
    r'^\s*SOURCE_URL="(https://storage\.googleapis\.com/alphafold/[^"\s]+\.tar)"\s*$',
    re.M,
)
_REPOSITORY = "google-deepmind/alphafold"
_SCRIPT_PATH = "scripts/download_alphafold_params.sh"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AlphaFoldParameterArchiveSourceAdapter:
    """Read the literal archive URL selected by AlphaFold's official script.

    The tar is not fetched or unpacked. The record represents the archive as
    published, without claiming individual parameter names or checksums.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the single archive URL assigned in the current official "
        "download_alphafold_params.sh at the pinned repository revision. It "
        "does not enumerate files inside the tar, deprecated archives, or "
        "weights from third-party AlphaFold implementations."
    )

    def __init__(
        self,
        *,
        name: str = "alphafold-parameter-archive",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_script_bytes: int = 256 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or max_script_bytes <= 0:
            raise ValueError("name, branch, and positive max_script_bytes are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_script_bytes = max_script_bytes
        self.client = client or HttpClient(max_response_bytes=max_script_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "alphafold-parameter-archive-v1",
            "repository": repository,
            "branch": branch,
            "script_path": _SCRIPT_PATH,
            "max_script_bytes": max_script_bytes,
        })

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage((), {**state, "checked_at": checked}, True,
                              upstream_count=state.get("archive_count", 1))

        script_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{revision}/{_SCRIPT_PATH}"
        )
        response = self.client.get(script_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: download script returned HTTP {response.status}")
        if len(response.body) > self.max_script_bytes:
            raise ValueError(f"{self.name}: download script exceeds response limit")
        matches = _SOURCE_URL.findall(response.text())
        if len(matches) != 1:
            raise ValueError(f"{self.name}: expected one literal official SOURCE_URL")
        archive_url = matches[0]
        parsed = urlsplit(archive_url)
        if parsed.netloc != "storage.googleapis.com" or not parsed.path.endswith(".tar"):
            raise ValueError(f"{self.name}: archive URL is outside the admitted endpoint")
        record = self._record(archive_url, script_url, revision)
        return SourcePage(
            (record,),
            {"completed_revision": revision, "checked_at": checked,
             "archive_count": 1, "archive_url": archive_url},
            True,
            upstream_count=1,
            authoritative_snapshot=False,
        )

    def _record(self, archive_url: str, script_url: str, revision: str) -> SourceRecord:
        filename = archive_url.rsplit("/", 1)[-1]
        match = re.fullmatch(r"alphafold_params_(\d{4}-\d{2}-\d{2})\.tar", filename)
        if not match:
            raise ValueError(f"{self.name}: unrecognized parameter archive filename")
        date = match.group(1)
        handle = f"alphafold_params_{date}"
        namespace = "alphafold:parameter-archive"
        model_local_id = f"model:{handle}"
        script_page = f"{self.repository_url}/blob/{revision}/{_SCRIPT_PATH}"
        model = ModelHint(
            model_local_id,
            f"AlphaFold parameters {date}",
            identifiers=(Identifier(namespace, handle),),
            aliases=(filename,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_local_id,
            version=date,
            revision=revision,
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={"archive_filename": filename, "archive_url": archive_url,
                      "source_script": _SCRIPT_PATH, "repository_revision": revision},
        )
        return SourceRecord(
            source_record_id=f"archive:{handle}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(archive_url),
            title=f"AlphaFold parameter archive {date}",
            raw={"repository": self.repository, "revision": revision,
                 "script_path": _SCRIPT_PATH, "script_url": script_url,
                 "archive_filename": filename, "archive_url": archive_url},
            text=f"Official AlphaFold parameter archive: {filename}",
            identifiers=(Identifier(namespace, handle),),
            links=(
                Link(script_page, "source_registry", crawl=False,
                     model_local_ids=(model_local_id,)),
                Link(archive_url, "weights", crawl=False,
                     model_local_ids=(model_local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


__all__ = ["AlphaFoldParameterArchiveSourceAdapter"]
