"""RadiologyNET's first-party model-to-archive download table."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

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
_REPOSITORY = "AIlab-RITEH/RadiologyNET-TL-models"
_README_PATH = "README.md"
_MAX_ENTRIES = 100
_DOWNLOAD_ROW = re.compile(
    r"^\|\s*(?P<name>[A-Za-z0-9][A-Za-z0-9_-]*)\s*\|\s*"
    r"(?P<size>[0-9.]+\s+(?:MiB|GiB))\s*\|\s*"
    r"\[Download\]\((?P<url>https://drive\.google\.com/uc\?[^)]+)\)\s*\|$"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RadiologyNETCheckpointSourceAdapter:
    """Enumerate exact model names and Google Drive archive IDs from README.

    The README says these are per-model `.tar.gz` archives. No archive contents
    or internal checkpoint paths are inferred or fetched.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the ten named PyTorch archives in RadiologyNET's README Download "
        "table. It does not inspect archive members, models absent from the table, "
        "or other project outputs."
    )

    def __init__(
        self,
        *,
        name: str = "radiologynet-first-party-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "master",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = _MAX_ENTRIES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "radiologynet-first-party-checkpoint-table-v1",
            "repository": repository,
            "branch": branch,
            "readme_path": _README_PATH,
            "max_response_bytes": max_response_bytes,
            "max_entries": max_entries,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        readme_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{self.branch}/{_README_PATH}"
        )
        response = self.client.get(readme_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        digest = hashlib.sha256(response.body).hexdigest()
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if digest == state.get("completed_revision"):
            return SourcePage(
                (), {**state, "checked_at": checked}, True,
                upstream_count=state.get("checkpoint_count"),
            )
        checkpoints = _parse_download_table(response.text(), self.max_entries)
        records = tuple(_record(item, readme_url, self.name) for item in checkpoints)
        return SourcePage(
            records,
            {"completed_revision": digest, "checked_at": checked,
             "checkpoint_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_download_table(readme: str, limit: int) -> list[dict[str, str]]:
    in_download_section = False
    saw_section = False
    rows: dict[str, dict[str, str]] = {}
    file_ids: set[str] = set()
    for line in readme.splitlines():
        if line.strip().casefold() == "## download":
            in_download_section = True
            saw_section = True
            continue
        if in_download_section and line.startswith("## "):
            break
        if not in_download_section:
            continue
        match = _DOWNLOAD_ROW.fullmatch(line)
        if not match:
            continue
        name, url = match.group("name"), match.group("url")
        parts = urlsplit(url)
        query = parse_qs(parts.query, strict_parsing=True)
        file_id = query.get("id", [None])[0]
        if (
            parts.netloc != "drive.google.com" or parts.path != "/uc"
            or query.get("export") != ["download"]
            or not isinstance(file_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", file_id)
            or name in rows or file_id in file_ids
        ):
            raise ValueError(f"RadiologyNET has an invalid or duplicate download row for {name}")
        rows[name] = {
            "name": name,
            "size": match.group("size"),
            "archive_url": url,
            "drive_file_id": file_id,
        }
        file_ids.add(file_id)
        if len(rows) > limit:
            raise ValueError("RadiologyNET checkpoint table exceeds entry limit")
    if not saw_section or not rows:
        raise ValueError("RadiologyNET README Download section has no admitted rows")
    return [rows[name] for name in sorted(rows)]


def _record(item: dict[str, str], readme_url: str, source_name: str) -> SourceRecord:
    name = item["name"]
    local_id = f"model:{name}"
    model = ModelHint(
        local_id, f"RadiologyNET {name}",
        identifiers=(Identifier("radiologynet:model", name),),
        aliases=(name,), status=ModelStatus.RELEASED,
    )
    release = ReleaseHint(
        f"archive:{name}", local_id,
        identifiers=(Identifier("radiologynet:google-drive-file", item["drive_file_id"]),),
        metadata={
            "architecture": name,
            "archive_format": "tar.gz",
            "declared_size": item["size"],
            "archive_url": item["archive_url"],
            "google_drive_file_id": item["drive_file_id"],
            "internal_checkpoint_path": None,
        },
        locator=f"RadiologyNET README Download table row: {name}",
    )
    return SourceRecord(
        source_record_id=f"{source_name}:archive:{name}",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=canonicalize_url(item["archive_url"]),
        title=f"RadiologyNET {name} pretrained weights archive",
        raw={"model": name, "archive_url": item["archive_url"],
             "drive_file_id": item["drive_file_id"], "declared_size": item["size"],
             "readme_url": readme_url},
        text=f"RadiologyNET pretrained PyTorch weights for {name}; archive {item['size']}.",
        identifiers=(Identifier("radiologynet:model", name),),
        links=(
            Link(readme_url, "model_card", crawl=False, model_local_ids=(local_id,)),
            Link(item["archive_url"], "weights", crawl=False, model_local_ids=(local_id,)),
        ),
        models=(model,), releases=(release,),
    )


__all__ = ["RadiologyNETCheckpointSourceAdapter"]
