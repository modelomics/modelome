"""Tencent MedicalNet's named 3D ResNet checkpoint bundle inventory."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

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
_REPOSITORY = "Tencent/MedicalNet"
_README_PATH = "README.md"
_ARCHIVE_NAME = "MedicalNet_pytorch_files2.zip"
_DRIVE_FILE_ID = "13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG"
_ARCHIVE_URL = f"https://drive.google.com/file/d/{_DRIVE_FILE_ID}/view?usp=sharing"
_CHECKPOINT_LINE = re.compile(
    r"^\s*(?P<filename>resnet_(?P<depth>10|18|34|50|101|152|200)"
    r"(?P<cohort>_23dataset)?\.pth):\s+"
    r"--model resnet --model_depth (?P<declared_depth>\d+)\s+"
    r"--resnet_shortcut (?P<shortcut>[AB])\s*$"
)
_MAX_CHECKPOINTS = 100


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MedicalNetRegistrySourceAdapter:
    """Record README-declared checkpoint member names and the linked archive.

    MedicalNet distributes the named files inside one Google Drive ZIP archive;
    this source does not guess individual file URLs or fetch archive contents.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the eleven ResNet checkpoint filenames explicitly listed in the "
        "first-party README and their shared Google Drive archive handle. It does "
        "not enumerate unlisted files in that archive."
    )

    def __init__(
        self,
        *,
        name: str = "medicalnet-pretrained-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "master",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = _MAX_CHECKPOINTS,
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
            "adapter": "medicalnet-first-party-checkpoint-members-v1",
            "repository": repository,
            "branch": branch,
            "readme_path": _README_PATH,
            "drive_file_id": _DRIVE_FILE_ID,
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
        checkpoints = _parse_checkpoints(response.text(), self.max_entries)
        record = _record(checkpoints, readme_url)
        return SourcePage(
            (record,),
            {"completed_revision": digest, "checked_at": checked,
             "checkpoint_count": len(checkpoints)},
            True,
            upstream_count=len(checkpoints),
            authoritative_snapshot=True,
        )


def _parse_checkpoints(readme: str, limit: int) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in readme.splitlines():
        match = _CHECKPOINT_LINE.fullmatch(line)
        if not match:
            continue
        filename = match.group("filename")
        depth = match.group("depth")
        if match.group("declared_depth") != depth:
            raise ValueError(f"MedicalNet checkpoint {filename} has mismatched depth")
        if filename in rows:
            raise ValueError(f"MedicalNet README repeats checkpoint {filename}")
        rows[filename] = {
            "filename": filename,
            "depth": depth,
            "cohort": "23-dataset" if match.group("cohort") else "legacy listed cohort",
            "shortcut": match.group("shortcut"),
        }
        if len(rows) > limit:
            raise ValueError("MedicalNet checkpoint list exceeds entry limit")
    if not rows:
        raise ValueError("MedicalNet README contains no admitted checkpoint rows")
    return [rows[name] for name in sorted(rows)]


def _record(checkpoints: list[dict[str, str]], readme_url: str) -> SourceRecord:
    namespace = "medicalnet:checkpoint"
    models: list[ModelHint] = []
    releases: list[ReleaseHint] = []
    for item in checkpoints:
        filename = item["filename"]
        local_id = f"checkpoint:{filename}"
        cohort_text = "23-dataset" if item["cohort"] == "23-dataset" else "legacy listed cohort"
        title = f"MedicalNet 3D-ResNet-{item['depth']} ({cohort_text})"
        models.append(ModelHint(
            local_id, title,
            identifiers=(Identifier(namespace, filename),),
            aliases=(filename,), status=ModelStatus.RELEASED,
        ))
        releases.append(ReleaseHint(
            f"archive-member:{filename}", local_id,
            identifiers=(Identifier(f"{namespace}:archive-member", filename),),
            metadata={
                "checkpoint_filename": filename,
                "checkpoint_depth": int(item["depth"]),
                "checkpoint_variant": item["cohort"],
                "shortcut": item["shortcut"],
                "archive_filename": _ARCHIVE_NAME,
                "archive_file_id": _DRIVE_FILE_ID,
                "archive_url": _ARCHIVE_URL,
                "archive_member_url": None,
            },
            locator=f"MedicalNet README checkpoint row: {filename}",
        ))
    return SourceRecord(
        source_record_id="medicalnet:pretrained-checkpoint-inventory",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=canonicalize_url(readme_url),
        title="MedicalNet named pretrained 3D ResNet checkpoints",
        raw={"repository": _REPOSITORY, "readme_url": readme_url,
             "archive_url": _ARCHIVE_URL, "checkpoints": checkpoints},
        text="\n".join(
            f"{item['filename']} depth={item['depth']} shortcut={item['shortcut']}"
            for item in checkpoints
        ),
        identifiers=(Identifier("medicalnet:archive-file-id", _DRIVE_FILE_ID),),
        links=(
            Link(readme_url, "model_card", crawl=False),
            Link(_ARCHIVE_URL, "weights", crawl=False),
        ),
        models=tuple(models),
        releases=tuple(releases),
    )


__all__ = ["MedicalNetRegistrySourceAdapter"]
