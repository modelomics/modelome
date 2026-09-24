"""Read the exact MillionAID checkpoint links in the first-party ViTAE RSP README."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from modelome.http import HttpClient
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

_REPOSITORY = "ViTAE-Transformer/RSP"
_README = f"https://raw.githubusercontent.com/{_REPOSITORY}/main/README.md"
_README_PAGE = f"https://github.com/{_REPOSITORY}"
_ROWS: dict[str, tuple[str, str]] = {
    "RSP-ResNet-50-E300": (
        "1K3P4_fDfcBRGqpKoSdSa6OXS4xC1xLC9",
        "RSP ResNet-50, MillionAID, 300 epochs",
    ),
    "RSP-Swin-T-E300": (
        "1G5wjbjIHepmT6VVOuW03bWmyvrhcfe1F",
        "RSP Swin-T, MillionAID, 300 epochs",
    ),
    "RSP-ViTAEv2-S-E100": (
        "1cDB69frN-NxCyoy8lghjx6NiH1JriYUc",
        "RSP ViTAEv2-S, MillionAID, 100 epochs",
    ),
}
_LINK_RE = re.compile(r'\[[^\]]+\]\(https://drive\.google\.com/file/d/(?P<id>[A-Za-z0-9_-]+)(?:/view)?(?:\?[^)]*)?\)')


class VitaeRSPCheckpointRegistrySourceAdapter:
    """Enumerate only the three README-declared remote-sensing pretrained checkpoints."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the three MillionAID pretrained checkpoints in the first-party RSP README. "
        "The README links to Google Drive files; this adapter records the declared links "
        "without resolving Drive metadata or fetching checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "vitae-rsp-checkpoint-registry",
        max_response_bytes: int = 1024 * 1024,
        client: Any | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        if isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "vitae-rsp-checkpoint-registry-v1",
                "repository": _REPOSITORY,
                "readme": _README,
                "file_ids": sorted(item[0] for item in _ROWS.values()),
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(_README, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds configured byte limit")
        found = _parse_readme(response.text(), self.name)
        digest = content_hash(response.body)
        if digest == state.get("completed_readme_sha256"):
            return SourcePage((), dict(state), True, upstream_count=len(found))
        records = tuple(
            _record(self.name, model_id, file_id, digest)
            for model_id, file_id in found.items()
        )
        return SourcePage(
            records,
            {"completed_readme_sha256": digest, "checkpoint_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_readme(document: str, source: str) -> dict[str, str]:
    if document.count("### MillionAID") != 1:
        raise ValueError(f"{source}: expected the MillionAID checkpoint section")
    section = False
    found: dict[str, str] = {}
    for line in document.splitlines():
        stripped = line.strip()
        if stripped == "### MillionAID":
            section = True
            continue
        if section and stripped.startswith("### "):
            break
        if not section:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        model_id = cells[0] if cells else ""
        if not model_id.startswith("RSP-"):
            continue
        if model_id not in _ROWS or model_id in found:
            raise ValueError(f"{source}: unexpected or duplicate checkpoint row {model_id!r}")
        matches = list(_LINK_RE.finditer(stripped))
        expected_id = _ROWS[model_id][0]
        if len(matches) != 1 or matches[0].group("id") != expected_id:
            raise ValueError(f"{source}: expected the exact Google Drive link for {model_id}")
        found[model_id] = expected_id
    if set(found) != set(_ROWS):
        raise ValueError(f"{source}: expected exactly three MillionAID checkpoints")
    return found


def _record(source: str, model_id: str, file_id: str, readme_sha256: str) -> SourceRecord:
    title = _ROWS[model_id][1]
    url = f"https://drive.google.com/file/d/{file_id}/view"
    model_identifier = Identifier("vitae-rsp:model", model_id)
    checkpoint_identifier = Identifier("vitae-rsp:drive-file", file_id)
    return SourceRecord(
        source_record_id=f"{source}:{model_id}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=url,
        title=title,
        raw={
            "repository": _REPOSITORY,
            "model_id": model_id,
            "training_dataset": "MillionAID",
            "google_drive_file_id": file_id,
            "readme_sha256": readme_sha256,
        },
        identifiers=(checkpoint_identifier, model_identifier),
        links=(
            Link(
                url,
                relation="weights",
                locator="first-party README MillionAID checkpoint row",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            Link(
                _README_PAGE,
                relation="documentation",
                locator="ViTAE RSP README pretrained checkpoint table",
                crawl=False,
                model_local_ids=(model_id,),
            ),
        ),
        models=(ModelHint(local_id=model_id, name=title, identifiers=(model_identifier,)),),
        releases=(
            ReleaseHint(
                local_id=f"{model_id}@drive-{file_id}",
                model_local_id=model_id,
                identifiers=(checkpoint_identifier,),
                metadata={"google_drive_file_id": file_id},
                locator="README-listed Google Drive checkpoint",
            ),
        ),
    )
