"""Read the two exact GeoLink pretrained checkpoint links from its official README."""

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

_REPOSITORY = "bailubin/GeoLink_NeurIPS2025"
_README = f"https://raw.githubusercontent.com/{_REPOSITORY}/main/README.md"
_README_PAGE = f"https://github.com/{_REPOSITORY}"
_MODELS = {
    "unimodal": (
        "geolink_vit_large_patch16_224.pth",
        "12u0goOohBYHjlkKIs11bVeTYyscd2Z9g",
        "Unimodal GeoLink ViT-L/16 checkpoint",
    ),
    "multimodal": (
        "geolink_mutimodal_vit_large_patch16_224.pth",
        "1bAeNurdrH9nEI7qzwNWyLBWBSZnfwj_9",
        "Multimodal GeoLink ViT-L/16 checkpoint",
    ),
}
_DRIVE_RE = re.compile(
    r"\[Google Drive\]\(https://drive\.google\.com/file/d/"
    r"(?P<file_id>[A-Za-z0-9_-]+)"
    r'/view(?:\?[^)]*)?\)'
)


class GeoLinkCheckpointRegistrySourceAdapter:
    """Enumerate the unimodal and multimodal pretrained GeoLink checkpoint files."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the two Google Drive checkpoint files named for pretrained models "
        "in the official GeoLink README. PKU Disk links and downstream fine-tuned weights "
        "are excluded; the adapter does not query Drive or fetch checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "geolink-checkpoint-registry",
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
                "adapter": "geolink-checkpoint-registry-v1",
                "repository": _REPOSITORY,
                "readme": _README,
                "file_ids": sorted(model[1] for model in _MODELS.values()),
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
            _record(self.name, kind, file_id, digest)
            for kind, file_id in found.items()
        )
        return SourcePage(
            records,
            {"completed_readme_sha256": digest, "checkpoint_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_readme(document: str, source: str) -> dict[str, str]:
    if document.count("## 🏋️‍♂️ Pre-trained Models") != 1:
        raise ValueError(f"{source}: expected the pretrained models section")
    section = False
    model_kind: str | None = None
    found: dict[str, str] = {}
    for line in document.splitlines():
        stripped = line.strip()
        if stripped == "## 🏋️‍♂️ Pre-trained Models":
            section = True
            continue
        if section and stripped.startswith("## "):
            break
        if not section:
            continue
        if stripped.startswith("### 1. Unimodal GeoLink(ViT)"):
            model_kind = "unimodal"
            continue
        if stripped.startswith("### 2. Multimodal GeoLink"):
            model_kind = "multimodal"
            continue
        if model_kind is None:
            continue
        matches = list(_DRIVE_RE.finditer(stripped))
        if not matches:
            continue
        if len(matches) != 1 or model_kind in found:
            raise ValueError(f"{source}: unexpected or duplicate link for {model_kind}")
        expected_id = _MODELS[model_kind][1]
        if matches[0].group("file_id") != expected_id:
            raise ValueError(f"{source}: unexpected Google Drive ID for {model_kind}")
        found[model_kind] = expected_id
    if set(found) != set(_MODELS):
        raise ValueError(f"{source}: expected exactly two pretrained GeoLink checkpoint links")
    return found


def _record(source: str, kind: str, file_id: str, readme_sha256: str) -> SourceRecord:
    filename, _, title = _MODELS[kind]
    model_id = f"geolink-{kind}-vit-large-patch16-224"
    url = f"https://drive.google.com/file/d/{file_id}/view"
    model_identifier = Identifier("geolink:model", model_id)
    checkpoint_identifier = Identifier("geolink:drive-file", file_id)
    return SourceRecord(
        source_record_id=f"{source}:{kind}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=url,
        title=title,
        raw={
            "repository": _REPOSITORY,
            "model_id": model_id,
            "filename": filename,
            "modality": kind,
            "google_drive_file_id": file_id,
            "readme_sha256": readme_sha256,
        },
        identifiers=(checkpoint_identifier, model_identifier),
        links=(
            Link(
                url,
                relation="weights",
                locator="official README pretrained model download entry",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            Link(
                _README_PAGE,
                relation="documentation",
                locator="GeoLink pretrained models README section",
                crawl=False,
                model_local_ids=(model_id,),
            ),
        ),
        models=(
            ModelHint(local_id=model_id, name=title, identifiers=(model_identifier,)),
        ),
        releases=(
            ReleaseHint(
                local_id=f"{model_id}@drive-{file_id}",
                model_local_id=model_id,
                identifiers=(checkpoint_identifier,),
                metadata={"filename": filename, "google_drive_file_id": file_id},
                locator="README-listed Google Drive checkpoint",
            ),
        ),
    )
