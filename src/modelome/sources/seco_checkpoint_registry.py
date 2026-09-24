"""Read the four exact Seasonal Contrast checkpoint URLs from its first-party README."""

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

_REPOSITORY = "ServiceNow/seasonal-contrast"
_README = f"https://raw.githubusercontent.com/{_REPOSITORY}/main/README.md"
_README_PAGE = f"https://github.com/{_REPOSITORY}"
_RECORD_ID = "4728033"
_ROWS: dict[str, tuple[str, str, str]] = {
    "seco_resnet18_100k.ckpt": (
        "SeCo-100K-ResNet-18",
        "SeCo-100K pretrained ResNet-18 checkpoint",
        "dcf336be31f6c6b0e77dcb6cc958fca8",
    ),
    "seco_resnet18_1m.ckpt": (
        "SeCo-1M-ResNet-18",
        "SeCo-1M pretrained ResNet-18 checkpoint",
        "53d5c41d0f479bdfd31d6746ad4126db",
    ),
    "seco_resnet50_100k.ckpt": (
        "SeCo-100K-ResNet-50",
        "SeCo-100K pretrained ResNet-50 checkpoint",
        "9672c303f6334ef816494c13b9d05753",
    ),
    "seco_resnet50_1m.ckpt": (
        "SeCo-1M-ResNet-50",
        "SeCo-1M pretrained ResNet-50 checkpoint",
        "7b09c54aed33c0c988b425c54f4ef948",
    ),
}
_FILE_RE = re.compile(
    r"https://zenodo\.org/record/(?P<record>\d+)/files/(?P<filename>[^?\s)]+)\?download=1"
)


class SeCoCheckpointRegistrySourceAdapter:
    """Enumerate only the four README-listed Zenodo pretrained checkpoints."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the four SeCo ResNet checkpoints in the first-party README table. "
        "The Zenodo file URLs and README-declared MD5 values are recorded as metadata; "
        "the adapter does not resolve Zenodo metadata or fetch checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "seco-checkpoint-registry",
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
                "adapter": "seco-checkpoint-registry-v1",
                "repository": _REPOSITORY,
                "readme": _README,
                "record_id": _RECORD_ID,
                "files": sorted(_ROWS),
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
            _record(self.name, filename, md5, digest)
            for filename, md5 in found.items()
        )
        return SourcePage(
            records,
            {"completed_readme_sha256": digest, "checkpoint_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_readme(document: str, source: str) -> dict[str, str]:
    if document.count("Pre-trained Models") != 1:
        raise ValueError(f"{source}: expected the pretrained checkpoint table")
    found: dict[str, str] = {}
    section = False
    for line in document.splitlines():
        stripped = line.strip()
        if "Pre-trained Models" in stripped:
            section = True
            continue
        if section and stripped.startswith("## "):
            break
        if not section:
            continue
        for match in _FILE_RE.finditer(stripped):
            record_id, filename = match.group("record", "filename")
            if record_id != _RECORD_ID or filename not in _ROWS or filename in found:
                raise ValueError(f"{source}: unexpected or duplicate checkpoint {filename!r}")
            found[filename] = _ROWS[filename][2]
    if set(found) != set(_ROWS):
        raise ValueError(f"{source}: expected exactly four SeCo checkpoint links")
    return found


def _record(source: str, filename: str, md5: str, readme_sha256: str) -> SourceRecord:
    model_id, title, _ = _ROWS[filename]
    url = f"https://zenodo.org/record/{_RECORD_ID}/files/{filename}?download=1"
    model_identifier = Identifier("seco:model", model_id)
    file_identifier = Identifier("seco:zenodo-file", f"{_RECORD_ID}/{filename}")
    return SourceRecord(
        source_record_id=f"{source}:{filename}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=url,
        title=title,
        raw={
            "repository": _REPOSITORY,
            "zenodo_record_id": _RECORD_ID,
            "filename": filename,
            "md5": md5,
            "readme_sha256": readme_sha256,
        },
        identifiers=(file_identifier, model_identifier),
        links=(
            Link(
                url,
                relation="weights",
                locator="first-party README pretrained checkpoint table",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            Link(
                _README_PAGE,
                relation="documentation",
                locator="Seasonal Contrast pretrained model table",
                crawl=False,
                model_local_ids=(model_id,),
            ),
        ),
        models=(ModelHint(local_id=model_id, name=title, identifiers=(model_identifier,)),),
        releases=(
            ReleaseHint(
                local_id=f"{model_id}@zenodo-{_RECORD_ID}",
                model_local_id=model_id,
                identifiers=(file_identifier,),
                metadata={"zenodo_record_id": _RECORD_ID, "filename": filename, "md5": md5},
                locator="README-listed Zenodo checkpoint",
            ),
        ),
    )
