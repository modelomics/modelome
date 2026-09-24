"""Read exact Fields of The World checkpoint links from its first-party README."""

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

_REPOSITORY = "terramira/ftw-baselines"
_README = f"https://raw.githubusercontent.com/{_REPOSITORY}/main/README.md"
_README_PAGE = f"https://github.com/{_REPOSITORY}"
_RELEASE_PAGE = "https://github.com/fieldsoftheworld/ftw-baselines/releases/tag/v1"
_ASSET = re.compile(
    r"https://github\.com/fieldsoftheworld/ftw-baselines/releases/download/"
    r"v1/(?P<filename>[A-Za-z0-9_.-]+\.ckpt)"
)
_EXPECTED: dict[str, tuple[str, str]] = {
    "2_Class_FULL_FTW_Pretrained.ckpt": (
        "ftw-full-2-class",
        "Fields of The World full-data checkpoint, 2 classes",
    ),
    "3_Class_FULL_FTW_Pretrained.ckpt": (
        "ftw-full-3-class",
        "Fields of The World full-data checkpoint, 3 classes",
    ),
    "2_Class_CCBY_FTW_Pretrained.ckpt": (
        "ftw-cc-by-2-class",
        "Fields of The World CC-BY-only checkpoint, 2 classes",
    ),
    "3_Class_CCBY_FTW_Pretrained.ckpt": (
        "ftw-cc-by-3-class",
        "Fields of The World CC-BY-only checkpoint, 3 classes",
    ),
}


class FTWReleaseCheckpointSourceAdapter:
    """Enumerate only the four first-party-documented FTW v1 checkpoint assets."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the four exact v1 .ckpt release URLs declared by the FTW baseline "
        "README: 2/3-class FULL and CC-BY-only variants. It does not enumerate other "
        "release attachments or download checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "ftw-release-checkpoints",
        max_response_bytes: int = 2 * 1024 * 1024,
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
                "adapter": "ftw-release-checkpoints-v1",
                "repository": _REPOSITORY,
                "readme": _README,
                "assets": sorted(_EXPECTED),
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(_README, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds configured byte limit")
        matches = _parse_readme(response.text(), self.name)
        digest = content_hash(response.body)
        if digest == state.get("completed_readme_sha256"):
            return SourcePage((), dict(state), True, upstream_count=len(matches))
        records = tuple(_record(self.name, name, url, digest) for name, url in matches.items())
        return SourcePage(
            records,
            {"completed_readme_sha256": digest, "checkpoint_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_readme(document: str, source: str) -> dict[str, str]:
    if document.count("## Inference") != 1 or document.count(
        "### CC-BY (or equivalent) trained models"
    ) != 1:
        raise ValueError(f"{source}: expected the documented inference and CC-BY sections")
    urls = _ASSET.findall(document)
    found: dict[str, str] = {}
    for match in _ASSET.finditer(document):
        filename = match.group("filename")
        if filename not in _EXPECTED or filename in found:
            raise ValueError(f"{source}: unexpected or duplicate checkpoint URL {filename!r}")
        found[filename] = match.group(0)
    if len(urls) != len(_EXPECTED) or set(found) != set(_EXPECTED):
        raise ValueError(f"{source}: expected exactly four documented v1 checkpoint URLs")
    return found


def _record(source: str, filename: str, url: str, readme_sha256: str) -> SourceRecord:
    model_id, title = _EXPECTED[filename]
    model_identifier = Identifier("ftw:model", model_id)
    checkpoint_identifier = Identifier("ftw:release-asset", f"v1/{filename}")
    return SourceRecord(
        source_record_id=f"{source}:{model_id}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=url,
        title=title,
        raw={
            "repository": _REPOSITORY,
            "release_tag": "v1",
            "filename": filename,
            "readme_sha256": readme_sha256,
        },
        identifiers=(checkpoint_identifier, model_identifier),
        links=(
            Link(
                url,
                relation="weights",
                locator="first-party FTW README download command",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            Link(
                _README_PAGE,
                relation="documentation",
                locator="FTW README checkpoint download instructions",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            Link(
                _RELEASE_PAGE,
                relation="release",
                locator="first-party GitHub release v1",
                crawl=False,
                model_local_ids=(model_id,),
            ),
        ),
        models=(
            ModelHint(
                local_id=model_id,
                name=title,
                identifiers=(model_identifier,),
                locator="FTW pretrained checkpoint instructions",
            ),
        ),
        releases=(
            ReleaseHint(
                local_id=f"{model_id}@v1",
                model_local_id=model_id,
                identifiers=(checkpoint_identifier,),
                metadata={"release_tag": "v1", "filename": filename},
                locator="FTW release v1 README link",
            ),
        ),
    )
