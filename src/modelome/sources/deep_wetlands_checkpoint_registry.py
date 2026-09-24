"""Read the four Deep Wetlands checkpoint links from its first-party README."""

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

_REPOSITORY = "melqkiades/deep-wetlands"
_README = f"https://raw.githubusercontent.com/{_REPOSITORY}/master/README.md"
_README_PAGE = f"https://github.com/{_REPOSITORY}"
_FILE_RE = re.compile(
    r"https://drive\.google\.com/file/d/(?P<file_id>[A-Za-z0-9_-]+)(?:/view)?(?:\?[^\s)]*)?"
)
_CHECKPOINTS: dict[tuple[str, str], tuple[str, str]] = {
    ("small", "2018-2019"): (
        "1gRj98jWhvRSeLoAzcNzwM6Gi9I8K__0-",
        "Deep Wetlands small model, 2018–2019 Sentinel-1 data",
    ),
    ("small", "2020-2022"): (
        "1N7ca5fKTGdazw7n8ALCYH6m3nsG2SUII",
        "Deep Wetlands small model, 2020–2022 Sentinel-1 data",
    ),
    ("large", "2018-2019"): (
        "11CFnSUrKsjvTo4JqcFxKXP7RwCmwSUze",
        "Deep Wetlands large model, 2018–2019 Sentinel-1 data",
    ),
    ("large", "2020-2022"): (
        "1fJeg6hPMORZoNkcUC-zh7XlaPL6Bj-FA",
        "Deep Wetlands large model, 2020–2022 Sentinel-1 data",
    ),
}


class DeepWetlandsCheckpointRegistrySourceAdapter:
    """Enumerate only the four small/large, year-specific checkpoint links."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the four Google Drive file IDs linked under Small Models and Large "
        "Models in the first-party Deep Wetlands README. The links identify public "
        "Google Drive files, but the adapter does not resolve Drive metadata or fetch bytes."
    )

    def __init__(
        self,
        *,
        name: str = "deep-wetlands-checkpoint-registry",
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
                "adapter": "deep-wetlands-checkpoint-registry-v1",
                "repository": _REPOSITORY,
                "readme": _README,
                "file_ids": sorted(value[0] for value in _CHECKPOINTS.values()),
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
            _record(self.name, size, years, file_id, digest)
            for (size, years), file_id in found.items()
        )
        return SourcePage(
            records,
            {"completed_readme_sha256": digest, "checkpoint_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_readme(document: str, source: str) -> dict[tuple[str, str], str]:
    if document.count("## Download Pre-trained Models") != 1:
        raise ValueError(f"{source}: expected the checkpoint section")
    section = False
    size: str | None = None
    years: str | None = None
    found: dict[tuple[str, str], str] = {}
    for line in document.splitlines():
        stripped = line.strip()
        if stripped == "## Download Pre-trained Models":
            section = True
            continue
        if section and stripped.startswith("## "):
            break
        if not section:
            continue
        if stripped == "### Small Models":
            size, years = "small", None
            continue
        if stripped == "### Large Models":
            size, years = "large", None
            continue
        year_heading = re.fullmatch(r"#### Years (?P<years>20[0-9]{2}-20[0-9]{2})", stripped)
        if year_heading is not None:
            years = year_heading.group("years")
            continue
        match = _FILE_RE.search(stripped)
        if match is None:
            continue
        key = (size, years) if size is not None and years is not None else None
        expected = _CHECKPOINTS.get(key) if key is not None else None
        file_id = match.group("file_id")
        if expected is None or expected[0] != file_id or key in found:
            raise ValueError(f"{source}: unexpected or duplicate checkpoint link in {key!r}")
        found[key] = file_id
    if set(found) != set(_CHECKPOINTS):
        raise ValueError(f"{source}: expected exactly four README checkpoint links")
    return found


def _record(source: str, size: str, years: str, file_id: str, readme_sha256: str) -> SourceRecord:
    model_id = f"{size}-{years}"
    title = _CHECKPOINTS[(size, years)][1]
    url = f"https://drive.google.com/file/d/{file_id}/view"
    model_identifier = Identifier("deep-wetlands:model", model_id)
    checkpoint_identifier = Identifier("deep-wetlands:drive-file", file_id)
    return SourceRecord(
        source_record_id=f"{source}:{model_id}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=url,
        title=title,
        raw={
            "repository": _REPOSITORY,
            "model_size": size,
            "training_years": years,
            "google_drive_file_id": file_id,
            "readme_sha256": readme_sha256,
        },
        identifiers=(checkpoint_identifier, model_identifier),
        links=(
            Link(
                url,
                relation="weights",
                locator="first-party README year and size entry",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            Link(
                _README_PAGE,
                relation="documentation",
                locator="Deep Wetlands pretrained model list",
                crawl=False,
                model_local_ids=(model_id,),
            ),
        ),
        models=(
            ModelHint(
                local_id=model_id,
                name=title,
                identifiers=(model_identifier,),
                locator="Deep Wetlands size and training-years heading",
            ),
        ),
        releases=(
            ReleaseHint(
                local_id=f"{model_id}@drive-{file_id}",
                model_local_id=model_id,
                identifiers=(checkpoint_identifier,),
                metadata={"google_drive_file_id": file_id, "training_years": years},
                locator="README-listed Google Drive checkpoint",
            ),
        ),
    )
