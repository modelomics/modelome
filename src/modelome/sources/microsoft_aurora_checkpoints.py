"""Project Microsoft's documented Aurora checkpoints from its official Hub repo.

The allowlist mirrors checkpoint filenames named by Microsoft's model
documentation. The adapter reads one Hub model metadata response and emits one
source record per present checkpoint, with URLs pinned to the repository SHA.
It does not enumerate arbitrary repository files or transfer checkpoint bytes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient, HttpResponse
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

_REPO = "microsoft/aurora"
_API = "https://huggingface.co/api/models/microsoft/aurora"
_PAGE = "https://huggingface.co/microsoft/aurora"
_DOCUMENTATION = "https://microsoft.github.io/aurora/models.html"
_REVISION = re.compile(r"^[0-9a-f]{40}$")

# Exact names from Microsoft's official available-model documentation/API.
_CHECKPOINTS = {
    "aurora-0.25-pretrained.ckpt": "Aurora 0.25° Pretrained",
    "aurora-0.25-small-pretrained.ckpt": "Aurora 0.25° Small Pretrained",
    "aurora-0.25-finetuned.ckpt": "Aurora 0.25° Fine-Tuned",
    "aurora-0.25-12h-pretrained.ckpt": "Aurora 0.25° 12-Hour Pretrained",
    "aurora-0.1-finetuned.ckpt": "Aurora 0.1° Fine-Tuned",
    "aurora-0.4-air-pollution.ckpt": "Aurora 0.4° Air Pollution",
    "aurora-0.25-wave.ckpt": "Aurora 0.25° Wave",
    "aurora-0.25-v1.5.ckpt": "Aurora 1.5 (0.25°)",
    "aurora-0.25-v1.5-ensemble.ckpt": "Aurora 1.5 Ensemble (0.25°)",
}


class MicrosoftAuroraCheckpointSourceAdapter:
    """Enumerate only the documented Aurora checkpoint files in the official repo."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the exact checkpoint files documented by Microsoft for Aurora. "
        "It does not project static data, test fixtures, undocumented files, "
        "or download weights."
    )

    def __init__(
        self,
        *,
        name: str = "microsoft-aurora-checkpoints",
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not name:
            raise ValueError("source name must not be empty")
        if isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.name = name
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "microsoft-aurora-checkpoints-v1",
                "repository": _REPO,
                "documented_checkpoints": sorted(_CHECKPOINTS),
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            _API,
            params={"full": "true", "config": "false"},
            headers={"Accept": "application/json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: Hub returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: Hub response exceeds configured byte limit")
        payload = response.json()
        if not isinstance(payload, Mapping) or payload.get("id") != _REPO:
            raise ValueError(f"{self.name}: Hub response repository identity is invalid")
        revision = payload.get("sha")
        siblings = payload.get("siblings")
        if (
            not isinstance(revision, str)
            or not _REVISION.fullmatch(revision)
            or not isinstance(siblings, list)
        ):
            raise ValueError(f"{self.name}: Hub response lacks revision or file inventory")
        files = {
            item.get("rfilename")
            for item in siblings
            if isinstance(item, Mapping) and isinstance(item.get("rfilename"), str)
        }
        declared = [filename for filename in _CHECKPOINTS if filename in files]
        if not declared:
            raise ValueError(f"{self.name}: no documented checkpoint files are present")

        prior_files = state.get("checkpoint_files")
        if (
            state.get("completed") is True
            and state.get("revision") == revision
            and prior_files == declared
        ):
            return SourcePage(
                records=(),
                next_state=dict(state),
                complete=True,
                upstream_count=len(declared),
            )

        records = tuple(
            _record(self.name, filename, _CHECKPOINTS[filename], revision) for filename in declared
        )
        next_state = {
            "completed": True,
            "repository": _REPO,
            "revision": revision,
            "checkpoint_files": declared,
            "checkpoint_count": len(records),
        }
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _record(source: str, filename: str, title: str, revision: str) -> SourceRecord:
    local_id = filename
    weight_url = (
        f"https://huggingface.co/{_REPO}/resolve/{quote(revision, safe='')}/"
        f"{quote(filename, safe='/')}"
    )
    return SourceRecord(
        source_record_id=f"{source}:{filename}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=weight_url,
        title=title,
        raw={
            "repository": _REPO,
            "revision": revision,
            "checkpoint_file": filename,
            "documentation_url": _DOCUMENTATION,
        },
        identifiers=(Identifier("microsoft:aurora-checkpoint", filename),),
        links=(
            Link(
                weight_url,
                relation="weights",
                locator="Hub.siblings.rfilename",
                crawl=False,
                model_local_ids=(local_id,),
            ),
            Link(
                _PAGE,
                relation="model_repository",
                locator="Hub.id",
                crawl=False,
                model_local_ids=(local_id,),
            ),
            Link(
                _DOCUMENTATION,
                relation="documentation",
                locator="official model table",
                crawl=False,
                model_local_ids=(local_id,),
            ),
        ),
        models=(
            ModelHint(
                local_id=local_id,
                name=title,
                identifiers=(Identifier("microsoft:aurora-model", filename),),
                locator="Microsoft Aurora available models documentation",
            ),
        ),
        releases=(
            ReleaseHint(
                local_id=f"{local_id}@{revision}",
                model_local_id=local_id,
                revision=revision,
                identifiers=(Identifier("microsoft:aurora-checkpoint", filename),),
                metadata={"checkpoint_file": filename, "weight_url": weight_url},
                locator="Hub.siblings.rfilename",
            ),
        ),
    )
