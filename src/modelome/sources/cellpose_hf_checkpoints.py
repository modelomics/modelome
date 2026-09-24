"""Enumerate Cellpose v4's named checkpoints from its official Hub repository."""

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

_REPOSITORY = "mouseland/cellpose-sam"
_API = f"https://huggingface.co/api/models/{_REPOSITORY}"
_REPOSITORY_URL = f"https://huggingface.co/{_REPOSITORY}"
_DOCUMENTATION = "https://cellpose.readthedocs.io/en/latest/models.html"
_REVISION = re.compile(r"^[0-9a-f]{40}$")

# This is the literal built-in model list from MouseLand's Cellpose models.py.
_CHECKPOINTS = {
    "cpsam_v2": "Cellpose-SAM v2",
    "cpdino": "CellposeDINO ViTL",
    "cpdino-vitb": "CellposeDINO ViTB",
    "cpsam": "Cellpose-SAM",
}


class CellposeHubCheckpointSourceAdapter:
    """Project the four named Cellpose built-ins as files at one pinned revision."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the four Cellpose built-in checkpoint files named in the official "
        "models module when present in its public Hub repository. It excludes user "
        "models, third-party resources, and legacy website checkpoints."
    )

    def __init__(
        self,
        *,
        name: str = "cellpose-hub-checkpoints",
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        if isinstance(max_response_bytes, bool) or max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.name = name
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "cellpose-hub-checkpoints-v1",
                "repository": _REPOSITORY,
                "checkpoints": _CHECKPOINTS,
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
        if not isinstance(payload, Mapping) or payload.get("id") != _REPOSITORY:
            raise ValueError(f"{self.name}: Hub response repository identity is invalid")
        revision = payload.get("sha")
        siblings = payload.get("siblings")
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            raise ValueError(f"{self.name}: Hub response lacks a valid repository revision")
        if not isinstance(siblings, list) or len(siblings) > 10_000:
            raise ValueError(f"{self.name}: Hub response lacks a bounded file inventory")
        files = {
            item.get("rfilename")
            for item in siblings
            if isinstance(item, Mapping) and isinstance(item.get("rfilename"), str)
        }
        present = [filename for filename in _CHECKPOINTS if filename in files]

        if (
            state.get("completed") is True
            and state.get("revision") == revision
            and state.get("checkpoint_files") == present
        ):
            return SourcePage(
                records=(),
                next_state=dict(state),
                complete=True,
                upstream_count=len(present),
            )

        records = tuple(
            _record(self.name, filename, _CHECKPOINTS[filename], revision)
            for filename in present
        )
        next_state = {
            "completed": True,
            "repository": _REPOSITORY,
            "revision": revision,
            "checkpoint_files": present,
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
        f"{_REPOSITORY_URL}/resolve/{quote(revision, safe='')}/"
        f"{quote(filename, safe='')}"
    )
    return SourceRecord(
        source_record_id=f"{source}:{filename}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=weight_url,
        title=title,
        raw={
            "repository": _REPOSITORY,
            "revision": revision,
            "checkpoint_file": filename,
            "documentation_url": _DOCUMENTATION,
        },
        identifiers=(Identifier("cellpose:checkpoint", filename),),
        links=(
            Link(
                weight_url,
                relation="weights",
                locator="Hub.siblings.rfilename",
                crawl=False,
                model_local_ids=(local_id,),
            ),
            Link(
                _REPOSITORY_URL,
                relation="model_repository",
                locator="Hub.id",
                crawl=False,
                model_local_ids=(local_id,),
            ),
            Link(
                _DOCUMENTATION,
                relation="documentation",
                locator="Cellpose models documentation",
                crawl=False,
                model_local_ids=(local_id,),
            ),
        ),
        models=(
            ModelHint(
                local_id=local_id,
                name=title,
                identifiers=(Identifier("cellpose:model", filename),),
                locator="Cellpose.models.MODEL_NAMES",
            ),
        ),
        releases=(
            ReleaseHint(
                local_id=f"{local_id}@{revision}",
                model_local_id=local_id,
                revision=revision,
                identifiers=(Identifier("cellpose:checkpoint", filename),),
                metadata={"checkpoint_file": filename, "weight_url": weight_url},
                locator="Cellpose.models._MODEL_URL",
            ),
        ),
    )


__all__ = ["CellposeHubCheckpointSourceAdapter"]
