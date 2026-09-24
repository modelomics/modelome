"""Official RoseTTAFold checkpoint bundles documented in first-party READMEs."""

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
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORIES = (
    {
        "repository": "RosettaCommons/RoseTTAFold",
        "name": "RoseTTAFold",
        "asset_name": "weights.tar.gz",
        "asset_url": "https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz",
        "models": (
            {"name": "RoseTTAFold", "filename": None},
            {"name": "RoseTTAFold-2track", "filename": "RF2t.pt"},
        ),
    },
    {
        "repository": "uw-ipd/RoseTTAFold2",
        "name": "RoseTTAFold2",
        "asset_name": "RF2_jan24.tgz",
        "asset_url": "https://files.ipd.uw.edu/dimaio/RF2_jan24.tgz",
        "models": ({"name": "RoseTTAFold2", "filename": None},),
    },
)
_MAX_SOURCE_BYTES = 256 * 1024


class RoseTTAFoldCheckpointAdapter:
    """Project exact public weight-bundle links from the two official READMEs.

    The upstream READMEs document bundles rather than a file-by-file manifest.
    Consequently this adapter indexes those archive artifacts and preserves
    named variants only where the README states them explicitly.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Indexes the RoseTTAFold and RoseTTAFold2 weight archives named in the "
        "official repositories. Archives are not inspected and their contents "
        "are not expanded into inferred checkpoint records."
    )

    def __init__(
        self,
        *,
        name: str = "rosettafold-checkpoint-bundles",
        max_source_bytes: int = _MAX_SOURCE_BYTES,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not name.strip() or max_source_bytes <= 0:
            raise ValueError("name and a positive max_source_bytes are required")
        self.name = name
        self.max_source_bytes = max_source_bytes
        self.client = client or HttpClient(max_response_bytes=max_source_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "rosettafold-checkpoint-bundles-v1",
                "repositories": [item["repository"] for item in _REPOSITORIES],
                "max_source_bytes": max_source_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        revisions: dict[str, str] = {}
        for source in _REPOSITORIES:
            repository = source["repository"]
            api_url = f"https://api.github.com/repos/{repository}/commits/main"
            commit_response = self.client.get(
                api_url, headers={"Accept": "application/vnd.github+json"}
            )
            if commit_response.status != 200:
                raise ValueError(
                    f"{self.name}: {repository} commit endpoint returned "
                    f"HTTP {commit_response.status}"
                )
            payload = commit_response.json()
            revision = payload.get("sha") if isinstance(payload, Mapping) else None
            if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
                raise ValueError(f"{self.name}: invalid {repository} revision")
            revisions[repository] = revision

        completed_revisions = state.get("completed_revisions", {})
        if not isinstance(completed_revisions, Mapping):
            raise ValueError("completed_revisions must be a mapping")
        if all(
            completed_revisions.get(repository) == revision
            for repository, revision in revisions.items()
        ):
            return SourcePage(
                (), {"completed_revisions": revisions}, True,
                upstream_count=len(_REPOSITORIES),
            )

        records: list[SourceRecord] = []
        for source in _REPOSITORIES:
            repository = str(source["repository"])
            revision = revisions[repository]
            raw_url = (
                f"https://raw.githubusercontent.com/{repository}/"
                f"{revision}/README.md"
            )
            response = self.client.get(raw_url, headers={"Accept": "text/plain"})
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: {repository} README returned HTTP {response.status}"
                )
            if len(response.body) > self.max_source_bytes:
                raise ValueError(f"{self.name}: {repository} README exceeds limit")
            source_text = response.text()
            if source["asset_url"] not in source_text:
                raise ValueError(
                    f"{self.name}: expected first-party asset link missing for {repository}"
                )
            for model in source["models"]:
                filename = model["filename"]
                if filename is not None and filename not in source_text:
                    raise ValueError(
                        f"{self.name}: expected first-party checkpoint name missing "
                        f"for {repository}"
                    )
            records.append(self._record(source, revision, raw_url))

        return SourcePage(
            tuple(records),
            {"completed_revisions": revisions},
            True,
            upstream_count=len(_REPOSITORIES),
            authoritative_snapshot=True,
        )

    def _record(
        self, source: Mapping[str, Any], revision: str, source_url: str
    ) -> SourceRecord:
        repository = str(source["repository"])
        artifact_name = str(source["asset_name"])
        artifact_url = str(source["asset_url"])
        readme_url = f"https://github.com/{repository}/blob/{revision}/README.md"
        models: list[ModelHint] = []
        releases: list[ReleaseHint] = []
        local_ids: list[str] = []
        for item in source["models"]:
            model_name = str(item["name"])
            filename = item["filename"]
            model_local_id = f"model:{model_name}"
            local_ids.append(model_local_id)
            models.append(
                ModelHint(
                    model_local_id,
                    model_name,
                    identifiers=(Identifier("rosettafold:model", model_name),),
                    status=ModelStatus.RELEASED,
                    locator="README.md",
                )
            )
            releases.append(
                ReleaseHint(
                    f"release:{model_name}",
                    model_local_id,
                    identifiers=(
                        Identifier("rosettafold:weights-archive", artifact_url),
                        Identifier("rosettafold:checkpoint", filename)
                        if filename is not None
                        else Identifier("rosettafold:archive-member", artifact_name),
                    ),
                    metadata={
                        "archive_filename": artifact_name,
                        "archive_url": artifact_url,
                        "checkpoint_filename": filename,
                        "archive_contents_enumerated": False,
                    },
                    locator="README.md download instructions",
                )
            )
        return SourceRecord(
            source_record_id=f"{self.name}:{repository}:{revision}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(readme_url),
            title=f"{source['name']} pretrained weights archive",
            raw={
                "repository": repository,
                "revision": revision,
                "readme_source_url": source_url,
                "archive_filename": artifact_name,
                "archive_url": artifact_url,
                "models": [
                    {"name": item["name"], "filename": item["filename"]}
                    for item in source["models"]
                ],
            },
            text=f"{source['name']} {artifact_name}: "
            + ", ".join(str(item["name"]) for item in source["models"]),
            identifiers=(Identifier("rosettafold:repository", repository),),
            links=(
                Link(readme_url, "model_card", crawl=False),
                Link(artifact_url, "weights", crawl=False, model_local_ids=tuple(local_ids)),
            ),
            models=tuple(models),
            releases=tuple(releases),
        )


__all__ = ["RoseTTAFoldCheckpointAdapter"]
