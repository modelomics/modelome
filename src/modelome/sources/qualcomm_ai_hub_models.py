"""Public Qualcomm AI Hub Models directory from the official repository."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_REPOSITORY = "qualcomm/ai-hub-models"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ROW = re.compile(
    r"^\s*\|\s*\[(?P<name>[^\]]+)\]"
    r"\(https://aihub\.qualcomm\.com/models/(?P<slug>[a-zA-Z0-9_-]+)\)\s*"
    r"\|\s*\[qai_hub_models\.models\.(?P<module>[a-zA-Z0-9_-]+)\]"
    r"\(src/qai_hub_models/models/(?P<path>[a-zA-Z0-9_-]+)/README\.md\)\s*\|\s*$"
)


class QualcommAIHubModelsSourceAdapter:
    """Enumerate the public model directory in Qualcomm's official Git repo.

    The source is the pinned root README, whose model table pairs each
    Qualcomm model-card URL with its exact package model ID. This catalog is
    separate from account-scoped AI Hub Workbench jobs and user-uploaded models.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers models in the public Qualcomm AI Hub Models repository's root "
        "README at one pinned commit. It excludes account-scoped Workbench "
        "uploads/jobs, third-party recipes, and historical repository states."
    )

    def __init__(
        self,
        *,
        name: str = "qualcomm-ai-hub-models",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_readme_bytes: int = 512 * 1024,
        max_entries: int = 1000,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or min(max_readme_bytes, max_entries) <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_readme_bytes = max_readme_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_readme_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "qualcomm-ai-hub-models-readme-v1",
                "repository": repository,
                "branch": branch,
                "model_row_pattern": _MODEL_ROW.pattern,
                "max_readme_bytes": max_readme_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid repository revision")
        if revision == state.get("completed_revision"):
            model_count = state.get("model_count")
            if not isinstance(model_count, int) or isinstance(model_count, bool) or model_count < 0:
                raise ValueError(f"{self.name}: invalid completed model count in checkpoint")
            return SourcePage(
                (),
                dict(state),
                True,
                upstream_count=model_count,
                authoritative_snapshot=False,
            )

        readme_url = f"https://raw.githubusercontent.com/{self.repository}/{revision}/README.md"
        response = self.client.get(readme_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_readme_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        try:
            readme = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: README is not UTF-8") from exc

        entries: dict[str, tuple[str, int]] = {}
        for line_number, line in enumerate(readme.splitlines(), start=1):
            match = _MODEL_ROW.fullmatch(line)
            if not match:
                if "aihub.qualcomm.com/models/" in line and "qai_hub_models.models." in line:
                    raise ValueError(
                        f"{self.name}: unrecognized model directory row on README line "
                        f"{line_number}"
                    )
                continue
            slug, module, path = match.group("slug", "module", "path")
            if slug != module or slug != path:
                raise ValueError(
                    f"{self.name}: inconsistent exact model identity on README line {line_number}"
                )
            if slug in entries:
                raise ValueError(f"{self.name}: duplicate model ID {slug!r}")
            entries[slug] = (match.group("name"), line_number)
            if len(entries) > self.max_entries:
                raise ValueError(f"{self.name}: model count exceeds configured limit")
        if not entries:
            raise ValueError(f"{self.name}: README model directory format was not recognized")

        records = tuple(
            self._record(slug, name, line_number, revision, readme_url)
            for slug, (name, line_number) in sorted(entries.items())
        )
        return SourcePage(
            records,
            {"completed_revision": revision, "model_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=False,
        )

    def _record(
        self, slug: str, name: str, line_number: int, revision: str, readme_url: str
    ) -> SourceRecord:
        model_url = f"https://aihub.qualcomm.com/models/{quote(slug, safe='')}"
        model_local_id = f"model:{slug}"
        model = ModelHint(
            model_local_id,
            name,
            identifiers=(Identifier("qualcomm:ai-hub-model", slug),),
            aliases=(slug,) if slug.casefold() != name.casefold() else (),
            status=ModelStatus.RELEASED,
            locator=f"README.md:{line_number}",
        )
        source_url = (
            f"https://github.com/{self.repository}/tree/{revision}/"
            f"src/qai_hub_models/models/{quote(slug, safe='')}"
        )
        return SourceRecord(
            source_record_id=slug,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(model_url),
            title=name,
            raw={"model_id": slug, "repository_revision": revision, "readme_line": line_number},
            text=f"Qualcomm AI Hub model {name} ({slug}).",
            identifiers=(Identifier("qualcomm:ai-hub-model", slug),),
            links=(
                Link(model_url, "model_page", crawl=False, model_local_ids=(model_local_id,)),
                Link(
                    source_url,
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(model_local_id,),
                ),
                Link(readme_url, "catalog_source", crawl=False),
            ),
            models=(model,),
        )
