"""Black Forest Labs' first-party hosted FLUX API model endpoints."""

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

_REPOSITORY = "black-forest-labs/skills"
_DOC_PATH = "skills/bfl-api/references/endpoints.md"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_HEADING = re.compile(r"^### (?P<name>FLUX[^\n]+?)\s*$")
_ROUTE = re.compile(r"^POST /v1/(?P<endpoint>[a-z0-9.-]+)\s*$")
# The open Klein weights are already enumerated by the official FLUX.2
# repository and HF. This source focuses on BFL-hosted offerings with no
# corresponding open-weight release in the official model tables.
_OPEN_WEIGHT_ENDPOINTS = frozenset({"flux-2-klein-4b", "flux-2-klein-9b"})


class BFLAPIModelsSourceAdapter:
    """Enumerate first-party API model IDs from BFL's versioned endpoint guide.

    Only API endpoints that are not the separately released Klein checkpoints
    are admitted. These are hosted inference products, not downloadable weights.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers named model endpoints in Black Forest Labs' API endpoint guide, "
        "excluding FLUX.2 Klein endpoints already represented by open weights. "
        "It does not enumerate downloadable checkpoints, authenticated account "
        "models, or third-party deployments."
    )

    def __init__(
        self,
        *,
        name: str = "bfl-api-models",
        repository: str = _REPOSITORY,
        branch: str = "master",
        source_path: str = _DOC_PATH,
        max_response_bytes: int = 512 * 1024,
        max_entries: int = 100,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY or source_path != _DOC_PATH:
            raise ValueError(
                "repository and source_path must identify the official BFL endpoint guide"
            )
        if not name.strip() or not branch.strip() or min(max_response_bytes, max_entries) <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.source_path = source_path
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "bfl-api-models-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "excluded_open_weight_endpoints": sorted(_OPEN_WEIGHT_ENDPOINTS),
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        branch = quote(self.branch, safe="")
        return f"https://api.github.com/repos/{self.repository}/commits/{branch}"

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )

    def blob_url(self, revision: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid repository revision")
        if revision == state.get("completed_revision"):
            count = state.get("model_count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{self.name}: invalid completed model count")
            return SourcePage((), dict(state), True, upstream_count=count)

        url = self.raw_url(revision)
        response = self.client.get(url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: endpoint guide returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: endpoint guide exceeds response limit")
        try:
            text = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: endpoint guide is not UTF-8") from exc
        models = _parse_endpoints(text, maximum=self.max_entries)
        if not models:
            raise ValueError(f"{self.name}: endpoint guide contains no admitted models")
        source_url = self.blob_url(revision)
        records = tuple(self._record(name, endpoint, revision, source_url, response.body)
                        for name, endpoint in models)
        next_state = {
            "completed_revision": revision,
            "model_count": len(records),
            "source_url": url,
            "source_sha256": content_hash(response.body),
        }
        return SourcePage(
            records,
            next_state,
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self, name: str, endpoint: str, revision: str, source_url: str, source: bytes
    ) -> SourceRecord:
        model_id = f"model:{endpoint}"
        model = ModelHint(
            local_id=model_id,
            name=name,
            aliases=(endpoint,),
            identifiers=(Identifier("bfl:api-model", endpoint),),
            status=ModelStatus.DOCUMENTED,
            locator=f"POST /v1/{endpoint}",
        )
        return SourceRecord(
            source_record_id=f"api-model:{endpoint}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(source_url),
            title=name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "api_model_id": endpoint,
                "method": "POST",
                "route": f"/v1/{endpoint}",
                "hosted_api_url": f"https://api.bfl.ai/v1/{endpoint}",
            },
            text=f"Black Forest Labs hosted API model: {name} ({endpoint})",
            identifiers=model.identifiers,
            links=(
                Link(source_url, relation="model_card", locator=model.locator, crawl=False,
                     model_local_ids=(model_id,)),
                Link(f"https://api.bfl.ai/v1/{endpoint}", relation="inference_endpoint",
                     locator=model.locator, crawl=False, model_local_ids=(model_id,)),
                Link(self.repository_url, relation="source_implementation", crawl=False),
            ),
            models=(model,),
        )


def _parse_endpoints(markdown: str, *, maximum: int) -> tuple[tuple[str, str], ...]:
    """Parse only heading-followed POST route pairs from the official guide."""
    lines = markdown.splitlines()
    rows: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        heading = _HEADING.fullmatch(line.strip())
        if heading is None:
            continue
        name = heading.group("name").strip()
        endpoint = None
        for candidate in lines[index + 1 : index + 7]:
            route = _ROUTE.fullmatch(candidate.strip())
            if route:
                endpoint = route.group("endpoint")
                break
            if candidate.startswith("### "):
                break
        if endpoint is None or endpoint in _OPEN_WEIGHT_ENDPOINTS:
            continue
        rows.append((name, endpoint))
        if len(rows) > maximum:
            raise ValueError("BFL endpoint guide exceeds entry limit")
    endpoints = [endpoint for _, endpoint in rows]
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("BFL endpoint guide contains duplicate API model IDs")
    return tuple(rows)
