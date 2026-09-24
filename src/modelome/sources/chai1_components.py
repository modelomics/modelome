"""Enumerate Chai-1's first-party exported model component artifacts."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

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

_REPOSITORY = "chaidiscovery/chai-lab"
_INFERENCE_PATH = "chai_lab/chai1.py"
_PATHS_PATH = "chai_lab/utils/paths.py"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_COMPONENT = re.compile(r"^[a-z][a-z0-9_]*\.pt$")
_MAX_SOURCE_BYTES = 512 * 1024
_MAX_COMPONENTS = 64


class Chai1ComponentRegistryAdapter:
    """Index Chai-1 component URLs explicitly assembled by first-party code.

    The adapter reads only the component-name call sites and URL template; it
    never downloads component bytes or interprets the model contents.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers literal `_component_moved_to` model components referenced by "
        "the official Chai-1 inference implementation at its current revision. "
        "It does not enumerate unrelated helper downloads or fetch weight files."
    )

    def __init__(
        self,
        *,
        name: str = "chai1-component-registry",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_source_bytes: int = _MAX_SOURCE_BYTES,
        max_components: int = _MAX_COMPONENTS,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip():
            raise ValueError("name and branch are required")
        if max_source_bytes <= 0 or not 1 <= max_components <= _MAX_COMPONENTS:
            raise ValueError("source and component limits are out of range")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_source_bytes = max_source_bytes
        self.max_components = max_components
        self.client = client or HttpClient(max_response_bytes=max_source_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "chai1-components-v1",
                "repository": repository,
                "branch": branch,
                "inference_path": _INFERENCE_PATH,
                "paths_path": _PATHS_PATH,
                "max_source_bytes": max_source_bytes,
                "max_components": max_components,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        commit_url = (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
        )
        commit = self.client.get(
            commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid repository revision")
        if revision == state.get("completed_revision"):
            return SourcePage(
                (), {**state}, True, upstream_count=state.get("component_count")
            )

        inference_url = self._raw_url(revision, _INFERENCE_PATH)
        paths_url = self._raw_url(revision, _PATHS_PATH)
        inference = self._get_text(inference_url, "inference source")
        paths = self._get_text(paths_url, "component URL source")
        components = _parse_components(inference, self.max_components)
        template = _parse_url_template(paths)
        records = (self._record(revision, inference_url, paths_url, template, components),)
        return SourcePage(
            records,
            {"completed_revision": revision, "component_count": len(components)},
            True,
            upstream_count=len(components),
            authoritative_snapshot=True,
        )

    def _raw_url(self, revision: str, path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{revision}/{path}"

    def _get_text(self, url: str, description: str) -> str:
        response = self.client.get(url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: {description} returned HTTP {response.status}")
        if len(response.body) > self.max_source_bytes:
            raise ValueError(f"{self.name}: {description} exceeds response limit")
        return response.text()

    def _record(
        self,
        revision: str,
        inference_url: str,
        paths_url: str,
        template: str,
        components: tuple[str, ...],
    ) -> SourceRecord:
        model_id = "model:chai1"
        model = ModelHint(
            model_id,
            "Chai-1",
            identifiers=(Identifier("chai:model", "chai1"),),
            status=ModelStatus.RELEASED,
            locator="chai_lab/chai1.py inference component call sites",
        )
        component_links: list[Link] = []
        releases: list[ReleaseHint] = []
        component_rows: list[dict[str, str]] = []
        for filename in components:
            asset_url = template.format(comp_key=filename)
            if not _is_allowed_asset_url(asset_url):
                raise ValueError(f"{self.name}: component URL is outside chaiassets.com")
            component_links.append(
                Link(
                    asset_url,
                    "weights",
                    crawl=False,
                    model_local_ids=(model_id,),
                )
            )
            releases.append(
                ReleaseHint(
                    f"component:{filename}",
                    model_id,
                    identifiers=(Identifier("chai:component", filename),),
                    metadata={"component_filename": filename, "artifact_url": asset_url},
                    locator="_component_moved_to literal in chai_lab/chai1.py",
                )
            )
            component_rows.append({"filename": filename, "url": asset_url})

        page_url = f"{self.repository_url}/tree/{revision}"
        return SourceRecord(
            source_record_id=f"{self.name}:{revision}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(page_url),
            title="Chai-1 exported model components",
            raw={
                "repository": self.repository,
                "revision": revision,
                "inference_source_url": inference_url,
                "component_url_source_url": paths_url,
                "component_url_template": template,
                "components": component_rows,
            },
            text="\n".join(f"{row['filename']} {row['url']}" for row in component_rows),
            identifiers=(Identifier("chai:registry-revision", revision),),
            links=(
                Link(inference_url, "source_registry", crawl=False, model_local_ids=(model_id,)),
                Link(paths_url, "source_registry", crawl=False, model_local_ids=(model_id,)),
                *component_links,
            ),
            models=(model,),
            releases=tuple(releases),
        )


def _parse_components(source: str, limit: int) -> tuple[str, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("Chai-1 inference source is invalid Python") from exc
    components: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "_component_moved_to"):
            continue
        if not node.args:
            raise ValueError("Chai-1 component call lacks a filename")
        try:
            filename = ast.literal_eval(node.args[0])
        except (ValueError, TypeError) as exc:
            raise ValueError("Chai-1 component filenames must be literals") from exc
        if not isinstance(filename, str) or not _COMPONENT.fullmatch(filename):
            raise ValueError("Chai-1 component filename is invalid")
        if filename not in components:
            components.append(filename)
            if len(components) > limit:
                raise ValueError("Chai-1 component inventory exceeds limit")
    if not components:
        raise ValueError("Chai-1 component inventory is empty")
    return tuple(sorted(components))


def _parse_url_template(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("Chai-1 paths source is invalid Python") from exc
    values: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "COMPONENT_URL"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Chai-1 COMPONENT_URL must be a literal string") from exc
        if isinstance(value, str):
            values.append(value)
    if len(values) != 1:
        raise ValueError("expected one literal Chai-1 COMPONENT_URL")
    template = values[0]
    if template != "https://chaiassets.com/chai1-inference-depencencies/models_v2/{comp_key}":
        raise ValueError("Chai-1 component URL template changed unexpectedly")
    if template.count("{comp_key}") != 1:
        raise ValueError("Chai-1 component URL template is invalid")
    return template


def _is_allowed_asset_url(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "chaiassets.com"
        and parsed.path.startswith("/chai1-inference-depencencies/models_v2/")
        and bool(_COMPONENT.fullmatch(parsed.path.rsplit("/", 1)[-1]))
        and not parsed.query
        and not parsed.fragment
    )


__all__ = ["Chai1ComponentRegistryAdapter"]
