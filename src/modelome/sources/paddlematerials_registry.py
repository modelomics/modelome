"""PaddleMaterials first-party pretrained model package registry."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

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

_REPOSITORY = "PaddlePaddle/PaddleMaterials"
_BRANCH = "develop"
_PATH = "ppmat/models/__init__.py"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_URL_PREFIXES = (
    "https://paddle-org.bj.bcebos.com/paddlematerial/",
    "https://paddle-org.bj.bcebos.com/paddlematerials/",
)
# These entries duplicate separately indexed first-party potential/checkpoint
# families; PaddleMaterials' remaining registry is independently enumerable.
_EXCLUDED = frozenset({"chgnet_mptrj", "mattersim_1M", "mattersim_5M"})


class PaddleMaterialsRegistryAdapter:
    """Index literal model-package names and URLs from PaddleMaterials source."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Indexes PaddleMaterials MODEL_REGISTRY zip package URLs, excluding its CHGNet and "
        "MatterSim aliases already covered by first-party model sources. URLs come from the "
        "current pinned Python source. Binary archives are not requested."
    )

    def __init__(
        self,
        *,
        name: str = "paddlematerials-model-registry",
        client: HttpClient | Any | None = None,
        max_response_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if not name.strip() or max_response_bytes <= 0:
            raise ValueError("name and positive response limit are required")
        self.name = name
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {"adapter": "paddlematerials-registry-v1", "repository": _REPOSITORY,
             "branch": _BRANCH, "path": _PATH, "excluded": sorted(_EXCLUDED)}
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/commits/{_BRANCH}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage((), next_state, True, upstream_count=state.get("model_count"))

        source_url = f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/{_PATH}"
        response = self.client.get(source_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: registry source returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: registry exceeds {self.max_response_bytes} bytes")
        rows = _parse_registry(response.text(), self.name)
        records = tuple(self._record(key, url, revision, source_url) for key, url in rows)
        return SourcePage(
            records,
            {"completed_revision": revision, "checked_at": checked_at,
             "source_url": source_url, "source_sha256": content_hash(response.body),
             "model_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, key: str, url: str, revision: str, source_url: str) -> SourceRecord:
        identity = Identifier("paddlematerials:model-package", key)
        model_id = f"model:{key}"
        model = ModelHint(
            model_id, f"PaddleMaterials pretrained package {key}", aliases=(key,),
            identifiers=(identity,), status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"package:{key}", model_id, revision=revision,
            identifiers=(Identifier("paddlematerials:model-package:release", key),),
            metadata={"repository": _REPOSITORY, "revision": revision,
                      "source_path": _PATH, "package_url": url,
                      "binary_reachability_checked": False},
        )
        return SourceRecord(
            source_record_id=f"package:{key}", kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url), title=model.name,
            raw={"package_name": key, "package_url": url},
            text=f"PaddleMaterials pretrained model package: {key}",
            identifiers=(identity,),
            links=(Link(source_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                   Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
                   Link(f"https://github.com/{_REPOSITORY}", "source_implementation", crawl=False)),
            models=(model,), releases=(release,),
        )


def _parse_registry(source_text: str, source: str) -> tuple[tuple[str, str], ...]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError as error:
        raise ValueError(f"{source}: registry source is not valid Python") from error
    registry: ast.Dict | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MODEL_REGISTRY"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                registry = node.value
            break
    if registry is None or not registry.keys:
        raise ValueError(f"{source}: expected a non-empty literal MODEL_REGISTRY")
    rows: list[tuple[str, str]] = []
    for key_node, url_node in zip(registry.keys, registry.values, strict=True):
        if (not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str)
                or not isinstance(url_node, ast.Constant) or not isinstance(url_node.value, str)):
            raise ValueError(f"{source}: registry entries must use literal string keys and URLs")
        key, url = key_node.value, url_node.value
        if key in _EXCLUDED:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_]+", key):
            raise ValueError(f"{source}: invalid package name {key!r}")
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not any(url.startswith(prefix) for prefix in _URL_PREFIXES)
                or not parsed.path.endswith(f"/{key}.zip") or parsed.query or parsed.fragment):
            raise ValueError(f"{source}: invalid first-party package URL for {key!r}")
        rows.append((key, url))
    if len(rows) < 20 or len({key for key, _ in rows}) != len(rows):
        raise ValueError(f"{source}: registry must contain 20+ unique supported package entries")
    return tuple(rows)


__all__ = ["PaddleMaterialsRegistryAdapter"]
