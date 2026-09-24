"""Exact RFdiffusion2 checkpoint files listed by its first-party installer."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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

Clock = Callable[[], datetime]
_REPOSITORY = "RosettaCommons/RFdiffusion2"
_PATH = "setup.py"
_BASE_URL = "https://files.ipd.uw.edu/pub/rfdiffusion2/"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_WEIGHT_PATH = re.compile(r"^model_weights/(?P<name>RFD_[0-9]+\.pt)$")
_MAX_ENTRIES = 32


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RFDiffusion2CheckpointRegistryAdapter:
    """Read model weight identities from RFdiffusion2's installer manifest.

    The installer lists both RFdiffusion2 weights and third-party LigandMPNN
    weights. This adapter deliberately emits only paths in its own
    ``model_weights/`` directory and records the exact first-party URLs without
    downloading their contents.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only .pt paths under model_weights/ listed in the official "
        "RFdiffusion2 setup.py WEIGHTS manifest. Third-party weights and SIF "
        "runtime images are excluded. Checkpoint bytes are not fetched."
    )

    def __init__(
        self,
        *,
        name: str = "rfdiffusion2-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_source_bytes: int = 512 * 1024,
        max_entries: int = _MAX_ENTRIES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or max_source_bytes <= 0 or max_entries <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_source_bytes = max_source_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_source_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "rfdiffusion2-checkpoint-registry-v1",
            "repository": repository,
            "branch": branch,
            "registry_path": _PATH,
            "max_source_bytes": max_source_bytes,
            "max_entries": max_entries,
        })

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage((), {**state, "checked_at": checked}, True,
                              upstream_count=state.get("checkpoint_count"))

        source_url = f"https://raw.githubusercontent.com/{self.repository}/{revision}/{_PATH}"
        response = self.client.get(source_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: installer manifest returned HTTP {response.status}")
        if len(response.body) > self.max_source_bytes:
            raise ValueError(f"{self.name}: installer manifest exceeds response limit")
        weights = _parse_manifest(response.text(), max_entries=self.max_entries)
        record = self._record(weights, revision, source_url)
        return SourcePage(
            (record,),
            {"completed_revision": revision, "checked_at": checked,
             "checkpoint_count": len(weights)},
            True,
            upstream_count=len(weights),
            authoritative_snapshot=True,
        )

    def _record(
        self, weights: tuple[tuple[str, str], ...], revision: str, source_url: str
    ) -> SourceRecord:
        page_url = f"{self.repository_url}/blob/{revision}/{_PATH}"
        models: list[ModelHint] = []
        releases: list[ReleaseHint] = []
        links: list[Link] = [Link(page_url, "model_card", crawl=False)]
        for path, url in weights:
            filename = path.rsplit("/", 1)[-1]
            handle = filename.removesuffix(".pt")
            local_id = f"model:{handle}"
            models.append(ModelHint(
                local_id, f"RFdiffusion2 {handle}",
                identifiers=(Identifier("rfdiffusion2:checkpoint", handle),),
                aliases=(filename,), status=ModelStatus.RELEASED,
                locator=f"{_PATH}: WEIGHTS",
            ))
            releases.append(ReleaseHint(
                f"release:{handle}", local_id,
                identifiers=(Identifier("rfdiffusion2:checkpoint-url", url),),
                metadata={"checkpoint_path": path, "checkpoint_filename": filename,
                          "checkpoint_url": url, "installer_revision": revision},
                locator=f"{_PATH}: WEIGHTS",
            ))
            links.append(Link(url, "weights", crawl=False, model_local_ids=(local_id,)))
        return SourceRecord(
            source_record_id=f"{self.name}:{revision}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(page_url),
            title="RFdiffusion2 pretrained checkpoints",
            raw={"repository": self.repository, "revision": revision,
                 "manifest_url": source_url,
                 "checkpoints": [{"path": path, "url": url} for path, url in weights]},
            text="\n".join(f"{path} {url}" for path, url in weights),
            identifiers=(Identifier("rfdiffusion2:registry-revision", revision),),
            links=tuple(links), models=tuple(models), releases=tuple(releases),
        )


def _parse_manifest(source: str, *, max_entries: int) -> tuple[tuple[str, str], ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("RFdiffusion2 installer is invalid Python") from exc
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"BASE_URL", "WEIGHTS"}:
                    if target.id in assignments:
                        raise ValueError(f"RFdiffusion2 {target.id} is assigned more than once")
                    assignments[target.id] = node.value
    if set(assignments) != {"BASE_URL", "WEIGHTS"}:
        raise ValueError("RFdiffusion2 BASE_URL or WEIGHTS manifest is missing")
    try:
        base_url = ast.literal_eval(assignments["BASE_URL"])
        raw_weights = ast.literal_eval(assignments["WEIGHTS"])
    except (ValueError, TypeError) as exc:
        raise ValueError("RFdiffusion2 manifest fields must be literals") from exc
    if base_url != _BASE_URL:
        raise ValueError("RFdiffusion2 base URL changed or is unsupported")
    if not isinstance(raw_weights, list) or not raw_weights:
        raise ValueError("RFdiffusion2 WEIGHTS must be a nonempty literal list")
    if len(raw_weights) > max_entries:
        raise ValueError("RFdiffusion2 WEIGHTS exceeds entry limit")
    admitted: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in raw_weights:
        if not isinstance(path, str):
            raise ValueError("RFdiffusion2 weight paths must be strings")
        match = _WEIGHT_PATH.fullmatch(path)
        if match is None:
            if path.startswith("model_weights/"):
                raise ValueError("RFdiffusion2 manifest has an unsupported checkpoint path")
            continue
        if path in seen:
            raise ValueError("RFdiffusion2 manifest repeats a checkpoint path")
        seen.add(path)
        admitted.append((path, _BASE_URL + path))
    if not admitted:
        raise ValueError("RFdiffusion2 manifest has no own model checkpoints")
    return tuple(sorted(admitted))


__all__ = ["RFDiffusion2CheckpointRegistryAdapter"]
