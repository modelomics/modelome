"""MACE-MP and MACE-Polar literal pretrained checkpoint inventories."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
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

Clock = Callable[[], datetime]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FAMILIES = {
    "mace_mp_urls": ("mace-mp", "mace:mp-checkpoint"),
    "polar_model_urls": ("mace-polar", "mace:polar-checkpoint"),
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MaceFoundationCheckpointRegistrySourceAdapter:
    """Read the literal MP and Polar maps from first-party MACE source.

    It parses two top-level static Python dictionaries in the pinned MACE loader
    without importing or executing the module. Only direct GitHub release assets
    under the two MACE foundation repositories are admitted.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers literal `mace_mp_urls` and `polar_model_urls` entries in the MACE "
        "foundation loader. It excludes MACE-OFF23 (indexed separately), MACE-ANI "
        "and MACE-MDP (not listed in these maps), arbitrary URLs, and user-trained "
        "checkpoints; model binaries are not fetched."
    )

    def __init__(
        self,
        *,
        name: str = "mace-foundation-checkpoints",
        repository: str = "ACEsuit/mace",
        branch: str = "develop",
        source_path: str = "mace/calculators/foundations_models.py",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 1_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if (not name.strip() or repository != "ACEsuit/mace"
                or source_path != "mace/calculators/foundations_models.py"
                or not branch.strip()):
            raise ValueError("name and the official MACE repository/path are required")
        if max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("response and entry limits must be positive")
        self.name, self.repository, self.branch = name, repository, branch
        self.source_path = source_path
        self.max_response_bytes, self.max_entries = max_response_bytes, max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "mace-foundation-literal-maps-v1",
            "repository": repository,
            "branch": branch,
            "source_path": source_path,
            "families": _FAMILIES,
            "max_response_bytes": max_response_bytes,
            "max_entries": max_entries,
        })

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}",
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
                              upstream_count=state.get("model_count"))

        source_url = (
            f"https://raw.githubusercontent.com/{self.repository}/{revision}/"
            f"{quote(self.source_path, safe='/')}"
        )
        response = self.client.get(source_url, headers={"Accept": "text/x-python,text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: source file returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source file exceeds response limit")
        entries = _parse_maps(response.text(), self.max_entries, self.name)
        records = tuple(
            self._record(family, namespace, handle, url, revision, source_url)
            for family, namespace, handle, url in entries
        )
        if not records:
            raise ValueError(f"{self.name}: no supported checkpoint entries found")
        return SourcePage(
            records,
            {"completed_revision": revision, "checked_at": checked,
             "source_sha256": content_hash(response.body), "model_count": len(records)},
            True, upstream_count=len(records), authoritative_snapshot=True,
        )

    def _record(
        self, family: str, namespace: str, handle: str, url: str,
        revision: str, source_url: str,
    ) -> SourceRecord:
        model_id = f"model:{family}:{handle}"
        source_page = f"{self.repository_url}/blob/{revision}/{self.source_path}"
        model = ModelHint(
            model_id, f"MACE {family.upper()} {handle}",
            identifiers=(Identifier(namespace, handle),), aliases=(handle,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{family}:{handle}", model_id, version=handle,
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={"repository": self.repository, "revision": revision,
                      "source_path": self.source_path, "family_map": _family_map(family),
                      "checkpoint_handle": handle, "weight_url": url},
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{family}:{handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(url),
            title=model.name,
            raw={"repository": self.repository, "revision": revision,
                 "source_path": self.source_path, "family": family,
                 "family_map": _family_map(family), "checkpoint_handle": handle,
                 "weight_url": url, "source_url": source_url},
            text=f"First-party MACE foundation checkpoint {family} / {handle}.",
            identifiers=(Identifier(namespace, handle),),
            links=(
                Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(source_page, "model_card", crawl=False,
                     model_local_ids=(model_id,)),
                Link(self.repository_url, "source_implementation", crawl=False,
                     model_local_ids=(model_id,)),
            ),
            models=(model,), releases=(release,),
        )


def _family_map(family: str) -> str:
    return "mace_mp_urls" if family == "mace-mp" else "polar_model_urls"


def _parse_maps(
    source_text: str, maximum: int, source: str
) -> tuple[tuple[str, str, str, str], ...]:
    try:
        module = ast.parse(source_text)
    except SyntaxError as exc:
        raise ValueError(f"{source}: source is not valid Python: {exc.msg}") from exc
    assignments: dict[str, ast.Dict] = {}
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in _FAMILIES:
            if target.id in assignments or not isinstance(statement.value, ast.Dict):
                raise ValueError(
                    f"{source}: {_family_map(target.id)} is not one literal dictionary"
                )
            assignments[target.id] = statement.value
    if set(assignments) != set(_FAMILIES):
        raise ValueError(f"{source}: expected literal MACE-MP and MACE-Polar maps")
    count = sum(len(mapping.keys) for mapping in assignments.values())
    if count > maximum:
        raise ValueError(f"{source}: registry exceeds {maximum} entries")

    results: list[tuple[str, str, str, str]] = []
    for map_name, (family, namespace) in _FAMILIES.items():
        mapping = assignments[map_name]
        if not mapping.keys:
            raise ValueError(f"{source}: {map_name} is empty")
        seen: set[str] = set()
        for key_node, value_node in zip(mapping.keys, mapping.values, strict=True):
            handle = _literal_string(key_node)
            url = _literal_string(value_node)
            valid_handle = (
                isinstance(handle, str)
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", handle)
            )
            if not valid_handle or url is None or handle in seen:
                raise ValueError(
                    f"{source}: {map_name} must contain unique literal handles and URLs"
                )
            seen.add(handle)
            if not _valid_foundation_url(map_name, handle, url):
                raise ValueError(
                    f"{source}: {map_name} {handle!r} has an unapproved checkpoint URL"
                )
            results.append((family, namespace, handle, url))
    return tuple(results)


def _valid_foundation_url(map_name: str, handle: str, url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.query or parts.fragment:
        return False
    segments = parts.path.split("/")
    if not segments or not segments[-1].endswith(".model"):
        return False
    if map_name == "polar_model_urls":
        return (
            parts.hostname == "github.com"
            and segments[:5] == ["", "ACEsuit", "mace-foundations", "releases", "download"]
            and segments[5] == "mace_polar_1"
            and segments[-1] == f"MACE-POLAR-1-{handle.removeprefix('polar-1-').upper()}.model"
            and handle in {"polar-1-s", "polar-1-m", "polar-1-l"}
        )
    return (
        parts.hostname == "github.com"
        and len(segments) == 7
        and segments[0] == ""
        and segments[1] == "ACEsuit"
        and segments[2] in {"mace-mp", "mace-foundations"}
        and segments[3:5] == ["releases", "download"]
    )


def _literal_string(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


__all__ = ["MaceFoundationCheckpointRegistrySourceAdapter"]
