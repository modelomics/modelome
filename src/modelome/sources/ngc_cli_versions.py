"""Guest-access historical NVIDIA NGC model-version enumeration via NGC CLI.

NVIDIA documents ``ngc registry model list 'org/model:*' --format_type json``
for model versions. This optional adapter wraps that installed CLI lazily; it
does not run or probe the CLI during import or construction. It is deliberately
not wired into the default catalog because it requires an external executable
and one CLI call per configured model card.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from modelome.models import (
    ArtifactKind,
    Identifier,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import content_hash


def ngc_model_versions_command(target: str, *, executable: str = "ngc") -> tuple[str, ...]:
    """Return the documented CLI invocation for all versions of one model."""
    target = _canonical_target(target)
    return (executable, "registry", "model", "list", f"{target}:*", "--format_type", "json")


def parse_ngc_model_versions_json(
    output: str | bytes,
    *,
    target: str,
    catalog_base_url: str = "https://catalog.ngc.nvidia.com",
) -> tuple[SourceRecord, ...]:
    """Parse documented NGC CLI JSON version rows into exact release records.

    The CLI's JSON rows expose ``versionId`` (plus release metadata such as
    ``createdDate``, ``status``, ``totalFileCount`` and ``totalSizeInBytes``).
    Empty arrays are valid and mean that the configured model has no visible
    versions. Missing or malformed row identity is an error, never a silent skip.
    """
    target = _canonical_target(target)
    try:
        payload = json.loads(output)
    except (TypeError, ValueError) as error:
        raise ValueError("NGC CLI output is not valid JSON") from error
    if not isinstance(payload, list):
        raise ValueError("NGC CLI version output must be a JSON array")
    base_url = _http_url(catalog_base_url, "catalog_base_url").rstrip("/")
    records: list[SourceRecord] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"NGC CLI version row {index} must be a JSON object")
        version = _nonempty(item.get("versionId"), f"row {index} versionId")
        if version in seen:
            raise ValueError(f"NGC CLI returned duplicate versionId {version!r}")
        seen.add(version)
        version_id = f"{target}:{version}"
        model_id = target
        model_local_id = f"ngc-model:{model_id}"
        version_url = f"{base_url}/{_catalog_path(target)}"
        identity = Identifier("ngc:model-version", version_id)
        metadata = {
            key: item[key]
            for key in (
                "createdDate",
                "status",
                "totalFileCount",
                "totalSizeInBytes",
                "releaseType",
                "releaseNotes",
                "isSigned",
            )
            if key in item
        }
        records.append(
            SourceRecord(
                source_record_id=version_id,
                kind=ArtifactKind.CATALOG_RECORD,
                canonical_url=version_url,
                title=f"{target} {version}",
                raw={"target": target, "version": dict(item)},
                published_at=_iso_text(item.get("createdDate")),
                identifiers=(identity,),
                models=(
                    ModelHint(
                        local_id=model_local_id,
                        name=target,
                        identifiers=(Identifier("ngc:model", model_id),),
                        status=ModelStatus.RELEASED,
                    ),
                ),
                releases=(
                    ReleaseHint(
                        local_id=f"ngc-release:{version_id}",
                        model_local_id=model_local_id,
                        version=version,
                        identifiers=(identity,),
                        released_at=_iso_text(item.get("createdDate")),
                        metadata=metadata,
                        locator="ngc-cli-json.versionId",
                    ),
                ),
            )
        )
    return tuple(records)


class NgcCliModelVersionsSourceAdapter:
    """Enumerate version rows for an explicit, bounded list of model targets.

    This invokes the CLI only when ``fetch_page`` is called. Supply ``runner``
    to integrate another execution mechanism; the default subprocess import is
    also deferred until then. CLI completion is one target per source page.
    """

    disable_derived_extraction = True
    authoritative_snapshot = False
    coverage_limitation = (
        "Covers version rows visible to NGC Catalog guest CLI for explicitly "
        "configured model targets. It requires the NGC CLI at fetch time and "
        "makes one list call per model; it does not discover model targets, "
        "include gated/private/removed versions, or enumerate files/weights."
    )

    def __init__(
        self,
        targets: Sequence[str],
        *,
        name: str = "ngc-cli-model-versions",
        executable: str = "ngc",
        catalog_base_url: str = "https://catalog.ngc.nvidia.com",
        max_versions_per_target: int = 5000,
        runner: Callable[..., Any] | None = None,
    ) -> None:
        if not targets:
            raise ValueError("targets must contain at least one model target")
        self.targets = tuple(_canonical_target(value) for value in targets)
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be unique")
        self.name = _nonempty(name, "name")
        self.executable = _nonempty(executable, "executable")
        self.catalog_base_url = _http_url(catalog_base_url, "catalog_base_url")
        if not isinstance(max_versions_per_target, int) or max_versions_per_target < 1:
            raise ValueError("max_versions_per_target must be a positive integer")
        self.max_versions_per_target = max_versions_per_target
        self.runner = runner
        self.checkpoint_signature = content_hash(
            {
                "adapter": "ngc-cli-model-versions-v1",
                "targets": self.targets,
                "executable": self.executable,
                "catalog_base_url": self.catalog_base_url,
                "max_versions_per_target": self.max_versions_per_target,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        index = state.get("target_index", 0)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("target_index must be a nonnegative integer")
        if index >= len(self.targets):
            raise ValueError("target_index is outside configured targets")
        target = self.targets[index]
        command = ngc_model_versions_command(target, executable=self.executable)
        runner = self.runner
        if runner is None:
            import subprocess

            runner = subprocess.run
        result = runner(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"{self.name}: NGC CLI exited with status {result.returncode}")
        records = parse_ngc_model_versions_json(
            result.stdout,
            target=target,
            catalog_base_url=self.catalog_base_url,
        )
        if len(records) > self.max_versions_per_target:
            raise ValueError(
                f"{self.name}: {target} has more than configured cap "
                f"{self.max_versions_per_target}; increase cap and restart"
            )
        next_index = index + 1
        complete = next_index == len(self.targets)
        return SourcePage(
            records=records,
            next_state={"target_index": next_index},
            complete=complete,
            upstream_count=None,
        )


def _canonical_target(value: Any) -> str:
    target = _nonempty(value, "target")
    if target.count("/") not in (1, 2) or ":" in target:
        raise ValueError("target must be org/[team/]model without a version")
    parts = target.split("/")
    if any(part in {".", ".."} or not part for part in parts):
        raise ValueError("target contains an invalid path component")
    return target


def _catalog_path(target: str) -> str:
    org, *tail = target.split("/")
    model = tail[-1]
    team_segment = f"{quote(tail[0], safe='')}/" if len(tail) == 2 else ""
    return f"orgs/{quote(org, safe='')}/{team_segment}models/{quote(model, safe='')}"


def _http_url(value: Any, field: str) -> str:
    text = _nonempty(value, field)
    if not (text.startswith("https://") or text.startswith("http://")):
        raise ValueError(f"{field} must be an HTTP(S) URL")
    return text


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _iso_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
