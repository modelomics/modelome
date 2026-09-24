"""First-party download-script registry for CompVis latent-diffusion models."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpResponse
from modelome.models import SourcePage
from modelome.normalize import canonicalize_url, content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _isoformat,
    _text,
)

_DOWNLOAD = re.compile(r"^models/ldm/(?P<handle>[A-Za-z0-9_.+-]+)/(?P<file>[^/]+)$")
_FILE_SUFFIXES = (".zip", ".ckpt", ".safetensors", ".pt", ".pth")
_URL = re.compile(r"^https://ommer-lab\.com/files/latent-diffusion/.+$")


class CompVisLatentDiffusionDownloadsSourceAdapter(
    StaticJsonCheckpointRegistrySourceAdapter
):
    """Read exact model-to-download mappings from CompVis's own shell script.

    Only simple literal ``wget -O models/ldm/<model>/<file> <url>`` rows are
    accepted. URLs must remain on the first-party Ommer Lab model host, and
    one output directory can identify only one downloadable archive/checkpoint.
    The reader resolves the repository revision first and never runs the script
    or downloads a model asset.
    """

    coverage_limitation = (
        "Covers direct model bundle URLs declared by CompVis latent-diffusion's "
        "download_models.sh at a pinned Git commit. Zip rows identify the upstream "
        "download bundle; internal files are not separately enumerated."
    )

    def __init__(
        self,
        *,
        name: str = "compvis-latent-diffusion-downloads",
        repository: str = "CompVis/latent-diffusion",
        branch: str = "main",
        source_path: str = "scripts/download_models.sh",
        provider_namespace: str = "compvis:latent-diffusion",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 10_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            repository=repository,
            branch=branch,
            source_path=source_path,
            provider_namespace=provider_namespace,
            max_response_bytes=max_response_bytes,
            max_entries=max_entries,
            **kwargs,
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "compvis-latent-diffusion-downloads-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "provider_namespace": provider_namespace,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "literal wget target path and first-party archive/checkpoint URL",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=state.get("model_count")
                if isinstance(state.get("model_count"), int)
                else None,
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: download script returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: download script exceeds byte limit")
        checkpoints = _parse_download_script(
            response.text(), source=self.name, path=self.source_path,
            maximum=self.max_entries,
        )
        records = tuple(
            self._record(checkpoint, revision, response.body)
            for checkpoint in checkpoints
        )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
        }
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_download_script(
    document: str,
    *,
    source: str,
    path: str,
    maximum: int,
) -> tuple[_Checkpoint, ...]:
    entries: list[_Checkpoint] = []
    seen: set[str] = set()
    for line_number, line in enumerate(document.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            fields = shlex.split(stripped, comments=True, posix=True)
        except ValueError as error:
            raise ValueError(f"{source}: invalid shell syntax on line {line_number}") from error
        if not fields or fields[0] != "wget":
            continue
        # Do not accept variables, shell expansions, alternate wget options,
        # or commands that could make the URL/path ambiguous.
        if len(fields) != 4 or fields[1] != "-O":
            raise ValueError(f"{source}: unsupported wget row on line {line_number}")
        target, url = fields[2], fields[3]
        match = _DOWNLOAD.fullmatch(target)
        if match is None:
            raise ValueError(f"{source}: invalid model output path on line {line_number}")
        handle = match.group("handle")
        parts = urlsplit(url)
        if (
            not _URL.fullmatch(url)
            or parts.hostname != "ommer-lab.com"
            or not parts.path.casefold().endswith(_FILE_SUFFIXES)
        ):
            raise ValueError(f"{source}: non-first-party model asset URL on line {line_number}")
        if handle in seen:
            raise ValueError(f"{source}: duplicate model download handle {handle!r}")
        seen.add(handle)
        entries.append(
            _Checkpoint(
                handle=handle,
                url=canonicalize_url(url),
                locator=f"{path}:line:{line_number}",
            )
        )
        if len(entries) > maximum:
            raise ValueError(f"{source}: download list exceeds {maximum} entries")
    if not entries:
        raise ValueError(f"{source}: no supported checkpoint download rows found")
    return tuple(entries)


__all__ = ["CompVisLatentDiffusionDownloadsSourceAdapter"]
