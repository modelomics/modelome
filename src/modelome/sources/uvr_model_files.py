"""Read Ultimate Vocal Remover's first-party VR and MDX model-file manifest."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpClient, HttpResponse
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

_MANIFEST_URL = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"
)
_RELEASE_ROOT = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
_REPOSITORY_URL = "https://github.com/Anjok07/ultimatevocalremovergui"
_GROUPS = (("vr_download_list", "VR"), ("mdx_download_list", "MDX"))


def _utcnow() -> datetime:
    return datetime.now(UTC)


class UvrModelFilesAdapter:
    """Index UVR's finite public VR and MDX model-name-to-filename lists.

    The manifest and GitHub release names are read as metadata. Neither model
    files nor sibling Demucs, VIP, MDX23, or Roformer groups are fetched here.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only regular public VR and MDX file rows in UVR's first-party "
        "download_checks.json. It excludes VIP, Demucs, MDX23, Roformer, and "
        "community models not in those groups; no model bytes are fetched."
    )

    def __init__(
        self,
        *,
        name: str = "uvr-public-vr-mdx-model-files",
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
        min_models: int = 20,
        max_models: int = 200,
        max_response_bytes: int = 512 * 1024,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        for label, value in (
            ("min_models", min_models),
            ("max_models", max_models),
            ("max_response_bytes", max_response_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if min_models > max_models:
            raise ValueError("min_models cannot exceed max_models")
        self.name = name.strip()
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.min_models = min_models
        self.max_models = max_models
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "uvr-vr-mdx-model-files-v1",
                "manifest_url": _MANIFEST_URL,
                "release_root": _RELEASE_ROOT,
                "groups": _GROUPS,
                "min_models": min_models,
                "max_models": max_models,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            _MANIFEST_URL, headers={"Accept": "application/json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: UVR manifest returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: UVR manifest exceeds response limit")
        digest = content_hash(response.body)
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if digest == state.get("manifest_sha256"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=state.get("model_count"),
            )
        payload = response.json()
        rows = _parse_manifest(payload, self.name)
        if len(rows) < self.min_models:
            raise ValueError(
                f"{self.name}: manifest exposed only {len(rows)} models; "
                f"expected at least {self.min_models}"
            )
        if len(rows) > self.max_models:
            raise ValueError(f"{self.name}: manifest exceeds {self.max_models} model limit")
        records = tuple(self._record(row, digest) for row in rows)
        return SourcePage(
            records=records,
            next_state={
                "manifest_sha256": digest,
                "checked_at": checked_at,
                "model_count": len(records),
                "groups": {
                    group: sum(1 for row in rows if row.architecture == architecture)
                    for group, architecture in _GROUPS
                },
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, row: _UvrRow, digest: str) -> SourceRecord:
        filename = row.filename
        slug = filename.rsplit(".", 1)[0]
        model_id = f"model:uvr:{row.architecture.lower()}:{slug}"
        identifier = Identifier("uvr:model-file", filename)
        model = ModelHint(
            local_id=model_id,
            name=slug,
            aliases=(row.display_name,),
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=filename,
        )
        release = ReleaseHint(
            local_id=f"release:uvr:{row.architecture.lower()}:{slug}",
            model_local_id=model_id,
            identifiers=(Identifier("uvr:release-asset", filename),),
            metadata={
                "architecture": row.architecture,
                "display_name": row.display_name,
                "filename": filename,
                "weight_url": row.url,
                "release_tag": "all_public_uvr_models",
                "manifest_sha256": digest,
            },
            locator=filename,
        )
        return SourceRecord(
            source_record_id=f"uvr:{row.architecture.lower()}:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(row.url),
            title=row.display_name,
            raw={
                "architecture": row.architecture,
                "display_name": row.display_name,
                "filename": filename,
                "weight_url": row.url,
            },
            text=f"UVR {row.architecture} checkpoint {row.display_name} ({filename}).",
            identifiers=(identifier,),
            links=(
                Link(_MANIFEST_URL, "source_index", crawl=False, model_local_ids=(model_id,)),
                Link(
                    _REPOSITORY_URL,
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
                Link(row.url, "weights", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


class _UvrRow:
    __slots__ = ("architecture", "display_name", "filename", "url")

    def __init__(self, architecture: str, display_name: str, filename: str, url: str) -> None:
        self.architecture = architecture
        self.display_name = display_name
        self.filename = filename
        self.url = url


def _parse_manifest(payload: Any, source: str) -> tuple[_UvrRow, ...]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: UVR manifest is not an object")
    rows: list[_UvrRow] = []
    seen_names: set[tuple[str, str]] = set()
    seen_urls: set[str] = set()
    for group, architecture in _GROUPS:
        entries = payload.get(group)
        if not isinstance(entries, Mapping) or not entries:
            raise ValueError(f"{source}: manifest group {group} is missing or empty")
        for raw_name, raw_filename in entries.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError(f"{source}: model display name is invalid")
            if (
                not isinstance(raw_filename, str)
                or not raw_filename.endswith((".pth", ".onnx"))
                or "/" in raw_filename
                or "\\" in raw_filename
                or raw_filename in {".", ".."}
            ):
                raise ValueError(f"{source}: invalid UVR filename {raw_filename!r}")
            url = f"{_RELEASE_ROOT}{raw_filename}"
            _validate_url(url, raw_filename, source)
            identity = (architecture, raw_filename)
            if identity in seen_names or url in seen_urls:
                raise ValueError(f"{source}: duplicate UVR checkpoint {raw_filename}")
            seen_names.add(identity)
            seen_urls.add(url)
            rows.append(_UvrRow(architecture, raw_name.strip(), raw_filename, url))
    return tuple(rows)


def _validate_url(url: str, filename: str, source: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.path != f"/TRvlvr/model_repo/releases/download/all_public_uvr_models/{filename}"
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"{source}: invalid UVR release asset URL")


__all__ = ["UvrModelFilesAdapter"]
