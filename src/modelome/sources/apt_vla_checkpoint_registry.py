"""APT's first-party VLA checkpoint list and published Hugging Face assets."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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

_REPOSITORY = "xukechun/APT"
_DOCUMENT = "README.md"
_HF_REPOSITORY = "KechunXu1/apt_models"
_APT_VA_ARCHIVE_REVISION = "b0efa7a0892cc5ad776c7fac1ca21c8500576207"
_APT_VA_BEST_OID = "71d232cc580cc6e9cfc63c397d1d6242fb20aa5d"
_APT_VA_BEST_SIZE = 676_848_250
_APT_VA_ARCHIVE_CONFIG = "202605111256.json"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*$")
_POLICY_DIRS = {"apt_vla", "apt_vla_ftlibero", "apt_vla_ftpp"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class APTVLACheckpointRegistryAdapter:
    """Join APT's exact README release rows to exact published HF checkpoint files."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers three APT VLA releases named in its first-party README, the published "
        "Stage-0 apt_va prior, and its separately named historical ckpt_best artifact. "
        "It does not index user runs or download weights."
    )

    def __init__(
        self,
        *,
        name: str = "apt-vla-checkpoint-registry",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 8,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        for value, label in (
            (max_response_bytes, "max_response_bytes"),
            (max_entries, "max_entries"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "apt-vla-checkpoint-registry-v1",
                "repository": _REPOSITORY,
                "document": _DOCUMENT,
                "hf_repository": _HF_REPOSITORY,
                "policy_directories": sorted(_POLICY_DIRS),
                "apt_va_archive_revision": _APT_VA_ARCHIVE_REVISION,
                "apt_va_best_oid": _APT_VA_BEST_OID,
                "apt_va_archive_config": _APT_VA_ARCHIVE_CONFIG,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "first-party README table rows joined to published ckpt_latest.pt",
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{_REPOSITORY}"

    @property
    def hf_repository_url(self) -> str:
        return f"https://huggingface.co/{_HF_REPOSITORY}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        document_url = f"https://raw.githubusercontent.com/{_REPOSITORY}/main/{_DOCUMENT}"
        document_response: HttpResponse = self.client.get(
            document_url, headers={"Accept": "text/markdown,text/plain"}
        )
        self._require_response(document_response, "first-party README")
        entries = _parse_inventory(document_response.text(), source=self.name)
        if not entries:
            raise ValueError(f"{self.name}: no first-party checkpoint entries found")
        if len(entries) > self.max_entries:
            raise ValueError(f"{self.name}: inventory exceeds {self.max_entries} entries")

        info_response: HttpResponse = self.client.get(
            f"https://huggingface.co/api/models/{_HF_REPOSITORY}",
            headers={"Accept": "application/json"},
        )
        self._require_response(info_response, "HF model metadata")
        info = info_response.json()
        revision = info.get("sha", "") if isinstance(info, Mapping) else ""
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: HF model metadata did not return a full commit SHA")

        records: list[SourceRecord] = []
        listing_hashes: list[bytes] = []
        for entry in entries:
            folder = entry[0]
            files = self._list_folder(revision, folder, listing_hashes)
            if "ckpt_latest.pt" not in files or "config.json" not in files:
                raise ValueError(f"{self.name}: {folder} is missing checkpoint/config assets")
            records.append(
                self._record(entry, revision, files, document_response.body, document_url)
            )

        # The README documents Stage 0 as the visual-action prior (`apt_va`).
        # Its published checkpoint is outside the three VLA rows above.
        va_files = self._list_folder(revision, "apt_va", listing_hashes)
        if "ckpt_latest.pt" not in va_files or "config.json" not in va_files:
            raise ValueError(f"{self.name}: apt_va is missing checkpoint/config assets")
        va_entry = (
            "apt_va",
            "APT Vision-Action prior",
            "Stage-0 visual-action policy prior without language conditioning",
            "Droid + AgiBotWorld + InternA1 + InternM1",
            0,
        )
        records.append(
            self._record(
                va_entry,
                revision,
                va_files,
                document_response.body,
                document_url,
                record_suffix="latest",
            )
        )

        # Preserve the distinct ckpt_best artifact from the official Hub history.
        archived_files = self._list_folder(_APT_VA_ARCHIVE_REVISION, "apt_va", listing_hashes)
        best = archived_files.get("ckpt_best.pt")
        if best is None or best["oid"] != _APT_VA_BEST_OID or best["size"] != _APT_VA_BEST_SIZE:
            raise ValueError(f"{self.name}: archived apt_va best checkpoint changed or disappeared")
        records.append(
            self._record(
                (
                    "apt_va",
                    "APT Vision-Action prior (historical best)",
                    "Stage-0 visual-action policy prior; first-party archived best checkpoint",
                    "Droid + AgiBotWorld + InternA1 + InternM1",
                    0,
                ),
                _APT_VA_ARCHIVE_REVISION,
                archived_files,
                document_response.body,
                document_url,
                checkpoint_name="ckpt_best.pt",
                config_name=_APT_VA_ARCHIVE_CONFIG,
                record_suffix="best-archived",
            )
        )

        checked_at = self.clock()
        if checked_at.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return SourcePage(
            records=tuple(records),
            next_state={
                "asset_revision": revision,
                "checked_at": checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "document_sha256": content_hash(document_response.body),
                "asset_listing_sha256": content_hash(b"\n".join(listing_hashes)),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        entry: tuple[str, str, str, str, int],
        revision: str,
        files: dict[str, dict[str, Any]],
        document: bytes,
        document_url: str,
        *,
        checkpoint_name: str = "ckpt_latest.pt",
        config_name: str = "config.json",
        record_suffix: str | None = None,
    ) -> SourceRecord:
        folder, label, purpose, datasets, line = entry
        checkpoint = files[checkpoint_name]
        config = files[config_name]
        model_local_id = f"model:{folder}"
        locator = f"{_DOCUMENT}:line:{line}"
        identifier = Identifier("apt:policy", folder)
        weights_url = self._asset_url(revision, checkpoint["path"])
        config_url = self._asset_url(revision, config["path"])
        metadata = {
            "repository": _REPOSITORY,
            "document_path": _DOCUMENT,
            "hf_repository": _HF_REPOSITORY,
            "asset_revision": revision,
            "checkpoint_path": checkpoint["path"],
            "checkpoint_oid": checkpoint["oid"],
            "checkpoint_size_bytes": checkpoint["size"],
            "config_path": config["path"],
            "config_oid": config["oid"],
            "config_size_bytes": config["size"],
            "release_label": label,
            "purpose": purpose,
            "datasets": datasets,
            "source_document_sha256": content_hash(document),
        }
        model = ModelHint(
            local_id=model_local_id,
            name=label,
            identifiers=(identifier,),
            aliases=(folder,),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{folder}:{record_suffix}" if record_suffix else f"release:{folder}",
            model_local_id=model_local_id,
            revision=revision,
            identifiers=(Identifier("apt:checkpoint", checkpoint["path"]),),
            metadata=metadata,
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"apt:{folder}:{record_suffix}" if record_suffix else f"apt:{folder}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(weights_url),
            title=f"APT {label}",
            raw=metadata,
            text=f"{label}; {purpose}; datasets: {datasets}",
            identifiers=(identifier,),
            links=(
                Link(
                    weights_url,
                    relation="model_artifact",
                    locator=checkpoint["path"],
                    crawl=False,
                    model_local_ids=(model_local_id,),
                ),
                Link(
                    config_url,
                    relation="model_configuration",
                    locator=config["path"],
                    crawl=False,
                    model_local_ids=(model_local_id,),
                ),
                Link(document_url, relation="model_catalog", locator=locator, crawl=False),
                Link(self.hf_repository_url, relation="asset_repository", crawl=False),
                Link(self.repository_url, relation="source_repository", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )

    def _asset_url(self, revision: str, path: str) -> str:
        return f"{self.hf_repository_url}/resolve/{revision}/{quote(path, safe='/')}"

    def _list_folder(
        self, revision: str, folder: str, listing_hashes: list[bytes]
    ) -> dict[str, dict[str, Any]]:
        url = (
            f"https://huggingface.co/api/models/{_HF_REPOSITORY}/tree/{revision}/"
            f"{quote(folder, safe='/')}"
        )
        response: HttpResponse = self.client.get(
            url,
            params={"recursive": "false", "expand": "false"},
            headers={"Accept": "application/json"},
        )
        self._require_response(response, f"HF file listing {folder}@{revision}")
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"{self.name}: HF file listing is not a list")
        listing_hashes.append(response.body)
        files: dict[str, dict[str, Any]] = {}
        for item in payload:
            if not isinstance(item, Mapping) or item.get("type") != "file":
                continue
            path = item.get("path")
            if not isinstance(path, str) or not _SAFE_PATH.fullmatch(path):
                raise ValueError(f"{self.name}: invalid path in HF listing")
            if path.rsplit("/", 1)[0] != folder:
                continue
            oid, size = item.get("oid"), item.get("size")
            if not isinstance(oid, str) or not oid.strip():
                raise ValueError(f"{self.name}: file lacks object id: {path}")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"{self.name}: file has invalid size: {path}")
            files[path.rsplit("/", 1)[-1]] = {"path": path, "oid": oid, "size": size}
        return files

    def _require_response(self, response: HttpResponse, label: str) -> None:
        if response.status != 200:
            raise ValueError(f"{self.name}: {label} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: {label} exceeds {self.max_response_bytes} bytes")


def _parse_inventory(document: str, *, source: str) -> tuple[tuple[str, str, str, str, int], ...]:
    entries: dict[str, tuple[str, str, str, str, int]] = {}
    active = False
    for line_number, line in enumerate(document.splitlines(), start=1):
        if "Pretrained Checkpoints" in line:
            active = True
            continue
        if active and line.startswith("## "):
            break
        if not active or "|" not in line:
            continue
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0].casefold() in {"stage", "---"}:
            continue
        match = re.search(
            r"(?:huggingface\.co/[^/]+/[^/]+/tree/[^/]+/)?(apt_vla(?:_ftlibero|_ftpp)?)", line
        )
        if not match:
            continue
        folder = match.group(1)
        if folder not in _POLICY_DIRS or folder in entries:
            raise ValueError(f"{source}: invalid or duplicate checkpoint row on line {line_number}")
        entries[folder] = (folder, cells[0], cells[1], cells[2], line_number)
    missing = _POLICY_DIRS - entries.keys()
    if missing:
        raise ValueError(
            f"{source}: first-party checkpoint table missing entries: {sorted(missing)}"
        )
    return tuple(entries[key] for key in sorted(entries))
