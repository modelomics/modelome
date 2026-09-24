"""Checkpoint file metadata for Zatom's public Zenodo record."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
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

Clock = Callable[[], datetime]
_RECORD_ID = "19766997"
_RECORD_URL = f"https://zenodo.org/records/{_RECORD_ID}"
_API_URL = f"https://zenodo.org/api/records/{_RECORD_ID}"
_README_URL = "https://github.com/Zatom-AI/zatom/blob/main/README.md"
_README_DECLARED_FILES = frozenset(
    {
        "zatom_1_joint_paper_weights.ckpt",
        "zatom_1_joint_mat_prop_paper_weights.ckpt",
        "zatom_1_joint_pretraining_paper_weights.ckpt",
        "zatom_1_l_joint_pretraining_paper_weights.ckpt",
        "zatom_1_xl_joint_pretraining_paper_weights.ckpt",
        "zatom_1_wd_joint_pretraining_paper_weights.ckpt",
        "platom_1_qm9_only_pretraining_paper_weights.ckpt",
        "zatom_1_mp20_only_pretraining_paper_weights.ckpt",
        "zatom_1_qm9_only_pretraining_paper_weights.ckpt",
        "zatom_1_joint_geom_pretraining_paper_weights.ckpt",
        "zatom_1_qmof_only_pretraining_paper_weights.ckpt",
        "zatom_1_omol25_only_pretraining_paper_weights.ckpt",
        "zatom_1_omol25_only_mlip_pretraining_paper_weights.ckpt",
        "zatom_1_omol25_pretrained_and_finetuned_mlip_paper_weights.ckpt",
        "zatom_1_joint_mol_prop_pred_paper_weights.ckpt",
        "zatom_1_non_pretrained_mol_prop_pred_paper_weights.ckpt",
        "zatom_1_qm9_only_mol_prop_pred_paper_weights.ckpt",
        "zatom_1_joint_k25_layer_mol_prop_pred_paper_weights.ckpt",
        "zatom_1_joint_mid_layer_mol_prop_pred_paper_weights.ckpt",
        "zatom_1_joint_k75_layer_mol_prop_pred_paper_weights.ckpt",
        "zatom_1_xl_joint_mol_prop_pred_paper_weights.ckpt",
        "zatom_1_joint_mol_and_mat_prop_pred_paper_weights.ckpt",
        "zatom_1_qm9_only_mol_and_mat_prop_pred_paper_weights.ckpt",
        "zatom_1_joint_paper_weights_equal_mol_mat_training_set_ratio.ckpt",
    }
)
_CHECKPOINT = re.compile(r"^[A-Za-z0-9_.-]+\.ckpt$")
_MD5 = re.compile(r"^md5:[0-9a-f]{32}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ZatomZenodoCheckpointRecordSourceAdapter:
    """Index checkpoint file metadata from Zenodo's official public record API."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only .ckpt file objects in Zenodo record 19766997. The README-membership "
        "label reflects the audited README checkpoint commands; other files are excluded. "
        "Only metadata is requested, never model file content."
    )

    def __init__(
        self,
        *,
        name: str = "zatom-zenodo-checkpoint-record",
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 100,
    ) -> None:
        if not name.strip() or max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("name and positive response/entry limits are required")
        self.name = name
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.checkpoint_signature = content_hash(
            {
                "adapter": "zatom-zenodo-checkpoint-record-v1",
                "record_id": _RECORD_ID,
                "readme_declared_files": sorted(_README_DECLARED_FILES),
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(_API_URL, headers={"Accept": "application/json"})
        if response.status != 200:
            raise ValueError(f"{self.name}: Zenodo API returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: Zenodo metadata exceeds {self.max_response_bytes} bytes"
            )
        digest = content_hash(response.body)
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if digest == state.get("record_sha256"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage((), next_state, True, upstream_count=state.get("model_count"))

        payload = response.json()
        if not isinstance(payload, Mapping) or str(payload.get("id", "")) != _RECORD_ID:
            raise ValueError(f"{self.name}: response is not Zenodo record {_RECORD_ID}")
        files = _parse_files(payload.get("files"), self.name, self.max_entries)
        records = tuple(self._record(item, digest) for item in files)
        return SourcePage(
            records,
            {
                "record_sha256": digest,
                "checked_at": checked_at,
                "source_url": _API_URL,
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, item: tuple[str, str, str], digest: str) -> SourceRecord:
        filename, checksum, weight_url = item
        handle = filename.removesuffix(".ckpt")
        model_id = f"model:{handle}"
        readme_status = (
            "declared_in_readme_checkpoint_section"
            if filename in _README_DECLARED_FILES
            else "record_only_not_in_readme_checkpoint_section"
        )
        identity = Identifier("zatom:zenodo-file", filename)
        checkpoint_identity = Identifier("zatom:checkpoint", handle)
        model = ModelHint(
            model_id,
            filename,
            aliases=(handle,),
            identifiers=(identity, checkpoint_identity),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            identifiers=(Identifier("zatom:zenodo-file:release", filename),),
            metadata={
                "zenodo_record_id": _RECORD_ID,
                "zenodo_file_key": filename,
                "checkpoint_url": weight_url,
                "checksum": checksum,
                "readme_coverage": readme_status,
                "record_metadata_sha256": digest,
                "binary_reachability_checked": False,
            },
        )
        return SourceRecord(
            source_record_id=f"zenodo-file:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(weight_url),
            title=filename,
            raw={
                "zenodo_record_id": _RECORD_ID,
                "zenodo_file_key": filename,
                "checkpoint_url": weight_url,
                "checksum": checksum,
                "readme_coverage": readme_status,
            },
            text=f"Zatom Zenodo checkpoint file: {filename} ({checksum})",
            identifiers=(identity,),
            links=(
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(_RECORD_URL, "source_record", crawl=False, model_local_ids=(model_id,)),
                Link(_README_URL, "source_documentation", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_files(
    raw_files: Any, source: str, maximum: int
) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(raw_files, list):
        raise ValueError(f"{source}: Zenodo record has no file list")
    results: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{source}: Zenodo file entry is not an object")
        key = raw.get("key")
        if not isinstance(key, str) or not key.endswith(".ckpt"):
            continue
        if not _CHECKPOINT.fullmatch(key) or key in seen:
            raise ValueError(f"{source}: invalid or duplicate checkpoint key {key!r}")
        checksum = raw.get("checksum")
        if not isinstance(checksum, str) or not _MD5.fullmatch(checksum):
            raise ValueError(f"{source}: checkpoint {key!r} lacks a verified MD5 checksum")
        links = raw.get("links")
        url = links.get("self") if isinstance(links, Mapping) else None
        expected_url = f"https://zenodo.org/api/records/{_RECORD_ID}/files/{key}/content"
        if url != expected_url or urlsplit(url).hostname != "zenodo.org":
            raise ValueError(f"{source}: checkpoint {key!r} has an unexpected file URL")
        seen.add(key)
        results.append((key, checksum, url))
        if len(results) > maximum:
            raise ValueError(f"{source}: checkpoint list exceeds {maximum} entries")
    if not results:
        raise ValueError(f"{source}: Zenodo record has no .ckpt file entries")
    extras = seen - _README_DECLARED_FILES
    if extras != {
        "platom_1_joint_pretraining_paper_weights.ckpt",
        "zatom_1_from_omol25_mol_prop_pred_paper_weights.ckpt",
    }:
        raise ValueError(f"{source}: record-only checkpoint inventory changed: {sorted(extras)}")
    if not seen >= _README_DECLARED_FILES:
        raise ValueError(f"{source}: one or more README-declared checkpoint files are absent")
    return tuple(results)


__all__ = ["ZatomZenodoCheckpointRecordSourceAdapter"]
