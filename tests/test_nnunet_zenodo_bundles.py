from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.nnunet_zenodo_bundles import NnUNetZenodoBundleRegistryAdapter


class _QueuedClient:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        return self.response


def _response(record_id: str, payload: Mapping[str, Any]) -> HttpResponse:
    return HttpResponse(
        200, {"ETag": f'"{record_id}-snapshot"'}, json.dumps(payload).encode(),
        f"https://zenodo.org/api/records/{record_id}",
    )


def _bundle(record_id: str, filename: str, checksum: str, size: int) -> dict[str, Any]:
    return {
        "key": filename,
        "checksum": f"md5:{checksum}",
        "size": size,
        "links": {
            "self": (
                f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"
            )
        },
    }


def _adapter(record_id: str, payload: Mapping[str, Any]) -> NnUNetZenodoBundleRegistryAdapter:
    return NnUNetZenodoBundleRegistryAdapter(
        record_id=record_id,
        client=_QueuedClient(_response(record_id, payload)),
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_newer_nnunet_v1_zenodo_record_exposes_task_archive_version_identities() -> None:
    record_id = "4003545"
    payload = {
        "metadata": {"doi": "10.5281/zenodo.4003545", "title": "nnU-Net v1", "version": "2"},
        "files": [
            _bundle(record_id, "Task075_Fluo_C3DH_A549_ManAndSim.zip", "a" * 32, 123),
            _bundle(record_id, "Task001_BrainTumour.zip", "b" * 32, 456),
            _bundle(record_id, "README.txt", "c" * 32, 10),
        ],
    }

    page = _adapter(record_id, payload).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    record = page.records[0]
    assert record.identifiers == (Identifier("doi", "10.5281/zenodo.4003545"),)
    assert {model.name for model in record.models} == {
        "Task001_BrainTumour", "Task075_Fluo_C3DH_A549_ManAndSim"
    }
    assert all(
        model.identifiers[0].namespace == "nnunet:model"
        for model in record.models
    )
    assert all(release.metadata["nnunet_generation"] == "v1" for release in record.releases)
    assert page.next_state["release_doi"] == "10.5281/zenodo.4003545"


def test_nnunet_v2_autopet_record_keeps_each_configuration_attached_to_its_file() -> None:
    record_id = "8362371"
    config_a = "3d_fullres_resenc_192x192x192_b24"
    config_b = "3d_fullres_resenc_bs80"
    payload = {
        "metadata": {
            "doi": "10.5281/zenodo.8362371",
            "title": "AutoPET II nnU-Net v2 models",
            "version": "1.0",
        },
        "files": [
            _bundle(
                record_id,
                f"nnUNetTrainer__nnUNetPlans__{config_a}_exported.zip",
                "d" * 32,
                3_800_000_000,
            ),
            _bundle(
                record_id,
                f"nnUNetTrainer__nnUNetPlans__{config_b}_exported.zip",
                "e" * 32,
                3_800_000_001,
            ),
        ],
    }

    page = _adapter(record_id, payload).fetch_page({})

    assert page.upstream_count == 2
    record = page.records[0]
    assert {model.name for model in record.models} == {config_a, config_b}
    model_local_ids = {model.name: model.local_id for model in record.models}
    releases = {release.model_local_id: release for release in record.releases}
    assert releases[model_local_ids[config_a]].metadata["archive_size"] == 3_800_000_000
    assert releases[model_local_ids[config_a]].version == "1.0"
    assert releases[model_local_ids[config_a]].metadata["nnunet_generation"] == "v2"
    assert {
        (link.model_local_ids[0], link.url)
        for link in record.links
    } == {
        (
            model_local_ids[config_a],
            f"https://zenodo.org/api/records/{record_id}/files/"
            f"nnUNetTrainer__nnUNetPlans__{config_a}_exported.zip/content",
        ),
        (
            model_local_ids[config_b],
            f"https://zenodo.org/api/records/{record_id}/files/"
            f"nnUNetTrainer__nnUNetPlans__{config_b}_exported.zip/content",
        ),
    }


def test_nnunet_bundle_adapter_rejects_other_records_and_foreign_file_urls() -> None:
    with pytest.raises(ValueError, match="only the verified"):
        NnUNetZenodoBundleRegistryAdapter(record_id="8360190")

    record_id = "8362371"
    filename = "nnUNetTrainer__nnUNetPlans__3d_fullres_resenc_bs80_exported.zip"
    payload = {
        "metadata": {"doi": "10.5281/zenodo.8362371", "title": "AutoPET", "version": "1"},
        "files": [{
            **_bundle(record_id, filename, "a" * 32, 5),
            "links": {"download": f"https://example.test/records/{record_id}/{filename}"},
        }],
    }
    with pytest.raises(ValueError, match="not in its record"):
        _adapter(record_id, payload).fetch_page({})

    payload["files"] = [{
        **_bundle(record_id, filename, "a" * 32, 5),
        "links": {
            "self": (
                f"https://zenodo.org/api/records/{record_id}/files/{filename}"
            )
        },
    }]
    with pytest.raises(ValueError, match="not in its record"):
        _adapter(record_id, payload).fetch_page({})
