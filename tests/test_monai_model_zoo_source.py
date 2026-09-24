from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier, ModelStatus
from modelome.normalize import content_hash
from modelome.sources.monai_model_zoo import MonaiModelZooSourceAdapter

NOW = datetime(2026, 9, 21, 20, 0, tzinfo=UTC)
FIRST_REVISION = "a" * 40
SECOND_REVISION = "b" * 40
REPOSITORY = "Project-MONAI/model-zoo"


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.params_calls: list[dict[str, Any] | None] = []

    def get(
        self,
        url: str,
        *,
        params: Any = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(headers or {})))
        self.params_calls.append(dict(params) if params is not None else None)
        return self.responses.pop(0)


def response(
    body: str | dict[str, Any] | list[Any], *, headers: dict[str, str] | None = None
) -> HttpResponse:
    payload = body if isinstance(body, str) else json.dumps(body)
    return HttpResponse(
        status=200,
        headers=headers or {},
        body=payload.encode(),
        url="https://example.test",
    )


def archive_asset(key: str, *, asset_id: int = 100) -> dict[str, Any]:
    filename = f"{key}.zip"
    return {
        "id": asset_id,
        "name": filename,
        "digest": "sha256:" + "c" * 64,
        "updated_at": "2025-01-02T03:04:05Z",
        "browser_download_url": (
            f"https://github.com/{REPOSITORY}/releases/download/"
            f"hosting_storage_v1/{filename}"
        ),
    }


def archive_release(assets_count: int = 0) -> dict[str, Any]:
    return {
        "id": 1234,
        "tag_name": "hosting_storage_v1",
        "assets_count": assets_count,
        "assets_url": (
            f"https://api.github.com/repos/{REPOSITORY}/releases/1234/assets"
        ),
        "assets": [],
    }


def _registry(*, include_spleen: bool = True) -> dict[str, Any]:
    registry = {
        "vista3d_v1.0.0": {
            "checksum": "2" * 40,
            "source": (
                "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/vista3d/"
                "versions/1.0.0/files/vista3d_v1.0.0.zip"
            ),
        }
    }
    if include_spleen:
        registry["spleen_ct_segmentation_v0.1.0"] = {
            "checksum": "1" * 40,
            "source": (
                "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/"
                "spleen_ct_segmentation/versions/0.1.0/files/"
                "spleen_ct_segmentation_v0.1.0.zip"
            ),
        }
    return registry


def test_monai_registry_binds_versioned_bundles_to_archives_and_checksums() -> None:
    client = QueuedClient(
        response({"sha": FIRST_REVISION}, headers={"ETag": '"commit-v1"'}),
        response(archive_release(1)),
        response([archive_asset("spleen_ct_segmentation_v0.1.0")]),
        response(_registry()),
    )
    adapter = MonaiModelZooSourceAdapter(
        name="monai-example",
        repository=REPOSITORY,
        client=client,
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["completed_revision"] == FIRST_REVISION
    assert page.next_state["record_count"] == 2
    assert page.next_state["archive_asset_count"] == 1
    assert set(page.next_state["known_records"]) == {
        "spleen_ct_segmentation_v0.1.0",
        "vista3d_v1.0.0",
    }

    spleen = page.records[0]
    assert spleen.source_record_id == "bundle:spleen_ct_segmentation_v0.1.0"
    assert spleen.models[0].name == "spleen_ct_segmentation"
    assert spleen.models[0].status is ModelStatus.RELEASED
    assert spleen.models[0].identifiers == (
        Identifier("monai:model", "spleen_ct_segmentation"),
    )
    assert spleen.releases[0].version == "0.1.0"
    assert spleen.releases[0].revision == FIRST_REVISION
    assert spleen.releases[0].identifiers == (
        Identifier("monai:bundle", "spleen_ct_segmentation_v0.1.0"),
        Identifier(
            "monai:bundle-archive",
            (
                "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/"
                "spleen_ct_segmentation/versions/0.1.0/files/"
                "spleen_ct_segmentation_v0.1.0.zip"
            ),
        ),
    )
    assert spleen.releases[0].metadata["archive_sha1"] == "1" * 40
    assert spleen.releases[0].metadata["github_archive_asset_id"] == 100
    assert spleen.releases[0].metadata["github_archive_asset_digest"] == (
        "sha256:" + "c" * 64
    )
    assert spleen.releases[0].metadata["github_archive_asset_updated_at"] == (
        "2025-01-02T03:04:05Z"
    )
    assert {
        (link.url, link.relation, link.crawl) for link in spleen.links
    } >= {
        (
            "https://api.ngc.nvidia.com/v2/models/nvidia/monaihosting/"
            "spleen_ct_segmentation/versions/0.1.0/files/"
            "spleen_ct_segmentation_v0.1.0.zip",
            "weights",
            False,
        ),
        ("https://github.com/Project-MONAI/model-zoo", "source_repository", True),
        (
            "https://github.com/Project-MONAI/model-zoo/releases/download/"
            "hosting_storage_v1/spleen_ct_segmentation_v0.1.0.zip",
            "github_archive",
            False,
        ),
    }


def test_monai_registry_skips_an_unchanged_public_revision() -> None:
    archive = archive_release()

    client = QueuedClient(
        response({"sha": FIRST_REVISION}, headers={"ETag": '"commit-v1"'}),
        response(archive),
    )
    adapter = MonaiModelZooSourceAdapter(
        name="monai-example",
        repository=REPOSITORY,
        client=client,
        clock=lambda: NOW,
    )

    page = adapter.fetch_page(
        {
            "completed_revision": FIRST_REVISION,
            "archive_assets_digest": content_hash({"release": archive, "assets": []}),
            "record_count": 2,
        }
    )

    assert page.records == ()
    assert page.complete is True
    assert page.upstream_count == 2
    assert page.next_state["checked_at"] == "2026-09-21T20:00:00Z"
    assert page.next_state["commit_etag"] == '"commit-v1"'
    assert client.calls == [
        (
            f"https://api.github.com/repos/{REPOSITORY}/commits/dev",
            {"Accept": "application/vnd.github+json"},
        ),
        (
            f"https://api.github.com/repos/{REPOSITORY}/releases/tags/hosting_storage_v1",
            {"Accept": "application/vnd.github+json"},
        ),
    ]


def test_monai_registry_tombstones_a_bundle_removed_from_a_new_revision() -> None:
    client = QueuedClient(
        response({"sha": FIRST_REVISION}),
        response(archive_release()),
        response(_registry()),
        response({"sha": SECOND_REVISION}),
        response(archive_release()),
        response(_registry(include_spleen=False)),
    )
    adapter = MonaiModelZooSourceAdapter(
        name="monai-example",
        repository=REPOSITORY,
        client=client,
        clock=lambda: NOW,
    )

    initial = adapter.fetch_page({})
    page = adapter.fetch_page(initial.next_state)

    assert page.upstream_count == 1
    assert [record.source_record_id for record in page.records] == [
        "bundle:vista3d_v1.0.0",
        "bundle:spleen_ct_segmentation_v0.1.0",
    ]
    tombstone = page.records[-1]
    assert tombstone.deleted is True
    assert tombstone.models == ()
    assert tombstone.releases == ()
    assert tombstone.raw["removal_observed"] is True
    assert set(page.next_state["known_records"]) == {"vista3d_v1.0.0"}


def test_monai_registry_rejects_a_non_versioned_bundle_key() -> None:
    client = QueuedClient(
        response({"sha": FIRST_REVISION}),
        response(archive_release()),
        response(
            {
                "spleen_ct_segmentation": {
                    "checksum": "1" * 40,
                    "source": "https://example.test/spleen.zip",
                }
            }
        ),
    )
    adapter = MonaiModelZooSourceAdapter(
        name="monai-example", repository=REPOSITORY, client=client
    )

    with pytest.raises(ValueError, match="lacks a version suffix"):
        adapter.fetch_page({})


def test_monai_registry_keeps_a_published_bundle_when_checksum_is_omitted() -> None:
    client = QueuedClient(
        response({"sha": FIRST_REVISION}),
        response(archive_release()),
        response(
            {
                "brain_image_synthesis_latent_diffusion_model_v1.0.1": {
                    "source": "https://example.test/latent-diffusion.zip"
                }
            }
        ),
    )
    adapter = MonaiModelZooSourceAdapter(
        name="monai-example", repository=REPOSITORY, client=client
    )

    page = adapter.fetch_page({})

    assert page.records[0].models[0].name == "brain_image_synthesis_latent_diffusion_model"
    assert page.records[0].releases[0].metadata["archive_sha1"] is None
    assert page.records[0].raw["checksum"] is None


def test_monai_registry_includes_versioned_release_assets_missing_from_current_index() -> None:
    key = "brats_mri_axial_slices_generative_diffusion_v1.0.0"
    asset = archive_asset(key, asset_id=321)
    client = QueuedClient(
        response({"sha": FIRST_REVISION}),
        response(archive_release(1)),
        response([asset]),
        response({}),
    )
    adapter = MonaiModelZooSourceAdapter(
        name="monai-example", repository=REPOSITORY, client=client
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.source_record_id == f"bundle:{key}"
    assert record.models[0].name == "brats_mri_axial_slices_generative_diffusion"
    assert record.releases[0].version == "1.0.0"
    assert record.releases[0].metadata["archive_url"] == asset["browser_download_url"]
    assert record.releases[0].metadata["archive_sha1"] is None
    assert record.raw["archive_asset_id"] == 321
    assert record.raw["archive_asset_digest"] == asset["digest"]
    assert asset["browser_download_url"] in {link.url for link in record.links}
    assert page.next_state["known_records"][key]["archive_asset_id"] == 321


def test_monai_registry_archive_asset_removal_creates_tombstone() -> None:
    key = "brats_mri_axial_slices_generative_diffusion_v1.0.0"
    client = QueuedClient(
        response({"sha": FIRST_REVISION}),
        response(archive_release(1)),
        response([archive_asset(key)]),
        response({}),
        response({"sha": SECOND_REVISION}),
        response(archive_release()),
        response({}),
    )
    adapter = MonaiModelZooSourceAdapter(
        name="monai-example", repository=REPOSITORY, client=client
    )

    initial = adapter.fetch_page({})
    page = adapter.fetch_page(initial.next_state)

    assert page.upstream_count == 0
    assert len(page.records) == 1
    assert page.records[0].source_record_id == f"bundle:{key}"
    assert page.records[0].deleted is True


def test_monai_archive_assets_are_fetched_across_all_pages() -> None:
    assets = [
        archive_asset(f"historical_bundle_{index:03}_v1.0.0", asset_id=index + 1)
        for index in range(310)
    ]
    registry = {"current_bundle_v1.0.0": {"source": "https://example.test/current.zip"}}
    client = QueuedClient(
        response({"sha": FIRST_REVISION}),
        response(archive_release(len(assets))),
        response(assets[:100]),
        response(assets[100:200]),
        response(assets[200:300]),
        response(assets[300:]),
        response(registry),
    )
    adapter = MonaiModelZooSourceAdapter(
        name="monai-example", repository=REPOSITORY, client=client
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 311
    assert page.next_state["archive_asset_count"] == 310
    assert "historical_bundle_309_v1.0.0" in page.next_state["known_records"]
    asset_page_calls = [
        params for (url, _headers), params in zip(client.calls, client.params_calls, strict=True)
        if url.endswith("/releases/1234/assets")
    ]
    assert asset_page_calls == [
        {"per_page": 100, "page": 1},
        {"per_page": 100, "page": 2},
        {"per_page": 100, "page": 3},
        {"per_page": 100, "page": 4},
    ]
