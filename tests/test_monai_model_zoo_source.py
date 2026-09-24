from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier, ModelStatus
from modelome.sources.monai_model_zoo import MonaiModelZooSourceAdapter

NOW = datetime(2026, 9, 21, 20, 0, tzinfo=UTC)
FIRST_REVISION = "a" * 40
SECOND_REVISION = "b" * 40
REPOSITORY = "Project-MONAI/model-zoo"


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Any = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        self.calls.append((url, dict(headers or {})))
        return self.responses.pop(0)


def response(body: str | dict[str, Any], *, headers: dict[str, str] | None = None) -> HttpResponse:
    payload = json.dumps(body) if isinstance(body, dict) else body
    return HttpResponse(
        status=200,
        headers=headers or {},
        body=payload.encode(),
        url="https://example.test",
    )


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
    }


def test_monai_registry_skips_an_unchanged_public_revision() -> None:
    client = QueuedClient(response({"sha": FIRST_REVISION}, headers={"ETag": '"commit-v1"'}))
    adapter = MonaiModelZooSourceAdapter(
        name="monai-example",
        repository=REPOSITORY,
        client=client,
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({"completed_revision": FIRST_REVISION, "record_count": 2})

    assert page.records == ()
    assert page.complete is True
    assert page.upstream_count == 2
    assert page.next_state["checked_at"] == "2026-09-21T20:00:00Z"
    assert page.next_state["commit_etag"] == '"commit-v1"'
    assert client.calls == [
        (
            f"https://api.github.com/repos/{REPOSITORY}/commits/dev",
            {"Accept": "application/vnd.github+json"},
        )
    ]


def test_monai_registry_tombstones_a_bundle_removed_from_a_new_revision() -> None:
    client = QueuedClient(
        response({"sha": FIRST_REVISION}),
        response(_registry()),
        response({"sha": SECOND_REVISION}),
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
