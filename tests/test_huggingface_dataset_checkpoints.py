from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, ModelStatus
from modelome.sources.catalog import create_source, load_source_configs
from modelome.sources.huggingface import HuggingFaceDatasetCheckpointSourceAdapter


class _Client:
    def __init__(
        self,
        listing: object,
        *,
        listing_headers: dict[str, str] | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        self.listing = listing
        self.listing_headers = listing_headers or {}
        self.details = details or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        if params:
            payload = self.listing
            response_headers = self.listing_headers
        else:
            payload = self.details[url]
            response_headers = {}
        return HttpResponse(
            status=200,
            headers=response_headers,
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_dataset_listing_is_unfiltered_and_emits_only_exact_weight_candidates() -> None:
    revision = "a" * 40
    client = _Client(
        [
            {
                "id": "data-lab/checkpoint-pack",
                "sha": revision,
                "createdAt": "2026-01-01T00:00:00Z",
                "lastModified": "2026-02-01T00:00:00Z",
                "gated": True,
            },
            {
                "id": "data-lab/text-only",
                "sha": revision,
            },
        ],
        listing_headers={"x-total-count": "2"},
        details={
            "https://huggingface.co/api/datasets/data-lab/checkpoint-pack": {
                "id": "data-lab/checkpoint-pack",
                "sha": revision,
                "gated": True,
                "siblings": [
                    {"rfilename": "model.safetensors"},
                    {"rfilename": "metadata.json"},
                    {"rfilename": "nested/model.gguf"},
                    {"rfilename": "unsafe/../pytorch_model.bin"},
                ],
            },
            "https://huggingface.co/api/datasets/data-lab/text-only": {
                "id": "data-lab/text-only",
                "sha": revision,
                "siblings": [{"rfilename": "README.md"}],
            },
        },
    )
    adapter = HuggingFaceDatasetCheckpointSourceAdapter(client=client)

    queued = adapter.fetch_page({})
    page = adapter.fetch_page(queued.next_state)
    drained = adapter.fetch_page(page.next_state)

    assert client.calls[0] == (
        "https://huggingface.co/api/datasets",
        {
            "limit": 10,
            "full": "true",
            "sort": "createdAt",
            "direction": 1,
        },
    )
    assert client.calls[1][0] == (
        "https://huggingface.co/api/datasets/data-lab/checkpoint-pack"
    )
    assert queued.records == ()
    assert page.complete is False
    assert len(page.records) == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.source_record_id == f"data-lab/checkpoint-pack@{revision}"
    assert record.models[0].status is ModelStatus.CANDIDATE
    assert record.models[0].identifiers[0].namespace == "huggingface:dataset-checkpoint-candidate"
    assert record.raw["gated"] is True
    assert record.raw["weight_files"] == ["model.safetensors", "nested/model.gguf"]
    assert record.releases[0].metadata["repo_type"] == "dataset"
    assert all(link.crawl is False for link in record.links)
    assert page.upstream_count == 2
    assert drained.records == ()
    assert drained.complete is True


def test_dataset_listing_follows_provider_cursor_and_caps_files_with_incomplete_flag() -> None:
    revision = "b" * 40
    next_url = "https://huggingface.co/api/datasets?cursor=opaque"
    client = _Client(
        [
            {
                "id": "dataset-owner/large-checkpoint-archive",
                "sha": revision,
            }
        ],
        listing_headers={"Link": f'<{next_url}>; rel="next"'},
        details={
            "https://huggingface.co/api/datasets/dataset-owner/large-checkpoint-archive": {
                "id": "dataset-owner/large-checkpoint-archive",
                "sha": revision,
                "siblings": [
                    {"rfilename": "a.safetensors"},
                    {"rfilename": "b.safetensors"},
                ],
            }
        },
    )
    adapter = HuggingFaceDatasetCheckpointSourceAdapter(
        client=client,
        max_checkpoint_files=1,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    queued = adapter.fetch_page({})
    page = adapter.fetch_page(queued.next_state)

    assert page.complete is False
    assert page.next_state["next_url"] == next_url
    assert page.records[0].raw["weight_files"] == ["a.safetensors"]
    assert page.records[0].raw["weight_files_complete"] is False


def test_dataset_detail_sha_drift_is_reported_and_does_not_admit_files() -> None:
    listed_sha = "c" * 40
    changed_sha = "d" * 40
    repo_id = "dataset-owner/changed-during-scan"
    client = _Client(
        [{"id": repo_id, "sha": listed_sha}],
        details={
            f"https://huggingface.co/api/datasets/{repo_id}": {
                "id": repo_id,
                "sha": changed_sha,
                "siblings": [{"rfilename": "weights.safetensors"}],
            }
        },
    )
    adapter = HuggingFaceDatasetCheckpointSourceAdapter(client=client)

    queued = adapter.fetch_page({})
    page = adapter.fetch_page(queued.next_state)

    assert page.records == ()
    assert page.complete is True
    assert page.advance_on_source_issues is True
    assert page.issues[0].summary == {
        "listed_sha": listed_sha,
        "detail_sha": changed_sha,
        "file_inventory_status": "revision_drift_incomplete",
    }


def test_live_acronym_dataset_detail_contains_no_checkpoint_candidate() -> None:
    repo_id = "amirveyseh/acronym_identification"
    revision = "15ef643450d589d5883e289ffadeb03563e80a9e"
    client = _Client(
        [{"id": repo_id, "sha": revision}],
        details={
            f"https://huggingface.co/api/datasets/{repo_id}": {
                "id": repo_id,
                "sha": revision,
                "siblings": [
                    {"rfilename": ".gitattributes"},
                    {"rfilename": "README.md"},
                    {"rfilename": "data/test-00000-of-00001.parquet"},
                    {"rfilename": "data/train-00000-of-00001.parquet"},
                    {"rfilename": "data/validation-00000-of-00001.parquet"},
                ],
            }
        },
    )
    adapter = HuggingFaceDatasetCheckpointSourceAdapter(client=client)

    queued = adapter.fetch_page({})
    checked = adapter.fetch_page(queued.next_state)

    assert client.calls[0][0] == "https://huggingface.co/api/datasets"
    assert client.calls[1][0] == f"https://huggingface.co/api/datasets/{repo_id}"
    assert queued.records == ()
    assert checked.records == ()
    assert checked.issues == ()
    assert checked.complete is True


def test_dataset_checkpoint_proposal_loads_through_catalog():
    config = load_source_configs(
        "config/proposals/huggingface_dataset_checkpoints.toml"
    )[0]

    source = create_source(config)

    assert isinstance(source, HuggingFaceDatasetCheckpointSourceAdapter)
    assert source.url == "https://huggingface.co/api/datasets"
    assert source.page_size == 10
    assert source.created_at_sweep_interval_days == 30
