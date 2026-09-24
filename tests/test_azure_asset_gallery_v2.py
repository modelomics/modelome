from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.pipeline import SyncEngine
from modelome.sources.azure_asset_gallery_v2 import AzureAssetGalleryV2Adapter
from modelome.storage import Database


def _summary(name: str, version: str = "1") -> dict[str, Any]:
    return {
        "assetId": (
            "azureml://registries/HuggingFace/models/" f"{name}/versions/{version}"
        ),
        "name": name,
        "displayName": name,
        "version": version,
        "registryName": "HuggingFace",
        "publisher": "Hugging Face",
        "labels": ["latest", "default"],
        "createdTime": "2024-08-02T08:21:41.4171379+00:00",
        "inferenceTasks": ["text-generation"],
    }


class FixtureClient:
    def __init__(
        self,
        responses: list[dict[str, Any]],
        details: dict[str, dict[str, Any]] | None = None,
        detail_statuses: dict[str, int] | None = None,
    ) -> None:
        self.responses = responses
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []
        self.details = details or {}
        self.detail_statuses = detail_statuses or {}
        self.detail_calls: list[str] = []

    def post(self, url: str, *, data: bytes, headers=None) -> HttpResponse:
        self.calls.append((url, data, dict(headers or {})))
        payload = self.responses[len(self.calls) - 1]
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
            url=url,
        )

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.detail_calls.append(url)
        payload = self.details[url]
        return HttpResponse(
            status=self.detail_statuses.get(url, 200),
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_asset_gallery_v2_uses_anonymous_cursor_pagination_and_exact_versions() -> None:
    client = FixtureClient(
        [
            {
                "totalCount": 3,
                "continuationToken": "opaque-next-page-token",
                "summaries": [_summary("01-ai-yi-1.5-34b"), _summary("01-ai-yi-1.5-34b-chat")],
            },
            {
                "totalCount": 3,
                "continuationToken": None,
                "summaries": [_summary("01-ai-yi-1.5-34b-chat-16k", "2")],
            },
        ]
    )
    adapter = AzureAssetGalleryV2Adapter(page_size=2, client=client)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert not first.complete
    assert second.complete
    assert [record.raw["assetId"] for record in (*first.records, *second.records)] == [
        "azureml://registries/HuggingFace/models/01-ai-yi-1.5-34b/versions/1",
        "azureml://registries/HuggingFace/models/01-ai-yi-1.5-34b-chat/versions/1",
        "azureml://registries/HuggingFace/models/01-ai-yi-1.5-34b-chat-16k/versions/2",
    ]
    versioned_model = second.records[0]
    assert versioned_model.models[0].identifiers[0].value == (
        "HuggingFace/01-ai-yi-1.5-34b-chat-16k"
    )
    assert versioned_model.releases[0].version == "2"
    assert versioned_model.releases[0].identifiers[0].value == versioned_model.raw["assetId"]
    assert json.loads(client.calls[0][1]) == {
        "order": [{"field": "name", "direction": "asc"}],
        "pageSize": 2,
        "includeTotalResultCount": True,
    }
    assert "continuationToken" not in json.loads(client.calls[0][1])
    assert json.loads(client.calls[1][1])["continuationToken"] == "opaque-next-page-token"
    assert all(
        "Authorization" not in headers
        for _, _, headers in client.calls
    )


def test_exact_original_model_card_detail_adds_hf_identity_and_model_card_link() -> None:
    summary = _summary("01-ai-yi-1.5-34b")
    detail_url = (
        "https://api.catalog.azureml.ms/asset-gallery/v1.0/HuggingFace/models/"
        "01-ai-yi-1.5-34b/version/1"
    )
    client = FixtureClient(
        [{"totalCount": 1, "continuationToken": None, "summaries": [summary]}],
        {
            detail_url: {
                "AssetId": (
                    "azureml://registries/HuggingFace/models/"
                    "01-ai-yi-1.5-34b/versions/1"
                ),
                "Name": "01-ai-yi-1.5-34b",
                "Version": "1",
                "RegistryName": "HuggingFace",
                "Description": (
                    "Read the original here: "
                    "[Original Model Card](https://huggingface.co/01-ai/Yi-1.5-34B)"
                ),
            }
        },
    )
    page = AzureAssetGalleryV2Adapter(
        page_size=1,
        client=client,
        max_hf_origin_details_per_page=1,
    ).fetch_page({})

    record = page.records[0]
    assert client.detail_calls == [detail_url]
    assert any(
        identifier.namespace == "huggingface:model"
        and identifier.value == "01-ai/Yi-1.5-34B"
        for identifier in record.models[0].identifiers
    )
    assert any(
        link.url == "https://huggingface.co/01-ai/Yi-1.5-34B"
        and link.relation == "model_card"
        and link.locator == "$.description.Original Model Card"
        for link in record.links
    )
    hf_seed = {
        "source": "huggingface",
        "source_record_id": "01-ai/Yi-1.5-34B",
        "canonical_url": "https://huggingface.co/01-ai/Yi-1.5-34B",
        "title": "Yi-1.5-34B",
        "kind": "model_card",
        "models": [
            {
                "local_id": "model",
                "name": "Yi-1.5-34B",
                "identifiers": [
                    {"namespace": "huggingface:model", "value": "01-ai/Yi-1.5-34B"}
                ],
            }
        ],
    }
    joined = build_entries(
        [
            source_record_to_entry_seed(record, source="azure-asset-gallery-v2"),
            hf_seed,
        ]
    )
    assert len(joined.entries) == 1
    assert {member.source for member in joined.entries[0].members} == {
        "azure-asset-gallery-v2",
        "huggingface",
    }


def test_detail_link_requires_exact_label_and_direct_hf_model_card() -> None:
    summary = _summary("example")
    detail_url = (
        "https://api.catalog.azureml.ms/asset-gallery/v1.0/HuggingFace/models/"
        "example/version/1"
    )
    for description in (
        "[Model Card](https://huggingface.co/org/model)",
        "[Original Model Card](https://huggingface.co/org/model/tree/main)",
        "[Original Model Card](https://huggingface.co/datasets/org/model)",
    ):
        client = FixtureClient(
            [{"totalCount": 1, "continuationToken": None, "summaries": [summary]}],
            {
                detail_url: {
                    "name": "example",
                    "version": "1",
                    "registryName": "HuggingFace",
                    "description": description,
                }
            },
        )
        page = AzureAssetGalleryV2Adapter(
            page_size=1,
            client=client,
            max_hf_origin_details_per_page=1,
        ).fetch_page({})
        assert not any(
            identifier.namespace == "huggingface:model"
            for identifier in page.records[0].models[0].identifiers
        )


def test_detail_enrichment_is_page_bounded_and_cursor_resume_keeps_rows() -> None:
    first_url = (
        "https://api.catalog.azureml.ms/asset-gallery/v1.0/HuggingFace/models/"
        "first/version/1"
    )
    second_url = (
        "https://api.catalog.azureml.ms/asset-gallery/v1.0/HuggingFace/models/"
        "second/version/1"
    )
    client = FixtureClient(
        [
            {"totalCount": 2, "continuationToken": "next", "summaries": [_summary("first")]},
            {"totalCount": 2, "continuationToken": None, "summaries": [_summary("second")]},
        ],
        {
            first_url: {
                "name": "first",
                "version": "1",
                "registryName": "HuggingFace",
                "description": "[Original Model Card](https://huggingface.co/org/first)",
            },
            second_url: {
                "name": "second",
                "version": "1",
                "registryName": "HuggingFace",
                "description": "[Original Model Card](https://huggingface.co/org/second)",
            },
        },
    )
    adapter = AzureAssetGalleryV2Adapter(
        page_size=1,
        client=client,
        max_hf_origin_details_per_page=1,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert len(first.records) == len(second.records) == 1
    assert [
        record.models[0].identifiers[-1].value
        for record in (first.records[0], second.records[0])
    ] == [
        "org/first",
        "org/second",
    ]
    assert client.detail_calls == [first_url, second_url]


def test_detail_failure_does_not_advance_the_cursor_checkpoint() -> None:
    summary = _summary("example")
    detail_url = (
        "https://api.catalog.azureml.ms/asset-gallery/v1.0/HuggingFace/models/"
        "example/version/1"
    )

    class FailOnceClient(FixtureClient):
        def get(self, url: str, *, headers=None) -> HttpResponse:
            self.detail_calls.append(url)
            if len(self.detail_calls) == 1:
                return HttpResponse(503, {}, b"temporary failure", url)
            return HttpResponse(
                200,
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "name": "example",
                        "version": "1",
                        "registryName": "HuggingFace",
                        "description": "[Original Model Card](https://huggingface.co/org/model)",
                    }
                ).encode(),
                url,
            )

    client = FailOnceClient(
        [
            {"totalCount": 1, "continuationToken": None, "summaries": [summary]},
            {"totalCount": 1, "continuationToken": None, "summaries": [summary]},
        ]
    )
    adapter = AzureAssetGalleryV2Adapter(
        page_size=1,
        client=client,
        max_hf_origin_details_per_page=1,
    )
    checkpoint: dict[str, Any] = {}

    with pytest.raises(ValueError, match="detail returned HTTP 503"):
        adapter.fetch_page(checkpoint)
    assert checkpoint == {}
    retry = adapter.fetch_page(checkpoint)

    assert retry.complete
    assert retry.records[0].models[0].identifiers[-1].value == "org/model"
    assert len(client.calls) == 2
    assert client.detail_calls == [detail_url, detail_url]


@pytest.mark.parametrize("status", [404, 410])
def test_missing_detail_keeps_listing_record_and_reports_issue(status: int) -> None:
    summary = _summary("deleted-detail")
    detail_url = (
        "https://api.catalog.azureml.ms/asset-gallery/v1.0/HuggingFace/models/"
        "deleted-detail/version/1"
    )
    client = FixtureClient(
        [{"totalCount": 1, "continuationToken": None, "summaries": [summary]}],
        {detail_url: {}},
        {detail_url: status},
    )

    page = AzureAssetGalleryV2Adapter(
        page_size=1,
        client=client,
        max_hf_origin_details_per_page=1,
    ).fetch_page({})

    assert page.complete
    assert len(page.records) == 1
    assert not any(
        identifier.namespace == "huggingface:model"
        for identifier in page.records[0].models[0].identifiers
    )
    assert page.advance_on_source_issues
    assert len(page.issues) == 1
    assert page.issues[0].source_record_id == (
        "azureml:asset-version:HuggingFace/deleted-detail@1"
    )
    assert page.issues[0].summary["detail_status"] == status


def test_detail_budget_reports_unchecked_hf_summaries_and_advances_page() -> None:
    first_url = (
        "https://api.catalog.azureml.ms/asset-gallery/v1.0/HuggingFace/models/"
        "first/version/1"
    )
    client = FixtureClient(
        [
            {
                "totalCount": 2,
                "continuationToken": "next",
                "summaries": [_summary("first"), _summary("second")],
            }
        ],
        {
            first_url: {
                "name": "first",
                "version": "1",
                "registryName": "HuggingFace",
                "description": "[Original Model Card](https://huggingface.co/org/first)",
            }
        },
    )

    page = AzureAssetGalleryV2Adapter(
        page_size=2,
        client=client,
        max_hf_origin_details_per_page=1,
    ).fetch_page({})

    assert not page.complete
    assert page.next_state["continuation_token"] == "next"
    assert [issue.source_record_id for issue in page.issues] == [
        "azureml:asset-version:HuggingFace/second@1"
    ]
    assert page.advance_on_source_issues
    assert client.detail_calls == [first_url]


def test_detail_budget_can_cover_larger_configured_pages() -> None:
    adapter = AzureAssetGalleryV2Adapter(
        page_size=1000,
        max_hf_origin_details_per_page=1000,
        client=FixtureClient([]),
    )

    assert adapter.max_hf_origin_details_per_page == adapter.page_size


def test_asset_gallery_restarts_when_total_changes_and_rejects_bad_identity() -> None:
    changed_total = FixtureClient(
        [
            {"totalCount": 2, "continuationToken": "next", "summaries": [_summary("a")]},
            {"totalCount": 3, "continuationToken": None, "summaries": [_summary("b")]},
        ]
    )
    adapter = AzureAssetGalleryV2Adapter(page_size=1, client=changed_total)
    first = adapter.fetch_page({})
    restarted = adapter.fetch_page(first.next_state)
    assert restarted.records == ()
    assert not restarted.complete
    assert restarted.next_state == {
        "page": 1,
        "total_count": 3,
        "records_seen": 0,
        "seen_cursor_hashes": [],
        "seen_asset_hashes": [],
        "scan_restarts": 1,
    }

    invalid = _summary("known")
    invalid["assetId"] = "azureml://registries/HuggingFace/models/other/versions/1"
    adapter = AzureAssetGalleryV2Adapter(
        page_size=1,
        client=FixtureClient(
            [{"totalCount": 1, "continuationToken": None, "summaries": [invalid]}]
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        adapter.fetch_page({})


def test_asset_gallery_rejects_cursor_cycle_and_count_drift() -> None:
    cyclic = FixtureClient(
        [
            {"totalCount": 3, "continuationToken": "first", "summaries": [_summary("a")]},
            {"totalCount": 3, "continuationToken": "second", "summaries": [_summary("b")]},
            {"totalCount": 3, "continuationToken": "first", "summaries": [_summary("c")]},
        ]
    )
    adapter = AzureAssetGalleryV2Adapter(page_size=1, client=cyclic)
    state = adapter.fetch_page({}).next_state
    state = adapter.fetch_page(state).next_state
    with pytest.raises(ValueError, match="repeated or cycled"):
        adapter.fetch_page(state)

    too_many = FixtureClient(
        [
            {
                "totalCount": 2,
                "continuationToken": "next",
                "summaries": [_summary("a"), _summary("b")],
            },
            {"totalCount": 2, "continuationToken": None, "summaries": [_summary("c")]},
        ]
    )
    adapter = AzureAssetGalleryV2Adapter(page_size=2, client=too_many)
    page = adapter.fetch_page(adapter.fetch_page({}).next_state)
    assert page.records == ()
    assert page.next_state["page"] == 1
    assert page.next_state["total_count"] == 2
    assert page.next_state["scan_restarts"] == 1


def test_asset_gallery_rejects_incomplete_final_count_and_duplicate_assets() -> None:
    incomplete = FixtureClient(
        [{"totalCount": 2, "continuationToken": None, "summaries": [_summary("a")]}]
    )
    page = AzureAssetGalleryV2Adapter(page_size=2, client=incomplete).fetch_page({})
    assert page.records == ()
    assert page.next_state["scan_restarts"] == 1

    duplicate = FixtureClient(
        [
            {"totalCount": 2, "continuationToken": "next", "summaries": [_summary("a")]},
            {"totalCount": 2, "continuationToken": None, "summaries": [_summary("a")]},
        ]
    )
    adapter = AzureAssetGalleryV2Adapter(page_size=1, client=duplicate)
    page = adapter.fetch_page(adapter.fetch_page({}).next_state)
    assert page.records == ()
    assert page.next_state["scan_restarts"] == 1


def test_partial_resume_does_not_tombstone_azure_catalog_assets(tmp_path) -> None:
    client = FixtureClient(
        [
            {"totalCount": 2, "continuationToken": "next", "summaries": [_summary("a")]},
            {"totalCount": 2, "continuationToken": None, "summaries": [_summary("b")]},
        ]
    )
    adapter = AzureAssetGalleryV2Adapter(
        name="azure-live-catalog", page_size=1, client=client
    )
    database = Database(tmp_path / "store")
    database.initialize()
    engine = SyncEngine(database, {adapter.name: adapter})

    partial = engine.sync_source(adapter, max_pages=1)
    resumed = engine.sync_source(adapter, max_pages=1)

    assert partial.status == "partial"
    assert resumed.status == "complete"
    assert database.stats()["active_artifacts"] == 2
    assert all(
        row["active"] == 1 and row["tombstoned_at"] is None
        for row in database.table_rows("artifacts")
    )
