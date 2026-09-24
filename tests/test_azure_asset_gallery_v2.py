from __future__ import annotations

import json
from typing import Any

import pytest

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
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []

    def post(self, url: str, *, data: bytes, headers=None) -> HttpResponse:
        self.calls.append((url, data, dict(headers or {})))
        payload = self.responses[len(self.calls) - 1]
        return HttpResponse(
            status=200,
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
