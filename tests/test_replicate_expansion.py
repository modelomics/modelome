from __future__ import annotations

import json
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.replicate import (
    ReplicateModelsSourceAdapter,
)


class _QueuedClient:
    def __init__(self, *payloads: dict[str, Any]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(headers or {})))
        payload = self.payloads.pop(0)
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
            url=url,
        )



def test_replicate_model_listing_drains_version_cursors_before_next_catalog_page() -> None:
    catalog_next = "https://api.replicate.com/v1/models?cursor=models-page-two"
    model_versions_next = (
        "https://api.replicate.com/v1/models/lab/vision/versions?cursor=versions-page-two"
    )
    client = _QueuedClient(
        {
            "results": [
                {
                    "owner": "lab",
                    "name": "vision",
                    "visibility": "public",
                    "latest_version": {"id": "latest-id"},
                }
            ],
            "next": catalog_next,
        },
        {
            "results": [{"id": "historical-id", "created_at": "2023-04-05T00:00:00Z"}],
            "next": model_versions_next,
        },
        {"results": [{"id": "latest-id"}], "next": None},
        {
            "results": [
                {
                    "owner": "lab",
                    "name": "audio",
                    "visibility": "public",
                }
            ],
            "next": None,
        },
        {"results": [{"id": "audio-version"}], "next": None},
    )
    adapter = ReplicateModelsSourceAdapter(token="replicate-secret", client=client)

    model_page = adapter.fetch_page({})
    historical_page_one = adapter.fetch_page(model_page.next_state)
    historical_page_two = adapter.fetch_page(historical_page_one.next_state)
    next_catalog_page = adapter.fetch_page(historical_page_two.next_state)
    final_versions_page = adapter.fetch_page(next_catalog_page.next_state)

    assert not model_page.complete
    assert model_page.records[0].releases[0].local_id == "lab/vision#model#version:latest-id"
    assert historical_page_one.records[0].source_record_id == "lab/vision@historical-id"
    assert historical_page_one.next_state["version_queue"][0]["next_url"] == model_versions_next
    assert (
        historical_page_two.records[0].releases[0].local_id
        == model_page.records[0].releases[0].local_id
    )
    assert historical_page_two.next_state["next_url"] == catalog_next
    assert next_catalog_page.records[0].source_record_id == "lab/audio"
    assert next_catalog_page.next_state["catalog_complete"] is True
    assert final_versions_page.records[0].source_record_id == "lab/audio@audio-version"
    assert final_versions_page.complete
    assert final_versions_page.next_state == {}
    assert [call[0] for call in client.calls] == [
        "https://api.replicate.com/v1/models",
        "https://api.replicate.com/v1/models/lab/vision/versions",
        model_versions_next,
        catalog_next,
        "https://api.replicate.com/v1/models/lab/audio/versions",
    ]
