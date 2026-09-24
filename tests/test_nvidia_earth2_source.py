from __future__ import annotations

import json
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.nvidia_earth2 import NvidiaEarth2SourceAdapter


class _RouteClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        params = dict(params or {})
        self.calls.append((url, params))
        if url == "https://huggingface.co/api/collections/nvidia/earth-2":
            payload = {
                "items": [
                    {"type": "model", "id": "nvidia/earth2-fixture"},
                    {"type": "paper", "id": "2601.00001"},
                    {"type": "model", "id": "nvidia/earth2-second"},
                ]
            }
        elif url == "https://huggingface.co/api/models/nvidia/earth2-fixture":
            payload = {
                "id": "nvidia/earth2-fixture",
                "sha": "immutable-sha",
                "pipeline_tag": "image-to-image",
                "siblings": [
                    {"rfilename": "checkpoint/model.pt"},
                    {"rfilename": "README.md"},
                ],
            }
        elif url == "https://huggingface.co/api/models/nvidia/earth2-second":
            payload = {
                "id": "nvidia/earth2-second",
                "sha": "second-sha",
                "siblings": [{"rfilename": "model.safetensors"}],
            }
        else:
            raise AssertionError(f"unexpected request: {url} {params}")
        return HttpResponse(
            status=200,
            headers={},
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_projects_nvidia_curated_earth2_models_with_exact_checkpoint_links() -> None:
    client = _RouteClient()
    adapter = NvidiaEarth2SourceAdapter(page_size=1, client=client)

    first = adapter.fetch_page({})
    assert [record.source_record_id for record in first.records] == [
        "nvidia/earth2-fixture"
    ]
    assert first.complete is False
    assert first.next_state["collection_model_ids"] == [
        "nvidia/earth2-fixture",
        "nvidia/earth2-second",
    ]

    assert first.records[0].releases[0].revision == "immutable-sha"
    weight_links = [
        (link.url, link.relation, link.crawl)
        for link in first.records[0].links
        if link.relation == "weights"
    ]
    assert weight_links == [
        (
            "https://huggingface.co/nvidia/earth2-fixture/resolve/immutable-sha/checkpoint/model.pt",
            "weights",
            False,
        )
    ]
    assert (
        "https://huggingface.co/collections/nvidia/earth-2",
        "curated_collection",
        False,
    ) in {
        (link.url, link.relation, link.crawl) for link in first.records[0].links
    }
    assert first.records[0].raw["curated_collection"] == (
        "https://huggingface.co/collections/nvidia/earth-2"
    )

    second = adapter.fetch_page(first.next_state)
    assert [record.source_record_id for record in second.records] == [
        "nvidia/earth2-second"
    ]
    assert second.records[0].releases[0].metadata["weight_files"] == [
        "model.safetensors"
    ]
    assert second.complete is True
    assert [call[0] for call in client.calls].count(
        "https://huggingface.co/api/collections/nvidia/earth-2"
    ) == 1
    assert all(
        params == {"full": "true", "cardData": "true", "config": "true"}
        for url, params in client.calls
        if url.startswith("https://huggingface.co/api/models/")
    )
