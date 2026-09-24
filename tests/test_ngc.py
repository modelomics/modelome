from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.entries import build_entries, plan_entry_seed, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.ngc import NgcModelsSourceAdapter


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(
    resources: list[Mapping[str, Any]],
    *,
    total: int,
    page: int,
    page_size: int = 2,
) -> HttpResponse:
    payload = {
        "params": {"page": page, "pageSize": page_size},
        "resultPageTotal": (total + page_size - 1) // page_size,
        "resultTotal": total,
        "results": [
            {
                "groupValue": "_scored",
                "totalCount": len(resources),
                "resources": resources,
            },
            {
                "groupValue": "MODEL",
                "totalCount": total,
                "resources": resources,
            },
        ],
    }
    return HttpResponse(
        200,
        {},
        json.dumps(payload).encode(),
        "https://api.ngc.nvidia.com/v2/search/catalog/resources/MODEL",
    )


_ACTION = {
    "resourceId": "nvidia/tao/actionrecognitionnet",
    "orgName": "nvidia",
    "teamName": "tao",
    "name": "actionrecognitionnet",
    "displayName": "ActionRecognitionNet",
    "description": "TAO model; source: https://github.com/NVIDIA/tao_toolkit.",
    "dateModified": "2026-09-20T04:00:00Z",
    "labels": [
        {
            "key": "general",
            "values": ["video", "action-recognition"],
            "unresolvedValues": ["video", "action-recognition"],
        }
    ],
    "guestAccess": True,
    "isPublic": True,
    "attributes": [
        {"key": "latestVersionIdStr", "value": "1.0.0"},
        {"key": "application", "value": "TAO Toolkit"},
        {"key": "format", "value": "TAO"},
        {"key": "latestVersionSizeInBytes", "value": 3145728},
    ],
}

_NEMOTRON = {
    "resourceId": "nvidia/nemotron",
    "orgName": "nvidia",
    "teamName": "",
    "name": "nemotron",
    "displayName": "Nemotron",
    "description": "",
    "dateModified": "2026-09-20T04:00:00Z",
    "attributes": [],
}

_SEGMENT = {
    "resourceId": "nvidia/tao/segmentation",
    "orgName": "nvidia",
    "teamName": "tao",
    "name": "segmentation",
    "displayName": "Segmentation",
    "description": "",
    "dateModified": "2026-09-20T04:00:00Z",
    "attributes": [],
}


def test_ngc_paginates_the_public_model_group_without_snapshot_deletion() -> None:
    client = _QueuedClient(
        _response([_ACTION, _NEMOTRON], total=3, page=0),
        _response([_SEGMENT], total=3, page=1),
    )
    adapter = NgcModelsSourceAdapter(
        page_size=2,
        client=client,
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.upstream_count is None
    assert first.next_state["page"] == 1
    assert first.next_state["provider_total"] == 3
    assert second.complete is True
    assert second.authoritative_snapshot is False
    assert second.upstream_count == 3
    assert second.next_state == {
        "completed_at": "2026-09-21T00:00:00Z",
        "provider_total": 3,
        "page_total": 2,
    }

    queries = [json.loads(call[1]["q"]) for call in client.calls]
    assert [query["page"] for query in queries] == [0, 1]
    assert all(query["pageSize"] == 2 for query in queries)
    assert all(
        query["orderBy"]
        == [
            {"field": "nameSort", "value": "ASC"},
            {"field": "resourceId", "value": "ASC"},
        ]
        for query in queries
    )

    record = first.records[0]
    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.source_record_id == "nvidia/tao/actionrecognitionnet"
    assert record.identifiers == (
        Identifier("ngc:model", "nvidia/tao/actionrecognitionnet"),
    )
    assert record.models[0].status is ModelStatus.RELEASED
    assert record.canonical_url == (
        "https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/actionrecognitionnet"
    )
    assert record.raw["catalog_page"] == 0
    assert record.raw["provider_total"] == 3
    assert record.releases[0].identifiers == (
        Identifier(
            "ngc:model-version",
            "nvidia/tao/actionrecognitionnet:1.0.0",
        ),
    )
    assert record.releases[0].metadata == {
        "application": "TAO Toolkit",
        "format": "TAO",
        "latestVersionSizeInBytes": 3145728,
    }
    assert (
        "https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/actionrecognitionnet",
        "model_card",
    ) in {(link.url, link.relation) for link in record.links}
    assert next(link for link in record.links if link.relation == "model_card").crawl is False
    assert (
        "https://api.ngc.nvidia.com/v2/models/nvidia/tao/"
        "actionrecognitionnet/versions/1.0.0",
        "model_card_metadata",
    ) in {(link.url, link.relation) for link in record.links}
    assert next(
        link for link in record.links if link.relation == "model_card_metadata"
    ).crawl is True
    seed = source_record_to_entry_seed(record, source="ngc-models")
    plan = plan_entry_seed(seed)
    assert any(
        action["action"] == "resolve_resource"
        and action["relation"] == "model_card_metadata"
        for action in plan["actions"]
    )
    entries = build_entries((seed,)).entries
    metadata_resource = next(
        resource
        for resource in entries[0].resources
        if resource.relation == "model_card_metadata"
    )
    assert metadata_resource.category == "model"
    assert (
        "https://github.com/NVIDIA/tao_toolkit",
        "documentation_reference",
    ) in {(link.url, link.relation) for link in record.links}
    assert first.records[1].canonical_url == "https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemotron"


def test_ngc_rejects_provider_total_drift_during_a_sweep() -> None:
    client = _QueuedClient(_response([_ACTION, _NEMOTRON], total=4, page=1))
    adapter = NgcModelsSourceAdapter(page_size=2, client=client)

    with pytest.raises(ValueError, match="provider total changed from 3 to 4"):
        adapter.fetch_page({"page": 1, "provider_total": 3})


def test_ngc_keeps_a_malformed_model_row_quarantined() -> None:
    malformed = dict(_ACTION)
    malformed.pop("orgName")
    client = _QueuedClient(_response([malformed], total=1, page=0, page_size=1))
    adapter = NgcModelsSourceAdapter(page_size=1, client=client)

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.records == ()
    assert len(page.issues) == 1
    assert page.issues[0].source_record_id == "nvidia/tao/actionrecognitionnet"
    assert page.issues[0].stage == "source_normalize"


def test_ngc_rejects_a_partial_page_or_wrong_model_group_total() -> None:
    response = _response([_ACTION], total=2, page=0)
    client = _QueuedClient(response)
    adapter = NgcModelsSourceAdapter(page_size=2, client=client)

    with pytest.raises(ValueError, match="expected 2 from provider total"):
        adapter.fetch_page({})
