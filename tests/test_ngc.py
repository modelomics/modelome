from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError

import pytest

from modelome.entries import build_entries, plan_entry_seed, source_record_to_entry_seed
from modelome.http import HttpFailure, HttpResponse
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


class _HttpFailureClient:
    def __init__(self, status: int) -> None:
        self.status = status
        self.calls = 0

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls += 1
        if self.calls == 1:
            return _response([_ACTION], total=1, page=0, page_size=200)
        error = HTTPError(
            "https://api.ngc.nvidia.com/v2/models/nvidia/tao/actionrecognitionnet/versions?token=secret",
            self.status,
            "failed",
            {},
            None,
        )
        raise HttpFailure(
            "GET https://api.ngc.nvidia.com/versions?token=secret failed"
        ) from error


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


def test_ngc_opt_in_expansion_records_all_guest_visible_versions_and_checksums() -> None:
    versions_payload = {
        "model": {"latestVersionIdStr": "1.0.0"},
        "modelVersions": [
            {
                "versionId": "0.5.0",
                "createdDate": "2024-08-29T17:59:14.698Z",
                "status": "UPLOAD_COMPLETE",
                "totalFileCount": 2,
                "totalSizeInBytes": 332020614,
                "isSigned": True,
                "customMetrics": [
                    {
                        "name": "SHA256 digests",
                        "attributes": [{"key": "weights.bin", "value": "abc123"}],
                    }
                ],
            },
            {
                "versionId": "1.0.0",
                "createdDate": "2023-02-01T00:00:00Z",
                "status": "UPLOAD_COMPLETE",
                "totalFileCount": 1,
                "totalSizeInBytes": 42,
            },
        ],
        "paginationInfo": {"index": 0, "size": 100, "totalResults": 2, "totalPages": 1},
        "requestStatus": {"statusCode": "SUCCESS"},
    }
    client = _QueuedClient(
        _response([_ACTION], total=1, page=0),
        HttpResponse(
            200,
            {},
            json.dumps(versions_payload).encode(),
            "https://api.ngc.nvidia.com/v2/models/nvidia/tao/actionrecognitionnet/versions",
        ),
    )

    page = NgcModelsSourceAdapter(
        client=client,
        page_size=2,
        include_all_versions=True,
    ).fetch_page({})

    assert len(page.records) == 1
    releases = page.records[0].releases
    assert [release.version for release in releases] == ["0.5.0", "1.0.0"]
    assert [release.revision for release in releases] == [None, None]
    assert [release.metadata["version_inventory_status"] for release in releases] == [
        "complete",
        "complete",
    ]
    assert releases[0].released_at == "2024-08-29T17:59:14.698Z"
    assert releases[0].metadata["totalFileCount"] == 2
    assert releases[0].metadata["customMetrics"][0]["name"] == "SHA256 digests"
    metadata_url = next(
        link.url for link in page.records[0].links if link.relation == "model_card_metadata"
    )
    assert metadata_url.endswith("/versions/1.0.0")
    assert client.calls[1][0] == (
        "https://api.ngc.nvidia.com/v2/models/nvidia/tao/actionrecognitionnet/versions"
    )
    assert client.calls[1][1] == {}


def test_ngc_opt_in_version_expansion_marks_unverified_second_page_unavailable() -> None:
    versions_payload = {
        "modelVersions": [{"versionId": "1.0.0"}],
        "paginationInfo": {"index": 0, "size": 1, "totalResults": 2, "totalPages": 2},
    }
    client = _QueuedClient(
        _response([_ACTION], total=1, page=0),
        HttpResponse(200, {}, json.dumps(versions_payload).encode(), "https://api.ngc.nvidia.com"),
    )

    page = NgcModelsSourceAdapter(
        client=client,
        page_size=2,
        include_all_versions=True,
    ).fetch_page({})

    assert len(page.records) == 1
    assert page.records[0].source_record_id == "nvidia/tao/actionrecognitionnet"
    assert page.records[0].raw["version_inventory_status"] == "unavailable_error"
    assert "continuation contract is not verified" in page.records[0].raw[
        "version_inventory_error"
    ]
    assert page.records[0].releases[0].version == "1.0.0"
    assert page.records[0].releases[0].metadata["version_inventory_status"] == (
        "unavailable_error"
    )


@pytest.mark.parametrize("total_pages", [0, 1])
def test_ngc_opt_in_expansion_retains_model_with_an_empty_version_inventory(
    total_pages: int,
) -> None:
    versions_payload = {
        "modelVersions": [],
        "paginationInfo": {
            "index": 0,
            "size": 100,
            "totalResults": 0,
            "totalPages": total_pages,
        },
    }
    client = _QueuedClient(
        _response([_ACTION], total=1, page=0),
        HttpResponse(200, {}, json.dumps(versions_payload).encode(), "https://api.ngc.nvidia.com"),
    )

    page = NgcModelsSourceAdapter(
        client=client,
        page_size=2,
        include_all_versions=True,
    ).fetch_page({})

    assert len(page.records) == 1
    assert page.records[0].releases == ()
    assert page.records[0].raw["version_inventory_status"] == "complete"


def test_ngc_one_inaccessible_version_inventory_does_not_drop_row_or_block_later_page() -> None:
    version_response = {
        "modelVersions": [{"versionId": "1.0.0", "status": "UPLOAD_COMPLETE"}],
        "paginationInfo": {"index": 0, "size": 100, "totalResults": 1, "totalPages": 1},
    }
    client = _QueuedClient(
        _response([_ACTION], total=2, page=0, page_size=1),
        HttpResponse(503, {}, b"temporarily unavailable", "https://api.ngc.nvidia.com"),
        _response([_NEMOTRON], total=2, page=1, page_size=1),
        HttpResponse(200, {}, json.dumps(version_response).encode(), "https://api.ngc.nvidia.com"),
    )
    adapter = NgcModelsSourceAdapter(
        client=client,
        page_size=1,
        include_all_versions=True,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert len(first.records) == 1
    assert first.records[0].raw["version_inventory_status"] == "unavailable_transient"
    assert first.records[0].releases[0].metadata["version_inventory_status"] == (
        "unavailable_transient"
    )
    assert first.complete is False
    assert first.next_state["page"] == 1
    assert len(second.records) == 1
    assert second.records[0].source_record_id == "nvidia/nemotron"
    assert second.records[0].raw["version_inventory_status"] == "complete"
    assert second.complete is True


@pytest.mark.parametrize("status", [401, 403])
def test_ngc_http_failure_auth_status_is_unavailable_unauthorized_without_leaking_tokens(
    status: int,
) -> None:
    page = NgcModelsSourceAdapter(
        client=_HttpFailureClient(status),
        include_all_versions=True,
    ).fetch_page({})

    assert len(page.records) == 1
    record = page.records[0]
    assert record.raw["version_inventory_status"] == "unavailable_unauthorized"
    assert record.raw["version_inventory_error"] == f"HTTP {status}"
    assert "secret" not in json.dumps(record.raw)
    assert record.releases[0].metadata["version_inventory_status"] == (
        "unavailable_unauthorized"
    )
    assert record.releases[0].metadata["version_inventory_error"] == f"HTTP {status}"


def test_ngc_http_failure_transient_status_remains_distinguishable() -> None:
    page = NgcModelsSourceAdapter(
        client=_HttpFailureClient(503),
        include_all_versions=True,
    ).fetch_page({})

    assert len(page.records) == 1
    assert page.records[0].raw["version_inventory_status"] == "unavailable_transient"
    assert page.records[0].raw["version_inventory_error"] == "HTTP 503"
