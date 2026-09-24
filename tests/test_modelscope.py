from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier, ModelStatus
from modelome.sources.modelscope import ModelScopeModelsSourceAdapter


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
    models: list[Mapping[str, Any]],
    *,
    total: int,
    page_number: int = 1,
    page_size: int = 2,
) -> HttpResponse:
    payload = {
        "success": True,
        "data": {
            "models": models,
            "total_count": total,
            "page_number": page_number,
            "page_size": page_size,
        },
    }
    return HttpResponse(200, {}, json.dumps(payload).encode(), "https://fixtures.test/models")


_QWEN = {
    "id": "Qwen/Qwen-Image-2.1",
    "display_name": "Qwen-Image-2.1",
    "description": "See https://github.com/QwenLM/Qwen2.5 for implementation.",
    "created_at": "2026-09-15T08:33:52Z",
    "last_modified": "2026-09-21T04:31:54Z",
    "tasks": ["text-to-image-synthesis"],
    "tags": ["library:diffusers", "custom_tag:image-generation"],
    "private": False,
}

_GLM = {
    "id": "ZhipuAI/GLM-5.3",
    "display_name": "GLM-5.3",
    "description": "",
    "created_at": "2026-09-01T00:00:00Z",
    "last_modified": "2026-09-18T00:00:00Z",
    "tasks": ["text-generation"],
    "tags": ["library:transformer"],
    "private": False,
}


def test_modelscope_collects_bounded_documented_sort_windows() -> None:
    client = _QueuedClient(
        _response([_QWEN, _GLM], total=256_960),
        _response([_GLM, _QWEN], total=256_960),
    )
    adapter = ModelScopeModelsSourceAdapter(
        page_size=2,
        max_pages_per_sort=1,
        sorts=("default", "likes"),
        client=client,
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.upstream_count is None
    assert first.next_state["sort_index"] == 1
    assert second.complete is True
    assert second.upstream_count == 4
    assert second.authoritative_snapshot is False
    assert second.next_state["provider_totals"] == {"default": 256_960, "likes": 256_960}
    assert [call[1] for call in client.calls] == [
        {"sort": "default", "page_number": 1, "page_size": 2},
        {"sort": "likes", "page_number": 1, "page_size": 2},
    ]
    record = first.records[0]
    assert record.source_record_id == "Qwen/Qwen-Image-2.1"
    assert record.identifiers == (Identifier("modelscope:model", "Qwen/Qwen-Image-2.1"),)
    assert record.models[0].status is ModelStatus.RELEASED
    assert record.raw["catalog_sort"] == "default"
    assert record.raw["provider_total_count"] == 256_960
    assert (
        "https://modelscope.cn/models/Qwen/Qwen-Image-2.1",
        "model_page",
    ) in {(link.url, link.relation) for link in record.links}
    assert (
        "https://github.com/QwenLM/Qwen2.5",
        "documentation_reference",
    ) in {(link.url, link.relation) for link in record.links}


def test_modelscope_rejects_pagination_past_provider_limit() -> None:
    with pytest.raises(ValueError, match="3,000-row per-sort ceiling"):
        ModelScopeModelsSourceAdapter(page_size=50, max_pages_per_sort=61)


def test_modelscope_rejects_a_mismatched_page_response() -> None:
    client = _QueuedClient(_response([_QWEN], total=1, page_number=2, page_size=1))
    adapter = ModelScopeModelsSourceAdapter(
        page_size=1,
        max_pages_per_sort=1,
        sorts=("default",),
        client=client,
    )

    with pytest.raises(ValueError, match="pagination response does not match request"):
        adapter.fetch_page({})


def test_modelscope_rejects_a_short_page_before_its_provider_total() -> None:
    client = _QueuedClient(_response([_QWEN], total=2, page_size=2))
    adapter = ModelScopeModelsSourceAdapter(
        page_size=2,
        max_pages_per_sort=1,
        sorts=("default",),
        client=client,
    )

    with pytest.raises(ValueError, match="expected 2 from provider total"):
        adapter.fetch_page({})


def test_modelscope_does_not_persist_an_unexpected_private_row() -> None:
    private = dict(_QWEN)
    private["private"] = True
    client = _QueuedClient(_response([private], total=1, page_size=1))
    adapter = ModelScopeModelsSourceAdapter(
        page_size=1,
        max_pages_per_sort=1,
        sorts=("default",),
        client=client,
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.records == ()
