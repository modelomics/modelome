from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.modelscope import ModelScopeModelsSourceAdapter


class _Client:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append(dict(params or {}))
        page = (params or {})["page_number"]
        size = (params or {})["page_size"]
        models = [{"id": "org/model"}] if page == 1 else []
        body = json.dumps({"success": True, "data": {
            "models": models, "total_count": 1,
            "page_number": page, "page_size": size,
        }}).encode()
        return HttpResponse(200, {}, body, url)


def test_search_queries_add_bounded_catalog_partitions_and_resume() -> None:
    client = _Client()
    adapter = ModelScopeModelsSourceAdapter(
        page_size=1,
        max_pages_per_sort=1,
        sorts=("default",),
        search_queries=("rare-model-family",),
        client=client,
    )

    base = adapter.fetch_page({})
    searched = adapter.fetch_page(base.next_state)

    assert base.complete is False
    assert base.next_state["query_index"] == 1
    assert searched.complete is True
    assert searched.next_state["provider_totals"] == {
        "default": 1,
        "search:rare-model-family:default": 1,
    }
    assert client.calls == [
        {"sort": "default", "page_number": 1, "page_size": 1},
        {"sort": "default", "page_number": 1, "page_size": 1,
         "search": "rare-model-family"},
    ]


@pytest.mark.parametrize("queries", ["qwen", ("qwen", "qwen")])
def test_search_queries_must_be_an_unique_sequence(queries) -> None:
    with pytest.raises(ValueError, match="search_queries"):
        ModelScopeModelsSourceAdapter(search_queries=queries)
