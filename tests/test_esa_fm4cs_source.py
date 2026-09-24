from __future__ import annotations

import json
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.esa_fm4cs import EsaFm4csSourceAdapter


class _RouteClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        params = dict(params or {})
        self.calls.append((url, params))
        if url == "https://huggingface.co/api/models?author=FM4CS":
            payload = [
                {
                    "id": "FM4CS/THOR-1.0-tiny",
                    "sha": "catalog-sha",
                    "pipeline_tag": "image-feature-extraction",
                    "cardData": {"license": "apache-2.0"},
                }
            ]
        elif url.endswith("/refs"):
            payload = {
                "branches": [{"ref": "refs/heads/main"}],
                "tags": [],
                "converts": [],
            }
        elif url.endswith("/commits/refs%2Fheads%2Fmain"):
            payload = [{"id": "stable-commit", "date": "2026-02-01T00:00:00Z"}]
        elif url.endswith("/tree/stable-commit?recursive=true&expand=false"):
            payload = [
                {"type": "file", "path": "thor_v1_vit_tiny.pt"},
                {"type": "file", "path": "README.md"},
            ]
        else:
            raise AssertionError(f"unexpected request: {url} {params}")
        return HttpResponse(
            status=200,
            headers={},
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_esa_fm4cs_scope_enumerates_exact_public_model_checkpoint() -> None:
    client = _RouteClient()
    adapter = EsaFm4csSourceAdapter(client=client)

    catalog_page = adapter.fetch_page({})
    assert catalog_page.records[0].source_record_id == "FM4CS/THOR-1.0-tiny"
    assert len(catalog_page.records[0].models) == 1
    assert catalog_page.records[0].models[0].name == "FM4CS/THOR-1.0-tiny"
    assert client.calls[0] == (
        "https://huggingface.co/api/models?author=FM4CS",
        {
            "limit": 100,
            "full": "true",
            "cardData": "true",
            "config": "true",
            "sort": "lastModified",
            "direction": -1,
        },
    )

    state = catalog_page.next_state
    while True:
        page = adapter.fetch_page(state)
        state = page.next_state
        if page.records:
            checkpoint = page.records[0]
            break

    assert checkpoint.source_record_id == "FM4CS/THOR-1.0-tiny@stable-commit"
    assert checkpoint.releases[0].revision == "stable-commit"
    assert checkpoint.releases[0].metadata["weight_files"] == ["thor_v1_vit_tiny.pt"]
    assert checkpoint.releases[0].metadata["weight_files_complete"] is True
    assert [(link.url, link.relation, link.crawl) for link in checkpoint.links] == [
        (
            "https://huggingface.co/FM4CS/THOR-1.0-tiny",
            "model_page",
            True,
        ),
        (
            "https://huggingface.co/FM4CS/THOR-1.0-tiny/resolve/stable-commit/thor_v1_vit_tiny.pt",
            "weights",
            False,
        ),
    ]
    assert page.complete is True
