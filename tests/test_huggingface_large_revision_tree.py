from __future__ import annotations

import json
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.huggingface import HuggingFaceSourceAdapter


class _RouteClient:
    def __init__(self, routes: dict[str, tuple[object, dict[str, str]]]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        if params:
            url = f"{url}?catalog=1"
        self.calls.append(url)
        payload, response_headers = self.routes[url]
        return HttpResponse(
            status=200,
            headers=response_headers,
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_three_page_revision_tree_keeps_all_large_checkpoint_inventory() -> None:
    repo = "lab/large-checkpoint-inventory"
    sha = "f" * 40
    tree_base = f"https://huggingface.co/api/models/{repo}/tree/{sha}"
    tree_urls = [
        f"{tree_base}?recursive=true&expand=false",
        f"{tree_base}?cursor=page-2&expand=false&recursive=true",
        f"{tree_base}?cursor=page-3&expand=false&recursive=true",
    ]
    filenames = [f"checkpoints/adapter-{index:04d}.pt" for index in range(2206)]
    routes: dict[str, tuple[object, dict[str, str]]] = {
        tree_urls[0]: (
            [{"type": "file", "path": name} for name in filenames[:1000]],
            {"Link": f'<{tree_urls[1]}>; rel="next"'},
        ),
        tree_urls[1]: (
            [{"type": "file", "path": name} for name in filenames[1000:2000]],
            {"Link": f'<{tree_urls[2]}>; rel="next"'},
        ),
        tree_urls[2]: (
            [{"type": "file", "path": name} for name in filenames[2000:]],
            {},
        ),
    }
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )
    state: dict[str, Any] = {
        "revision_listing_complete": True,
        "revision_queue": [
            {
                "model_id": repo,
                "phase": "commits",
                "commits_exhausted": True,
                "tree_queue": [
                    {"commit": {"id": sha}, "weight_candidates": []}
                ],
            }
        ],
    }

    pages = []
    for _ in tree_urls:
        page = adapter.fetch_page(state)
        pages.append(page)
        state = dict(page.next_state)

    assert [len(page.records) for page in pages] == [0, 0, 1]
    assert pages[-1].complete is True
    record = pages[-1].records[0]
    assert record.releases[0].revision == sha
    assert record.releases[0].metadata["weight_files"] == filenames
    assert record.releases[0].metadata["weight_files_complete"] is True
    links = [link for link in record.links if link.relation == "weights"]
    assert len(links) == 2206
    assert all(f"/resolve/{sha}/" in link.url for link in links)
    assert client.calls == tree_urls
