from __future__ import annotations

import json
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.huggingface import HuggingFaceSourceAdapter


class _RouteClient:
    def __init__(self, routes: dict[str, tuple[int, object, dict[str, str]]]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        if params:
            url = f"{url}?catalog=1"
        self.calls.append(url)
        status, payload, response_headers = self.routes[url]
        return HttpResponse(
            status=status,
            headers=response_headers,
            body=json.dumps(payload).encode(),
            url=url,
        )


def _revision_routes(
    repo: str = "lab/revision-model",
) -> dict[str, tuple[int, object, dict[str, str]]]:
    catalog = "https://huggingface.co/api/models?catalog=1"
    refs = f"https://huggingface.co/api/models/{repo}/refs"
    commits = (
        f"https://huggingface.co/api/models/{repo}/commits/"
        "refs%2Fheads%2Fmain"
    )
    return {
        catalog: (200, [{"id": repo, "sha": "head"}], {}),
        refs: (200, {"branches": [{"ref": "refs/heads/main"}], "tags": [], "converts": []}, {}),
        commits: (
            200,
            [{"id": "commit-a", "date": "2025-01-01T00:00:00Z", "title": "A"}],
            {},
        ),
    }


def _drain_to_tree(adapter: HuggingFaceSourceAdapter, state: dict[str, Any]) -> dict[str, Any]:
    catalog = adapter.fetch_page(state)
    refs = adapter.fetch_page(catalog.next_state)
    commits = adapter.fetch_page(refs.next_state)
    return dict(commits.next_state)


def test_revision_weight_files_follow_tree_pagination_and_resume() -> None:
    repo = "lab/revision-model"
    tree = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    following = (
        f"https://huggingface.co/api/models/{repo}/tree/commit-a"
        "?cursor=page-2&expand=false&recursive=true"
    )
    routes = _revision_routes(repo)
    routes[tree] = (
        200,
        [
            {
                "type": "file",
                "path": "model.safetensors",
                "size": 1234,
                "lfs": {"size": 1234, "sha256": "A" * 64, "pointer_size": 128},
            },
            {"type": "file", "path": "docs/README.md"},
        ],
        {"Link": f'<{following}>; rel="next"'},
    )
    routes[following] = (
        200,
        [
            {"type": "file", "path": "shards/part-00001.safetensors"},
            {"type": "file", "path": "../escape.safetensors"},
        ],
        {},
    )
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )

    state = _drain_to_tree(adapter, {})
    first_tree_page = adapter.fetch_page(state)
    assert first_tree_page.records == ()
    assert first_tree_page.complete is False
    assert first_tree_page.next_state["revision_queue"][0]["tree_queue"][0]["next_url"] == following

    # A new adapter instance can resume from the checkpointed cursor.
    resumed = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )
    final_tree_page = resumed.fetch_page(first_tree_page.next_state)
    assert final_tree_page.complete is True
    record = final_tree_page.records[0]
    assert record.releases[0].revision == "commit-a"
    assert record.releases[0].metadata["weight_files"] == [
        "model.safetensors",
        "shards/part-00001.safetensors",
    ]
    assert record.releases[0].metadata["weight_files_complete"] is True
    assert record.releases[0].metadata["weight_file_metadata"] == {
        "model.safetensors": {"size_bytes": 1234, "lfs_sha256": "a" * 64}
    }
    assert {
        (link.url, link.relation, link.crawl)
        for link in record.links
        if link.relation == "weights"
    } == {
        (
            f"https://huggingface.co/{repo}/resolve/commit-a/model.safetensors",
            "weights",
            False,
        ),
        (
            f"https://huggingface.co/{repo}/resolve/commit-a/shards/part-00001.safetensors",
            "weights",
            False,
        ),
    }
    assert len(client.calls) == 5


def test_revision_tree_page_cap_marks_observed_files_incomplete() -> None:
    repo = "lab/revision-model"
    tree = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    following = (
        f"https://huggingface.co/api/models/{repo}/tree/commit-a"
        "?cursor=page-2&expand=false&recursive=true"
    )
    routes = _revision_routes(repo)
    routes[tree] = (
        200,
        [{"type": "file", "path": "observed.safetensors"}],
        {"Link": f'<{following}>; rel="next"'},
    )
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
        max_revision_tree_pages=1,
    )

    state = _drain_to_tree(adapter, {})
    result = adapter.fetch_page(state)
    record = result.records[0]
    assert result.complete is True
    assert record.releases[0].metadata["weight_files"] == ["observed.safetensors"]
    assert record.releases[0].metadata["weight_files_complete"] is False
    assert len(client.calls) == 4
    assert following not in client.calls


def test_unavailable_historical_tree_does_not_block_later_revisions() -> None:
    repo = "lab/revision-model"
    commits = f"https://huggingface.co/api/models/{repo}/commits/refs%2Fheads%2Fmain"
    first_tree = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    second_tree = f"https://huggingface.co/api/models/{repo}/tree/commit-b?recursive=true&expand=false"
    routes = _revision_routes(repo)
    routes[commits] = (
        200,
        [{"id": "commit-a"}, {"id": "commit-b"}],
        {},
    )
    routes[first_tree] = (404, {"error": "revision unavailable"}, {})
    routes[second_tree] = (
        200,
        [{"type": "file", "path": "model.safetensors"}],
        {},
    )
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )

    state = _drain_to_tree(adapter, {})
    first = adapter.fetch_page(state)
    assert [record.releases[0].revision for record in first.records] == ["commit-a"]
    assert first.records[0].releases[0].metadata["weight_files_complete"] is False
    second = adapter.fetch_page(first.next_state)
    assert [record.releases[0].revision for record in second.records] == ["commit-b"]
    assert second.records[0].releases[0].metadata["weight_files_complete"] is True
    assert second.records[0].releases[0].metadata["weight_files"] == ["model.safetensors"]


def test_weight_candidate_state_byte_cap_marks_file_list_incomplete() -> None:
    repo = "lab/revision-model"
    tree = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    routes = _revision_routes(repo)
    routes[tree] = (
        200,
        [
            {"type": "file", "path": "first.safetensors"},
            {"type": "file", "path": "second.safetensors"},
        ],
        {},
    )
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
        max_revision_weight_file_state_bytes=len(b"first.safetensors"),
    )

    state = _drain_to_tree(adapter, {})
    result = adapter.fetch_page(state)

    assert result.complete is True
    assert result.records[0].releases[0].metadata["weight_files"] == ["first.safetensors"]
    assert result.records[0].releases[0].metadata["weight_files_complete"] is False


def test_escaped_unicode_filename_is_percent_encoded_and_surrogates_are_rejected() -> None:
    repo = "lab/revision-model"
    tree = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    routes = _revision_routes(repo)
    routes[tree] = (
        200,
        [
            {"type": "file", "path": "folder/model #1.safetensors"},
            {"type": "file", "path": "bad\udcff.safetensors"},
        ],
        {},
    )
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )

    state = _drain_to_tree(adapter, {})
    result = adapter.fetch_page(state)
    record = result.records[0]

    assert record.releases[0].metadata["weight_files"] == ["folder/model #1.safetensors"]
    assert [link.url for link in record.links if link.relation == "weights"] == [
        f"https://huggingface.co/{repo}/resolve/commit-a/folder/model%20%231.safetensors"
    ]


def test_lfs_metadata_checkpoint_is_byte_bounded_and_truncation_is_explicit() -> None:
    repo = "lab/revision-model"
    tree = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    routes = _revision_routes(repo)
    routes[tree] = (
        200,
        [
            {
                "type": "file",
                "path": "model.safetensors",
                "size": 1234,
                "lfs": {"size": 1234, "sha256": "b" * 64},
            }
        ],
        {},
    )
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
        max_revision_weight_file_state_bytes=32,
    )

    state = _drain_to_tree(adapter, {})
    result = adapter.fetch_page(state)
    metadata = result.records[0].releases[0].metadata

    assert metadata["weight_files"] == ["model.safetensors"]
    assert metadata.get("weight_file_metadata", {}) == {}
    assert metadata["weight_file_metadata_truncated"] is True


def test_lfs_metadata_stays_with_each_commit_across_queue_resume() -> None:
    repo = "lab/revision-model"
    commits = f"https://huggingface.co/api/models/{repo}/commits/refs%2Fheads%2Fmain"
    tree_a = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    tree_b = f"https://huggingface.co/api/models/{repo}/tree/commit-b?recursive=true&expand=false"
    routes = _revision_routes(repo)
    routes[commits] = (200, [{"id": "commit-a"}, {"id": "commit-b"}], {})
    routes[tree_a] = (
        200,
        [
            {
                "type": "file",
                "path": "model.safetensors",
                "size": 111,
                "lfs": {"size": 111, "sha256": "a" * 64},
            }
        ],
        {},
    )
    routes[tree_b] = (
        200,
        [
            {
                "type": "file",
                "path": "model.safetensors",
                "size": 222,
                "lfs": {"size": 222, "sha256": "b" * 64},
            }
        ],
        {},
    )
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )

    state = _drain_to_tree(adapter, {})
    first_commit = adapter.fetch_page(state)
    assert first_commit.records[0].releases[0].revision == "commit-a"
    assert first_commit.records[0].releases[0].metadata["weight_file_metadata"] == {
        "model.safetensors": {"size_bytes": 111, "lfs_sha256": "a" * 64}
    }

    # The queued second commit resumes independently and cannot inherit the
    # first commit's file metadata despite reusing the same filename.
    resumed = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )
    second_commit = resumed.fetch_page(first_commit.next_state)
    assert second_commit.records[0].releases[0].revision == "commit-b"
    assert second_commit.records[0].releases[0].metadata["weight_file_metadata"] == {
        "model.safetensors": {"size_bytes": 222, "lfs_sha256": "b" * 64}
    }
