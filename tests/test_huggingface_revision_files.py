from __future__ import annotations

import json
from typing import Any

import pytest

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
            {"type": "file", "path": "docs/README.md"},
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


def test_live_resnet18_listing_and_revision_trees_match_checkpoint_history() -> None:
    """Snapshot of first-party Hub listing, refs, commits, and tree responses."""
    repo = "microsoft/resnet-18"
    latest = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
    snapshots = {
        latest: ["model.safetensors", "pytorch_model.bin", "tf_model.h5"],
        "f45a6faa12e0381f5620c5c8d7e916bca90f2c44": [
            "model.safetensors",
            "pytorch_model.bin",
            "tf_model.h5",
        ],
        "2f536bd335677c6b111b3d103af458ef57a6145e": [
            "pytorch_model.bin",
            "tf_model.h5",
        ],
        "06959238e40d10272a27632d7458258a78a11840": ["pytorch_model.bin"],
        "cb2984207d7196932d571bd7732a0a644b471372": ["pytorch_model.bin"],
        "b84c5cd73e9544fa1b67d690748d13a4bdb29267": ["pytorch_model.bin"],
        "e32f4f8ec69beca53a7ef46172b32363b480a335": ["pytorch_model.bin"],
        "f66f0aca93e014a4f12dd56f097b26111a48551d": [],
    }
    catalog = "https://huggingface.co/api/models?catalog=1"
    refs_url = f"https://huggingface.co/api/models/{repo}/refs"
    commits_url = (
        f"https://huggingface.co/api/models/{repo}/commits/refs%2Fheads%2Fmain"
    )
    routes = {
        catalog: (
            200,
            [
                {
                    "id": repo,
                    "sha": latest,
                    "siblings": [
                        {"rfilename": name} for name in snapshots[latest]
                    ],
                }
            ],
            {},
        ),
        refs_url: (
            200,
            {
                "branches": [
                    {
                        "name": "main",
                        "ref": "refs/heads/main",
                        "targetCommit": latest,
                    }
                ],
                "tags": [],
                "converts": [],
            },
            {},
        ),
        commits_url: (
            200,
            [{"id": sha} for sha in snapshots],
            {},
        ),
    }
    for sha, filenames in snapshots.items():
        tree_url = (
            f"https://huggingface.co/api/models/{repo}/tree/{sha}"
            "?recursive=true&expand=false"
        )
        routes[tree_url] = (
            200,
            [{"type": "file", "path": filename} for filename in filenames],
            {},
        )
    adapter = HuggingFaceSourceAdapter(
        client=_RouteClient(routes),
        include_revisions=True,
        include_revision_files=True,
    )

    listed = adapter.fetch_page({})
    assert listed.records[0].releases[0].revision == latest
    assert listed.records[0].releases[0].metadata["weight_files"] == snapshots[latest]
    refs = adapter.fetch_page(listed.next_state)
    commits = adapter.fetch_page(refs.next_state)
    state = dict(commits.next_state)

    observed: dict[str, list[str]] = {}
    for _ in snapshots:
        result = adapter.fetch_page(state)
        assert len(result.records) == 1
        record = result.records[0]
        observed[record.releases[0].revision] = record.releases[0].metadata[
            "weight_files"
        ]
        assert all(
            f"/resolve/{record.releases[0].revision}/" in link.url
            for link in record.links
            if link.relation == "weights"
        )
        state = dict(result.next_state)

    assert observed == snapshots


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
    routes[following] = (
        200,
        [{"type": "file", "path": "historical/late.safetensors"}],
        {},
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
    assert "historical/late.safetensors" not in record.releases[0].metadata["weight_files"]
    assert len(client.calls) == 4
    assert following not in client.calls

    # Raising the explicit per-commit cap retrieves the path on the later
    # historical tree page and lets the release report a complete file list.
    client.calls.clear()
    larger_cap = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
        max_revision_tree_pages=2,
    )
    larger_state = _drain_to_tree(larger_cap, {})
    page_one = larger_cap.fetch_page(larger_state)
    assert page_one.records == ()
    page_two = larger_cap.fetch_page(page_one.next_state)
    assert page_two.records[0].releases[0].metadata["weight_files"] == [
        "historical/late.safetensors",
        "observed.safetensors",
    ]
    assert page_two.records[0].releases[0].metadata["weight_files_complete"] is True


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
    assert record.releases[0].metadata["weight_files_complete"] is False
    assert [link.url for link in record.links if link.relation == "weights"] == [
        f"https://huggingface.co/{repo}/resolve/commit-a/folder/model%20%231.safetensors"
    ]


def test_overlong_weight_path_cannot_be_silently_reported_as_complete() -> None:
    repo = "lab/revision-model"
    tree = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    long_weight_path = "/".join(["a" * 200] * 6) + "/model.safetensors"
    routes = _revision_routes(repo)
    routes[tree] = (
        200,
        [
            {"type": "file", "path": "model.safetensors"},
            {"type": "file", "path": long_weight_path},
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
    record = adapter.fetch_page(state).records[0]

    assert record.releases[0].metadata["weight_files"] == ["model.safetensors"]
    assert record.releases[0].metadata["weight_files_complete"] is False


def test_default_tree_cap_marks_weight_after_twenty_thousand_entries_partial() -> None:
    repo = "lab/revision-model"
    tree_base = f"https://huggingface.co/api/models/{repo}/tree/commit-a"
    tree_urls = [
        f"{tree_base}?recursive=true&expand=false",
        *(
            f"{tree_base}?cursor=page-{page}&expand=false&recursive=true"
            for page in range(2, 22)
        ),
    ]
    routes = _revision_routes(repo)
    for page_index, url in enumerate(tree_urls[:-1]):
        routes[url] = (
            200,
            [
                {"type": "file", "path": f"docs/page-{page_index:02d}-file-{index:04d}.txt"}
                for index in range(1000)
            ],
            {"Link": f'<{tree_urls[page_index + 1]}>; rel="next"'},
        )
    routes[tree_urls[-1]] = (
        200,
        [{"type": "file", "path": "historical/late.safetensors"}],
        {},
    )
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )

    state = _drain_to_tree(adapter, {})
    result = None
    for _ in range(20):
        result = adapter.fetch_page(state)
        state = dict(result.next_state)
    assert result is not None and result.records
    partial = result.records[0].releases[0].metadata
    assert partial["weight_files"] == []
    assert partial["weight_files_complete"] is False
    assert tree_urls[-1] not in client.calls

    # Raising the configured finite page cap reaches the exact historical file.
    client.calls.clear()
    larger_cap = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
        max_revision_tree_pages=21,
    )
    state = _drain_to_tree(larger_cap, {})
    result = None
    for _ in range(21):
        result = larger_cap.fetch_page(state)
        state = dict(result.next_state)
    assert result is not None and result.records
    complete = result.records[0].releases[0].metadata
    assert complete["weight_files"] == ["historical/late.safetensors"]
    assert complete["weight_files_complete"] is True


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


def test_refs_with_same_exact_target_commit_share_one_history_scan() -> None:
    repo = "lab/revision-model"
    refs = f"https://huggingface.co/api/models/{repo}/refs"
    commits = f"https://huggingface.co/api/models/{repo}/commits/refs%2Fheads%2Fmain"
    tree = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    routes = _revision_routes(repo)
    routes[refs] = (
        200,
        {
            "branches": [
                {"ref": "refs/heads/main", "target_commit": "commit-a"},
            ],
            "tags": [
                {"ref": "refs/tags/v1", "target_commit": "commit-a"},
            ],
            "converts": [],
        },
        {},
    )
    routes[commits] = (200, [{"id": "commit-a"}], {})
    routes[tree] = (
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

    catalog = adapter.fetch_page({})
    ref_page = adapter.fetch_page(catalog.next_state)
    commit_page = adapter.fetch_page(ref_page.next_state)
    tree_page = adapter.fetch_page(commit_page.next_state)

    assert [record.releases[0].revision for record in tree_page.records] == ["commit-a"]
    assert tree_page.records[0].releases[0].metadata["weight_files"] == [
        "model.safetensors"
    ]
    assert tree_page.records[0].releases[0].metadata["git_refs"] == [
        "refs/heads/main",
        "refs/tags/v1",
    ]
    assert client.calls.count(commits) == 1
    assert client.calls.count(tree) == 1


def test_ref_names_attach_only_to_their_exact_target_commit() -> None:
    repo = "lab/revision-model"
    refs = f"https://huggingface.co/api/models/{repo}/refs"
    commits = f"https://huggingface.co/api/models/{repo}/commits/refs%2Fheads%2Fmain"
    tree_target = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    tree_parent = f"https://huggingface.co/api/models/{repo}/tree/commit-parent?recursive=true&expand=false"
    routes = _revision_routes(repo)
    routes[refs] = (
        200,
        {
            "branches": [{"ref": "refs/heads/main", "target_commit": "commit-a"}],
            "tags": [],
            "converts": [],
        },
        {},
    )
    routes[commits] = (200, [{"id": "commit-a"}, {"id": "commit-parent"}], {})
    routes[tree_target] = (200, [{"type": "file", "path": "model.safetensors"}], {})
    routes[tree_parent] = (200, [{"type": "file", "path": "old-model.safetensors"}], {})
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )

    catalog = adapter.fetch_page({})
    ref_page = adapter.fetch_page(catalog.next_state)
    commit_page = adapter.fetch_page(ref_page.next_state)
    target = adapter.fetch_page(commit_page.next_state)
    parent = adapter.fetch_page(target.next_state)

    assert target.records[0].releases[0].metadata["git_refs"] == ["refs/heads/main"]
    assert target.records[0].releases[0].metadata["weight_files"] == ["model.safetensors"]
    assert "git_refs" not in parent.records[0].releases[0].metadata
    assert parent.records[0].releases[0].metadata["weight_files"] == [
        "old-model.safetensors"
    ]


def test_malformed_commit_page_cannot_silently_complete_revision_history() -> None:
    repo = "lab/revision-model"
    commits = f"https://huggingface.co/api/models/{repo}/commits/refs%2Fheads%2Fmain"
    routes = _revision_routes(repo)
    routes[commits] = (200, {"error": "truncated"}, {})
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )

    catalog = adapter.fetch_page({})
    ref_page = adapter.fetch_page(catalog.next_state)

    with pytest.raises(ValueError, match="revision commits must be a JSON array"):
        adapter.fetch_page(ref_page.next_state)


def test_commit_cursor_resumes_after_current_tree_tasks_drain() -> None:
    repo = "lab/revision-model"
    commits = f"https://huggingface.co/api/models/{repo}/commits/refs%2Fheads%2Fmain"
    commits_next = f"{commits}?cursor=page-2"
    tree_a = f"https://huggingface.co/api/models/{repo}/tree/commit-a?recursive=true&expand=false"
    tree_b = f"https://huggingface.co/api/models/{repo}/tree/commit-b?recursive=true&expand=false"
    routes = _revision_routes(repo)
    routes[commits] = (
        200,
        [{"id": "commit-a"}],
        {"Link": f'<{commits_next}>; rel="next"'},
    )
    routes[commits_next] = (200, [{"id": "commit-b"}], {})
    routes[tree_a] = (200, [{"type": "file", "path": "old.safetensors"}], {})
    routes[tree_b] = (200, [{"type": "file", "path": "new.safetensors"}], {})
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_revisions=True,
        include_revision_files=True,
    )

    catalog = adapter.fetch_page({})
    ref_page = adapter.fetch_page(catalog.next_state)
    first_commits_page = adapter.fetch_page(ref_page.next_state)
    first_tree = adapter.fetch_page(first_commits_page.next_state)
    assert first_tree.records[0].releases[0].metadata["weight_files"] == [
        "old.safetensors"
    ]
    assert first_tree.next_state["revision_queue"][0]["next_url"] == commits_next

    second_commits_page = adapter.fetch_page(first_tree.next_state)
    second_tree = adapter.fetch_page(second_commits_page.next_state)
    assert second_tree.records[0].releases[0].metadata["weight_files"] == [
        "new.safetensors"
    ]
    assert second_tree.complete is True
