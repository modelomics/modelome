from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.github_repositories import GitHubPublicRepositoriesSourceAdapter

NOW = datetime(2026, 9, 21, 20, 0, tzinfo=UTC)


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return self.responses.pop(0)


def response(
    body: list[dict[str, Any]], *, headers: dict[str, str] | None = None, url: str = ""
) -> HttpResponse:
    return HttpResponse(
        status=200,
        headers=headers or {},
        body=json.dumps(body).encode(),
        url=url or "https://api.github.com/repositories",
    )


def repository(repository_id: int, name: str) -> dict[str, Any]:
    return {
        "id": repository_id,
        "node_id": f"R_{repository_id}",
        "full_name": name,
        "html_url": f"https://github.com/{name}",
        "url": f"https://api.github.com/repos/{name}",
        "description": "A public research repository.",
        "private": False,
        "fork": False,
        "archived": False,
        "disabled": False,
        "created_at": "2026-09-20T01:02:03Z",
        "updated_at": "2026-09-21T04:05:06Z",
    }


def test_direct_github_catalog_follows_the_provider_since_cursor() -> None:
    next_url = "https://api.github.com/repositories?since=102&per_page=2"
    canonical_next_url = "https://api.github.com/repositories?per_page=2&since=102"
    client = QueuedClient(
        response(
            [repository(101, "lab/first"), repository(102, "lab/second")],
            headers={"Link": f'<{next_url}>; rel="next"'},
        ),
        response([repository(103, "lab/third")], url=canonical_next_url),
    )
    adapter = GitHubPublicRepositoriesSourceAdapter(
        name="github-fixture",
        page_size=2,
        token="public-metadata-token",
        client=client,
        clock=lambda: NOW,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state == {
        "next_url": canonical_next_url,
        "last_repository_id": 102,
        "raw_items_seen": 2,
    }
    assert client.calls[0][1] == {"per_page": 2}
    assert client.calls[1][0] == canonical_next_url
    assert client.calls[1][1] == {}
    assert all(
        call[2]["Authorization"] == "Bearer public-metadata-token" for call in client.calls
    )
    assert all(call[2]["X-GitHub-Api-Version"] == "2026-03-10" for call in client.calls)
    assert second.complete is True
    assert second.next_state == {
        "completed_at": "2026-09-21T20:00:00Z",
        "last_repository_id": 103,
    }

    record = first.records[0]
    assert record.kind is ArtifactKind.CODE_REPOSITORY
    assert record.canonical_url == "https://api.github.com/repos/lab/first"
    assert record.identifiers == (
        Identifier("github:repository-id", "101"),
        Identifier("github:repository", "lab/first"),
    )
    assert record.links[0].url == "https://github.com/lab/first"
    assert record.links[0].relation == "repository_catalog"
    assert record.models == ()


def test_direct_github_catalog_rejects_a_cursor_that_does_not_match_the_page_tail() -> None:
    client = QueuedClient(
        response(
            [repository(101, "lab/first")],
            headers={
                "Link": (
                    '<https://api.github.com/repositories?since=100&per_page=1>; '
                    'rel="next"'
                )
            },
        )
    )
    adapter = GitHubPublicRepositoriesSourceAdapter(
        name="github-fixture", page_size=1, client=client
    )

    with pytest.raises(ValueError, match="does not match the final repository ID"):
        adapter.fetch_page({})


def test_direct_github_catalog_retries_a_malformed_provider_row_without_advancing() -> None:
    client = QueuedClient(response([repository(101, "lab/first"), {"id": 102}]))
    adapter = GitHubPublicRepositoriesSourceAdapter(
        name="github-fixture", page_size=2, client=client
    )

    page = adapter.fetch_page({"last_repository_id": 100})

    assert page.complete is False
    assert page.next_state == {"last_repository_id": 100}
    assert page.retry_state == {"last_repository_id": 100}
    assert len(page.records) == 1
    assert page.issues[0].source_record_id == "github-repository-id:102"


def test_direct_github_catalog_rejects_private_rows_and_cross_origin_cursors() -> None:
    private = repository(101, "lab/first")
    private["private"] = True
    private_client = QueuedClient(response([private]))
    adapter = GitHubPublicRepositoriesSourceAdapter(
        name="github-fixture", page_size=1, client=private_client
    )
    page = adapter.fetch_page({})
    assert page.complete is False
    assert "private repository" in page.issues[0].error

    cursor_client = QueuedClient(
        response(
            [repository(101, "lab/first")],
            headers={
                "Link": (
                    '<https://example.test/repositories?since=101&per_page=1>; rel="next"'
                )
            },
        )
    )
    adapter = GitHubPublicRepositoriesSourceAdapter(
        name="github-fixture", page_size=1, client=cursor_client
    )
    with pytest.raises(ValueError, match="escaped the catalog endpoint"):
        adapter.fetch_page({})
