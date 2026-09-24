from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.github_repositories import GitHubPublicRepositoriesSourceAdapter


class OnePageClient:
    def __init__(self, body: list[dict[str, Any]], headers=None) -> None:
        self.body = body
        self.headers = headers or {}

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        return HttpResponse(
            status=200,
            headers=self.headers,
            body=json.dumps(self.body).encode(),
            url=url,
        )


class CursorClient:
    def __init__(self, pages: dict[int | None, list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, dict[str, int] | None]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, params))
        cursor = params.get("since") if params is not None else None
        return HttpResponse(
            status=200,
            headers={},
            body=json.dumps(self.pages[cursor]).encode(),
            url=url,
        )


def _repository(repository_id: Any) -> dict[str, Any]:
    return {
        "id": repository_id,
        "full_name": "lab/model",
        "html_url": "https://github.com/lab/model",
        "url": "https://api.github.com/repos/lab/model",
        "private": False,
    }


@pytest.mark.parametrize("repository_id", [True, 101.5])
def test_catalog_does_not_coerce_malformed_ids_into_a_discovery_cursor(
    repository_id: Any,
) -> None:
    adapter = GitHubPublicRepositoriesSourceAdapter(
        page_size=2,
        client=OnePageClient([_repository(repository_id)]),
    )

    page = adapter.fetch_page({"last_repository_id": 100})

    assert page.complete is False
    assert page.records == ()
    assert page.next_state == {"last_repository_id": 100}
    assert page.retry_state == {"last_repository_id": 100}
    assert len(page.issues) == 1
    assert "positive integer" in page.issues[0].error


@pytest.mark.parametrize("page_size", [True, 1.5])
def test_catalog_configuration_rejects_lossy_page_size_coercion(page_size: Any) -> None:
    with pytest.raises(ValueError, match="page_size must be a positive integer"):
        GitHubPublicRepositoriesSourceAdapter(page_size=page_size)


def test_catalog_rejects_a_fractional_durable_seen_count() -> None:
    adapter = GitHubPublicRepositoriesSourceAdapter(
        page_size=2,
        client=OnePageClient(
            [_repository(101)],
            {"Link": '<https://api.github.com/repositories?since=101&per_page=2>; rel="next"'},
        ),
    )

    with pytest.raises(ValueError, match="raw_items_seen must be an integer"):
        adapter.fetch_page({"last_repository_id": 100, "raw_items_seen": 1.5})


def test_catalog_can_scan_a_closed_historical_id_range_without_following_past_it() -> None:
    adapter = GitHubPublicRepositoriesSourceAdapter(
        page_size=3,
        initial_since=100,
        max_repository_id=102,
        client=OnePageClient(
            [_repository(101), _repository(102), _repository(103)],
            {"Link": '<https://api.github.com/repositories?since=103&per_page=3>; rel="next"'},
        ),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert [record.title for record in page.records] == ["lab/model", "lab/model"]
    assert page.next_state["last_repository_id"] == 102
    assert page.upstream_count == 2


def test_catalog_rejects_a_historical_ceiling_before_its_start() -> None:
    with pytest.raises(ValueError, match="max_repository_id must be >= initial_since"):
        GitHubPublicRepositoriesSourceAdapter(initial_since=200, max_repository_id=199)


def test_exact_full_terminal_page_uses_since_cursor_probe() -> None:
    client = CursorClient(
        {
            None: [_repository(101), _repository(102)],
            102: [],
        }
    )
    adapter = GitHubPublicRepositoriesSourceAdapter(page_size=2, client=client)

    full_page = adapter.fetch_page({})
    terminal_probe = adapter.fetch_page(full_page.next_state)

    assert len(full_page.records) == 2
    assert full_page.complete is False
    assert full_page.next_state["last_repository_id"] == 102
    assert terminal_probe.complete is True
    assert client.calls == [
        ("https://api.github.com/repositories", {"per_page": 2}),
        ("https://api.github.com/repositories", {"per_page": 2, "since": 102}),
    ]
