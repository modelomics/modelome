from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.github_historical_release_assets import (
    GitHubHistoricalReleaseAssetsSourceAdapter,
)


class RouteClient:
    def __init__(self, routes: dict[str, tuple[Any, dict[str, str]]]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        body, response_headers = self.routes[url]
        return HttpResponse(
            status=200,
            headers=response_headers,
            body=json.dumps(body).encode(),
            url=url,
        )


def test_historical_scan_factory_preserves_bounded_scope() -> None:
    source = create_source(
        {
            "name": "github-historical-release-assets-100-200",
            "adapter": "github_historical_release_assets",
            "initial_since": 100,
            "max_repository_id": 200,
            "max_repositories": 50,
            "max_releases_per_repository": 20,
            "max_assets_per_release": 25,
        },
        client=RouteClient({}),
        environ={},
    )
    assert isinstance(source, GitHubHistoricalReleaseAssetsSourceAdapter)
    assert source.name == "github-historical-release-assets-100-200"
    assert source.max_repository_id == 200
    assert source.max_repositories == 50
    assert source.max_releases_per_repository == 20
    assert source.max_assets_per_release == 25


def test_bounded_historical_scan_pages_repository_release_and_assets() -> None:
    repository_url = "https://api.github.com/repositories?per_page=2&since=100"
    releases_url = "https://api.github.com/repos/lab/model/releases?per_page=2&page=1"
    assets_first = "https://api.github.com/repos/lab/model/releases/501/assets?per_page=100&page=1"
    assets_second = "https://api.github.com/repos/lab/model/releases/501/assets?per_page=100&page=2"
    client = RouteClient(
        {
            repository_url: (
                [{"id": 101, "full_name": "lab/model", "private": False}],
                {},
            ),
            releases_url: (
                [
                    {
                        "id": 501,
                        "tag_name": "v2",
                        "name": "Model release",
                        "body": "Neural model checkpoint",
                        "html_url": "https://github.com/lab/model/releases/tag/v2",
                        "published_at": "2022-03-04T00:00:00Z",
                    }
                ],
                {},
            ),
            assets_first: (
                [
                    {
                        "id": 601,
                        "name": "LICENSE.txt",
                        "browser_download_url": "https://github.com/lab/model/releases/download/v2/LICENSE.txt",
                    }
                ],
                {"Link": f'<{assets_second}>; rel="next"'},
            ),
            assets_second: (
                [
                    {
                        "id": 602,
                        "name": "resnet50.safetensors",
                        "browser_download_url": "https://github.com/lab/model/releases/download/v2/resnet50.safetensors",
                        "size": 1024,
                        "content_type": "application/octet-stream",
                    }
                ],
                {},
            ),
        }
    )
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        max_repositories=1,
        page_size=2,
        client=client,
    )

    state: dict[str, Any] = {}
    pages = []
    for _ in range(10):
        page = adapter.fetch_page(state)
        pages.append(page)
        state = dict(page.next_state)
        if page.complete:
            break
    else:
        pytest.fail("bounded scan did not complete")

    candidates = [record for page in pages for record in page.records]
    print(
        "scan debug",
        client.calls,
        [
            (
                p.complete,
                p.next_state.get("current_release"),
                p.next_state.get("asset_next_url"),
                p.next_state.get("release_queue"),
            )
            for p in pages
        ],
    )
    assert [record.source_record_id for record in candidates] == [
        "github-release-asset:101:501:602"
    ]
    assert candidates[0].raw["discovery_basis"] == "github_historical_release_asset_metadata"
    assert candidates[0].models[0].confidence == 0.2
    assert candidates[0].links[0].crawl is False
    assert client.calls == [repository_url, releases_url, assets_first, assets_second]
    assert state["coverage_status"] == "repository_limit_reached"


def test_historical_scan_caps_are_explicit_and_bounded() -> None:
    with pytest.raises(ValueError, match="max_repositories"):
        GitHubHistoricalReleaseAssetsSourceAdapter(
            initial_since=0,
            max_repository_id=10,
            max_repositories=10_001,
        )
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=0,
        max_repository_id=10,
        max_repositories=5,
        page_size=2,
        max_releases_per_repository=20,
        max_assets_per_release=100,
        max_asset_pages_per_release=1,
    )
    assert adapter.max_page_requests == 3 + 5 * (10 + 20)
    assert adapter.max_api_requests == adapter.max_page_requests * 4


def test_named_id_ranges_have_distinct_checkpoints() -> None:
    lower = GitHubHistoricalReleaseAssetsSourceAdapter(
        name="github-history-0-999",
        initial_since=0,
        max_repository_id=999,
        max_repositories=1,
    )
    upper = GitHubHistoricalReleaseAssetsSourceAdapter(
        name="github-history-1000-1999",
        initial_since=1_000,
        max_repository_id=1_999,
        max_repositories=1,
    )

    assert lower.name != upper.name
    assert lower.checkpoint_signature != upper.checkpoint_signature


def test_numeric_cursor_allows_id_gaps_but_rejects_rows_below_lower_cursor() -> None:
    url = "https://api.github.com/repositories?per_page=2&since=100"
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        max_repositories=2,
        page_size=2,
        client=RouteClient(
            {url: ([{"id": 107, "full_name": "lab/after-gap", "private": False}], {})}
        ),
    )

    page = adapter.fetch_page({})

    assert page.next_state["last_repository_id"] == 107
    assert page.next_state["repo_queue"] == [{"id": 107, "full_name": "lab/after-gap"}]

    invalid = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        max_repositories=2,
        page_size=2,
        client=RouteClient(
            {url: ([{"id": 99, "full_name": "lab/before-cursor", "private": False}], {})}
        ),
    )
    with pytest.raises(ValueError, match="IDs did not advance"):
        invalid.fetch_page({})


def test_public_repository_since_link_is_checkpointed_as_text_cursor() -> None:
    first_url = "https://api.github.com/repositories?per_page=1&since=100"
    next_url = "https://api.github.com/repositories?per_page=1&since=107"
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        max_repositories=2,
        page_size=1,
        client=RouteClient(
            {
                first_url: (
                    [{"id": 107, "full_name": "lab/after-gap", "private": False}],
                    {"Link": f'<{next_url}>; rel="next"'},
                )
            }
        ),
    )

    page = adapter.fetch_page({})

    assert page.complete is False
    assert page.next_state["last_repository_id"] == 107
    assert page.next_state["repo_next_url"] == next_url


def test_empty_repository_page_cannot_claim_completion_with_next_cursor() -> None:
    url = "https://api.github.com/repositories?per_page=2&since=100"
    next_url = "https://api.github.com/repositories?per_page=2&since=100"
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        page_size=2,
        client=RouteClient({url: ([], {"Link": f'<{next_url}>; rel="next"'})}),
    )

    with pytest.raises(ValueError, match="next cursor without a row"):
        adapter.fetch_page({})


def test_exact_full_terminal_pages_without_link_are_accepted() -> None:
    repo_url = "https://api.github.com/repositories?per_page=1&since=100"
    releases_url = "https://api.github.com/repos/lab/model/releases?per_page=1&page=1"
    assets_url = "https://api.github.com/repos/lab/model/releases/501/assets?per_page=100&page=1"
    assets = [
        {
            "id": 600 + index,
            "name": ("resnet50.safetensors" if index == 99 else f"readme-{index}.txt"),
            "browser_download_url": (
                "https://github.com/lab/model/releases/download/v2/"
                f"{'resnet50.safetensors' if index == 99 else f'readme-{index}.txt'}"
            ),
        }
        for index in range(100)
    ]
    client = RouteClient(
        {
            repo_url: ([{"id": 101, "full_name": "lab/model", "private": False}], {}),
            releases_url: (
                [
                    {
                        "id": 501,
                        "tag_name": "v2",
                        "name": "Model release",
                        "body": "Neural checkpoint",
                        "html_url": "https://github.com/lab/model/releases/tag/v2",
                        "published_at": "2022-03-04T00:00:00Z",
                    }
                ],
                {},
            ),
            assets_url: (assets, {}),
        }
    )
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        max_repositories=1,
        page_size=1,
        max_assets_per_release=100,
        client=client,
    )

    state: dict[str, Any] = {}
    pages = []
    for _ in range(10):
        page = adapter.fetch_page(state)
        pages.append(page)
        state = dict(page.next_state)
        if page.complete:
            break

    assert page.complete is True
    assert [record.source_record_id for item in pages for record in item.records] == [
        "github-release-asset:101:501:699"
    ]
    assert client.calls == [repo_url, releases_url, assets_url]


def test_release_cap_is_reported_and_scan_advances_to_later_repository() -> None:
    repository_url = "https://api.github.com/repositories?per_page=2&since=100"
    first_releases = "https://api.github.com/repos/lab/first/releases?per_page=2&page=1"
    first_releases_next = "https://api.github.com/repos/lab/first/releases?per_page=2&page=2"
    first_assets = "https://api.github.com/repos/lab/first/releases/501/assets?per_page=100&page=1"
    second_releases = "https://api.github.com/repos/lab/second/releases?per_page=2&page=1"
    client = RouteClient(
        {
            repository_url: (
                [
                    {"id": 101, "full_name": "lab/first", "private": False},
                    {"id": 102, "full_name": "lab/second", "private": False},
                ],
                {},
            ),
            first_releases: (
                [
                    {
                        "id": 501,
                        "tag_name": "v1",
                        "name": "First release",
                        "body": "",
                        "html_url": "https://github.com/lab/first/releases/tag/v1",
                    },
                    {
                        "id": 502,
                        "tag_name": "v2",
                        "name": "Second release",
                        "body": "",
                        "html_url": "https://github.com/lab/first/releases/tag/v2",
                    },
                ],
                {"Link": f'<{first_releases_next}>; rel="next"'},
            ),
            first_releases_next: ([], {}),
            first_assets: ([], {}),
            second_releases: ([], {}),
        }
    )
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        max_repositories=2,
        page_size=2,
        max_releases_per_repository=1,
        client=client,
    )

    state: dict[str, Any] = {}
    for _ in range(20):
        page = adapter.fetch_page(state)
        state = dict(page.next_state)
        if page.complete:
            break
    else:
        pytest.fail("bounded scan did not complete after release cap")

    assert page.complete is True
    assert state["coverage_status"] == "bounded_scan_complete_with_truncation"
    assert state["truncated_release_count"] == 1
    assert second_releases in client.calls
    assert first_releases_next not in client.calls


def test_asset_cap_is_reported_and_later_release_is_still_scanned() -> None:
    repository_url = "https://api.github.com/repositories?per_page=2&since=100"
    releases_url = "https://api.github.com/repos/lab/model/releases?per_page=2&page=1"
    first_assets = "https://api.github.com/repos/lab/model/releases/501/assets?per_page=100&page=1"
    second_assets = "https://api.github.com/repos/lab/model/releases/502/assets?per_page=100&page=1"

    def asset(asset_id: int, name: str) -> dict[str, Any]:
        return {
            "id": asset_id,
            "name": name,
            "browser_download_url": (
                f"https://github.com/lab/model/releases/download/v{asset_id}/{name}"
            ),
        }

    client = RouteClient(
        {
            repository_url: ([{"id": 101, "full_name": "lab/model", "private": False}], {}),
            releases_url: (
                [
                    {
                        "id": 501,
                        "tag_name": "v1",
                        "name": "Model release",
                        "body": "Neural checkpoint",
                        "html_url": "https://github.com/lab/model/releases/tag/v1",
                    },
                    {
                        "id": 502,
                        "tag_name": "v2",
                        "name": "Model release",
                        "body": "Neural checkpoint",
                        "html_url": "https://github.com/lab/model/releases/tag/v2",
                    },
                ],
                {},
            ),
            first_assets: (
                [asset(601, "resnet50.safetensors"), asset(602, "bert-base.safetensors")],
                {},
            ),
            second_assets: ([asset(603, "vit-base.safetensors")], {}),
        }
    )
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        max_repositories=1,
        page_size=2,
        max_releases_per_repository=2,
        max_assets_per_release=1,
        client=client,
    )

    state: dict[str, Any] = {}
    records: list[str] = []
    for _ in range(15):
        page = adapter.fetch_page(state)
        state = dict(page.next_state)
        records.extend(record.source_record_id for record in page.records)
        if page.complete:
            break
    else:
        pytest.fail("bounded scan did not advance past asset cap")

    assert page.complete is True
    assert state["truncated_asset_release_count"] == 1
    assert records == [
        "github-release-asset:101:501:601",
        "github-release-asset:101:502:603",
    ]
    assert second_assets in client.calls
