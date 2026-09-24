from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpFailure, HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.github_historical_release_assets import (
    GitHubHistoricalReleaseAssetsSourceAdapter,
    plan_github_repository_id_ranges,
    plan_github_repository_id_ranges_with_budget,
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


class RateLimitOnceClient:
    def __init__(self, url: str, body: list[dict[str, Any]]) -> None:
        self.url = url
        self.body = body
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        if len(self.calls) == 1:
            raise HttpFailure("simulated GitHub 429 after retry budget")
        return HttpResponse(
            status=200,
            headers={},
            body=json.dumps(self.body).encode(),
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


def test_range_planner_makes_disjoint_bounded_slices_with_cost_estimates() -> None:
    plans = plan_github_repository_id_ranges(
        initial_since=10,
        max_repository_id=26,
        shard_count=3,
        source_name_prefix="history-part",
        page_size=4,
        max_releases_per_repository=2,
        max_assets_per_release=3,
        max_release_pages_per_repository=2,
        max_asset_pages_per_release=2,
        max_http_attempts=2,
    )

    assert [(plan.initial_since, plan.max_repository_id) for plan in plans] == [
        (10, 16),
        (16, 21),
        (21, 26),
    ]
    assert [plan.name for plan in plans] == [
        "history-part-11-16",
        "history-part-17-21",
        "history-part-22-26",
    ]
    assert [plan.max_repositories for plan in plans] == [6, 5, 5]
    assert [plan.max_page_requests for plan in plans] == [38, 32, 32]
    assert [plan.max_api_requests for plan in plans] == [76, 64, 64]

    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(**plans[0].adapter_kwargs())
    assert adapter.initial_since == plans[0].initial_since
    assert adapter.max_repository_id == plans[0].max_repository_id
    assert adapter.max_repositories == plans[0].max_repositories
    assert adapter.max_http_attempts == plans[0].max_http_attempts == 2
    assert adapter.max_api_requests == plans[0].max_api_requests


def test_range_planner_supports_documented_page_and_item_caps() -> None:
    plan = plan_github_repository_id_ranges(
        initial_since=10,
        max_repository_id=11,
        shard_count=1,
        max_releases_per_repository=1_000,
        max_assets_per_release=1_000,
        max_release_pages_per_repository=100,
        max_asset_pages_per_release=100,
        max_http_attempts=3,
    )[0]
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(**plan.adapter_kwargs())

    assert adapter.max_releases_per_repository == 1_000
    assert adapter.max_assets_per_release == 1_000
    assert adapter.max_release_pages_per_repository == 100
    assert adapter.max_asset_pages_per_release == 100
    assert adapter.max_http_attempts == 3
    assert adapter.max_api_requests == plan.max_api_requests


def test_planner_bounds_thousand_release_scan_by_retry_inclusive_budget() -> None:
    plan = plan_github_repository_id_ranges(
        initial_since=10,
        max_repository_id=12,
        shard_count=1,
        max_releases_per_repository=1_000,
        max_release_pages_per_repository=10,
        max_asset_pages_per_release=1,
        max_http_attempts=4,
        max_api_requests_per_range=8_084,
    )[0]
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(**plan.adapter_kwargs())

    assert plan.max_releases_per_repository == 1_000
    assert plan.max_page_requests == 2_021
    assert plan.max_api_requests == adapter.max_api_requests == 8_084

    repo_url = "https://api.github.com/repositories?per_page=100&since=10"
    releases_url = "https://api.github.com/repos/lab/model/releases?per_page=100&page=1"
    releases_next = "https://api.github.com/repos/lab/model/releases?per_page=100&page=2"
    release_rows = [
        {
            "id": 1000 + index,
            "tag_name": f"v{index}",
            "name": "Model release",
            "body": "",
            "html_url": f"https://github.com/lab/model/releases/tag/v{index}",
        }
        for index in range(100)
    ]
    bounded_adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        name="github-wider-range",
        initial_since=10,
        max_repository_id=12,
        max_repositories=1,
        max_releases_per_repository=1_000,
        max_release_pages_per_repository=10,
        max_asset_pages_per_release=1,
        client=RouteClient(
            {
                repo_url: ([{"id": 11, "full_name": "lab/model", "private": False}], {}),
                releases_url: (release_rows, {"Link": f'<{releases_next}>; rel="next"'}),
            }
        ),
    )
    repository_page = bounded_adapter.fetch_page({})
    selected_repo = bounded_adapter.fetch_page(repository_page.next_state)
    release_page = bounded_adapter.fetch_page(selected_repo.next_state)
    assert len(release_page.next_state["release_queue"]) == 100
    assert release_page.next_state["release_next_url"] == releases_next

    with pytest.raises(ValueError, match="above max_api_requests_per_range"):
        plan_github_repository_id_ranges(
            initial_since=10,
            max_repository_id=12,
            shard_count=1,
            max_releases_per_repository=200,
            max_release_pages_per_repository=2,
            max_asset_pages_per_release=10,
            max_http_attempts=4,
            max_api_requests_per_range=8_083,
        )


def test_budget_planner_uses_full_or_largest_fitting_release_cap() -> None:
    common = {
        "initial_since": 10,
        "max_repository_id": 12,
        "shard_count": 1,
        "max_releases_per_repository": 1_000,
        "max_assets_per_release": 1_000,
        "max_release_pages_per_repository": 10,
        "max_asset_pages_per_release": 10,
        "max_http_attempts": 4,
    }
    full = plan_github_repository_id_ranges_with_budget(
        **common, max_api_requests_per_range=80_084
    )[0]
    assert full.max_releases_per_repository == 1_000
    assert full.max_page_requests == 20_021
    assert full.max_api_requests == 80_084

    partial = plan_github_repository_id_ranges_with_budget(
        **common, max_api_requests_per_range=20_000
    )[0]
    assert partial.max_releases_per_repository == 248
    assert partial.max_api_requests == 19_924
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(**partial.adapter_kwargs())
    assert adapter.max_releases_per_repository == 248
    assert adapter.max_api_requests == partial.max_api_requests


def test_budget_planner_rejects_budget_below_one_release_per_repository() -> None:
    with pytest.raises(ValueError, match="minimum is 164 requests"):
        plan_github_repository_id_ranges_with_budget(
            initial_since=10,
            max_repository_id=12,
            shard_count=1,
            max_api_requests_per_range=163,
            max_releases_per_repository=1_000,
            max_release_pages_per_repository=10,
            max_asset_pages_per_release=10,
            max_http_attempts=4,
        )


@pytest.mark.parametrize(
    ("initial_since", "max_repository_id", "shard_count"),
    [(3, 3, 1), (0, 10, 11), (0, 20_001, 1)],
)
def test_range_planner_rejects_empty_or_unbounded_slices(
    initial_since: int, max_repository_id: int, shard_count: int
) -> None:
    with pytest.raises(ValueError):
        plan_github_repository_id_ranges(
            initial_since=initial_since,
            max_repository_id=max_repository_id,
            shard_count=shard_count,
        )


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


def test_default_asset_item_cap_matches_ten_documented_pages() -> None:
    repo_url = "https://api.github.com/repositories?per_page=1&since=100"
    releases_url = "https://api.github.com/repos/lab/model/releases?per_page=1&page=1"
    assets_first = "https://api.github.com/repos/lab/model/releases/501/assets?per_page=100&page=1"
    assets_second = "https://api.github.com/repos/lab/model/releases/501/assets?per_page=100&page=2"
    ordinary_assets = [
        {
            "id": 600 + index,
            "name": f"readme-{index}.txt",
            "browser_download_url": f"https://github.com/lab/model/download/readme-{index}.txt",
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
                        "tag_name": "v1",
                        "name": "Model checkpoint release",
                        "body": "Neural model weights",
                        "html_url": "https://github.com/lab/model/releases/tag/v1",
                    }
                ],
                {},
            ),
            assets_first: (ordinary_assets, {"Link": f'<{assets_second}>; rel="next"'}),
            assets_second: (
                [
                    {
                        "id": 700,
                        "name": "resnet50.safetensors",
                        "browser_download_url": (
                            "https://github.com/lab/model/releases/download/v1/resnet50.safetensors"
                        ),
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
        page_size=1,
        client=client,
    )

    state: dict[str, Any] = {}
    records = []
    for _ in range(10):
        page = adapter.fetch_page(state)
        state = dict(page.next_state)
        records.extend(page.records)
        if page.complete:
            break
    else:
        pytest.fail("bounded scan did not complete after the second asset page")

    assert adapter.max_assets_per_release == 1_000
    assert [record.source_record_id for record in records] == ["github-release-asset:101:501:700"]
    assert state["truncated_asset_release_count"] == 0
    assert assets_second in client.calls


def test_default_release_cap_includes_checkpoint_on_eleventh_public_release() -> None:
    repo_url = "https://api.github.com/repositories?per_page=20&since=100"
    releases_url = "https://api.github.com/repos/lab/model/releases?per_page=20&page=1"
    release_rows = [
        {
            "id": 500 + index,
            "tag_name": f"b{index}",
            "name": f"Build {index}",
            "body": "Neural model checkpoint release",
            "html_url": f"https://github.com/lab/model/releases/tag/b{index}",
            "published_at": "2026-01-01T00:00:00Z",
        }
        for index in range(1, 12)
    ]
    client_routes: dict[str, tuple[Any, dict[str, str]]] = {
        repo_url: ([{"id": 101, "full_name": "lab/model", "private": False}], {}),
        releases_url: (release_rows, {}),
    }
    for index in range(1, 12):
        asset_url = (
            f"https://api.github.com/repos/lab/model/releases/{500 + index}"
            "/assets?per_page=100&page=1"
        )
        assets = (
            [
                {
                    "id": 700,
                    "name": "resnet50.safetensors",
                    "browser_download_url": (
                        f"https://github.com/lab/model/releases/download/b{index}/"
                        "resnet50.safetensors"
                    ),
                }
            ]
            if index == 11
            else []
        )
        client_routes[asset_url] = (assets, {})
    client = RouteClient(client_routes)
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        max_repositories=1,
        page_size=20,
        client=client,
    )

    state: dict[str, Any] = {}
    records = []
    for _ in range(30):
        page = adapter.fetch_page(state)
        state = dict(page.next_state)
        records.extend(page.records)
        if page.complete:
            break
    else:
        pytest.fail("bounded scan did not complete after 11 releases")

    assert adapter.max_releases_per_repository == 100
    assert [record.source_record_id for record in records] == ["github-release-asset:101:511:700"]
    assert state["truncated_release_count"] == 0
    assert len(client.calls) == 13


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


def test_deleted_last_release_asset_advances_to_the_next_repository() -> None:
    repository_url = "https://api.github.com/repositories?per_page=2&since=100"
    first_releases = "https://api.github.com/repos/lab/first/releases?per_page=2&page=1"
    deleted_assets = (
        "https://api.github.com/repos/lab/first/releases/501/assets?per_page=100&page=1"
    )
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
                    }
                ],
                {},
            ),
            second_releases: ([], {}),
        }
    )
    # RouteClient emits HTTP 200 for configured entries; override this one
    # response to model a release deleted between the two API requests.
    original_get = client.get

    def get(url: str, *, headers=None) -> HttpResponse:
        if url == deleted_assets:
            client.calls.append(url)
            return HttpResponse(status=404, headers={}, body=b"{}", url=url)
        return original_get(url, headers=headers)

    client.get = get  # type: ignore[method-assign]
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        max_repositories=2,
        page_size=2,
        client=client,
    )

    state: dict[str, Any] = {}
    for _ in range(20):
        page = adapter.fetch_page(state)
        state = dict(page.next_state)
        if page.complete:
            break
    else:
        pytest.fail("scan stalled after the last release disappeared")

    assert page.complete is True
    assert client.calls.count(first_releases) == 1
    assert second_releases in client.calls


def test_rate_limit_failure_leaves_asset_checkpoint_resumable() -> None:
    asset_url = "https://api.github.com/repos/lab/model/releases/501/assets?per_page=100&page=1"
    client = RateLimitOnceClient(
        asset_url,
        [
            {
                "id": 601,
                "name": "resnet50.safetensors",
                "browser_download_url": (
                    "https://github.com/lab/model/releases/download/v1/resnet50.safetensors"
                ),
            }
        ],
    )
    adapter = GitHubHistoricalReleaseAssetsSourceAdapter(
        initial_since=100,
        max_repository_id=200,
        max_repositories=1,
        client=client,
    )
    state: dict[str, Any] = {
        "current_repo": {"id": 101, "full_name": "lab/model"},
        "current_release": {
            "id": 501,
            "tag_name": "v1",
            "name": "Model release",
            "body": "Neural model checkpoint",
            "html_url": "https://github.com/lab/model/releases/tag/v1",
        },
        "repo_queue": [],
        "release_queue": [],
        "repos_seen": 1,
        "asset_page": 1,
        "assets_seen": 0,
    }
    original_state = dict(state)

    with pytest.raises(HttpFailure, match="429"):
        adapter.fetch_page(state)
    assert state == original_state

    resumed = adapter.fetch_page(state)
    assert resumed.records[0].source_record_id == "github-release-asset:101:501:601"
    assert client.calls == [asset_url, asset_url]
