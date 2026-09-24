"""Bounded metadata scan of GitHub public repositories and release assets.

This adapter supplements GH Archive: it can discover assets on old releases and
assets added after the original ReleaseEvent. It calls only GitHub's public REST
API and never fetches release bytes. The scan is deliberately finite: callers
choose an exclusive lower repository-ID cursor, an inclusive upper ID, and a
maximum repository count.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import parse_qsl, quote, urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, SourcePage, SourceRecord
from modelome.normalize import content_hash
from modelome.sources.github_release_assets import project_github_release_assets

_API = "https://api.github.com"
_PAGE_SIZE = 100
_API_VERSION = "2026-03-10"
_LINK_RE = re.compile(r"<([^>]+)>\s*((?:;\s*[^,]+)*)")
_REL_RE = re.compile(r'\brel\s*=\s*(?:"([^"]+)"|([^;\s,]+))', re.I)


@dataclass(frozen=True, slots=True)
class GitHubRepositoryIdRangePlan:
    """One contiguous, disjoint slice compatible with the scanner options."""

    name: str
    initial_since: int
    max_repository_id: int
    max_repositories: int
    page_size: int
    max_releases_per_repository: int
    max_assets_per_release: int
    max_release_pages_per_repository: int
    max_asset_pages_per_release: int
    max_http_attempts: int
    max_page_requests: int
    max_api_requests: int

    def adapter_kwargs(self) -> dict[str, Any]:
        """Return options that instantiate this exact bounded slice."""

        return {
            "name": self.name,
            "initial_since": self.initial_since,
            "max_repository_id": self.max_repository_id,
            "max_repositories": self.max_repositories,
            "page_size": self.page_size,
            "max_releases_per_repository": self.max_releases_per_repository,
            "max_assets_per_release": self.max_assets_per_release,
            "max_release_pages_per_repository": self.max_release_pages_per_repository,
            "max_asset_pages_per_release": self.max_asset_pages_per_release,
            "http_attempts": self.max_http_attempts,
        }


def plan_github_repository_id_ranges(
    *,
    initial_since: int,
    max_repository_id: int,
    shard_count: int,
    source_name_prefix: str = "github-historical-release-assets",
    page_size: int = _PAGE_SIZE,
    max_releases_per_repository: int = 100,
    max_assets_per_release: int = 1_000,
    max_release_pages_per_repository: int = 10,
    max_asset_pages_per_release: int = 10,
    max_http_attempts: int = 4,
    max_api_requests_per_range: int | None = None,
) -> tuple[GitHubRepositoryIdRangePlan, ...]:
    """Partition ``(initial_since, max_repository_id]`` into safe scan slices.

    Each slice uses the previous slice's inclusive upper ID as its exclusive
    ``initial_since``. A slice's repository count cap equals its numeric width,
    so missing IDs cannot consume the cap and make the slice stop early. Widths
    above the adapter's 10,000-repository bound require more shards.
    """

    lower = _integer(initial_since, "initial_since", minimum=0)
    upper = _integer(max_repository_id, "max_repository_id", minimum=1)
    if upper <= lower:
        raise ValueError("max_repository_id must be greater than initial_since")
    shard_count = _integer(shard_count, "shard_count", minimum=1)
    span = upper - lower
    if shard_count > span:
        raise ValueError("shard_count cannot exceed the number of IDs in the range")
    if not isinstance(source_name_prefix, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", source_name_prefix.strip()
    ):
        raise ValueError("source_name_prefix must be a simple source-name token")
    prefix = source_name_prefix.strip()
    page_size = _integer(page_size, "page_size", minimum=1, maximum=_PAGE_SIZE)
    max_releases = _integer(
        max_releases_per_repository,
        "max_releases_per_repository",
        minimum=1,
        maximum=1_000,
    )
    max_assets = _integer(
        max_assets_per_release,
        "max_assets_per_release",
        minimum=1,
        maximum=1_000,
    )
    release_pages = _integer(
        max_release_pages_per_repository,
        "max_release_pages_per_repository",
        minimum=1,
        maximum=100,
    )
    asset_pages = _integer(
        max_asset_pages_per_release,
        "max_asset_pages_per_release",
        minimum=1,
        maximum=100,
    )
    attempts = _integer(max_http_attempts, "max_http_attempts", minimum=1, maximum=10)
    request_budget = (
        None
        if max_api_requests_per_range is None
        else _integer(
            max_api_requests_per_range,
            "max_api_requests_per_range",
            minimum=1,
        )
    )

    base_width, remainder = divmod(span, shard_count)
    cursor = lower
    plans: list[GitHubRepositoryIdRangePlan] = []
    for index in range(shard_count):
        width = base_width + (1 if index < remainder else 0)
        if width > 10_000:
            raise ValueError("each range must cover at most 10000 IDs; increase shard_count")
        end = cursor + width
        page_requests = math.ceil(width / page_size) + width * (
            release_pages + max_releases * asset_pages
        )
        api_requests = page_requests * attempts
        if request_budget is not None and api_requests > request_budget:
            raise ValueError(
                f"planned range requires up to {api_requests} API requests, above "
                f"max_api_requests_per_range {request_budget}; increase shard_count "
                "or lower the scan caps"
            )
        plans.append(
            GitHubRepositoryIdRangePlan(
                name=f"{prefix}-{cursor + 1}-{end}",
                initial_since=cursor,
                max_repository_id=end,
                max_repositories=width,
                page_size=page_size,
                max_releases_per_repository=max_releases,
                max_assets_per_release=max_assets,
                max_release_pages_per_repository=release_pages,
                max_asset_pages_per_release=asset_pages,
                max_http_attempts=attempts,
                max_page_requests=page_requests,
                max_api_requests=api_requests,
            )
        )
        cursor = end
    return tuple(plans)


def plan_github_repository_id_ranges_with_budget(
    *,
    initial_since: int,
    max_repository_id: int,
    shard_count: int,
    max_api_requests_per_range: int,
    source_name_prefix: str = "github-historical-release-assets",
    page_size: int = _PAGE_SIZE,
    max_releases_per_repository: int = 1_000,
    max_assets_per_release: int = 1_000,
    max_release_pages_per_repository: int = 10,
    max_asset_pages_per_release: int = 10,
    max_http_attempts: int = 4,
) -> tuple[GitHubRepositoryIdRangePlan, ...]:
    """Maximize releases scanned per ID range under a retry-inclusive API budget.

    The requested release and asset counts are ceilings. For each numeric
    repository shard, this chooses the largest release cap that fits the
    conservative request estimate while retaining the configured release and
    asset page limits. It raises if the budget cannot cover even one release
    per repository in the shard. `adapter_kwargs()` on each result carries the
    selected cap and retry count.
    """

    budget = _integer(
        max_api_requests_per_range,
        "max_api_requests_per_range",
        minimum=1,
    )
    plans = plan_github_repository_id_ranges(
        initial_since=initial_since,
        max_repository_id=max_repository_id,
        shard_count=shard_count,
        source_name_prefix=source_name_prefix,
        page_size=page_size,
        max_releases_per_repository=max_releases_per_repository,
        max_assets_per_release=max_assets_per_release,
        max_release_pages_per_repository=max_release_pages_per_repository,
        max_asset_pages_per_release=max_asset_pages_per_release,
        max_http_attempts=max_http_attempts,
    )
    result: list[GitHubRepositoryIdRangePlan] = []
    for plan in plans:
        page_budget = budget // plan.max_http_attempts
        repository_pages = math.ceil(plan.max_repositories / plan.page_size)
        fixed_pages = repository_pages + (
            plan.max_repositories * plan.max_release_pages_per_repository
        )
        pages_per_release = plan.max_repositories * plan.max_asset_pages_per_release
        available_releases = (page_budget - fixed_pages) // pages_per_release
        if available_releases < 1:
            minimum_api_requests = (fixed_pages + pages_per_release) * plan.max_http_attempts
            raise ValueError(
                f"API request budget {budget} cannot cover one release per repository; "
                f"minimum is {minimum_api_requests} requests for this range"
            )
        release_page_capacity = plan.page_size * plan.max_release_pages_per_repository
        selected_releases = min(
            plan.max_releases_per_repository,
            release_page_capacity,
            available_releases,
        )
        page_requests = fixed_pages + pages_per_release * selected_releases
        result.append(
            replace(
                plan,
                max_releases_per_repository=selected_releases,
                max_page_requests=page_requests,
                max_api_requests=page_requests * plan.max_http_attempts,
            )
        )
    return tuple(result)


class GitHubHistoricalReleaseAssetsSourceAdapter:
    """Scan a bounded public-repository ID range for explicit release assets.

    One ``fetch_page`` call makes exactly one API request. Checkpoint state holds
    the repository, release, and asset cursors plus bounded queues (at most one
    GitHub API page each). The finite API-call ceiling is
    ``ceil(max_repositories / page_size) + max_repositories *
    (max_release_pages_per_repository + max_releases_per_repository *
    max_asset_pages_per_release)`` endpoint requests. ``max_api_requests`` also
    includes the retry-attempt ceiling of the default HTTP client. Page caps are
    separate from item caps because an API page can contain fewer rows than
    requested.
    """

    name = "github-historical-release-assets"
    disable_derived_extraction = True
    coverage_limitation = (
        "Scans only public repositories with IDs greater than initial_since and "
        "at most max_repository_id, "
        "up to max_repositories in GitHub's public-repository order. It inspects "
        "release pages and their explicit asset metadata only; it does not scan "
        "repository files, download assets, or prove that a release remains public. "
        "A range ending or repository cap is completion of the configured bounded "
        "scan, not a global historical census."
    )

    def __init__(
        self,
        *,
        name: str = "github-historical-release-assets",
        initial_since: int,
        max_repository_id: int,
        max_repositories: int = 10,
        page_size: int = _PAGE_SIZE,
        max_releases_per_repository: int = 100,
        max_assets_per_release: int = 1_000,
        max_release_pages_per_repository: int = 10,
        max_asset_pages_per_release: int = 10,
        http_attempts: int = 4,
        token: str | None = None,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be nonempty text")
        self.name = name.strip()
        self.initial_since = _integer(initial_since, "initial_since", minimum=0)
        self.max_repository_id = _integer(max_repository_id, "max_repository_id", minimum=1)
        if self.max_repository_id < self.initial_since:
            raise ValueError("max_repository_id must be >= initial_since")
        self.max_repositories = _integer(
            max_repositories, "max_repositories", minimum=1, maximum=10_000
        )
        self.page_size = _integer(page_size, "page_size", minimum=1, maximum=_PAGE_SIZE)
        self.max_releases_per_repository = _integer(
            max_releases_per_repository,
            "max_releases_per_repository",
            minimum=1,
            maximum=1_000,
        )
        self.max_assets_per_release = _integer(
            max_assets_per_release,
            "max_assets_per_release",
            minimum=1,
            maximum=1_000,
        )
        self.max_release_pages_per_repository = _integer(
            max_release_pages_per_repository,
            "max_release_pages_per_repository",
            minimum=1,
            maximum=100,
        )
        self.max_asset_pages_per_release = _integer(
            max_asset_pages_per_release,
            "max_asset_pages_per_release",
            minimum=1,
            maximum=100,
        )
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise ValueError("token must be nonempty text when provided")
        self.token = token.strip() if token is not None else None
        configured_http_attempts = _integer(http_attempts, "http_attempts", minimum=1, maximum=10)
        self.client = (
            client if client is not None else HttpClient(attempts=configured_http_attempts)
        )
        self.max_page_requests = math.ceil(self.max_repositories / self.page_size) + (
            self.max_repositories
            * (
                self.max_release_pages_per_repository
                + self.max_releases_per_repository * self.max_asset_pages_per_release
            )
        )
        # HttpClient retries transient/rate-limit responses without committing
        # adapter state. Custom clients are assumed single-attempt unless they
        # expose their retry-attempt count.
        self.max_http_attempts = _integer(
            getattr(self.client, "attempts", configured_http_attempts),
            "HTTP client attempts",
            minimum=1,
            maximum=10,
        )
        self.max_api_requests = self.max_page_requests * self.max_http_attempts
        self.checkpoint_signature = content_hash(
            {
                "adapter": "github-historical-release-assets-v1",
                "name": self.name,
                "initial_since": self.initial_since,
                "max_repository_id": self.max_repository_id,
                "max_repositories": self.max_repositories,
                "page_size": self.page_size,
                "max_releases_per_repository": self.max_releases_per_repository,
                "max_assets_per_release": self.max_assets_per_release,
                "max_release_pages_per_repository": self.max_release_pages_per_repository,
                "max_asset_pages_per_release": self.max_asset_pages_per_release,
                "max_http_attempts": self.max_http_attempts,
                "max_api_requests": self.max_api_requests,
                "authenticated": bool(self.token),
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if not isinstance(state, Mapping):
            raise TypeError("github-historical-release-assets: state must be a mapping")
        current_repo = _repo_state(state.get("current_repo"))
        repo_queue = _repo_queue(state.get("repo_queue", []))
        repo_url = _optional_text(state.get("repo_next_url"))
        repo_started = _boolean(state.get("repo_started", False), "repo_started")
        repos_seen = _integer(state.get("repos_seen", 0), "repos_seen", minimum=0)
        current_release = _release_state(state.get("current_release"))
        release_queue = _release_queue(state.get("release_queue", []))
        release_url = _optional_text(state.get("release_next_url"))
        release_page = _integer(state.get("release_page", 1), "release_page", minimum=1)
        releases_seen = _integer(state.get("releases_seen", 0), "releases_seen", minimum=0)
        asset_url = _optional_text(state.get("asset_next_url"))
        asset_page = _integer(state.get("asset_page", 1), "asset_page", minimum=1)
        assets_seen = _integer(state.get("assets_seen", 0), "assets_seen", minimum=0)
        truncated_release_count = _integer(
            state.get("truncated_release_count", 0),
            "truncated_release_count",
            minimum=0,
        )
        truncated_asset_release_count = _integer(
            state.get("truncated_asset_release_count", 0),
            "truncated_asset_release_count",
            minimum=0,
        )

        if current_release is not None:
            return self._asset_page(
                state,
                current_repo=current_repo,
                repo_queue=repo_queue,
                repo_url=repo_url,
                repo_started=repo_started,
                repos_seen=repos_seen,
                current_release=current_release,
                release_queue=release_queue,
                release_url=release_url,
                release_page=release_page,
                releases_seen=releases_seen,
                asset_url=asset_url,
                asset_page=asset_page,
                assets_seen=assets_seen,
                truncated_release_count=truncated_release_count,
                truncated_asset_release_count=truncated_asset_release_count,
            )

        if release_queue:
            current_release, release_queue = release_queue[0], release_queue[1:]
            return self._state_page(
                state,
                current_repo=current_repo,
                repo_queue=repo_queue,
                repo_url=repo_url,
                repo_started=repo_started,
                repos_seen=repos_seen,
                current_release=current_release,
                release_queue=release_queue,
                release_url=release_url,
                release_page=release_page,
                releases_seen=releases_seen,
                truncated_release_count=truncated_release_count,
                truncated_asset_release_count=truncated_asset_release_count,
                asset_next_url=None,
                asset_page=1,
                assets_seen=0,
            )

        if current_repo is not None:
            return self._release_page(
                state,
                current_repo=current_repo,
                repo_queue=repo_queue,
                repo_url=repo_url,
                repo_started=repo_started,
                repos_seen=repos_seen,
                release_url=release_url,
                release_page=release_page,
                releases_seen=releases_seen,
                truncated_release_count=truncated_release_count,
                truncated_asset_release_count=truncated_asset_release_count,
            )

        if repo_queue:
            current_repo, repo_queue = repo_queue[0], repo_queue[1:]
            return self._state_page(
                state,
                current_repo=current_repo,
                repo_queue=repo_queue,
                repo_url=repo_url,
                repo_started=repo_started,
                repos_seen=repos_seen,
                release_queue=[],
                release_next_url=None,
                release_page=1,
                releases_seen=0,
                current_release=None,
                asset_next_url=None,
                asset_page=1,
                assets_seen=0,
                truncated_release_count=truncated_release_count,
                truncated_asset_release_count=truncated_asset_release_count,
            )

        if repos_seen >= self.max_repositories or (repo_started and not repo_url):
            return self._complete_page(
                state,
                repos_seen=repos_seen,
                coverage_status=(
                    "bounded_scan_complete_with_truncation"
                    if truncated_release_count or truncated_asset_release_count
                    else "repository_limit_reached"
                    if repos_seen >= self.max_repositories
                    else "repository_id_range_exhausted"
                ),
                truncated_release_count=truncated_release_count,
                truncated_asset_release_count=truncated_asset_release_count,
            )
        return self._repository_page(
            state,
            repo_url=repo_url,
            repo_started=repo_started,
            repos_seen=repos_seen,
            truncated_release_count=truncated_release_count,
            truncated_asset_release_count=truncated_asset_release_count,
        )

    def _repository_page(
        self,
        state: Mapping[str, Any],
        *,
        repo_url: str | None,
        repo_started: bool,
        repos_seen: int,
        truncated_release_count: int,
        truncated_asset_release_count: int,
    ) -> SourcePage:
        since = self.initial_since
        if repos_seen:
            since = _integer(
                state.get("last_repository_id"),
                "last_repository_id",
                minimum=1,
            )
        request_url = (
            self._safe_url(repo_url, f"{_API}/repositories")
            if repo_url
            else f"{_API}/repositories?per_page={self.page_size}&since={since}"
        )
        response = self._get(request_url)
        if response.status != 200:
            raise ValueError(f"GitHub public repository listing returned HTTP {response.status}")
        payload = response.json()
        if not _sequence(payload) or len(payload) > self.page_size:
            raise ValueError("GitHub public repository response is not a bounded JSON array")
        queue: list[dict[str, Any]] = []
        last_id = _integer(
            state.get("last_repository_id", self.initial_since),
            "last_repository_id",
            minimum=0,
        )
        range_done = False
        for row in payload:
            if not isinstance(row, Mapping):
                raise ValueError("GitHub public repository row is not an object")
            repo_id = _integer(row.get("id"), "repository id", minimum=1)
            full_name = _repository_name(row.get("full_name"))
            if repo_id <= last_id:
                raise ValueError("GitHub public repository IDs did not advance")
            if row.get("private") is True:
                raise ValueError("GitHub public repository listing returned a private repository")
            if repo_id > self.max_repository_id:
                range_done = True
                break
            if repos_seen + len(queue) >= self.max_repositories:
                break
            queue.append({"id": repo_id, "full_name": full_name})
            last_id = repo_id
        next_link = _link_next(_header(response.headers, "link"))
        if not queue:
            if next_link and not range_done:
                raise ValueError("GitHub repository page supplied a next cursor without a row")
            next_link = None
        elif range_done or repos_seen + len(queue) >= self.max_repositories:
            next_link = None
        elif next_link:
            next_link = self._safe_url(next_link, f"{_API}/repositories")
            cursor = _query_positive_integer(
                dict(parse_qsl(urlsplit(next_link).query)).get("since"),
                "repository next cursor",
            )
            if cursor != last_id:
                raise ValueError("GitHub repository cursor does not match last observed ID")
        next_state = dict(state)
        next_state.update(
            {
                "repo_started": True,
                "repo_next_url": next_link,
                "repo_queue": queue,
                "repos_seen": repos_seen + len(queue),
                "last_repository_id": last_id,
                "coverage_status": ("repository_id_range_exhausted" if range_done else None),
            }
        )
        if not queue and not next_link:
            return self._complete_page(
                state,
                repos_seen=repos_seen,
                coverage_status="repository_id_range_exhausted",
                truncated_release_count=truncated_release_count,
                truncated_asset_release_count=truncated_asset_release_count,
            )
        return SourcePage((), next_state, complete=False, upstream_count=None)

    def _release_page(
        self,
        state: Mapping[str, Any],
        *,
        current_repo: dict[str, Any],
        repo_queue: list[dict[str, Any]],
        repo_url: str | None,
        repo_started: bool,
        repos_seen: int,
        release_url: str | None,
        release_page: int,
        releases_seen: int,
        truncated_release_count: int,
        truncated_asset_release_count: int,
    ) -> SourcePage:
        endpoint = f"{_API}/repos/{quote(current_repo['full_name'], safe='/')}/releases"
        request_url = (
            self._safe_url(release_url, endpoint)
            if release_url
            else f"{endpoint}?per_page={min(self.page_size, _PAGE_SIZE)}&page={release_page}"
        )
        response = self._get(request_url)
        if response.status == 404:
            return self._state_page(
                state,
                current_repo=None,
                repo_queue=repo_queue,
                repo_url=repo_url,
                repo_started=repo_started,
                repos_seen=repos_seen,
                truncated_release_count=truncated_release_count,
                truncated_asset_release_count=truncated_asset_release_count,
            )
        if response.status != 200:
            raise ValueError(f"GitHub releases returned HTTP {response.status}")
        payload = response.json()
        if not _sequence(payload) or len(payload) > self.page_size:
            raise ValueError("GitHub releases response is not a bounded JSON array")
        remaining_releases = self.max_releases_per_repository - releases_seen
        releases = [_release(item) for item in payload[:remaining_releases]]
        next_link = _link_next(_header(response.headers, "link"))
        if next_link:
            next_link = self._safe_url(next_link, endpoint)
            _validate_page_cursor(next_link, release_page, "release")
        release_truncated = bool(
            len(payload) > remaining_releases
            or next_link
            and (
                releases_seen + len(releases) >= self.max_releases_per_repository
                or release_page >= self.max_release_pages_per_repository
            )
        )
        if release_truncated:
            next_link = None
        next_state = dict(state)
        next_state.update(
            {
                "current_repo": current_repo if payload or next_link else None,
                "repo_queue": repo_queue,
                "repo_next_url": repo_url,
                "repo_started": repo_started,
                "repos_seen": repos_seen,
                "release_queue": releases,
                "release_next_url": next_link,
                "release_page": release_page + 1,
                "releases_seen": releases_seen + len(releases),
                "truncated_release_count": truncated_release_count + int(release_truncated),
                "truncated_asset_release_count": truncated_asset_release_count,
                "current_release": None,
                "asset_next_url": None,
                "asset_page": 1,
                "assets_seen": 0,
            }
        )
        if not payload and not next_link:
            next_state["current_repo"] = None
            next_state["releases_seen"] = 0
            next_state["release_page"] = 1
        return SourcePage((), next_state, complete=False, upstream_count=len(payload))

    def _asset_page(
        self,
        state: Mapping[str, Any],
        *,
        current_repo: dict[str, Any] | None,
        repo_queue: list[dict[str, Any]],
        repo_url: str | None,
        repo_started: bool,
        repos_seen: int,
        current_release: dict[str, Any],
        release_queue: list[dict[str, Any]],
        release_url: str | None,
        release_page: int,
        releases_seen: int,
        asset_url: str | None,
        asset_page: int,
        assets_seen: int,
        truncated_release_count: int,
        truncated_asset_release_count: int,
    ) -> SourcePage:
        if current_repo is None:
            raise ValueError("asset cursor has no current repository")
        endpoint = (
            f"{_API}/repos/{quote(current_repo['full_name'], safe='/')}/releases/"
            f"{current_release['id']}/assets"
        )
        request_url = (
            self._safe_url(asset_url, endpoint)
            if asset_url
            else f"{endpoint}?per_page={_PAGE_SIZE}&page={asset_page}"
        )
        response = self._get(request_url)
        if response.status == 404:
            more_releases = bool(release_queue or release_url)
            return self._state_page(
                state,
                current_repo=current_repo if more_releases else None,
                repo_queue=repo_queue,
                repo_url=repo_url,
                repo_started=repo_started,
                repos_seen=repos_seen,
                release_queue=release_queue,
                release_url=release_url,
                release_page=release_page if more_releases else 1,
                releases_seen=releases_seen if more_releases else 0,
                current_release=None,
                asset_next_url=None,
                asset_page=1,
                assets_seen=0,
                truncated_release_count=truncated_release_count,
                truncated_asset_release_count=truncated_asset_release_count,
            )
        if response.status != 200:
            raise ValueError(f"GitHub release assets returned HTTP {response.status}")
        payload = response.json()
        if not _sequence(payload) or len(payload) > _PAGE_SIZE:
            raise ValueError("GitHub release assets response is not a bounded JSON array")
        remaining_assets = self.max_assets_per_release - assets_seen
        page_assets = payload[:remaining_assets]
        projected = self._project_assets(current_repo, current_release, page_assets)
        next_link = _link_next(_header(response.headers, "link"))
        if next_link:
            next_link = self._safe_url(next_link, endpoint)
            _validate_page_cursor(next_link, asset_page, "asset")
        assets_truncated = bool(
            len(payload) > remaining_assets
            or next_link
            and (
                assets_seen + len(page_assets) >= self.max_assets_per_release
                or asset_page >= self.max_asset_pages_per_release
            )
        )
        if assets_truncated:
            next_link = None
        next_state = dict(state)
        next_state.update(
            {
                "current_repo": current_repo,
                "repo_queue": repo_queue,
                "repo_next_url": repo_url,
                "repo_started": repo_started,
                "repos_seen": repos_seen,
                "current_release": current_release if next_link else None,
                "release_queue": release_queue,
                "release_next_url": release_url,
                "release_page": release_page,
                "releases_seen": releases_seen,
                "asset_next_url": next_link,
                "asset_page": asset_page + 1,
                "assets_seen": assets_seen + len(page_assets),
                "truncated_release_count": truncated_release_count,
                "truncated_asset_release_count": truncated_asset_release_count
                + int(assets_truncated),
            }
        )
        if not next_link:
            if release_queue:
                next_state["current_release"] = release_queue[0]
                next_state["release_queue"] = release_queue[1:]
                next_state["asset_next_url"] = None
                next_state["asset_page"] = 1
                next_state["assets_seen"] = 0
            elif release_url:
                next_state["current_release"] = None
            else:
                next_state["current_repo"] = None
                next_state["releases_seen"] = 0
        return SourcePage(projected, next_state, complete=False, upstream_count=len(payload))

    def _project_assets(
        self, repo: Mapping[str, Any], release: Mapping[str, Any], assets: Sequence[Any]
    ) -> tuple[SourceRecord, ...]:
        record = SourceRecord(
            source_record_id=f"github-historical-release:{repo['id']}:{release['id']}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=release["html_url"],
            title=f"{repo['full_name']} {release['tag_name']}",
            raw={
                "record_type": "gharchive_public_event",
                "repository": {"id": repo["id"], "name": repo["full_name"]},
                "event": {
                    "id": f"historical-{repo['id']}-{release['id']}",
                    "type": "ReleaseEvent",
                    "created_at": release.get("published_at") or "",
                    "payload": {
                        "action": "published",
                        "release": {**release, "assets": list(assets)},
                    },
                },
            },
        )
        return tuple(
            _with_historical_scope(candidate)
            for candidate in project_github_release_assets(record, max_assets=_PAGE_SIZE)
        )

    def _get(self, url: str) -> HttpResponse:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": _API_VERSION}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response: HttpResponse = self.client.get(url, headers=headers)
        if not isinstance(response.status, int):
            raise ValueError("GitHub response has invalid status")
        return response

    @staticmethod
    def _safe_url(value: str, endpoint: str) -> str:
        url = urljoin(endpoint, value)
        parts, expected = urlsplit(url), urlsplit(endpoint)
        if (
            parts.scheme != "https"
            or parts.hostname != "api.github.com"
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
            or parts.path.rstrip("/") != expected.path.rstrip("/")
        ):
            raise ValueError("GitHub pagination URL escaped its endpoint")
        return url

    @staticmethod
    def _state_page(state: Mapping[str, Any], **updates: Any) -> SourcePage:
        next_state = dict(state)
        next_state.update(updates)
        return SourcePage((), next_state, complete=False, upstream_count=0)

    @staticmethod
    def _complete_page(
        state: Mapping[str, Any],
        *,
        repos_seen: int,
        coverage_status: str,
        truncated_release_count: int = 0,
        truncated_asset_release_count: int = 0,
    ) -> SourcePage:
        next_state = dict(state)
        next_state.update(
            {
                "repos_seen": repos_seen,
                "coverage_status": coverage_status,
                "complete": True,
                "truncated_release_count": truncated_release_count,
                "truncated_asset_release_count": truncated_asset_release_count,
            }
        )
        return SourcePage((), next_state, complete=True, upstream_count=repos_seen)


def _release(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("GitHub release row is not an object")
    return {
        "id": _integer(value.get("id"), "release id", minimum=1),
        "tag_name": _text(value.get("tag_name"), "release tag"),
        "name": _optional_text(value.get("name")) or "",
        "body": _optional_text(value.get("body")) or "",
        "html_url": _https_url(value.get("html_url"), "github.com"),
        "published_at": _optional_text(value.get("published_at")) or "",
    }


def _repo_state(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("current_repo must be an object")
    return {
        "id": _integer(value.get("id"), "current repository ID", minimum=1),
        "full_name": _repository_name(value.get("full_name")),
    }


def _repo_queue(value: Any) -> list[dict[str, Any]]:
    if not _sequence(value) or len(value) > _PAGE_SIZE:
        raise ValueError("repo_queue must be a bounded array")
    return [_repo_state(item) for item in value]  # type: ignore[list-item]


def _release_state(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("current_release must be an object")
    return _release(value)


def _release_queue(value: Any) -> list[dict[str, Any]]:
    if not _sequence(value) or len(value) > _PAGE_SIZE:
        raise ValueError("release_queue must be a bounded array")
    return [_release(item) for item in value]


def _with_historical_scope(record: SourceRecord) -> SourceRecord:
    return SourceRecord(
        source_record_id=record.source_record_id,
        kind=record.kind,
        canonical_url=record.canonical_url,
        title=record.title,
        raw={
            **record.raw,
            "discovery_basis": "github_historical_release_asset_metadata",
            "candidate_scope": "github-historical-release-assets",
        },
        text=record.text,
        published_at=record.published_at,
        modified_at=record.modified_at,
        identifiers=record.identifiers,
        links=record.links,
        models=record.models,
        model_relations=record.model_relations,
        releases=record.releases,
        deleted=record.deleted,
    )


def _repository_name(value: Any) -> str:
    text = _text(value, "repository full_name")
    if text.count("/") != 1 or any(not part or part in {".", ".."} for part in text.split("/")):
        raise ValueError("repository full_name is invalid")
    return text


def _https_url(value: Any, host: str) -> str:
    text = _text(value, "URL")
    parts = urlsplit(text)
    if (
        parts.scheme != "https"
        or parts.hostname != host
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"URL must be credential-free HTTPS on {host}")
    return text


def _integer(value: Any, label: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(
            f"{label} must be from {minimum} to {maximum}"
            if maximum
            else f"{label} must be at least {minimum}"
        )
    return value


def _text(value: Any, label: str = "text") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _header(headers: Mapping[str, Any], name: str) -> str:
    return next(
        (_text(value) for key, value in headers.items() if str(key).casefold() == name.casefold()),
        "",
    )


def _link_next(value: str) -> str | None:
    for url, params in _LINK_RE.findall(value):
        relation = _REL_RE.search(params)
        if relation and "next" in (relation.group(1) or relation.group(2) or "").casefold().split():
            return url
    return None


def _validate_page_cursor(url: str, current_page: int, label: str) -> None:
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    value = query.get("page")
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{label} next page must be a positive decimal integer")
    next_page = int(value)
    if next_page < 1:
        raise ValueError(f"{label} next page must be a positive decimal integer")
    if next_page != current_page + 1:
        raise ValueError(f"GitHub {label} cursor did not advance by one page")


def _query_positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"{label} must be a positive decimal integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be a positive decimal integer")
    return result


__all__ = [
    "GitHubHistoricalReleaseAssetsSourceAdapter",
    "GitHubRepositoryIdRangePlan",
    "plan_github_repository_id_ranges",
    "plan_github_repository_id_ranges_with_budget",
]
