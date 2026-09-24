from __future__ import annotations

from datetime import UTC, datetime

from modelome.sources.catalog import load_sources
from modelome.sources.github_historical_release_assets import (
    GitHubCuratedReleaseAssetsSourceAdapter,
)


def _fixed_clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_load_sources_routes_curated_github_assets_without_token(tmp_path) -> None:
    source_config = tmp_path / "github-curated.toml"
    source_config.write_text(
        '''
[[source]]
name = "github-phhofm-models-release-assets"
adapter = "github_curated_release_assets"
enabled = true
repository_names = ["Phhofm/models"]
page_size = 100
max_releases_per_repository = 120
max_assets_per_release = 100
max_release_pages_per_repository = 2
max_asset_pages_per_release = 1
''',
        encoding="utf-8",
    )

    class FakeClient:
        pass

    shared_client = FakeClient()
    source = load_sources(
        source_config,
        client=shared_client,
        clock=_fixed_clock,
        environ={},
    )["github-phhofm-models-release-assets"]

    assert isinstance(source, GitHubCuratedReleaseAssetsSourceAdapter)
    assert source.repository_names == ("Phhofm/models",)
    assert source.page_size == 100
    assert source.max_releases_per_repository == 120
    assert source.max_assets_per_release == 100
    assert source.max_release_pages_per_repository == 2
    assert source.max_asset_pages_per_release == 1
    assert source.token is None
    assert source.client is shared_client
    assert callable(source.fetch_page)


def test_load_sources_uses_only_supplied_environment_for_shared_github_client(
    tmp_path, monkeypatch
) -> None:
    source_config = tmp_path / "github-curated-env.toml"
    source_config.write_text(
        '\n'.join(
            (
                "[[source]]",
                'name = "github-env-test"',
                'adapter = "github_curated_release_assets"',
                "enabled = true",
                'repository_names = ["Phhofm/models"]',
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-secret")

    isolated = load_sources(source_config, environ={})["github-env-test"]
    supplied = load_sources(
        source_config,
        environ={"GITHUB_TOKEN": "provided-secret"},
    )["github-env-test"]
    injected_client = object()
    injected = load_sources(
        source_config,
        client=injected_client,
        environ={"GITHUB_TOKEN": "provided-secret"},
    )["github-env-test"]

    assert isolated.client._github_token is None
    assert supplied.client._github_token == "provided-secret"
    assert injected.client is injected_client
