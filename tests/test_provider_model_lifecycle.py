from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source


class FixtureClient:
    def __init__(self, body: bytes, url: str) -> None:
        self.body = body
        self.url = url

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=self.body,
            url=url,
        )


def _source_config() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "config/proposals/provider_model_lifecycle.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)["source"][0]


def test_public_foundry_retirement_table_retains_model_versions_and_status_rows() -> None:
    config = _source_config()
    html = b"""
    <html><body>
      <table><thead><tr><th>Model</th><th>Version</th><th>Lifecycle</th>
      <th>Retirement date</th><th>Replacement</th></tr></thead><tbody>
        <tr><td>gpt-4o</td><td>2024-05-13</td><td>Deprecated</td>
        <td>2026-10-01</td><td>gpt-5.1</td></tr>
        <tr><td>gpt-4o</td><td>2024-08-06</td><td>Deprecated</td>
        <td>2027-04-14</td><td>gpt-5.1</td></tr>
        <tr><td>gpt-5.1</td><td>2025-11-13</td><td>GA</td>
        <td>2027-05-15</td><td>&mdash;</td></tr>
      </tbody></table>
      <table><thead><tr><th>Model</th><th>Description</th></tr></thead>
      <tbody><tr><td>not-a-lifecycle-row</td><td>Ignored</td></tr></tbody></table>
    </body></html>
    """
    adapter = create_source(
        config,
        client=FixtureClient(html, config["url"]),
        environ={},
    )

    page = adapter.fetch_page({})

    assert len(page.records) == 1
    catalog = page.records[0]
    assert {model.identifiers[0].value for model in catalog.models} == {
        "gpt-4o",
        "gpt-5.1",
    }
    entries = {entry["identity"]: entry for entry in catalog.raw["entries"]}
    gpt4o_rows = entries["gpt-4o"]["occurrences"]
    assert [row["cells"] for row in gpt4o_rows] == [
        ["gpt-4o", "2024-05-13", "Deprecated", "2026-10-01", "gpt-5.1"],
        ["gpt-4o", "2024-08-06", "Deprecated", "2027-04-14", "gpt-5.1"],
    ]
    assert entries["gpt-5.1"]["occurrences"][0]["cells"] == [
        "gpt-5.1",
        "2025-11-13",
        "GA",
        "2027-05-15",
        "—",
    ]


def test_foundry_model_lifecycle_proposal_remains_disabled_and_is_not_project_api() -> None:
    config = _source_config()

    assert config["enabled"] is False
    assert config["url"] == (
        "https://learn.microsoft.com/en-us/azure/foundry/openai/"
        "concepts/model-retirement-schedule"
    )


def test_anthropic_lifecycle_table_includes_exact_historical_api_model_ids() -> None:
    path = Path(__file__).parents[1] / "config/proposals/provider_model_lifecycle.toml"
    with path.open("rb") as handle:
        config = tomllib.load(handle)["source"][1]
    html = b"""
    <html><body>
      <table><thead><tr><th>API model name</th><th>Current state</th>
      <th>Deprecated</th><th>Tentative retirement date</th></tr></thead><tbody>
        <tr><td>claude-opus-4-1-20250805</td><td>Retired</td>
        <td>June 5, 2026</td><td>August 5, 2026</td></tr>
        <tr><td>claude-sonnet-4-6</td><td>Active</td><td>N/A</td>
        <td>Not sooner than February 17, 2027</td></tr>
      </tbody></table>
      <table><thead><tr><th>Retirement date</th><th>Deprecated model</th>
      <th>Recommended replacement</th></tr></thead><tbody>
        <tr><td>July 21, 2025</td><td><code>claude-2.0</code></td>
        <td><code>claude-opus-4-8</code></td></tr>
      </tbody></table>
    </body></html>
    """
    adapter = create_source(
        config,
        client=FixtureClient(html, config["url"]),
        environ={},
    )

    page = adapter.fetch_page({})

    assert len(page.records) == 1
    catalog = page.records[0]
    assert {model.identifiers[0].value for model in catalog.models} == {
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-6",
        "claude-2.0",
    }
    entries = {entry["identity"]: entry for entry in catalog.raw["entries"]}
    assert entries["claude-2.0"]["occurrences"][0]["cells"] == [
        "July 21, 2025",
        "claude-2.0",
        "claude-opus-4-8",
    ]
