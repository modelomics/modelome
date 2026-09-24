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
    path = Path(__file__).parents[1] / "config/proposals/openai_model_snapshot_history.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)["source"][0]


def test_public_openai_deprecations_capture_exact_snapshot_ids_and_replacements() -> None:
    config = _source_config()
    html = b"""
    <html><body>
      <table><thead><tr><th>Shutdown date</th><th>Model snapshot</th>
      <th>Substitute model</th></tr></thead><tbody>
        <tr><td>October 23, 2026</td><td><code>gpt-4o-2024-05-13</code></td>
        <td>gpt-5.6-sol</td></tr>
        <tr><td>October 23, 2026</td><td><code>o3-mini-2025-01-31</code></td>
        <td>gpt-5.6-sol</td></tr>
      </tbody></table>
      <table><thead><tr><th>Shutdown date</th><th>Deprecated model</th>
      <th>Deprecated model price</th><th>Recommended replacement</th></tr></thead>
      <tbody><tr><td>June 15, 2025</td><td><code>gpt-4-32k-0613</code></td>
      <td>$60 / 1M tokens</td><td>gpt-4o</td></tr></tbody></table>
      <table><thead><tr><th>Shutdown date</th><th>Model / system</th>
      <th>Recommended replacement</th></tr></thead><tbody>
        <tr><td>September 24, 2026</td><td>Videos API</td><td>&mdash;</td></tr>
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
        "gpt-4o-2024-05-13",
        "o3-mini-2025-01-31",
        "gpt-4-32k-0613",
    }
    entries = {entry["identity"]: entry for entry in catalog.raw["entries"]}
    assert entries["o3-mini-2025-01-31"]["occurrences"][0]["cells"] == [
        "October 23, 2026",
        "o3-mini-2025-01-31",
        "gpt-5.6-sol",
    ]
    assert entries["gpt-4-32k-0613"]["occurrences"][0]["cells"] == [
        "June 15, 2025",
        "gpt-4-32k-0613",
        "$60 / 1M tokens",
        "gpt-4o",
    ]
    assert "Videos API" not in entries


def test_openai_snapshot_history_is_disabled_and_uses_public_first_party_docs() -> None:
    config = _source_config()

    assert config["enabled"] is False
    assert config["url"] == "https://developers.openai.com/api/docs/deprecations"
