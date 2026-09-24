from __future__ import annotations

import tomllib
from pathlib import Path

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter


class FixtureClient:
    def __init__(self, url: str) -> None:
        self.url = url

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        html = b"""
        <html><body>
          <h2>Gemini models</h2>
          <table><thead><tr><th>Model</th><th>Release date</th>
          <th>Shutdown date</th><th>Recommended replacement</th></tr></thead>
          <tbody>
            <tr><td><code>gemini-3.1-flash-lite</code></td>
              <td>May 7, 2026</td><td>May 7, 2027</td>
              <td><code>gemini-3.5-flash-lite</code></td></tr>
            <tr><td><code>gemini-2.0-flash</code></td>
              <td>February 5, 2025</td><td>June 1, 2026</td>
              <td><code>gemini-3.6-flash</code></td></tr>
          </tbody></table>
          <h2>Managed agents</h2>
          <table><thead><tr><th>Agent</th><th>Release date</th>
          <th>Shutdown date</th><th>Recommended replacement</th></tr></thead>
          <tbody><tr><td><code>antigravity-preview-09-2026</code></td>
            <td>September 17, 2026</td><td>No shutdown date announced</td><td></td></tr>
          </tbody></table>
        </body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=html,
            url=url,
        )


def test_gemini_lifecycle_tables_capture_exact_models_and_exclude_agents() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/google_gemini_model_lifecycle.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    adapter = create_source(config, client=FixtureClient(config["url"]), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert len(page.records) == 1
    catalog = page.records[0]
    assert {model.identifiers[0].value for model in catalog.models} == {
        "gemini-3.1-flash-lite",
        "gemini-2.0-flash",
    }
    entries = {entry["identity"]: entry for entry in catalog.raw["entries"]}
    assert entries["gemini-3.1-flash-lite"]["occurrences"][0]["cells"] == [
        "gemini-3.1-flash-lite",
        "May 7, 2026",
        "May 7, 2027",
        "gemini-3.5-flash-lite",
    ]
    assert entries["gemini-2.0-flash"]["occurrences"][0]["cells"][2] == (
        "June 1, 2026"
    )
