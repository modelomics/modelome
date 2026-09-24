from __future__ import annotations

import tomllib
from pathlib import Path

from modelome.http import HttpResponse
from modelome.models import ModelStatus
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter


class FixtureClient:
    def __init__(self, url: str) -> None:
        self.url = url

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        body = b"""
        <html><body><table>
          <thead><tr><th>Model provider</th><th>Model name</th><th>Model ID</th>
          <th>Regions</th><th>Legacy date</th><th>EOL date</th>
          <th>Public extended access start date</th></tr></thead>
          <tbody>
            <tr><td>Anthropic</td><td>Claude Sonnet 4</td>
              <td><code>anthropic.claude-sonnet-4-20250514-v1:0</code></td>
              <td>us-east-1, eu-west-1</td><td>April 14, 2026</td>
              <td>October 14, 2026</td><td>July 14, 2026</td></tr>
            <tr><td>Amazon</td><td>Nova Reel</td>
              <td><code>amazon.nova-reel-v1:1</code></td><td>us-east-1</td>
              <td>March 30, 2026</td><td>September 30, 2026</td><td>&mdash;</td></tr>
          </tbody>
        </table></body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_bedrock_lifecycle_table_preserves_exact_hosted_ids_and_dates() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/aws_bedrock_lifecycle_models.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    adapter = create_source(config, client=FixtureClient(config["url"]), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert len(page.records) == 1
    catalog = page.records[0]
    assert {model.identifiers[0].value for model in catalog.models} == {
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "amazon.nova-reel-v1:1",
    }
    assert all(model.status is ModelStatus.DOCUMENTED for model in catalog.models)
    entries = {entry["identity"]: entry for entry in catalog.raw["entries"]}
    assert entries["anthropic.claude-sonnet-4-20250514-v1:0"]["name"] == (
        "Claude Sonnet 4"
    )
    assert entries["anthropic.claude-sonnet-4-20250514-v1:0"]["occurrences"][0][
        "cells"
    ][4:] == ["April 14, 2026", "October 14, 2026", "July 14, 2026"]
    assert entries["amazon.nova-reel-v1:1"]["occurrences"][0]["cells"][5] == (
        "September 30, 2026"
    )
