from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source


class FixtureClient:
    def __init__(self, body: bytes, url: str, content_type: str) -> None:
        self.body = body
        self.url = url
        self.content_type = content_type

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        return HttpResponse(
            status=200,
            headers={"content-type": self.content_type},
            body=self.body,
            url=url,
        )


def _configs() -> list[dict[str, Any]]:
    path = Path(__file__).parents[1] / "config/proposals/vendor_apis_wave6.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)["source"]


def test_xai_detailed_list_preserves_documented_version_alias_and_fingerprint() -> None:
    config = _configs()[0]
    payload = {
        "models": [
            {
                "id": "grok-fixture",
                "version": "2.1",
                "fingerprint": "fp-fixture",
                "aliases": ["grok-fixture-latest"],
                "created": 1776556800,
                "object": "model",
                "owned_by": "xai",
            }
        ]
    }
    adapter = create_source(
        config,
        client=FixtureClient(
            json.dumps(payload).encode(), config["url"], "application/json"
        ),
        environ={"XAI_API_KEY": "fixture-secret"},
    )

    page = adapter.fetch_page({})

    assert page.records[0].identifiers[0].value == "grok-fixture"
    assert page.records[0].raw["version"] == "2.1"
    assert page.records[0].raw["fingerprint"] == "fp-fixture"
    assert page.records[0].raw["aliases"] == ["grok-fixture-latest"]
    assert page.records[0].links[0].crawl is False


def test_groq_deprecation_table_preserves_retired_ids_shutdown_and_replacement() -> None:
    config = _configs()[1]
    html = b"""
    <html><body><table>
      <thead><tr><th>Deprecated Model</th><th>Shutdown Date</th>
      <th>Recommended Replacement Model ID</th></tr></thead>
      <tbody><tr><td><code>groq/compound-mini</code></td><td>09/21/26</td>
      <td>&mdash;</td></tr>
      <tr><td><code>llama-3.1-8b-instant</code></td><td>08/16/26</td>
      <td><code>openai/gpt-oss-20b</code></td></tr></tbody>
    </table></body></html>
    """
    adapter = create_source(
        config,
        client=FixtureClient(html, config["url"], "text/html"),
        environ={},
    )

    page = adapter.fetch_page({})

    assert len(page.records) == 1
    catalog = page.records[0]
    assert {model.identifiers[0].value for model in catalog.models} == {
        "groq/compound-mini",
        "llama-3.1-8b-instant",
    }
    entries = {entry["identity"]: entry for entry in catalog.raw["entries"]}
    assert entries["groq/compound-mini"]["occurrences"][0]["cells"] == [
        "groq/compound-mini",
        "09/21/26",
        "—",
    ]
    assert entries["llama-3.1-8b-instant"]["occurrences"][0]["cells"] == [
        "llama-3.1-8b-instant",
        "08/16/26",
        "openai/gpt-oss-20b",
    ]
