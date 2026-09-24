from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/google_gemini_15_retired_lifecycle.toml"


class _FixtureClient:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert url == "https://ai.google.dev/gemini-api/docs/changelog"
        assert params is None
        assert "text/html" in (headers or {}).get("Accept", "")
        body = b"""
        <html><body>
          <h2>September 29, 2025</h2>
          <p>The following Gemini 1.5 models are now shut down:</p>
          <ul>
            <li><code>gemini-1.5-pro</code></li>
            <li><code>gemini-1.5-flash-8b</code></li>
            <li><code>gemini-1.5-flash</code></li>
          </ul>
          <p>Replacement: <code>gemini-2.5-flash</code></p>
        </body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_gemini_changelog_captures_only_the_three_retired_exact_ids() -> None:
    with _PROPOSAL.open("rb") as handle:
        source = tomllib.load(handle)["source"][0]
    adapter = create_source(source, client=_FixtureClient(), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})
    models = [model for record in page.records for model in record.models]

    assert [model.name for model in models] == [
        "gemini-1.5-pro",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash",
    ]
    assert [identifier.value for model in models for identifier in model.identifiers] == [
        model.name for model in models
    ]
    assert "retired-on:2025-09-29" in source["entry_tags"]
