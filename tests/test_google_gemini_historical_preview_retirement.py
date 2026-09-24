from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = (
    Path(__file__).parents[1] / "config/proposals/google_gemini_historical_preview_retirement.toml"
)


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
          <p>Deprecated, shutdown November 14:
            <code>gemini-2.0-flash-exp-image-generation</code>
          </p>
          <p>Separate catalog row: <code>gemini-2.0-flash-preview-image-generation</code></p>
          <p>Unrelated model: <code>gemini-2.0-flash-exp</code></p>
        </body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_gemini_changelog_captures_only_retired_exact_snapshot_id() -> None:
    with _PROPOSAL.open("rb") as handle:
        source = tomllib.load(handle)["source"][0]
    adapter = create_source(source, client=_FixtureClient(), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})
    models = [model for record in page.records for model in record.models]

    assert [model.name for model in models] == ["gemini-2.0-flash-exp-image-generation"]
    assert [identifier.value for model in models for identifier in model.identifiers] == [
        "gemini-2.0-flash-exp-image-generation"
    ]
    assert "retired-on:2025-11-14" in source["entry_tags"]
