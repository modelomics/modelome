from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/ai21_jamba_deprecated_endpoints.toml"


class _FixtureClient:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert url == "https://docs.ai21.com/docs/jamba-foundation-models"
        assert params is None
        assert "text/html" in (headers or {}).get("Accept", "")
        body = b"""
        <html><body>
          <h2>Model Deprecation</h2>
          <p>Jamba Mini 1.7 / 2025-07 / API Endpoint
            <code>jamba-mini-1.7-2025-08</code> / 2026-02-01</p>
          <p>Jamba Large 1.6 / 2025-03 / API Endpoint
            <code>jamba-large-1.6-2025-03</code> / 2025-08-03</p>
          <p>Jamba Mini 1.6 / 2025-03 / API Endpoint
            <code>jamba-mini-1.6-2025-03</code> / 2025-08-03</p>
          <p>Jamba Large 1.5 / 2024-08 / API Endpoint
            <code>jamba-large-1.5-2024-08</code> / 2025-05-06</p>
          <p>Jamba Mini 1.5 / 2024-08 / API Endpoint
            <code>jamba-mini-1.5-2024-08</code> / 2025-05-06</p>
          <p>Current endpoint: <code>jamba-large-1.7-2025-07</code></p>
        </body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_ai21_jamba_deprecation_table_yields_only_exact_deprecated_api_ids() -> None:
    with _PROPOSAL.open("rb") as handle:
        sources = tomllib.load(handle)["source"]
    client = _FixtureClient()
    observed: dict[str, str] = {}

    for source in sources:
        adapter = create_source(source, client=client, environ={})
        assert isinstance(adapter, HtmlCatalogSourceAdapter)
        page = adapter.fetch_page({})
        models = [model for record in page.records for model in record.models]
        expected_date = next(
            tag.removeprefix("deprecated-on:")
            for tag in source["entry_tags"]
            if tag.startswith("deprecated-on:")
        )
        for model in models:
            assert [identifier.value for identifier in model.identifiers] == [model.name]
            observed[model.name] = expected_date

    assert observed == {
        "jamba-mini-1.7-2025-08": "2026-02-01",
        "jamba-large-1.6-2025-03": "2025-08-03",
        "jamba-mini-1.6-2025-03": "2025-08-03",
        "jamba-large-1.5-2024-08": "2025-05-06",
        "jamba-mini-1.5-2024-08": "2025-05-06",
    }
