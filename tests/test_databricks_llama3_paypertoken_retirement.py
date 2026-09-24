from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = (
    Path(__file__).parents[1] / "config/proposals/databricks_llama3_paypertoken_retirement.toml"
)


class _FixtureClient:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert url == "https://docs.databricks.com/aws/en/machine-learning/retired-models-policy"
        assert params is None
        assert "text/html" in (headers or {}).get("Accept", "")
        body = b"""
        <html><body>
          <h3>Foundation Model APIs retirements</h3>
          <table><tr><th>Open model</th><th>Retirement date</th><th>Replacement</th></tr>
          <tr><td>Meta Llama 3 (70B)</td>
          <td>Pay-per-token: July 23, 2024 (Meta-Llama-3-70B-Instruct)</td>
          <td>Meta-Llama-4-Maverick</td></tr></table>
          <p>Model update example: <code>meta-llama/Meta-Llama-3.3-70B-030424</code></p>
        </body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_databricks_retirement_table_captures_exact_model_id() -> None:
    with _PROPOSAL.open("rb") as handle:
        source = tomllib.load(handle)["source"][0]
    adapter = create_source(source, client=_FixtureClient(), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})
    models = [model for record in page.records for model in record.models]

    assert [model.name for model in models] == ["Meta-Llama-3-70B-Instruct"]
    assert [identifier.value for model in models for identifier in model.identifiers] == [
        "Meta-Llama-3-70B-Instruct"
    ]
    assert "retired-on:2024-07-23" in source["entry_tags"]
    assert "offering:pay-per-token" in source["entry_tags"]
