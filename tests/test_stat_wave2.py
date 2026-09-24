from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/stat_wave2.toml"
_URL = "https://uber.github.io/orbit/orbit.models.html"


class _FakeClient:
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert url == _URL
        assert params is None
        assert "text/html" in (headers or {}).get("Accept", "")
        body = """<html><body>
        <a href="#orbit.models.dlt.DLTAggregated">DLTAggregated</a>
        <a href="#orbit.models.dlt.DLTFull">DLTFull</a>
        <a href="#orbit.models.dlt.DLTMAP">DLTMAP</a>
        <a href="#orbit.models.dlt.BaseDLT">BaseDLT</a>
        <a href="#orbit.models.lgt.LGTAggregated">LGTAggregated</a>
        <a href="#orbit.models.lgt.LGTFull">LGTFull</a>
        <a href="#orbit.models.lgt.LGTMAP">LGTMAP</a>
        <a href="#orbit.models.lgt.BaseLGT">BaseLGT</a>
        <a href="#orbit.estimators.pyro_estimator.PyroEstimator">PyroEstimator</a>
        </body></html>"""
        return HttpResponse(
            200,
            {"Content-Type": "text/html"},
            body.encode(),
            _URL,
        )


def test_orbit_proposal_extracts_only_documented_forecasting_models() -> None:
    proposal = tomllib.loads(_PROPOSAL.read_text())
    source = proposal["source"][0]
    adapter = HtmlCatalogSourceAdapter(
        name=source["name"],
        url=source["url"],
        provider_namespace=source["provider_namespace"],
        rules=source["rules"],
        artifact_kind=source["artifact_kind"],
        model_status=source["model_status"],
        client=_FakeClient(),
    )

    page = adapter.fetch_page({})

    models = [model for record in page.records for model in record.models]
    assert page.upstream_count == 6
    assert {model.name for model in models} == {
        "DLTAggregated",
        "DLTFull",
        "DLTMAP",
        "LGTAggregated",
        "LGTFull",
        "LGTMAP",
    }
    assert {
        identifier.value
        for model in models
        for identifier in model.identifiers
    } == {
        "orbit.models.dlt.DLTAggregated",
        "orbit.models.dlt.DLTFull",
        "orbit.models.dlt.DLTMAP",
        "orbit.models.lgt.LGTAggregated",
        "orbit.models.lgt.LGTFull",
        "orbit.models.lgt.LGTMAP",
    }
