from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSALS = Path(__file__).parents[1] / "config/proposals/classic_libraries_v2.toml"


class _FakeClient:
    def __init__(self, body: str, response_url: str) -> None:
        self.body = body.encode()
        self.response_url = response_url

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        assert "text/html" in (headers or {}).get("Accept", "")
        return HttpResponse(200, {"Content-Type": "text/html"}, self.body, self.response_url)


def _adapter(name: str, html: str, url: str) -> HtmlCatalogSourceAdapter:
    proposals = tomllib.loads(_PROPOSALS.read_text())
    source = next(item for item in proposals["source"] if item["name"] == name)
    return HtmlCatalogSourceAdapter(
        name=source["name"],
        url=source["url"],
        provider_namespace=source["provider_namespace"],
        rules=source["rules"],
        artifact_kind=source["artifact_kind"],
        model_status=source["model_status"],
        client=_FakeClient(html, url),
    )


def _model_names(page) -> list[str]:
    return [model.name for record in page.records for model in record.models]


@pytest.mark.parametrize(
    ("source_name", "catalog_url", "html", "expected"),
    [
        (
            "xgboost-python-estimators",
            "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
            """<html><body>
            <a href="#xgboost.XGBClassifier">XGBClassifier</a>
            <a href="#xgboost.XGBRegressor">XGBRegressor</a>
            <a href="#xgboost.XGBRanker">XGBRanker</a>
            <a href="#xgboost.DMatrix">DMatrix</a>
            </body></html>""",
            ["XGBClassifier", "XGBRegressor", "XGBRanker"],
        ),
        (
            "catboost-python-estimators",
            "https://catboost.ai/docs/en/concepts/python-reference_catboost",
            """<html><body>
            <a href="python-reference_catboostclassifier">CatBoostClassifier</a>
            <a href="python-reference_catboostregressor">CatBoostRegressor</a>
            <a href="python-reference_catboostranker">CatBoostRanker</a>
            <a href="python-reference_catboost">CatBoost</a>
            </body></html>""",
            ["CatBoostClassifier", "CatBoostRegressor", "CatBoostRanker"],
        ),
        (
            "river-ml-technique-overview",
            "https://riverml.xyz/latest/api/overview/",
            """<html><body>
            <a href="/latest/api/cluster/KMeans/">KMeans</a>
            <a href="/latest/api/cluster/DBSTREAM/">DBSTREAM</a>
            <a href="/latest/api/anomaly/HalfSpaceTrees/">HalfSpaceTrees</a>
            <a href="/latest/api/linear_model/LinearRegression/">LinearRegression</a>
            </body></html>""",
            ["KMeans", "DBSTREAM", "HalfSpaceTrees"],
        ),
        (
            "optuna-samplers",
            "https://optuna.readthedocs.io/en/stable/reference/samplers/index.html",
            """<html><body>
            <a href="generated/optuna.samplers.RandomSampler.html">RandomSampler</a>
            <a href="generated/optuna.samplers.TPESampler.html">TPESampler</a>
            <a href="generated/optuna.samplers.GPSampler.html">GPSampler</a>
            <a href="generated/optuna.samplers.BaseSampler.html">BaseSampler</a>
            </body></html>""",
            ["RandomSampler", "TPESampler", "GPSampler"],
        ),
        (
            "gpflow-model-classes",
            "https://gpflow.github.io/GPflow/develop/api/gpflow/models/index.html",
            """<html><body>
            <a href="#gpflow.models.CGLB">CGLB</a>
            <a href="#gpflow.models.SGPR">SGPR</a>
            <a href="#gpflow.models.SVGP">SVGP</a>
            <a href="#gpflow.models.BayesianModel">BayesianModel</a>
            </body></html>""",
            ["CGLB", "SGPR", "SVGP"],
        ),
    ],
)
def test_additional_live_catalog_rules_extract_documented_techniques(
    source_name: str,
    catalog_url: str,
    html: str,
    expected: list[str],
) -> None:
    page = _adapter(source_name, html, catalog_url).fetch_page({})

    assert page.upstream_count == len(expected)
    assert _model_names(page) == expected
    assert all(model.identifiers for record in page.records for model in record.models)
