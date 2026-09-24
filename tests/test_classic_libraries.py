from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import load_source_configs
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_CATALOG = Path(__file__).parents[1] / "config/sources.toml"


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
    source = next(item for item in load_source_configs(_CATALOG) if item["name"] == name)
    return HtmlCatalogSourceAdapter(
        name=source["name"],
        url=source["url"],
        provider_namespace=source["provider_namespace"],
        rules=source["rules"],
        artifact_kind=source["artifact_kind"],
        model_status=source["model_status"],
        client=_FakeClient(html, url),
    )


def test_lightgbm_catalog_selects_documented_classifier_regressor_and_ranker() -> None:
    catalog_url = "https://lightgbm.readthedocs.io/en/stable/Python-API.html"
    html = """<!doctype html><html><head><title>Python API</title></head><body>
    <a href="pythonapi/lightgbm.LGBMClassifier.html"><code>LGBMClassifier</code></a>
    <a href="pythonapi/lightgbm.LGBMRegressor.html"><code>LGBMRegressor</code></a>
    <a href="pythonapi/lightgbm.LGBMRanker.html"><code>LGBMRanker</code></a>
    <a href="pythonapi/lightgbm.Dataset.html">Dataset</a>
    </body></html>"""

    page = _adapter("lightgbm-python-estimators", html, catalog_url).fetch_page({})

    assert page.upstream_count == 3
    assert [model.name for record in page.records for model in record.models] == [
        "LGBMClassifier",
        "LGBMRegressor",
        "LGBMRanker",
    ]
    assert [model.identifiers[0].value for record in page.records for model in record.models] == [
        "LGBMClassifier",
        "LGBMRegressor",
        "LGBMRanker",
    ]


def test_imbalanced_learn_catalog_selects_documented_over_sampling_techniques() -> None:
    catalog_url = "https://imbalanced-learn.org/stable/references/over_sampling.html"
    html = """<!doctype html><html><head><title>Over-sampling methods</title></head><body>
    <a href="generated/imblearn.over_sampling.RandomOverSampler.html">RandomOverSampler</a>
    <a href="generated/imblearn.over_sampling.SMOTE.html">SMOTE</a>
    <a href="generated/imblearn.over_sampling.SMOTENC.html">SMOTENC</a>
    <a href="generated/imblearn.over_sampling.ADASYN.html">ADASYN</a>
    <a href="../under_sampling.html">Under-sampling methods</a>
    </body></html>"""

    page = _adapter("imbalanced-learn-over-sampling", html, catalog_url).fetch_page({})

    assert page.upstream_count == 4
    assert [model.name for record in page.records for model in record.models] == [
        "RandomOverSampler",
        "SMOTE",
        "SMOTENC",
        "ADASYN",
    ]
    assert [model.identifiers[0].value for record in page.records for model in record.models] == [
        "RandomOverSampler",
        "SMOTE",
        "SMOTENC",
        "ADASYN",
    ]
