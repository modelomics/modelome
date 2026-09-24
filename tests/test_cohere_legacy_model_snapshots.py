from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/cohere_legacy_model_snapshots.toml"


class _FixtureClient:
    _BODIES = {
        "https://docs.cohere.com/v1/changelog/model-sizing-update-improvements": (
            """<p>Small and large models are deprecated on December 2, 2022.</p>
            <p>Generate large-20220926 routes to xlarge-20221108.</p>
            <p>Generate small-20220926 routes to medium-20221108.</p>
            <p>Model IDs not in source: <code>command-a-03-2025</code>.</p>"""
        ),
        (
            "https://docs.cohere.com/v1/changelog/"
            "improvements-to-current-models-new-beta-model-command"
        ): (
            """<p>Older model versions xlarge-20220609 and medium-20220926</p>
            <p>will be deprecated on December 2, 2022.</p>
            <p>New versions xlarge-20221108 and medium-20221108 were released.</p>
            <p>Command beta command-xlarge-20221108.</p>"""
        ),
    }

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        assert "text/html" in (headers or {}).get("Accept", "")
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=self._BODIES[url].encode(),
            url=url,
        )


def test_archived_cohere_changelogs_capture_only_exact_retired_snapshot_ids() -> None:
    with _PROPOSAL.open("rb") as handle:
        sources = tomllib.load(handle)["source"]
    expected = {
        "cohere-legacy-snapshots-retired-2022-12-02-sizing-update": {
            "large-20220926",
            "small-20220926",
        },
        "cohere-legacy-snapshots-retired-2022-12-02-model-upgrades": {
            "xlarge-20220609",
            "medium-20220926",
        },
    }
    assert {source["name"] for source in sources} == set(expected)

    for source in sources:
        adapter = create_source(source, client=_FixtureClient(), environ={})
        assert isinstance(adapter, HtmlCatalogSourceAdapter)
        page = adapter.fetch_page({})
        models = [model for record in page.records for model in record.models]
        assert {model.name for model in models} == expected[source["name"]]
        assert all(
            identifier.value == model.name and identifier.namespace == "cohere:model"
            for model in models
            for identifier in model.identifiers
        )
        assert "retired-on:2022-12-02" in source["entry_tags"]
