from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_URL = "https://github.com/allenai/scispacy/blob/main/README.md"


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
        body = """<html><body><table>
        <tr><th>Model</th><th>Description</th><th>Install URL</th></tr>
        <tr><td>en_core_sci_sm</td><td>Biomedical pipeline</td>
        <td><a href="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz">Download</a></td></tr>
        <tr><td>en_ner_craft_md</td><td>CRAFT NER pipeline</td>
        <td><a href="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_craft_md-0.5.4.tar.gz">Download</a></td></tr>
        </table></body></html>"""
        return HttpResponse(200, {"Content-Type": "text/html"}, body.encode(), _URL)


def test_scispacy_proposal_extracts_versioned_first_party_pipelines() -> None:
    proposal = tomllib.loads(
        (Path(__file__).parents[1] / "config/proposals/scispacy_models.toml").read_text()
    )
    source = proposal["source"][0]
    assert source["enabled"] is False
    adapter = HtmlCatalogSourceAdapter(
        name=source["name"],
        url=source["url"],
        provider_namespace=source["provider_namespace"],
        rules=source["rules"],
        artifact_kind=source["artifact_kind"],
        model_status=source["model_status"],
        allowed_origins=source["allowed_origins"],
        client=_FakeClient(),
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    models = [model for record in page.records for model in record.models]
    assert {model.name for model in models} == {"en_core_sci_sm", "en_ner_craft_md"}
    assert {
        identifier.value for model in models for identifier in model.identifiers
    } == {
        "v0.5.4/en_core_sci_sm-0.5.4.tar.gz",
        "v0.5.4/en_ner_craft_md-0.5.4.tar.gz",
    }
