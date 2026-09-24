from __future__ import annotations

import tomllib
from pathlib import Path

from modelome.http import HttpResponse
from modelome.sources.catalog import create_source
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter


class FixtureClient:
    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        body = b"""
        <html><body><h1>Retrieval APIs</h1>
        <h2>Models</h2><h3>baai</h3>
        <table><thead><tr><th>Model</th><th>Endpoint</th></tr></thead><tbody>
          <tr><td><a href='/nim/baai-bge-m3'>baai / bge-m3</a></td><td>Embed</td></tr>
        </tbody></table>
        <h3>nvidia</h3>
        <table><thead><tr><th>Model</th><th>Endpoint</th></tr></thead><tbody>
          <tr><td><a href='/nim/nvidia-embed-qa-4'>nvidia / embed-qa-4</a></td><td>Embed</td></tr>
          <tr>
            <td><a href='/nim/nvidia-llama-nemotron-embed-1b-v2'>
              nvidia / llama-nemotron-embed-1b-v2
            </a></td><td>Embed</td>
          </tr>
          <tr>
            <td><a href='/nim/nvidia-nv-rerankqa-mistral-4b-v3'>
              nvidia / nv-rerankqa-mistral-4b-v3
            </a></td><td>Rerank</td>
          </tr>
        </tbody></table>
        <h3>snowflake</h3><a href='/nim/snowflake-arctic-embed-l'>snowflake / arctic-embed-l</a>
        </body></html>
        """
        return HttpResponse(
            status=200,
            headers={"content-type": "text/html"},
            body=body,
            url=url,
        )


def test_nvidia_nim_retrieval_docs_select_exact_nvidia_ids_only() -> None:
    proposal_path = (
        Path(__file__).parents[1]
        / "config/proposals/nvidia_nim_retrieval_models.toml"
    )
    with proposal_path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]
    adapter = create_source(config, client=FixtureClient(), environ={})

    assert isinstance(adapter, HtmlCatalogSourceAdapter)
    page = adapter.fetch_page({})

    assert len(page.records) == 1
    catalog = page.records[0]
    assert {model.identifiers[0].value for model in catalog.models} == {
        "nvidia/embed-qa-4",
        "nvidia/llama-nemotron-embed-1b-v2",
        "nvidia/nv-rerankqa-mistral-4b-v3",
    }
    assert all(
        model.identifiers[0].namespace == "nvidia:nim-model"
        for model in catalog.models
    )
