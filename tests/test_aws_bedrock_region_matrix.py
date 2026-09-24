from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.aws_bedrock_region_matrix import AwsBedrockRegionMatrixAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/aws_bedrock_region_matrix.toml"
_URL = "https://docs.aws.amazon.com/bedrock/latest/userguide/models-region-compatibility.html"


class _Client:
    def __init__(self, body: str, *, status: int = 200) -> None:
        self.body = body
        self.status = status

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert url == _URL
        assert params is None
        assert headers == {"Accept": "text/html"}
        return HttpResponse(self.status, {"Content-Type": "text/html"}, self.body.encode(), url)


def test_bedrock_region_matrix_uses_exact_model_card_link_and_keeps_lifecycle() -> None:
    proposal = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    body = """<html><body>
      <h2>AI21 Labs</h2>
      <h3><a href="model-card-ai21-labs-jamba-1-5-large.html">Jamba 1.5 Large</a></h3>
      <table>
        <tr><th>Region</th><th>In-Region</th><th>Geo</th><th>Global</th></tr>
        <tr><td><code>us-east-1</code> (N. Virginia)</td><td>Legacy (EOL: 2026-11-26)</td>
            <td><img alt="not-supported"></td><td><img alt="not-supported"></td></tr>
        <tr><td><code>us-west-2</code> (Oregon)</td><td>not-supported</td>
            <td>supported</td><td>not-supported</td></tr>
      </table>
      <a href="model-card-openai-gpt-6-astra.html">GPT-6 Astra</a>
      <table>
        <tr><th>Region</th><th>In-Region</th><th>Geo</th><th>Global</th></tr>
        <tr><td><code>us-east-1</code> (N. Virginia)</td><td>not-supported</td>
            <td>supported</td><td>supported</td></tr>
      </table>
    </body></html>"""
    adapter = AwsBedrockRegionMatrixAdapter(
        name=proposal["name"], url=proposal["url"], client=_Client(body),
        max_response_bytes=proposal["max_response_bytes"],
        max_entries=proposal["max_entries"], max_cells=proposal["max_cells"],
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    jamba = next(record for record in page.records if record.title == "Jamba 1.5 Large")
    assert jamba.kind is ArtifactKind.CATALOG_RECORD
    assert jamba.source_record_id == "aws-bedrock-region:ai21-labs-jamba-1-5-large"
    assert jamba.models[0].identifiers == (
        Identifier("aws:bedrock-model-card", "ai21-labs-jamba-1-5-large"),
    )
    assert jamba.raw["availability"] == [
        {
            "region": "us-east-1 (N. Virginia)",
            "in_region": "Legacy (EOL: 2026-11-26)",
            "geo": "not_supported",
            "global": "not_supported",
        },
        {
            "region": "us-west-2 (Oregon)",
            "in_region": "not_supported",
            "geo": "supported",
            "global": "not_supported",
        },
    ]
    assert jamba.links[0].url.endswith("model-card-ai21-labs-jamba-1-5-large.html")
    assert jamba.links[0].crawl is False


def test_bedrock_region_matrix_fails_closed_without_linked_model_matrices() -> None:
    adapter = AwsBedrockRegionMatrixAdapter(client=_Client(
        "<table><tr><th>Region</th><th>In-Region</th><th>Geo</th><th>Global</th></tr>"
        "<tr><td>us-east-1</td><td>supported</td><td>supported</td><td>supported</td></tr></table>"
    ))
    with pytest.raises(ValueError, match="no model-card-linked region matrices"):
        adapter.fetch_page({})


def test_bedrock_region_matrix_rejects_nonempty_pagination_state() -> None:
    adapter = AwsBedrockRegionMatrixAdapter(client=_Client(""))
    with pytest.raises(ValueError, match="does not accept pagination state"):
        adapter.fetch_page({"cursor": "next"})
