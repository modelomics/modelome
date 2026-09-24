from __future__ import annotations

import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_NOW = datetime(2026, 9, 23, tzinfo=UTC)
_HTML = b"""<html><body>
<ul>
<li><strong>NEW!!</strong> 2024 Dolma (220B tokens, 1.2M vocab, uncased, 300d vectors):
<a href="https://nlp.stanford.edu/data/wordvecs/glove.2024.dolma.300d.zip">glove.2024.dolma.300d.zip</a></li>
<li><strong>NEW!!</strong> 2024 Wikipedia + Gigaword 5 (11.9B tokens, 1.2M vocab,
uncased, 100d vectors):
<a href="https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip">glove.2024.wikigiga.100d.zip</a></li>
<li>Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d,
&amp; 300d vectors):
<a href="https://nlp.stanford.edu/data/glove.6B.zip">glove.6B.zip</a></li>
</ul>
<a href="https://opendatacommons.org/licenses/pddl/1.0/">license</a>
<a href="https://arxiv.org/abs/1405.4053">GloVe 2014 paper</a>
<a href="https://arxiv.org/abs/2507.18103">2024 vectors report</a>
</body></html>"""


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(200, {}, _HTML, url)


def test_stanford_glove_page_indexes_current_and_historical_vector_archives() -> None:
    config = tomllib.loads(
        (Path(__file__).parents[1] / "config/proposals/stanford_glove_vectors.toml").read_text()
    )["source"][0]
    assert config["enabled"] is False
    client = _FakeClient()
    adapter = HtmlCatalogSourceAdapter(
        name=config["name"],
        url=config["url"],
        provider_namespace=config["provider_namespace"],
        artifact_kind=config["artifact_kind"],
        model_status=config["model_status"],
        allowed_origins=config["allowed_origins"],
        rules=config["rules"],
        shared_link_rules=config["shared_links"],
        max_response_bytes=config["max_response_bytes"],
        max_entries=config["max_entries"],
        client=client,
        clock=lambda: _NOW,
    )

    page = adapter.fetch_page({})

    assert page.complete and page.upstream_count == 3
    record = page.records[0]
    models_by_id = {model.identifiers[0].value: model for model in record.models}
    assert {model.name for model in record.models} == {
        "GloVe 2024 Dolma (220B tokens, 1.2M vocab, uncased, 300d vectors)",
        "GloVe 2024 Wikipedia + Gigaword 5 (11.9B tokens, 1.2M vocab, uncased, 100d vectors)",
        "GloVe Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, "
        "50d, 100d, 200d, &amp; 300d vectors)",
    }
    assert set(models_by_id) == {
        "glove.2024.dolma.300d.zip",
        "glove.2024.wikigiga.100d.zip",
        "glove.6B.zip",
    }
    weight_urls = {link.url for link in record.links if link.relation == "weights"}
    assert weight_urls == {
        "https://nlp.stanford.edu/data/wordvecs/glove.2024.dolma.300d.zip",
        "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip",
        "https://nlp.stanford.edu/data/glove.6B.zip",
    }
    assert {link.relation for link in record.links} >= {
        "weights",
        "license",
        "paper_reference",
    }
    assert len(client.calls) == 1
