from __future__ import annotations

import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_NOW = datetime(2026, 9, 23, tzinfo=UTC)
_HTML = b"""<html><body><table><tr>
<td>Abkhazian: <a href="https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ab.zip">bin+text</a>,
<a href="https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ab.vec">text</a></td>
<td>English: <a href="https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip">bin+text</a>,
<a href="https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec">text</a></td>
</tr></table>
<a href="https://arxiv.org/abs/1607.04606">paper</a>
<a href="https://github.com/facebookresearch/fastText">implementation</a>
<a href="https://creativecommons.org/licenses/by-sa/3.0/">license</a>
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


def test_official_wiki_vectors_index_groups_exact_language_archives() -> None:
    config = tomllib.loads(
        (Path(__file__).parents[1] / "config/proposals/fasttext_wiki_vectors.toml").read_text()
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

    assert page.complete and page.upstream_count == 2
    record = page.records[0]
    assert {model.name for model in record.models} == {
        "fastText Wiki 300 Abkhazian",
        "fastText Wiki 300 English",
    }
    models_by_id = {
        model.identifiers[0].value: model
        for model in record.models
    }
    assert set(models_by_id) == {"ab", "en"}
    weight_links = {
        (link.url, link.model_local_ids)
        for link in record.links
        if link.relation == "weights"
    }
    urls_by_code = {
        "ab": (
            "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ab.zip",
            "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ab.vec",
        ),
        "en": (
            "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip",
            "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec",
        ),
    }
    assert weight_links == {
        (url, (models_by_id[code].local_id,))
        for code, urls in urls_by_code.items()
        for url in urls
    }
    assert len(client.calls) == 1
