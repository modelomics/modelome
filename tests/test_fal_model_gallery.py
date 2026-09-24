from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.fal_model_gallery import FalModelGalleryAdapter

_BASE = "https://fal.ai/explore/search"
_PAGE_2 = f"{_BASE}?page=2"


class _Client:
    def __init__(self, responses: Mapping[str, tuple[str, str]]) -> None:
        self.responses = dict(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        assert headers == {"Accept": "text/html"}
        self.calls.append(url)
        if url not in self.responses:
            raise AssertionError(f"unexpected gallery request: {url}")
        body, final_url = self.responses[url]
        return HttpResponse(200, {"content-type": "text/html"}, body.encode(), final_url)


def _cards(first: int, last: int) -> str:
    cards = "".join(
        f'<a class="page-model-card flex" href="/models/fal-ai/demo/model-{number}">'
        f"fal-ai/demo/model-{number} Description for model {number}</a>"
        for number in range(first, last + 1)
    )
    return (
        '<nav><a href="/models/fal-ai/featured">Featured model</a></nav>'
        f"<main>{cards}<p>Showing {first} to {last} of 25 results</p></main>"
    )


def test_gallery_paginates_exact_provider_endpoint_ids() -> None:
    client = _Client(
        {
            _BASE: (_cards(1, 24), _BASE),
            _PAGE_2: (_cards(25, 25), _PAGE_2),
        }
    )
    adapter = FalModelGalleryAdapter(client=client)

    first_page = adapter.fetch_page({})
    second_page = adapter.fetch_page(first_page.next_state)

    assert client.calls == [_BASE, _PAGE_2]
    assert not first_page.complete
    assert first_page.upstream_count == second_page.upstream_count == 25
    assert len(first_page.records) == 24
    assert second_page.complete
    assert len(second_page.records) == 1
    record = second_page.records[0]
    assert record.kind is ArtifactKind.PROVIDER_PAGE
    assert record.canonical_url == "https://fal.ai/models/fal-ai/demo/model-25"
    assert record.identifiers == (Identifier("fal:endpoint", "fal-ai/demo/model-25"),)
    assert record.models[0].identifiers == (
        Identifier("fal:endpoint", "fal-ai/demo/model-25"),
    )
    assert record.releases == ()
    assert record.links[0].crawl is False


def test_gallery_allows_total_drift_and_deduplicates_overlap_between_pages() -> None:
    page_2 = (
        '<a class="page-model-card" href="/models/fal-ai/demo/model-24">overlap</a>'
        + "".join(
            f'<a class="page-model-card" '
            f'href="/models/fal-ai/demo/model-{number}">model {number}</a>'
            for number in range(25, 48)
        )
        + "<p>Showing 25 to 48 of 48 results</p>"
    )
    client = _Client(
        {
            _BASE: (_cards(1, 24), _BASE),
            _PAGE_2: (page_2, _PAGE_2),
        }
    )
    adapter = FalModelGalleryAdapter(client=client)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert not first.complete
    assert first.upstream_count == 25
    assert second.complete
    assert second.upstream_count == 48
    assert len(second.records) == 23
    assert "fal-ai/demo/model-24" not in [record.title for record in second.records]
    assert "fal-ai/demo/model-47" in [record.title for record in second.records]


def test_gallery_rejects_page_with_no_new_ids_after_page_shift() -> None:
    body = _cards(1, 24).replace("1 to 24 of 25", "25 to 48 of 48")
    client = _Client({_PAGE_2: (body, _PAGE_2)})

    with pytest.raises(ValueError, match="made no model-ID progress"):
        FalModelGalleryAdapter(client=client).fetch_page(
            {"page": 2, "seen_ids": [f"fal-ai/demo/model-{i}" for i in range(1, 25)]}
        )


def test_gallery_rejects_missing_or_inconsistent_pagination_evidence() -> None:
    client = _Client({_BASE: ("<main>some models, but no result range</main>", _BASE)})
    with pytest.raises(ValueError, match="no gallery result range"):
        FalModelGalleryAdapter(client=client).fetch_page({})


def test_gallery_parses_live_ssr_card_class_and_split_result_range() -> None:
    body = """<header><a href="/models/fal-ai/featured">Featured</a></header>
      <div class="grid"><a class="page-model-card flex"
        href="/models/fal-ai/flux/schnell"><div><img alt="FLUX preview">
        <h3>flux/schnell</h3><p>FLUX.1 [schnell] text-to-image</p></div></a></div>
      <div>Showing<!-- --> <span>1</span> <!-- -->to<!-- --> <span>1</span>
      <!-- -->of <span>1</span> <!-- -->results</div>"""
    client = _Client({_BASE: (body, _BASE)})

    page = FalModelGalleryAdapter(client=client).fetch_page({})

    assert page.complete
    assert page.upstream_count == 1
    assert [record.title for record in page.records] == ["fal-ai/flux/schnell"]


def test_gallery_rejects_page_card_count_mismatch() -> None:
    body = (
        '<a class="page-model-card" href="/models/fal-ai/demo/only">only</a>'
        '<p>Showing 1 to 2 of 2 results</p>'
    )
    client = _Client({_BASE: (body, _BASE)})
    with pytest.raises(ValueError, match="card count does not match"):
        FalModelGalleryAdapter(client=client).fetch_page({})


def test_gallery_rejects_redirect_outside_requested_page() -> None:
    client = _Client({_BASE: (_cards(1, 24), "https://example.org/explore/search")})
    with pytest.raises(ValueError, match="redirected outside"):
        FalModelGalleryAdapter(client=client).fetch_page({})
