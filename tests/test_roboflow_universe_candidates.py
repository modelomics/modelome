from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.roboflow_universe_candidates import (
    RoboflowUniverseCandidatesAdapter,
)

_BASE = "https://universe.roboflow.com/search?p=0&q=object+detection"
_PROJECT = "https://universe.roboflow.com/leo-ueno/people-detection-o4rdr"


class _Client:
    def __init__(self, responses: Mapping[str, str]) -> None:
        self.responses = dict(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Mapping[str, Any] | None = None,
            headers: Mapping[str, str] | None = None) -> HttpResponse:
        assert params is None
        assert headers == {"Accept": "text/html"}
        self.calls.append(url)
        if url not in self.responses:
            raise AssertionError(f"unexpected public page request: {url}")
        return HttpResponse(200, {"content-type": "text/html"},
                            self.responses[url].encode(), url)


def test_search_page_fetches_explicit_selected_model_ids_from_public_details() -> None:
    listing = """<main><h1>Search Results for object detection</h1>
      <a href="/leo-ueno/people-detection-o4rdr">People Detection</a>
      <div>1 results per page</div><div>Showing 1 - 1 of 1</div></main>"""
    detail = """<main><h1>People Detection Computer Vision Model</h1>
      <div>Model type: RF-DETR NAS</div>
      <code>model_id="people-detection-o4rdr/12"</code></main>"""
    client = _Client({_BASE: listing, _PROJECT: detail})
    adapter = RoboflowUniverseCandidatesAdapter(query="object detection", client=client)

    page = adapter.fetch_page({})

    assert client.calls == [_BASE, _PROJECT]
    assert page.complete
    assert page.upstream_count == 1
    assert page.authoritative_snapshot is False
    assert len(page.records) == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.PROVIDER_PAGE
    assert record.canonical_url == _PROJECT
    assert record.title == "people-detection-o4rdr/12"
    assert record.identifiers == (Identifier("roboflow:model", "people-detection-o4rdr/12"),)
    assert record.raw["model_type"] == "RF-DETR NAS"
    assert record.raw["candidate_scope"] == "public-search-selected-deployment"
    assert record.releases == ()
    assert record.links[0].crawl is False


def test_search_pagination_preserves_query_and_stops_at_displayed_last_page() -> None:
    second = "https://universe.roboflow.com/search?p=1&q=object+detection"
    listing1 = '<a href="/team/a">A</a><div>1 results per page</div><div>Showing 1 - 1 of 2</div>'
    listing2 = '<a href="/team/b">B</a><div>1 results per page</div><div>Showing 2 - 2 of 2</div>'
    detail_a = '<div>model_id="a/3"</div>'
    detail_b = '<div>model_id="b/4"</div>'
    client = _Client({
        _BASE: listing1,
        "https://universe.roboflow.com/team/a": detail_a,
        second: listing2,
        "https://universe.roboflow.com/team/b": detail_b,
    })
    adapter = RoboflowUniverseCandidatesAdapter(
        query="object detection", client=client, max_projects_per_page=1
    )

    first_page = adapter.fetch_page({})
    next_page = adapter.fetch_page(first_page.next_state)

    assert first_page.complete is False
    assert first_page.next_state == {"page": 2}
    assert [record.title for record in first_page.records] == ["a/3"]
    assert next_page.complete is True
    assert [record.title for record in next_page.records] == ["b/4"]


def test_page_two_live_project_shape_emits_exact_deployed_model_id() -> None:
    # The live UI's second page displays "Showing 50 - 100 of 300" and "50
    # results per page". Its project page inference snippet declares this ID.
    page_two = "https://universe.roboflow.com/search?p=1&q=object+detection"
    project_url = "https://universe.roboflow.com/emotions-dectection/human-face-emotions"
    project_urls = [project_url] + [
        f"https://universe.roboflow.com/team/project-{index}" for index in range(49)
    ]
    anchors = "".join(
        f'<a href="{url.removeprefix("https://universe.roboflow.com")}">Project</a>'
        for url in project_urls
    )
    listing = f'{anchors}<div>50 results per page</div><div>Showing 50 - 100 of 300</div>'
    detail = """<main>
      <div>human-face-emotions/28</div>
      <div>Model type: yolov8n Model Upload</div>
      <code>CLIENT.infer(\"image.jpg\", model_id=\"human-face-emotions/28\")</code>
    </main>"""
    client = _Client({
        page_two: listing,
        project_url: detail,
        **{url: "<main>No deployed project model</main>" for url in project_urls[1:]},
    })
    adapter = RoboflowUniverseCandidatesAdapter(query="object detection", client=client)

    page = adapter.fetch_page({"page": 2})

    assert page.complete is False
    assert page.next_state == {"page": 3}
    assert client.calls[0] == page_two
    assert len(client.calls) == 51
    assert len(page.records) == 1
    assert page.records[0].identifiers == (
        Identifier("roboflow:model", "human-face-emotions/28"),
    )
    assert page.records[0].raw["model_type"] == "yolov8n Model Upload"


def test_rejects_listing_without_range_or_exact_project_count() -> None:
    client = _Client({_BASE: '<a href="/team/a">A</a>'})
    with pytest.raises(ValueError, match="no displayed result range"):
        RoboflowUniverseCandidatesAdapter(query="object detection", client=client).fetch_page({})

    client = _Client({_BASE: '<div>1 results per page</div><div>Showing 1 - 1 of 2</div>'})
    with pytest.raises(ValueError, match="range does not match project links"):
        RoboflowUniverseCandidatesAdapter(query="object detection", client=client).fetch_page({})


def test_requires_nonempty_query_and_positive_bounds() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        RoboflowUniverseCandidatesAdapter(query=" ")
    with pytest.raises(ValueError, match="max_pages"):
        RoboflowUniverseCandidatesAdapter(query="x", max_pages=0)
