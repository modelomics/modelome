from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.pmlr import PmlrSourceAdapter


class QueueClient:
    def __init__(self, *bodies: bytes) -> None:
        self.bodies = list(bodies)
        self.calls: list[str] = []

    def get(self, url: str, *, headers: Any = None, params: Any = None) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(200, {"content-type": "text/html"}, self.bodies.pop(0), url)


INDEX = b"""<html><body><a href="/v97">Volume 97: ICML</a></body></html>"""
VOLUME = b"""<html><body>
<h2>Volume 97: International Conference on Machine Learning</h2>
<dt>Graph Element Networks: adaptive, structured computation and memory</dt>
<dd>Ferran Alet; Proceedings of the 36th International Conference on Machine Learning</dd>
<a href="alet19a.html">abs</a>
<a href="alet19a.pdf">Download PDF</a>
<a href="alet19a-supp.pdf">Supplementary PDF</a>
<a href="https://github.com/example/graphs">Code</a>
</body></html>"""


def test_walks_official_index_then_emits_paper_and_exact_source_declared_links() -> None:
    client = QueueClient(INDEX, VOLUME)
    source = PmlrSourceAdapter(client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC))

    page = source.fetch_page({})

    assert client.calls == ["https://proceedings.mlr.press/", "https://proceedings.mlr.press/v97/"]
    assert page.complete is True
    assert len(page.records) == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.PAPER
    assert record.source_record_id == "v97/alet19a"
    assert record.canonical_url == "https://proceedings.mlr.press/v97/alet19a.html"
    assert record.title == "Graph Element Networks: adaptive, structured computation and memory"
    assert Identifier("pmlr", "v97/alet19a") in record.identifiers
    assert {(link.relation, link.url, link.crawl) for link in record.links} == {
        ("full_text", "https://proceedings.mlr.press/v97/alet19a.pdf", False),
        ("supplementary_material", "https://proceedings.mlr.press/v97/alet19a-supp.pdf", False),
        ("implementation", "https://github.com/example/graphs", True),
    }
    assert record.raw["volume_id"] == "v97"


def test_does_not_treat_generic_supplement_as_a_checkpoint() -> None:
    source = PmlrSourceAdapter(client=QueueClient(INDEX, VOLUME))
    page = source.fetch_page({})
    record = page.records[0]

    assert not record.models
    assert not record.releases
    supplement = next(link for link in record.links if link.relation == "supplementary_material")
    assert supplement.crawl is False


def test_includes_first_party_reissue_volumes_and_preserves_the_r_series_id() -> None:
    index = b"""<a href="/v97">Volume 97</a><a href="/r6">Volume R6</a>"""
    reissue = b"""<html><body>
    <dt>Adaptive inference on general graphical models</dt>
    <dd>Umut A. Acar; Proceedings of the 24th Conference on Uncertainty in Artificial
    Intelligence, PMLR R6:1-8</dd>
    <a href="acar08a.html">abs</a>
    <a href="https://raw.githubusercontent.com/mlresearch/r6/main/assets/acar08a/acar08a.pdf">
      Download PDF
    </a>
    </body></html>"""
    client = QueueClient(index, VOLUME, reissue)
    source = PmlrSourceAdapter(client=client)

    page = source.fetch_page({})

    assert client.calls == ["https://proceedings.mlr.press/", "https://proceedings.mlr.press/v97/"]
    # The index is in newest-first order. Advance the frozen manifest to R6.
    state = {
        **page.next_state,
        "volume_index": 1,
        "paper_count": page.next_state["paper_count"],
    }
    reissue_page = source.fetch_page(state)
    assert client.calls[-1] == "https://proceedings.mlr.press/r6/"
    record = reissue_page.records[0]
    assert record.source_record_id == "r6/acar08a"
    assert record.canonical_url == "https://proceedings.mlr.press/r6/acar08a.html"
    assert Identifier("pmlr", "r6/acar08a") in record.identifiers
    assert record.raw["volume_id"] == "r6"
    assert (
        record.links[0].url
        == "https://raw.githubusercontent.com/mlresearch/r6/main/assets/acar08a/acar08a.pdf"
    )
