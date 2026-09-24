from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from modelome.models import ArtifactKind
from modelome.semantic_scholar_citations import SemanticScholarCitationGraphAdapter
from modelome.sources.catalog import create_source


class _Response:
    def __init__(self, value):
        self.value = value

    def json(self):
        return self.value


class _Client:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def get(self, url, *, params=None, headers=None):
        self.calls.append((url, params, headers))
        return _Response(self.pages.pop(0))


def _adapter(client, **kwargs):
    return SemanticScholarCitationGraphAdapter(
        paper_id="CorpusId:42",
        paper_url="https://www.semanticscholar.org/paper/anchor",
        paper_title="Anchor paper",
        client=client,
        **kwargs,
    )


def test_custom_catalog_can_construct_a_targeted_citation_source():
    source = create_source(
        {
            "name": "paper-42-references",
            "adapter": "semantic_scholar_citation_graph",
            "paper_id": "CorpusId:42",
            "paper_url": "https://www.semanticscholar.org/paper/anchor",
            "paper_title": "Anchor paper",
            "direction": "references",
        },
        client=_Client([]),
        environ={},
    )
    assert isinstance(source, SemanticScholarCitationGraphAdapter)
    assert source.name == "paper-42-references"
    assert source.paper_id == "CorpusId:42"


def test_references_emit_exact_citation_edge_and_resume_from_next_offset():
    page1 = {
        "offset": 0,
        "next": 1,
        "data": [
            {
                "citedPaper": {
                    "paperId": "a" * 40,
                    "corpusId": 91,
                    "url": "https://www.semanticscholar.org/paper/target",
                    "title": "Target paper",
                },
                "contexts": ["We build on this."],
                "intents": ["methodology"],
            }
        ],
    }
    page2 = {"offset": 1, "next": 0, "data": []}
    client = _Client([page1, page2])
    adapter = _adapter(client, direction="references", api_key="secret")

    first = adapter.fetch_page({})
    assert first.complete is False
    assert first.next_state["offset"] == 1
    record = first.records[0]
    assert record.kind is ArtifactKind.PAPER
    assert record.canonical_url == "https://www.semanticscholar.org/paper/anchor"
    assert record.links[0].url == "https://www.semanticscholar.org/paper/target"
    assert record.links[0].relation == "cites"
    assert record.links[0].crawl is False
    assert record.raw["citing_paper_id"] == "CorpusId:42"
    assert record.raw["cited_paper_id"] == "a" * 40
    assert record.raw["edge"]["contexts"] == ["We build on this."]
    assert ("semantic-scholar:paper-id", "a" * 40) in {
        (item.namespace, item.value) for item in record.identifiers
    }

    second = adapter.fetch_page(first.next_state)
    assert second.complete is True
    assert second.records == ()
    endpoint, params, headers = client.calls[0]
    assert urlsplit(endpoint).path.endswith("/paper/CorpusId%3A42/references")
    assert params["offset"] == 0 and params["limit"] == 1000
    assert "citedPaper.paperId" in params["fields"]
    assert headers == {"x-api-key": "secret"}


def test_citations_use_cited_by_relation_and_keep_both_exact_endpoint_ids():
    client = _Client(
        [
            {
                "offset": 0,
                "next": None,
                "data": [
                    {
                        "citingPaper": {
                            "paperId": "b" * 40,
                            "corpusId": None,
                            "url": "https://www.semanticscholar.org/paper/citing",
                            "title": "Citing paper",
                        }
                    }
                ],
            }
        ]
    )
    record = _adapter(client, direction="citations").fetch_page({}).records[0]
    assert record.raw["citing_paper_id"] == "b" * 40
    assert record.raw["cited_paper_id"] == "CorpusId:42"
    assert record.links[0].relation == "cited_by"


@pytest.mark.parametrize(
    "payload",
    [
        {"offset": 1, "next": 2, "data": []},
        {"offset": 0, "next": 2, "data": [{"citedPaper": {}}]},
        {"offset": 0, "next": 1, "data": []},
        {"offset": 0, "next": False, "data": []},
        {"offset": 0, "next": 0.0, "data": []},
        {"offset": 0, "data": [{"citedPaper": {"paperId": "x", "url": "https://x.org"}}]},
    ],
)
def test_malformed_or_gapped_pages_fail_closed(payload):
    adapter = _adapter(_Client([payload]))
    with pytest.raises(ValueError):
        adapter.fetch_page({})


def test_checkpoint_cannot_be_reused_for_another_paper_or_direction():
    state = {"signature": "different", "offset": 10}
    with pytest.raises(ValueError, match="another query"):
        _adapter(_Client([])).fetch_page(state)


def test_unsigned_completed_checkpoint_cannot_skip_the_citation_bootstrap():
    adapter = _adapter(_Client([]))
    with pytest.raises(ValueError, match="missing its query signature"):
        adapter.fetch_page({"done": True})


def test_invalid_urls_and_page_size_are_rejected():
    with pytest.raises(ValueError, match="HTTPS"):
        SemanticScholarCitationGraphAdapter(
            paper_id="CorpusId:42", paper_url="http://example.org/paper", paper_title="Paper"
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        _adapter(_Client([]), page_size=1001)
