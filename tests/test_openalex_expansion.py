from __future__ import annotations

import json

from modelome.artifact_relations import ArtifactRelationMaterializer
from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.openalex import OpenAlexSourceAdapter
from modelome.storage import Database


class OnePageClient:
    def get(self, url, *, params, headers):
        payload = {
            "meta": {"count": 1, "next_cursor": None},
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Work with Wikidata identity",
                    "publication_date": "2026-09-01",
                    "ids": {"openalex": "https://openalex.org/W123", "wikidata": "Q12345"},
                }
            ],
        }
        return HttpResponse(200, {}, json.dumps(payload).encode(), url)


def test_openalex_preserves_wikidata_work_identifier() -> None:
    page = OpenAlexSourceAdapter(client=OnePageClient()).fetch_page({})

    assert page.complete is True
    assert Identifier("wikidata", "Q12345") in page.records[0].identifiers


def test_openalex_reference_joins_exact_target_when_canonical_url_is_a_doi(tmp_path) -> None:
    adapter = OpenAlexSourceAdapter()
    citing = adapter._record(
        {
            "id": "https://openalex.org/W100",
            "title": "Citing work",
            "doi": "https://doi.org/10.1234/citing",
            "referenced_works": ["https://openalex.org/W200"],
        }
    )
    cited = adapter._record(
        {
            "id": "https://openalex.org/W200",
            "title": "Cited work",
            "doi": "https://doi.org/10.1234/cited",
        }
    )
    unrelated = adapter._record(
        {
            "id": "https://openalex.org/W300",
            "title": "Cited work",
            "doi": "https://doi.org/10.1234/unrelated",
        }
    )

    database = Database(tmp_path / "store")
    database.initialize()
    database.ingest_page("openalex", (citing, cited, unrelated), extractor="fixture")
    materializer = ArtifactRelationMaterializer(database.root)
    receipt = materializer.materialize()
    rows = [
        row
        for batch in materializer.iter_batches(receipt)
        for row in batch.to_pylist()
    ]

    citations = [row for row in rows if row["predicate"] == "cites"]
    assert len(citations) == 1
    assert citations[0]["subject_source_record_id"] == "W100"
    assert citations[0]["target_source_record_id"] == "W200"
    assert citations[0]["target_canonical_url"] == "https://doi.org/10.1234/cited"
    assert citations[0]["symmetric"] is False
    database.close()
