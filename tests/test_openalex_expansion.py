from __future__ import annotations

import json

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.openalex import OpenAlexSourceAdapter


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
