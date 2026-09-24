from __future__ import annotations

import json
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.zenodo import ZenodoModelRecordsSourceAdapter


class _Client:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        return HttpResponse(200, {}, json.dumps(self.payload).encode(), url)


def test_zenodo_preserves_version_graph_and_related_identifier_relation() -> None:
    record = {
        "id": 42,
        "doi_url": "https://doi.org/10.5281/zenodo.42",
        "metadata": {
            "title": "Versioned model",
            "resource_type": {"type": "model"},
            "related_identifiers": [
                {
                    "identifier": "10.5281/zenodo.41",
                    "relation": "IsNewVersionOf",
                    "resource_type": "dataset",
                }
            ],
        },
        "links": {
            "self": "https://zenodo.org/api/records/42",
            "self_html": "https://zenodo.org/records/42",
            "parent": "https://zenodo.org/api/records/40",
            "parent_html": "https://zenodo.org/records/40",
            "latest": "https://zenodo.org/api/records/42",
            "latest_html": "https://zenodo.org/records/42",
            "versions": "https://zenodo.org/api/records/40/versions",
        },
    }
    payload = {"hits": {"hits": [record], "total": 1}, "links": {}}

    page = ZenodoModelRecordsSourceAdapter(client=_Client(payload)).fetch_page({})

    assert len(page.records) == 1
    links = {(link.url, link.relation) for link in page.records[0].links}
    assert ("https://zenodo.org/api/records/40", "version_parent_metadata") in links
    assert ("https://zenodo.org/records/40", "version_parent") in links
    assert ("https://zenodo.org/api/records/42", "latest_version_metadata") in links
    assert ("https://zenodo.org/records/42", "latest_version") in links
    assert ("https://zenodo.org/api/records/40/versions", "versions") in links
    assert ("https://doi.org/10.5281/zenodo.41", "version_of") in links
