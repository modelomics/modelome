from __future__ import annotations

import json

from modelome.http import HttpResponse
from modelome.sources.espnet_model_zoo import EspnetModelZooSourceAdapter


class _Client:
    def __init__(self, *bodies: bytes) -> None:
        self.bodies = list(bodies)

    def get(self, url, *, params=None, headers=None) -> HttpResponse:
        return HttpResponse(200, {}, self.bodies.pop(0), url)


def test_recipe_commit_is_exposed_as_a_first_party_source_revision_link() -> None:
    revision = "d" * 40
    table = (
        b"corpus,task,name,url,fs,lang,gender,pytorch,espnet,commit,valid\n"
        b"wsj,asr,org/wsj,https://zenodo.org/record/1/files/model.zip,16000,en,,"
        b"1.6.0,0.9.1,e67a1ad,true\n"
    )
    adapter = EspnetModelZooSourceAdapter(
        client=_Client(json.dumps({"sha": revision}).encode(), table)
    )

    page = adapter.fetch_page({})
    record = page.records[0]

    assert record.releases[0].metadata["recipe_commit"] == "e67a1ad"
    assert (
        "https://github.com/espnet/espnet/commit/e67a1ad",
        "source_revision",
        False,
    ) in {(link.url, link.relation, link.crawl) for link in record.links}
