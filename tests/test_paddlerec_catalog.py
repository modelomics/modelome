# ruff: noqa: E501

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.paddlerec_catalog import PaddleRecCatalogSourceAdapter

_REVISION = "5" * 40
_DOCUMENT = b"""
# Support Model List

| Type | Algorithm | Online Environment | Parameter-Server | Multi-GPU | version | Paper |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Recall | [ExampleRec](models/recall/example/)<br>([doc](https://paddlerec.readthedocs.io/en/latest/models/example.html)) | [online](https://aistudio.baidu.com/aistudio/projectdetail/1) | x | x | >=2.1.0 | [KDD][Example paper](https://arxiv.org/abs/1234.56789) |
| Rank | [LegacyRank](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5/models/rank/legacy/) | - | x | x | [1.8.5](https://github.com/PaddlePaddle/PaddleRec/tree/release/1.8.5) | [Paper](https://papers.legacy.test/legacy.pdf) |

| Type | Implementation |
| --- | --- |
| Unrelated | [do not add](https://example.test) |
"""


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append((url, headers))
        return self.responses.pop(0)


def _response(
    body: bytes, *, url: str = "https://fixtures.test/paddlerec"
) -> HttpResponse:
    return HttpResponse(200, {"etag": '"fixture"'}, body, url)


def _commit() -> HttpResponse:
    return _response(("{\"sha\": \"" + _REVISION + "\"}").encode())


def test_paddlerec_support_table_preserves_code_docs_and_papers_per_row() -> None:
    client = _QueuedClient(_commit(), _response(_DOCUMENT))
    adapter = PaddleRecCatalogSourceAdapter(
        repository="example/PaddleRec",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot
    assert page.upstream_count == 2
    assert client.calls[1][0].endswith(f"/{_REVISION}/README_EN.md")
    example = next(record for record in page.records if record.title == "ExampleRec")
    assert example.identifiers == (
        Identifier("paddlerec:recommendation-algorithm", "Recall\nExampleRec"),
    )
    assert example.models[0].status.value == "documented"
    assert example.releases == ()
    links = {(link.url, link.relation, link.crawl) for link in example.links}
    assert (
        "https://github.com/example/PaddleRec/blob/"
        f"{_REVISION}/models/recall/example",
        "source_implementation",
        False,
    ) in links
    assert (
        "https://paddlerec.readthedocs.io/en/latest/models/example.html",
        "model_card",
        False,
    ) in links
    assert ("https://arxiv.org/abs/1234.56789", "paper_reference", False) in links
    assert (
        "https://aistudio.baidu.com/aistudio/projectdetail/1",
        "related_resource",
        False,
    ) in links
    assert all(link.model_local_ids == (example.models[0].local_id,) for link in example.links)
    legacy = next(record for record in page.records if record.title == "LegacyRank")
    legacy_urls = {link.url for link in legacy.links}
    assert "https://paddlerec.readthedocs.io/en/latest/models/example.html" not in legacy_urls
    assert "https://arxiv.org/abs/1234.56789" not in legacy_urls
    assert (
        "https://papers.legacy.test/legacy.pdf",
        "paper_reference",
        False,
    ) in {(link.url, link.relation, link.crawl) for link in legacy.links}


def test_paddlerec_catalog_skips_when_commit_is_unchanged() -> None:
    client = _QueuedClient(_commit())
    adapter = PaddleRecCatalogSourceAdapter(client=client)

    page = adapter.fetch_page({"completed_revision": _REVISION, "model_count": 41})

    assert page.records == ()
    assert page.complete
    assert page.upstream_count == 41
    assert len(client.calls) == 1


def test_paddlerec_catalog_rejects_an_unrelated_table() -> None:
    document = b"""
| Type | Implementation |
| --- | --- |
| Recall | [ExampleRec](models/recall/example/) |
"""
    client = _QueuedClient(_commit(), _response(document))

    with pytest.raises(ValueError, match="no supported-algorithm rows"):
        PaddleRecCatalogSourceAdapter(client=client).fetch_page({})
