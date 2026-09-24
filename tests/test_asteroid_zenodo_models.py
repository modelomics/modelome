from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.asteroid_zenodo_models import AsteroidZenodoModelsAdapter


def _hit(record_id: int, title: str, filename: str = "model.pth") -> dict[str, Any]:
    return {
        "id": record_id,
        "metadata": {
            "title": title,
            "doi": f"10.5281/zenodo.{record_id}",
            "creators": [{"name": "Asteroid contributor"}],
        },
        "communities": {"entries": {"asteroid-models": {"id": "asteroid-models"}}},
        "files": [
            {
                "key": filename,
                "size": 1234,
                "checksum": "md5:0123456789abcdef0123456789abcdef",
                "links": {
                    "self": f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"
                },
            }
        ],
    }


class _Client:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        del headers
        self.calls.append(url)
        index = int(url.split("page=")[1].split("&")[0]) - 1
        body = json.dumps(self.pages[index]).encode()
        return HttpResponse(200, {"content-type": "application/json"}, body, url)


def _payload(total: int, hits: list[dict[str, Any]]) -> dict[str, Any]:
    return {"hits": {"total": total, "hits": hits}}


def test_enumerates_all_community_pages_and_excludes_dataset_records() -> None:
    model1, model2 = _hit(101, "ConvTasNet Libri1Mix"), _hit(102, "DPRNN WHAM")
    dataset = _hit(103, "MiniLibriMix dataset", "MiniLibriMix.zip")
    client = _Client([_payload(3, [model1, dataset]), _payload(3, [model2])])
    adapter = AsteroidZenodoModelsAdapter(
        client=client,
        page_size=2,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert len(client.calls) == 2
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert page.next_state["community_record_count"] == 3
    assert {record.title for record in page.records} == {"ConvTasNet Libri1Mix", "DPRNN WHAM"}
    assert all(
        record.releases[0].metadata["weight_url"].endswith("/content") for record in page.records
    )
    assert all(record.links[-1].crawl is False for record in page.records)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["hits"].update(hits=[]), "incomplete"),
        (
            lambda p: p["hits"]["hits"][0]["files"][0]["links"].update(
                self="https://example.com/model.pth"
            ),
            "invalid Zenodo checkpoint content URL",
        ),
        (lambda p: p["hits"]["hits"][0]["communities"]["entries"].clear(), "outside the Asteroid"),
    ],
)
def test_fails_closed_on_inconsistent_or_untrusted_metadata(mutate, message: str) -> None:
    first = _payload(1, [_hit(101, "ConvTasNet")])
    mutate(first)
    with pytest.raises(ValueError, match=message):
        AsteroidZenodoModelsAdapter(client=_Client([first])).fetch_page({})


def test_fails_if_community_total_changes_between_pages() -> None:
    client = _Client([_payload(2, [_hit(101, "first")]), _payload(3, [_hit(102, "second")])])
    with pytest.raises(ValueError, match="total changed"):
        AsteroidZenodoModelsAdapter(client=client, page_size=1).fetch_page({})


def test_enforces_record_limit_before_paging() -> None:
    with pytest.raises(ValueError, match="exceeds 1 record limit"):
        AsteroidZenodoModelsAdapter(client=_Client([_payload(2, [])]), max_records=1).fetch_page({})
