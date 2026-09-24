from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.fourcastnet_checkpoint_registry import (
    _INDEX_URL,
    FourCastNetCheckpointRegistrySourceAdapter,
    _parse_index,
)

REV = "a" * 40
BASE = "https://portal.nersc.gov/project/m4134/FCN_weights_v0/"
INDEX = """<!doctype html><html><body>
<h1>Index of /project/m4134/FCN_weights_v0</h1>
<a href="backbone.ckpt">backbone.ckpt</a>
<a href="precip.ckpt">precip.ckpt</a>
<a href="stats_v0/">stats_v0/</a>
</body></html>"""


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.urls.append(url)
        return self.responses.pop(0)


def response(value: Any) -> HttpResponse:
    body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
    return HttpResponse(200, {}, body, "https://fixture.test")


def test_public_fourcastnet_index_emits_two_direct_checkpoint_links() -> None:
    client = Client(response({"sha": REV}), response(INDEX))
    adapter = FourCastNetCheckpointRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot and page.upstream_count == 2
    names = {record.models[0].name for record in page.records}
    assert names == {"FourCastNet backbone", "FourCastNet precipitation diagnostic"}
    weights = {
        link.url
        for record in page.records
        for link in record.links
        if link.relation == "weights"
    }
    assert weights == {BASE + "backbone.ckpt", BASE + "precip.ckpt"}
    assert client.urls == [adapter.commit_url, _INDEX_URL]


def test_unchanged_repository_revision_still_checks_remote_index() -> None:
    client = Client(
        response({"sha": REV}),
        response(INDEX),
        response({"sha": REV}),
        response(INDEX + "\n<!-- index revalidated -->"),
    )
    adapter = FourCastNetCheckpointRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    first_page = adapter.fetch_page({})
    second_page = adapter.fetch_page(first_page.next_state)

    assert len(first_page.records) == 2
    assert len(second_page.records) == 2
    assert second_page.next_state["completed_index_sha256"] != first_page.next_state[
        "completed_index_sha256"
    ]
    assert client.urls == [adapter.commit_url, _INDEX_URL, adapter.commit_url, _INDEX_URL]


@pytest.mark.parametrize(
    "document",
    [
        INDEX.replace('href="precip.ckpt"', 'href="https://evil.example/precip.ckpt"'),
        INDEX.replace('<a href="precip.ckpt">precip.ckpt</a>', ""),
        INDEX.replace('href="backbone.ckpt"', 'href="backbone.ckpt?download=1"'),
        INDEX.replace(
            "</body>",
            '<a href="extra.ckpt">extra.ckpt</a></body>',
        ),
        INDEX.replace(
            '<a href="precip.ckpt">precip.ckpt</a>',
            '<a href="precip.ckpt">precip.ckpt</a><a href="precip.ckpt">again</a>',
        ),
    ],
)
def test_index_parser_rejects_incomplete_or_untrusted_checkpoint_links(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_index(document, "test")
