from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.utile_checkpoint_registry import (
    UTilizeCheckpointRegistrySourceAdapter,
    _parse_directory,
)

FILES = (
    "utilise_earthnet2021.pth",
    "utilise_sen12mscrts_w_s1.pth",
    "utilise_sen12mscrts_wo_s1.pth",
)
INDEX = (
    "<html><body>"
    + "".join(f'<a href="{name}">{name}</a>' for name in FILES)
    + "</body></html>"
)


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


def response(body: str, status: int = 200) -> HttpResponse:
    return HttpResponse(status, {"content-type": "text/html"}, body.encode(), "https://fixture.test")


def test_adapter_emits_exact_first_party_checkpoint_urls_without_fetching_bytes() -> None:
    client = Client(response(INDEX))
    page = UTilizeCheckpointRegistrySourceAdapter(client=client).fetch_page({})
    expected = {
        f"https://share.phys.ethz.ch/~pf/stuckercdata/u-tilise/checkpoints/{name}"
        for name in FILES
    }
    observed = {
        link.url
        for record in page.records
        for link in record.links
        if link.relation == "weights"
    }

    assert page.authoritative_snapshot and page.complete and page.upstream_count == 3
    assert observed == expected
    assert client.urls == [
        "https://share.phys.ethz.ch/~pf/stuckercdata/u-tilise/checkpoints/"
    ]
    assert len({record.models[0].local_id for record in page.records}) == 3


def test_adapter_uses_index_digest_for_freshness() -> None:
    client = Client(response(INDEX), response(INDEX))
    adapter = UTilizeCheckpointRegistrySourceAdapter(client=client)
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    assert len(first.records) == 3
    assert second.records == ()
    assert second.next_state == first.next_state


@pytest.mark.parametrize(
    "document",
    [
        INDEX.replace(FILES[0], "other.pth", 1),
        INDEX.replace(f'href="{FILES[0]}"', f'href="https://evil.example/{FILES[0]}"'),
        INDEX + f'<a href="{FILES[1]}">duplicate</a>',
        INDEX.replace(f'href="{FILES[0]}"', f'href="{FILES[0]}?download=1"'),
    ],
)
def test_parser_rejects_incomplete_noncanonical_or_duplicate_links(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_directory(document, "test")


def test_adapter_rejects_failed_index_response() -> None:
    adapter = UTilizeCheckpointRegistrySourceAdapter(client=Client(response(INDEX, 404)))
    with pytest.raises(ValueError, match="HTTP 404"):
        adapter.fetch_page({})
