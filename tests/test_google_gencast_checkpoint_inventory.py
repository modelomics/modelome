from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.google_gencast_checkpoint_inventory import (
    _PREFIX,
    GoogleGenCastCheckpointInventorySourceAdapter,
    _record,
)

BASE = "https://storage.googleapis.com/download/storage/v1/b/dm_graphcast/o/"


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any] | None]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, params))
        return self.responses.pop(0)


def response(value: Any) -> HttpResponse:
    body = json.dumps(value).encode()
    return HttpResponse(200, {}, body, "https://fixture.test")


def test_gencast_inventory_requests_only_the_first_party_prefix_and_paginates() -> None:
    client = Client(
        response(
            {
                "items": [
                    {
                        "name": _PREFIX + "GenCast 1p0deg Mini <2019.npz",
                        "generation": "123",
                        "size": "456",
                    },
                    {
                        "name": _PREFIX + "other-checkpoint.npz",
                        "generation": "124",
                    },
                ],
                "nextPageToken": "next",
            }
        ),
        response(
            {
                "items": [
                    {
                        "name": _PREFIX + "GenCast 0p25deg Operational <2022.npz",
                        "generation": "789",
                    }
                ]
            }
        ),
    )
    adapter = GoogleGenCastCheckpointInventorySourceAdapter(client=client)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert not first.complete and second.complete
    assert first.upstream_count == 2 and second.upstream_count == 1
    record = first.records[0]
    assert record.models[0].name == "GenCast 1p0deg Mini <2019"
    assert record.releases[0].metadata["size_bytes"] == 456
    assert record.links[0].url == (
        BASE + "gencast%2Fparams%2FGenCast%201p0deg%20Mini%20%3C2019.npz"
        "?alt=media&generation=123"
    )
    assert client.calls[0][1]["prefix"] == _PREFIX
    assert client.calls[1][1]["pageToken"] == "next"
    assert all(call[0].endswith("/o") for call in client.calls)


@pytest.mark.parametrize(
    "item",
    [
        {"name": _PREFIX + "GenCast 1p0deg <2019.npz"},
        {"name": "graphcast/params/GenCast 1p0deg <2019.npz", "generation": "123"},
    ],
)
def test_gencast_emits_only_recognized_models_and_requires_generation(item: dict[str, Any]) -> None:
    if item.get("name", "").startswith(_PREFIX) and item["name"].endswith(".npz"):
        with pytest.raises(ValueError, match="generation"):
            _record("test", item)
    else:
        assert _record("test", item) is None


def test_invalid_page_token_is_rejected_before_request() -> None:
    client = Client()
    adapter = GoogleGenCastCheckpointInventorySourceAdapter(client=client)
    with pytest.raises(ValueError, match="page token"):
        adapter.fetch_page({"page_token": ""})
    assert client.calls == []
