from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from modelome.models import ArtifactKind, ModelStatus
from modelome.sources.grand_challenge_algorithms import (
    GrandChallengeAlgorithmsSourceAdapter,
    _record,
)


def _row(pk: str = "9e3de616-b207-4f21-a484-34c0f8774cf2") -> dict[str, object]:
    return {
        "pk": pk,
        "slug": "ct-lesion-model",
        "url": "https://grand-challenge.org/algorithms/ct-lesion-model/",
        "api_url": f"https://grand-challenge.org/api/v1/algorithms/{pk}/",
        "title": "CT lesion model",
        "description": "Detect lesions in CT scans.",
        "interfaces": [{"inputs": [], "outputs": []}],
    }


class _Client:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, *, params: dict[str, object], headers: dict[str, str]):
        self.calls.append((url, params))
        payload = self.payloads.pop(0)
        body = json.dumps(payload).encode()
        return SimpleNamespace(status=200, body=body, json=lambda: payload)


def _listing(rows: list[dict[str, object]], count: int, next_url: str | None):
    return {"count": count, "next": next_url, "previous": None, "results": rows}


def test_fetch_page_respects_limit_offset_and_exact_count_continuity() -> None:
    first_url = "https://grand-challenge.org/api/v1/algorithms/?limit=1&offset=1"
    client = _Client(
        [
            _listing([_row()], 2, first_url),
            _listing([_row("second-id")], 2, None),
        ]
    )
    adapter = GrandChallengeAlgorithmsSourceAdapter(client=client, page_size=1)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert not first.complete and first.next_state == {"offset": 1, "count": 2}
    assert second.complete and second.next_state == {"offset": 2, "count": 2}
    assert first.upstream_count == second.upstream_count == 2
    assert client.calls == [
        ("https://grand-challenge.org/api/v1/algorithms/", {"limit": 1, "offset": 0}),
        ("https://grand-challenge.org/api/v1/algorithms/", {"limit": 1, "offset": 1}),
    ]


def test_record_preserves_public_card_identity_without_claiming_weights() -> None:
    record = _record(_row(), "grand-challenge-public-algorithms")

    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.canonical_url == "https://grand-challenge.org/algorithms/ct-lesion-model"
    assert record.models[0].identifiers[0].value == "9e3de616-b207-4f21-a484-34c0f8774cf2"
    assert record.models[0].status is ModelStatus.DOCUMENTED
    assert {link.relation for link in record.links} == {
        "provider_model_card",
        "provider_api_record",
    }
    assert not record.releases


def test_invalid_next_link_and_total_changes_fail_closed() -> None:
    client = _Client(
        [_listing([_row()], 2, "https://example.org/api/v1/algorithms/?limit=1&offset=1")]
    )
    adapter = GrandChallengeAlgorithmsSourceAdapter(client=client, page_size=1)
    with pytest.raises(ValueError, match="escaped its API endpoint"):
        adapter.fetch_page({})

    client = _Client(
        [_listing([_row()], 3, "https://grand-challenge.org/api/v1/algorithms/?limit=1&offset=1")]
    )
    adapter = GrandChallengeAlgorithmsSourceAdapter(client=client, page_size=1)
    with pytest.raises(ValueError, match="total changed"):
        adapter.fetch_page({"offset": 1, "count": 2})


def test_listing_cannot_end_before_declared_count() -> None:
    adapter = GrandChallengeAlgorithmsSourceAdapter(
        client=_Client([_listing([_row()], 2, None)]), page_size=2
    )

    with pytest.raises(ValueError, match="ended before"):
        adapter.fetch_page({})
