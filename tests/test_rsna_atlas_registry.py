from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from modelome.models import ArtifactKind, ModelStatus
from modelome.sources.rsna_atlas_registry import (
    RsnaAtlasRegistrySourceAdapter,
    _availability_urls,
    _record,
)


def _card(
    *,
    card_id: str = "123e4567-e89b-12d3-a456-426614174000",
    provider_id: str = "1763759773566",
    name: str = "Example CT model",
    availability: str = "Weights info: https://example.org/model.pth.",
    schema_version: str = "2025-11/model.json",
    publishing_status: str = "published",
) -> dict[str, object]:
    return {
        "id": card_id,
        "indexCardId": provider_id,
        "title": name + " model card",
        "schemaVersion": schema_version,
        "publishingStatus": publishing_status,
        "roadmapObject": json.dumps({
            "$schema": "https://atlas.rsna.org/schemas/2025-11/model.json",
            "Model": {
                "Name": name,
                "Link": "https://github.com/example/model",
                "Descriptors": {"Version": "v2"},
                "Model properties": {"Availability": availability},
            },
        }),
    }


class _Client:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def get(self, url: str, *, params: dict[str, object], headers: dict[str, str]):
        self.calls.append((url, params, headers))
        payload = self.payloads.pop(0)
        body = json.dumps(payload).encode()
        return SimpleNamespace(status=200, body=body, json=lambda: payload)


def _page(items: list[dict[str, object]], next_token: str | None = None):
    return {"data": {"listIndexCards": {"items": items, "nextToken": next_token}}}


def test_fetch_page_paginates_and_only_emits_published_model_cards() -> None:
    client = _Client([
        _page([
            _card(),
            _card(card_id="dataset-row", schema_version="2025-11/dataset.json"),
            _card(card_id="draft-row", publishing_status="draft"),
        ], "opaque-next-token"),
        _page([_card(card_id="later-model")]),
    ])
    adapter = RsnaAtlasRegistrySourceAdapter(client=client, page_size=3)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert len(first.records) == 1
    assert not first.complete
    assert first.next_state == {"next_token": "opaque-next-token", "page_index": 1}
    assert len(second.records) == 1
    assert second.complete
    assert second.next_state == {"next_token": None, "page_index": 2}
    first_variables = json.loads(client.calls[0][1]["variables"])
    second_variables = json.loads(client.calls[1][1]["variables"])
    assert first_variables == {"limit": 3}
    assert second_variables == {"limit": 3, "nextToken": "opaque-next-token"}
    assert client.calls[0][2]["x-api-key"]


def test_schema_mislabeled_dataset_row_is_excluded_from_model_cards() -> None:
    malformed = _card(card_id="bad-schema-row")
    malformed["roadmapObject"] = json.dumps({"Dataset": {"Name": "not a model"}})
    client = _Client([_page([_card(), malformed])])
    adapter = RsnaAtlasRegistrySourceAdapter(client=client, page_size=2)

    page = adapter.fetch_page({})

    assert len(page.records) == 1
    assert not page.issues
    assert not page.advance_on_source_issues


def test_record_preserves_card_identity_and_untyped_availability_links() -> None:
    record = _record(_card())

    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.source_record_id == "rsna-atlas:model-card:123e4567-e89b-12d3-a456-426614174000"
    assert record.models[0].identifiers[0].value == "123e4567-e89b-12d3-a456-426614174000"
    assert record.models[0].status is ModelStatus.DOCUMENTED
    assert record.releases[0].version == "v2"
    availability_link = next(
        link for link in record.links if link.relation == "availability_reference"
    )
    assert availability_link.url == "https://example.org/model.pth"
    assert availability_link.crawl is False
    assert "weight" not in availability_link.relation


def test_card_without_availability_is_documented_and_has_no_release_link() -> None:
    card = _card(availability="Not publicly available")
    record = _record(card)

    assert record.models[0].status is ModelStatus.DOCUMENTED
    assert not any(link.relation == "availability_reference" for link in record.links)


def test_availability_url_extraction_is_literal_deduplicated_and_https_only() -> None:
    assert _availability_urls(
        "weights at https://example.org/a.pth. repeated https://example.org/a.pth; "
        "ignore http://example.org/insecure"
    ) == ("https://example.org/a.pth",)


def test_invalid_model_card_and_response_errors_fail_closed() -> None:
    card = _card()
    card["roadmapObject"] = "not-json"
    with pytest.raises(ValueError, match="invalid ROADMAP JSON"):
        _record(card)

    client = _Client([{"errors": [{"message": "bad query"}]}])
    adapter = RsnaAtlasRegistrySourceAdapter(client=client)
    with pytest.raises(ValueError, match="GraphQL returned errors"):
        adapter.fetch_page({})
