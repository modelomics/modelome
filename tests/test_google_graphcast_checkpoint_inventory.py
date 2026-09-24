from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.google_graphcast_checkpoint_inventory import (
    GoogleGraphCastCheckpointInventorySourceAdapter,
)


class _Client:
    def __init__(self, *payloads: Any) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        return HttpResponse(200, {}, json.dumps(self.payloads.pop(0)).encode(), url)


def _item(name: str, generation: str = "1730412300000000") -> dict[str, str]:
    return {
        "name": f"graphcast/params/{name}",
        "generation": generation,
        "updated": "2026-09-24T12:00:00Z",
        "size": "3000000000",
    }


def test_projects_only_graphcast_checkpoint_objects_from_official_prefix() -> None:
    client = _Client(
        {
            "items": [
                _item("GraphCast - ERA5 1979-2017 - resolution 0.25 - params.npz"),
                _item("GraphCast_small - ERA5 1979-2015 - resolution 1.0 - params.npz"),
                _item("GraphCast_operational - ERA5-HRES 1979-2021 - params.npz"),
                _item("stats/stats_mean_by_level.nc"),
                _item("other-model-params.npz"),
                {"name": "dataset/era5.nc", "generation": "1"},
            ]
        }
    )
    adapter = GoogleGraphCastCheckpointInventorySourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete is True and page.upstream_count == 6
    assert [record.title for record in page.records] == [
        "GraphCast checkpoint",
        "GraphCast_small checkpoint",
        "GraphCast_operational checkpoint",
    ]
    operational = page.records[2]
    assert operational.models[0].identifiers == (
        Identifier("google:graphcast-model", "GraphCast_operational"),
    )
    assert operational.releases[0].revision == "1730412300000000"
    assert operational.identifiers == (
        Identifier(
            "google:graphcast-checkpoint",
            "GraphCast_operational - ERA5-HRES 1979-2021 - params.npz",
        ),
    )
    assert operational.links[0].url.endswith(
        "graphcast%2Fparams%2FGraphCast_operational%20-%20ERA5-HRES%201979-2021%20-%20params.npz"
        "?alt=media&generation=1730412300000000"
    )
    assert client.calls[0][0] == ("https://storage.googleapis.com/storage/v1/b/dm_graphcast/o")
    assert client.calls[0][1]["prefix"] == "graphcast/params/"


def test_build_entries_groups_graphcast_releases_by_family_and_keeps_variants_separate() -> None:
    page = GoogleGraphCastCheckpointInventorySourceAdapter(
        client=_Client(
            {
                "items": [
                    _item("GraphCast - ERA5 1979-2017 - resolution 0.25 - params.npz"),
                    _item(
                        "GraphCast - ERA5 1979-2021 - resolution 0.25 - params.npz",
                        generation="1730412300000001",
                    ),
                    _item("GraphCast_small - ERA5 1979-2015 - resolution 1.0 - params.npz"),
                ]
            }
        )
    ).fetch_page({})

    result = build_entries(
        source_record_to_entry_seed(record, source="google-graphcast-checkpoint-inventory")
        for record in page.records
    )

    assert len(result.entries) == 2
    graphcast = next(entry for entry in result.entries if entry.canonical_name == "GraphCast")
    small = next(
        entry for entry in result.entries if entry.canonical_name == "GraphCast_small"
    )
    assert len(graphcast.members) == 2
    assert len(graphcast.releases) == 2
    assert len(small.members) == 1
    assert len(small.releases) == 1
    assert {
        resource.source_record_id
        for resource in graphcast.resources
        if resource.relation == "documentation"
    } == {member.source_record_id for member in graphcast.members}


def test_checkpoint_listing_pages_follow_provider_token() -> None:
    first_item = _item("GraphCast_small - ERA5 1979-2015.npz")
    second_item = _item("GraphCast - ERA5 1979-2017.npz")
    client = _Client(
        {"items": [first_item], "nextPageToken": "next-page"},
        {"items": [second_item]},
    )
    adapter = GoogleGraphCastCheckpointInventorySourceAdapter(page_size=1, client=client)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state["page_token"] == "next-page"
    assert second.complete is True
    assert [record.models[0].name for record in first.records + second.records] == [
        "GraphCast_small",
        "GraphCast",
    ]
    assert client.calls[1][1]["pageToken"] == "next-page"


def test_rejects_invalid_checkpoint_generation() -> None:
    client = _Client({"items": [_item("GraphCast sample.npz", "not-a-generation")]})
    with pytest.raises(ValueError, match="generation"):
        GoogleGraphCastCheckpointInventorySourceAdapter(client=client).fetch_page({})


def test_disabled_proposal_matches_adapter_settings() -> None:
    proposal_path = (
        Path(__file__).parents[1] / "config/proposals/google_graphcast_checkpoint_inventory.toml"
    )
    source = tomllib.loads(proposal_path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = GoogleGraphCastCheckpointInventorySourceAdapter(
        name=source["name"],
        page_size=source["page_size"],
        max_response_bytes=source["max_response_bytes"],
    )
    assert adapter.name == source["name"]
    assert adapter.page_size == source["page_size"]
    assert adapter.max_response_bytes == source["max_response_bytes"]
