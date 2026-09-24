from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.models import ArtifactKind, Identifier
from modelome.sources.gharchive import GhArchiveSourceAdapter

NOW = datetime(2026, 9, 4, 12, 37, 19, tzinfo=UTC)


def _source(**kwargs):
    return GhArchiveSourceAdapter(
        page_size=2,
        initial_lookback_hours=5,
        availability_lag_hours=1,
        clock=lambda: NOW,
        **kwargs,
    )


def test_enumerates_every_eligible_hour_and_resumes_without_network_queries() -> None:
    source = _source()

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)
    third = source.fetch_page(second.next_state)
    records = (*first.records, *second.records, *third.records)

    assert first.complete is False
    assert second.complete is False
    assert third.complete is True
    assert first.upstream_count == second.upstream_count == third.upstream_count == 5
    assert [record.source_record_id for record in records] == [
        "gharchive:hour:2026-09-04-6",
        "gharchive:hour:2026-09-04-7",
        "gharchive:hour:2026-09-04-8",
        "gharchive:hour:2026-09-04-9",
        "gharchive:hour:2026-09-04-10",
    ]
    assert third.next_state["next_hour"] == "2026-09-04T11:00:00Z"
    assert third.next_state["last_hour"] == "2026-09-04T10:00:00Z"
    assert "stage" not in third.next_state

    record = records[0]
    assert record.kind is ArtifactKind.CATALOG_RECORD
    assert record.canonical_url == "https://data.gharchive.org/2026-09-04-6.json.gz"
    assert record.identifiers == (Identifier("gharchive:hour", "2026-09-04-6"),)
    assert record.links[0].relation == "bulk_payload"
    assert record.links[0].crawl is False
    assert record.models == ()
    assert record.raw == {
        "record_type": "gharchive_hour_shard",
        "hour_key": "2026-09-04-6",
        "hour_start": "2026-09-04T06:00:00Z",
        "hour_end": "2026-09-04T07:00:00Z",
        "archive_path": "2026-09-04-6.json.gz",
        "stable_object_url": "https://data.gharchive.org/2026-09-04-6.json.gz",
        "archive_format": "gzip_json_lines",
        "coverage_scope": "public_github_event_activity",
        "historical_repository_census": False,
    }


def test_completed_watermark_picks_up_only_newly_closed_hours() -> None:
    clock = [NOW]
    source = GhArchiveSourceAdapter(
        page_size=24,
        initial_lookback_hours=2,
        availability_lag_hours=0,
        clock=lambda: clock[0],
    )
    first = source.fetch_page({})
    assert first.complete is True
    assert [record.raw["hour_key"] for record in first.records] == [
        "2026-09-04-10",
        "2026-09-04-11",
    ]

    unchanged = source.fetch_page(first.next_state)
    assert unchanged.complete is True
    assert unchanged.records == ()
    assert unchanged.upstream_count == 0

    clock[0] = datetime(2026, 9, 4, 15, 1, tzinfo=UTC)
    update = source.fetch_page(unchanged.next_state)
    assert [record.raw["hour_key"] for record in update.records] == [
        "2026-09-04-12",
        "2026-09-04-13",
        "2026-09-04-14",
    ]


def test_frozen_scan_ignores_clock_movement_until_resume_finishes() -> None:
    clock = [NOW]
    source = GhArchiveSourceAdapter(
        page_size=1,
        initial_lookback_hours=2,
        availability_lag_hours=0,
        clock=lambda: clock[0],
    )
    first = source.fetch_page({})
    clock[0] = datetime(2026, 9, 5, 0, 0, tzinfo=UTC)

    second = source.fetch_page(first.next_state)

    assert [record.raw["hour_key"] for record in second.records] == [
        "2026-09-04-11"
    ]
    assert second.complete is True
    assert second.next_state["next_hour"] == "2026-09-04T12:00:00Z"


def test_configuration_signature_prevents_cross_configuration_resume() -> None:
    state = _source().fetch_page({}).next_state
    changed = GhArchiveSourceAdapter(
        page_size=2,
        initial_lookback_hours=6,
        availability_lag_hours=1,
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="different adapter configuration"):
        changed.fetch_page(state)


def test_checkpoint_rejects_skips_and_noncanonical_hours() -> None:
    source = _source()

    with pytest.raises(ValueError, match="scan total"):
        source.fetch_page(
            {
                "checkpoint_signature": source.checkpoint_signature,
                "stage": "closed_hours",
                "scan_from": "2026-09-04T01:00:00Z",
                "scan_until": "2026-09-04T03:00:00Z",
                "scan_cursor": 0,
                "scan_total_hours": 3,
            }
        )
    with pytest.raises(ValueError, match="exact UTC-hour"):
        source.fetch_page(
            {
                "checkpoint_signature": source.checkpoint_signature,
                "stage": "closed_hours",
                "scan_from": "2026-09-04T01:30:00Z",
                "scan_until": "2026-09-04T03:00:00Z",
                "scan_cursor": 0,
                "scan_total_hours": 2,
            }
        )


def test_coverage_semantics_explicitly_disclaim_historical_census() -> None:
    semantics = _source().coverage_semantics

    assert semantics["historical_repository_census"] is False
    assert semantics["discovery_basis"] == "repository activity"
    assert "all public GitHub events" in semantics["enumerates"]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"page_size": 0}, "page size"),
        ({"initial_lookback_hours": 0}, "initial lookback"),
        ({"availability_lag_hours": -1}, "availability lag"),
        ({"data_url": "http://data.gharchive.org"}, "HTTPS"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, error) -> None:
    defaults = {
        "page_size": 2,
        "initial_lookback_hours": 5,
        "availability_lag_hours": 1,
        "clock": lambda: NOW,
    }
    defaults.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=error):
        GhArchiveSourceAdapter(**defaults)
