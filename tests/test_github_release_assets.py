from __future__ import annotations

from typing import Any

import pytest

from modelome.models import ArtifactKind, SourceRecord
from modelome.sources.github_release_assets import project_github_release_assets


def release_event_record(assets: list[dict[str, Any]]) -> SourceRecord:
    release = {
        "id": 77,
        "tag_name": "v1.2.0",
        "html_url": "https://github.com/lab/model/releases/tag/v1.2.0",
        "published_at": "2026-09-20T12:00:00Z",
        "assets": assets,
    }
    return SourceRecord(
        source_record_id="gharchive:event:12345",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url="https://data.gharchive.org/2026-09-20-12.json.gz",
        title="GitHub public event fixture",
        raw={
            "record_type": "gharchive_public_event",
            "event_id": "12345",
            "repository": {"id": 42, "name": "lab/model"},
            "event": {
                "id": "12345",
                "type": "ReleaseEvent",
                "created_at": "2026-09-20T12:00:00Z",
                "payload": {"action": "published", "release": release},
            },
        },
    )


def asset(asset_id: int, name: str, url: str | None = None) -> dict[str, Any]:
    return {
        "id": asset_id,
        "name": name,
        "content_type": "application/octet-stream",
        "size": 1234,
        "digest": "sha256:" + "a" * 64,
        "browser_download_url": url
        or f"https://github.com/lab/model/releases/download/v1.2.0/{name}",
    }


def test_projects_checkpoint_assets_from_a_release_event_without_api_calls() -> None:
    event = release_event_record(
        [
            asset(1, "weights.safetensors"),
            asset(2, "README.txt"),
            asset(3, "foreign.pt", "https://example.test/lab/model/releases/download/v1/foreign.pt"),
        ]
    )

    candidates = project_github_release_assets(event)

    assert len(candidates) == 1
    record = candidates[0]
    assert record.source_record_id == "github-release-asset:42:77:1"
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url.endswith("/weights.safetensors")
    assert record.identifiers[-1].value == "1"
    assert record.links[0].relation == "source_release"
    assert record.raw["is_verified_model_checkpoint"] is False


def test_projects_only_the_bounded_prefix_of_embedded_assets() -> None:
    event = release_event_record([asset(i, f"{i}.pt") for i in range(1, 5)])

    candidates = project_github_release_assets(event, max_assets=2)

    assert [item.title for item in candidates] == ["1.pt", "2.pt"]


def test_ignores_non_release_events_and_rejects_invalid_asset_limit() -> None:
    event = release_event_record([asset(1, "weights.pt")])
    raw = dict(event.raw)
    raw["event"] = {**raw["event"], "type": "PushEvent"}
    non_release = SourceRecord(
        source_record_id=event.source_record_id,
        kind=event.kind,
        canonical_url=event.canonical_url,
        title=event.title,
        raw=raw,
    )

    assert project_github_release_assets(non_release) == ()
    with pytest.raises(ValueError, match="max_assets must be a positive integer"):
        project_github_release_assets(event, max_assets=0)
