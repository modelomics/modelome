from __future__ import annotations

import gzip
import io
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.frontier import FrontierCrawler
from modelome.gharchive_bulk import GhArchiveEventBulkLoader
from modelome.gharchive_projection import (
    GhArchiveRepositoryProjector,
    run_gharchive_repository_projection,
)
from modelome.lake import ParquetLandingZone
from modelome.models import ArtifactKind, ModelStatus, SourceRecord
from modelome.sources.gharchive import GhArchiveSourceAdapter
from modelome.storage import Database

HOUR = "2026-09-04-10"
URL = f"https://data.gharchive.org/{HOUR}.json.gz"


class Response(io.BytesIO):
    status = 200
    url = URL

    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}


class Transport:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls = 0

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> Response:
        self.calls += 1
        return Response(self.body)


def _event(event_id: str, repo_id: int, name: str, minute: int) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "PushEvent",
        "actor": {"id": repo_id + 10_000, "login": f"actor-{repo_id}"},
        "repo": {
            "id": repo_id,
            "name": name,
            "url": f"https://api.github.com/repos/{name}",
        },
        "payload": {"size": 1},
        "public": True,
        "created_at": f"2026-09-04T10:{minute:02}:00Z",
    }


def _release_event(event_id: str, repo_id: int, name: str, minute: int) -> dict[str, Any]:
    event = _event(event_id, repo_id, name, minute)
    event["type"] = "ReleaseEvent"
    event["payload"] = {
        "action": "published",
        "release": {
            "id": 700 + repo_id,
            "tag_name": "v1.0.0",
            "name": "ResNet50 model release",
            "body": "Pretrained neural model checkpoint for image classification.",
            "html_url": f"https://github.com/{name}/releases/tag/v1.0.0",
            "published_at": f"2026-09-04T10:{minute:02}:30Z",
            "assets": [
                {
                    "id": 800 + repo_id,
                    "name": "resnet50.safetensors",
                    "content_type": "application/octet-stream",
                    "size": 4096,
                    "digest": "sha256:" + "a" * 64,
                    "browser_download_url": (
                        f"https://github.com/{name}/releases/download/"
                        "v1.0.0/resnet50.safetensors"
                    ),
                },
                {
                    "id": 900 + repo_id,
                    "name": "checksums.txt",
                    "content_type": "text/plain",
                    "size": 64,
                    "browser_download_url": (
                        f"https://github.com/{name}/releases/download/"
                        "v1.0.0/checksums.txt"
                    ),
                },
            ],
        },
    }
    return event


def _sealed_lake(tmp_path: Path, *events: Mapping[str, Any]) -> ParquetLandingZone:
    jsonl = b"".join(json.dumps(event).encode() + b"\n" for event in events)
    body = gzip.compress(jsonl, mtime=0)
    source = GhArchiveSourceAdapter(
        initial_lookback_hours=1,
        availability_lag_hours=0,
        clock=lambda: datetime(2026, 9, 4, 11, 1, tzinfo=UTC),
    )
    control = source.fetch_page({}).records[0]
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = GhArchiveEventBulkLoader(
        lake,
        transport=Transport(body),
    ).load(control)
    lake.seal_release(
        source=receipt.source,
        dataset=receipt.dataset,
        release=receipt.release,
        expected_shards={receipt.shard: receipt.upstream_sha256},
    )
    return lake


def test_projector_emits_distinct_exact_candidates_with_digest_locators(
    tmp_path: Path,
) -> None:
    lake = _sealed_lake(
        tmp_path,
        _event("1", 101, "lab/active-model", 1),
        _event("2", 101, "lab/active-model", 2),
        _event("3", 102, "physics/new-surrogate", 3),
    )

    page = GhArchiveRepositoryProjector(lake).page(HOUR)

    assert page.complete is True
    assert page.rows_examined == 3
    assert page.total_rows == 3
    assert [candidate.repository_id for candidate in page.candidates] == ["101", "102"]
    first = page.candidates[0]
    assert first.repository_name == "lab/active-model"
    assert first.repository_url == "https://github.com/lab/active-model"
    assert first.repository_api_url == "https://api.github.com/repos/lab/active-model"
    assert first.event_source_record_id == "gharchive:event:1"
    assert len(first.event_json_sha256) == 64
    assert len(first.landing_record_sha256) == 64
    assert len(first.archive_sha256) == 64
    assert first.event_json_sha256 in first.locator

    record = first.as_source_record()
    assert record.source_record_id == "github-repository-id:101"
    assert record.kind is ArtifactKind.CATALOG_RECORD
    assert [(item.namespace, item.value) for item in record.identifiers] == [
        ("github:repository-id", "101"),
        ("github:repository", "lab/active-model"),
    ]
    assert record.links[0].url == "https://github.com/lab/active-model"
    assert record.links[0].crawl is True
    assert record.raw["historical_repository_census"] is False


def test_projector_bounds_work_and_resumes_at_exact_unconsumed_row(tmp_path: Path) -> None:
    lake = _sealed_lake(
        tmp_path,
        _event("11", 201, "lab/one", 1),
        _event("12", 202, "lab/two", 2),
        _event("13", 203, "lab/three", 3),
    )
    projector = GhArchiveRepositoryProjector(lake)

    first = projector.page(HOUR, max_repositories=1, max_event_rows=10)
    second = projector.page(
        HOUR,
        start_row=first.next_row,
        max_repositories=1,
        max_event_rows=10,
    )
    third = projector.page(
        HOUR,
        start_row=second.next_row,
        max_repositories=1,
        max_event_rows=10,
    )

    assert (first.start_row, first.next_row, first.complete) == (0, 1, False)
    assert (second.start_row, second.next_row, second.complete) == (1, 2, False)
    assert (third.start_row, third.next_row, third.complete) == (2, 3, True)
    assert [page.candidates[0].repository_name for page in (first, second, third)] == [
        "lab/one",
        "lab/two",
        "lab/three",
    ]


def test_projector_materializes_release_asset_candidates_without_frontier_expansion(
    tmp_path: Path,
) -> None:
    lake = _sealed_lake(
        tmp_path,
        _release_event("r1", 250, "lab/release-model", 1),
    )
    page = GhArchiveRepositoryProjector(lake).page(HOUR)

    assert len(page.candidates) == 1
    assert len(page.release_asset_records) == 1
    candidate = page.release_asset_records[0]
    assert candidate.source_record_id == "github-release-asset:250:950:1050"
    assert candidate.kind is ArtifactKind.WEIGHTS
    assert candidate.canonical_url == (
        "https://github.com/lab/release-model/releases/download/"
        "v1.0.0/resnet50.safetensors"
    )
    assert candidate.raw["candidate_confidence"] == 0.2
    assert candidate.raw["candidate_scope"] == "gharchive-event-release-asset"
    assert len(candidate.models) == 1
    assert candidate.models[0].name == "resnet50"
    assert candidate.models[0].status is ModelStatus.CANDIDATE
    assert candidate.models[0].confidence == 0.2
    assert candidate.models[0].identifiers == ()
    assert candidate.links[0].crawl is False
    assert len(page.records) == 2

    database = Database(tmp_path / "store")
    database.initialize()
    outcome = run_gharchive_repository_projection(database, lake, release=HOUR)

    assert outcome.status == "complete"
    assert {row["source_record_id"] for row in database.table_rows("artifacts")} == {
        "github-repository-id:250",
        "github-release-asset:250:950:1050",
    }
    assert [row["url"] for row in database.list_frontier()] == [
        "https://github.com/lab/release-model"
    ]
    [model] = database.search_models("resnet50")
    assert model["canonical_name"] == "resnet50"


def test_runtime_enqueues_exact_urls_and_is_idempotent(tmp_path: Path) -> None:
    lake = _sealed_lake(
        tmp_path,
        _event("21", 301, "bio/code-only-network", 1),
        _event("22", 302, "chemistry/learned-potential", 2),
    )
    database = Database(tmp_path / "store")
    database.initialize()

    outcome = run_gharchive_repository_projection(
        database,
        lake,
        release=HOUR,
    )
    repeat = run_gharchive_repository_projection(
        database,
        lake,
        release=HOUR,
    )

    assert outcome.status == "complete"
    assert outcome.repositories == 2
    assert outcome.links_discovered == 2
    assert [row["url"] for row in database.list_frontier()] == [
        "https://github.com/bio/code-only-network",
        "https://github.com/chemistry/learned-potential",
    ]
    assert repeat.status == "complete"
    assert repeat.run_id is None
    assert repeat.repositories == 0
    assert len(database.table_rows("artifacts")) == 2


def test_runtime_treats_previously_completed_unpadded_hour_as_monotonic_noop(
    tmp_path: Path,
) -> None:
    lake = _sealed_lake(
        tmp_path,
        _event("25", 351, "lab/current", 1),
    )
    database = Database(tmp_path / "store")
    database.initialize()
    current = run_gharchive_repository_projection(database, lake, release=HOUR)
    state = database.get_source_state("gharchive-repositories")

    older = run_gharchive_repository_projection(
        database,
        lake,
        release="2026-09-04-9",
    )

    assert current.status == "complete"
    assert older.status == "complete"
    assert older.run_id is None
    assert older.rows_examined == 0
    assert database.get_source_state("gharchive-repositories") == state
    assert len(database.table_rows("artifacts")) == 1


def test_runtime_checkpoint_prevents_skipping_busy_hour(tmp_path: Path) -> None:
    lake = _sealed_lake(
        tmp_path,
        _event("31", 401, "lab/one", 1),
        _event("32", 402, "lab/two", 2),
    )
    database = Database(tmp_path / "store")
    database.initialize()

    partial = run_gharchive_repository_projection(
        database,
        lake,
        release=HOUR,
        max_repositories=1,
    )

    assert partial.status == "partial"
    assert database.get_source_state("gharchive-repositories") == {
        "release": HOUR,
        "next_row": 1,
        "total_rows": 2,
        "complete": False,
    }
    try:
        run_gharchive_repository_projection(
            database,
            lake,
            release="2026-09-04-11",
        )
    except ValueError as error:
        assert "is incomplete" in str(error)
    else:
        raise AssertionError("projection advanced past an incomplete hour")

    completed = run_gharchive_repository_projection(
        database,
        lake,
        release=HOUR,
        max_repositories=1,
    )
    assert completed.status == "complete"
    assert [row["url"] for row in database.list_frontier()] == [
        "https://github.com/lab/one",
        "https://github.com/lab/two",
    ]


class CodeOnlyRepositoryFetcher:
    def accepts(self, url: str) -> bool:
        return url == "https://github.com/bio/code-only-network"

    def fetch(self, url: str) -> SourceRecord:
        return SourceRecord(
            source_record_id="bio/code-only-network",
            kind=ArtifactKind.CODE_REPOSITORY,
            canonical_url=url,
            title="bio/code-only-network",
            text=(
                "# CellSignalNet-7\n\n"
                "A deep neural network for predicting cell-to-cell signaling."
            ),
            raw={"full_name": "bio/code-only-network", "source": "README"},
        )


def test_code_only_repository_reaches_readme_extraction_end_to_end(tmp_path: Path) -> None:
    lake = _sealed_lake(
        tmp_path,
        _event("41", 501, "bio/code-only-network", 1),
    )
    database = Database(tmp_path / "store")
    database.initialize()
    projected = run_gharchive_repository_projection(database, lake, release=HOUR)

    crawled = FrontierCrawler(database, [CodeOnlyRepositoryFetcher()]).crawl(
        limit=10,
        max_depth=0,
    )

    assert projected.status == "complete"
    assert crawled.status == "complete"
    assert database.list_frontier(status="done")[0]["url"] == (
        "https://github.com/bio/code-only-network"
    )
    model = database.search_models("CellSignalNet-7")
    assert len(model) == 1
    assert model[0]["canonical_name"] == "CellSignalNet-7"
