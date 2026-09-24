from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.gharchive_bulk import (
    GHARCHIVE_DEFAULT_NEW_SHARD_BUDGET,
    GhArchiveBulkLimits,
    GhArchiveEventBulkLoader,
)
from modelome.lake import ParquetLandingZone
from modelome.sources.gharchive import GhArchiveSourceAdapter

HOUR = "2026-09-04-10"
URL = f"https://data.gharchive.org/{HOUR}.json.gz"


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        url: str = URL,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        maximum_read: int = 17,
    ) -> None:
        super().__init__(body)
        self.url = url
        self.status = status
        self.headers = dict(headers or {"Content-Length": str(len(body))})
        self.maximum_read = maximum_read
        self.largest_read_request = 0

    def read(self, size: int = -1) -> bytes:
        self.largest_read_request = max(self.largest_read_request, size)
        if size < 0:
            size = self.maximum_read
        return super().read(min(size, self.maximum_read))


class FakeTransport:
    def __init__(
        self,
        bodies: list[bytes],
        *,
        final_url: str = URL,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.bodies = list(bodies)
        self.final_url = final_url
        self.status = status
        self.headers = headers
        self.calls: list[tuple[str, Mapping[str, str], Callable[[str], None]]] = []
        self.responses: list[FakeResponse] = []

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> FakeResponse:
        self.calls.append((url, dict(headers), redirect_validator))
        if not self.bodies:
            raise AssertionError("unexpected GH Archive download")
        response = FakeResponse(
            self.bodies.pop(0),
            url=self.final_url,
            status=self.status,
            headers=self.headers,
        )
        self.responses.append(response)
        return response


def _event(
    event_id: str,
    repository_id: int,
    repository_name: str,
    *,
    event_type: str = "PushEvent",
    created_at: str = "2026-09-04T10:12:34Z",
    repo_url: str | None = None,
) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": event_type,
        "actor": {"id": 91, "login": "scientist"},
        "repo": {
            "id": repository_id,
            "name": repository_name,
            "url": repo_url or f"https://api.github.com/repos/{repository_name}",
        },
        "payload": {"distinct_size": 1, "arbitrary": {"nested": True}},
        "public": True,
        "created_at": created_at,
    }


def _body(*events: Mapping[str, Any]) -> bytes:
    jsonl = b"".join(
        json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events
    )
    return gzip.compress(jsonl, mtime=0)


def _control():
    source = GhArchiveSourceAdapter(
        initial_lookback_hours=1,
        availability_lag_hours=0,
        clock=lambda: datetime(2026, 9, 4, 11, 30, tzinfo=UTC),
    )
    page = source.fetch_page({})
    assert page.complete
    assert len(page.records) == 1
    return page.records[0]


def _rows(lake: ParquetLandingZone, receipt: Any) -> list[dict[str, Any]]:
    lake.seal_release(
        source=receipt.source,
        dataset=receipt.dataset,
        release=receipt.release,
        expected_shards={receipt.shard: receipt.upstream_sha256},
    )
    return [
        row
        for batch in lake.iter_release_batches(
            source=receipt.source,
            dataset=receipt.dataset,
            release=receipt.release,
        )
        for row in batch.to_pylist()
    ]


def test_streams_every_event_and_preserves_exact_repository_evidence(tmp_path: Path) -> None:
    first = _event("1001", 501, "bio-lab/folding-system")
    second = _event(
        "1002",
        502,
        "materials-group/crystal-predictor",
        event_type="ReleaseEvent",
    )
    compressed = _body(first, second)
    content_md5 = base64.b64encode(hashlib.md5(compressed).digest()).decode()
    transport = FakeTransport(
        [compressed],
        headers={
            "Content-Length": str(len(compressed)),
            "Content-MD5": content_md5,
            "ETag": '"hour-object"',
        },
    )
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = GhArchiveEventBulkLoader(
        lake,
        transport=transport,
        limits=replace(
            GhArchiveBulkLimits(),
            download_chunk_bytes=23,
            parquet_batch_rows=1,
        ),
    )

    receipt = loader.load(_control())
    rows = _rows(lake, receipt)

    assert receipt.release == HOUR
    assert receipt.dataset == "events"
    assert receipt.hour_key == HOUR
    assert receipt.compressed_bytes == len(compressed)
    assert receipt.uncompressed_bytes == len(gzip.decompress(compressed))
    assert receipt.event_count == receipt.row_count == 2
    assert receipt.upstream_sha256 == hashlib.sha256(compressed).hexdigest()
    assert receipt.response_etag == '"hour-object"'
    assert transport.responses[0].largest_read_request == 23
    assert transport.calls[0][1]["Accept-Encoding"] == "identity"

    assert [row["source_record_id"] for row in rows] == [
        "gharchive:event:1001",
        "gharchive:event:1002",
    ]
    payload = json.loads(rows[0]["payload_json"])
    assert payload["event"] == first
    assert payload["event_type"] == "PushEvent"
    assert payload["repository"] == {
        "id": "501",
        "name": "bio-lab/folding-system",
        "api_url": "https://api.github.com/repos/bio-lab/folding-system",
        "html_url": "https://github.com/bio-lab/folding-system",
        "identifier": {
            "namespace": "github:repository",
            "value": "bio-lab/folding-system",
        },
        "stable_identifier": {
            "namespace": "github:repository-id",
            "value": "501",
        },
    }
    assert payload["evidence"]["archive_hour"] == HOUR
    assert payload["evidence"]["archive_sha256"] == receipt.upstream_sha256
    assert payload["evidence"]["event_json_sha256"] == hashlib.sha256(
        json.dumps(first, separators=(",", ":")).encode()
    ).hexdigest()
    assert "line:1:sha256:" in payload["evidence"]["locator"]


def test_no_keyword_or_event_type_filter_excludes_a_repository(tmp_path: Path) -> None:
    events = (
        _event("2001", 601, "group/ordinary-name", event_type="ForkEvent"),
        _event("2002", 602, "group/another-project", event_type="IssuesEvent"),
        _event("2003", 603, "group/unfamiliar-event", event_type="FutureEventType"),
    )
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = GhArchiveEventBulkLoader(
        lake,
        transport=FakeTransport([_body(*events)]),
    ).load(_control())

    payloads = [json.loads(row["payload_json"]) for row in _rows(lake, receipt)]

    assert [payload["repository"]["name"] for payload in payloads] == [
        "group/ordinary-name",
        "group/another-project",
        "group/unfamiliar-event",
    ]
    assert [payload["event_type"] for payload in payloads] == [
        "ForkEvent",
        "IssuesEvent",
        "FutureEventType",
    ]


def test_cached_control_is_an_offline_idempotent_noop(tmp_path: Path) -> None:
    compressed = _body(_event("3001", 701, "lab/repository"))
    transport = FakeTransport([compressed])
    loader = GhArchiveEventBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=transport,
    )

    first = loader.load(_control())
    second = loader.load(_control())

    assert first.already_committed is False
    assert second.already_committed is True
    assert second.upstream_sha256 == first.upstream_sha256
    assert second.row_count == first.row_count
    assert second.event_count is None
    assert transport.bodies == []
    assert len(transport.calls) == 1


def test_plan_uses_one_release_per_hour_and_stable_control_identity(tmp_path: Path) -> None:
    loader = GhArchiveEventBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=FakeTransport([]),
    )

    plan = loader.plan_shard(_control())

    assert plan is not None
    assert plan.source == "gharchive"
    assert plan.dataset == "events"
    assert plan.release == HOUR
    assert plan.shard == f"gharchive:hour:{HOUR}"
    assert plan.control_sha256 is not None
    assert plan.upstream_sha256 is None


def test_source_specific_budget_covers_a_day_plus_lag_recovery(tmp_path: Path) -> None:
    loader = GhArchiveEventBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=FakeTransport([]),
    )

    assert GHARCHIVE_DEFAULT_NEW_SHARD_BUDGET == 48
    assert loader.shard_budget() == 48
    assert loader.shard_budget(72) == 72
    assert loader.shard_budget(0) == 0

    with pytest.raises(ValueError, match="nonnegative"):
        loader.shard_budget(-1)


@pytest.mark.parametrize(
    "event,error",
    [
        (
            _event(
                "4001",
                801,
                "lab/outside-hour",
                created_at="2026-09-04T09:59:59Z",
            ),
            "outside control hour",
        ),
        (
            _event(
                "4002",
                802,
                "lab/name-mismatch",
                repo_url="https://api.github.com/repos/other/repository",
            ),
            "disagrees with repository name",
        ),
        (
            {
                "id": "4003",
                "type": "PushEvent",
                "created_at": "2026-09-04T10:00:00Z",
            },
            "no repository object",
        ),
    ],
)
def test_malformed_event_fails_the_shard_atomically(
    tmp_path: Path,
    event: Mapping[str, Any],
    error: str,
) -> None:
    lake = ParquetLandingZone(tmp_path / error.replace(" ", "-"))
    loader = GhArchiveEventBulkLoader(lake, transport=FakeTransport([_body(event)]))

    with pytest.raises(ValueError, match=error):
        loader.load(_control())

    assert lake.list_releases(source="gharchive", dataset="events") == ()


def test_truncated_gzip_and_transfer_bounds_fail_closed(tmp_path: Path) -> None:
    compressed = _body(_event("5001", 901, "lab/repository"))
    lake = ParquetLandingZone(tmp_path / "truncated")
    loader = GhArchiveEventBulkLoader(
        lake,
        transport=FakeTransport([compressed[:-6]]),
    )
    with pytest.raises(ValueError, match="truncated GH Archive gzip"):
        loader.load(_control())

    bounded = GhArchiveEventBulkLoader(
        ParquetLandingZone(tmp_path / "bounded"),
        transport=FakeTransport(
            [compressed],
            headers={"Content-Length": str(len(compressed))},
        ),
        limits=replace(
            GhArchiveBulkLimits(),
            max_compressed_bytes=len(compressed) - 1,
            download_chunk_bytes=min(64, len(compressed) - 1),
        ),
    )
    with pytest.raises(ValueError, match="max_compressed_bytes"):
        bounded.load(_control())


def test_control_tampering_and_redirects_are_rejected(tmp_path: Path) -> None:
    control = _control()
    tampered = replace(
        control,
        raw={**control.raw, "historical_repository_census": True},
    )
    loader = GhArchiveEventBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=FakeTransport([]),
    )
    with pytest.raises(ValueError, match="must not claim historical census"):
        loader.load(tampered)

    body = _body(_event("6001", 1001, "lab/repository"))
    redirected = GhArchiveEventBulkLoader(
        ParquetLandingZone(tmp_path / "redirected"),
        transport=FakeTransport(
            [body],
            final_url="https://objects.example.test/2026-09-04-10.json.gz",
        ),
    )
    with pytest.raises(ValueError, match="changed from its exact control"):
        redirected.load(control)
