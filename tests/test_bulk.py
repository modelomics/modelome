from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from modelome.bulk import BulkControlOrchestrator, BulkShardPlan
from modelome.lake import (
    LakeRecord,
    ParquetLandingZone,
    ReleaseReceipt,
    ShardApplicationOrder,
)
from modelome.models import ArtifactKind, SourceRecord
from modelome.storage import Database

SOURCE = "bulk-controls"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


@dataclass(frozen=True, slots=True)
class FakeShardReceipt:
    source: str
    dataset: str
    release: str
    shard: str
    control_sha256: str
    upstream_sha256: str
    row_count: int
    already_committed: bool


class RecordingLake:
    def __init__(self, root: Path, *, fail: bool = False) -> None:
        self.root = root
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def seal_release(self, **kwargs: Any) -> ReleaseReceipt:
        self.calls.append(kwargs)
        if self.fail:
            raise ValueError("release cannot be sealed")
        expected = kwargs["expected_shards"]
        return ReleaseReceipt(
            source=kwargs["source"],
            dataset=kwargs["dataset"],
            release=kwargs["release"],
            shard_count=len(expected),
            row_count=sum(
                item.row_count if hasattr(item, "row_count") else int(shard.rpartition("-")[2]) + 1
                for shard, item in expected.items()
            ),
            application_mode="snapshot",
            path=self.root / kwargs["release"] / "RELEASE.json",
        )


class FakeLoader:
    def __init__(
        self,
        *,
        cached: set[str] | None = None,
        fail: set[str] | None = None,
        digest_overrides: dict[str, str] | None = None,
    ) -> None:
        self.cached = cached or set()
        self.fail = fail or set()
        self.digest_overrides = digest_overrides or {}
        self.calls: list[str] = []

    def __call__(self, record: SourceRecord) -> FakeShardReceipt:
        record_id = record.source_record_id
        self.calls.append(record_id)
        if record_id in self.fail:
            raise RuntimeError("transfer failed")
        raw = record.raw
        return FakeShardReceipt(
            source=str(raw["source"]),
            dataset=str(raw["dataset"]),
            release=str(raw["release"]),
            shard=str(raw["shard"]),
            control_sha256=_control_digest(record_id),
            upstream_sha256=self.digest_overrides.get(record_id, str(raw["upstream_sha256"])),
            row_count=int(raw["rows"]),
            already_committed=record_id in self.cached,
        )


@pytest.fixture
def database(tmp_path: Path) -> Database:
    result = Database(tmp_path / "store")
    result.initialize()
    return result


def _control(
    record_id: str,
    order: int,
    *,
    dataset: str = "papers",
    release: str = "2026-09-01",
    shard: str | None = None,
    digest: str = SHA_A,
    selected: bool = True,
) -> SourceRecord:
    return SourceRecord(
        source_record_id=record_id,
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=f"https://controls.example/{record_id}",
        title=f"Control {record_id}",
        raw={
            "dataset": dataset,
            "order": order,
            "release": release,
            "rows": order + 1,
            "selected": selected,
            "shard": shard or f"shard-{order}",
            "source": "bulk-data",
            "upstream_sha256": digest,
        },
    )


def _seed(
    database: Database,
    controls: tuple[SourceRecord, ...],
    *,
    complete: bool = True,
) -> None:
    database.ingest_page(
        SOURCE,
        controls,
        {"completed_at": "2026-09-02T12:00:00Z"},
        complete=complete,
        upstream_count=len(controls),
        extractor="fixture",
    )


def _select(record: SourceRecord) -> BulkShardPlan | None:
    raw = record.raw
    if raw.get("selected") is not True:
        return None
    return BulkShardPlan(
        source=str(raw["source"]),
        dataset=str(raw["dataset"]),
        release=str(raw["release"]),
        shard=str(raw["shard"]),
        control_sha256=_control_digest(record.source_record_id),
        upstream_sha256=str(raw["upstream_sha256"]),
    )


def _control_digest(record_id: str) -> str:
    return hashlib.sha256(record_id.encode()).hexdigest()


def _run(
    database: Database,
    lake: RecordingLake,
    loader: FakeLoader,
    *,
    max_new_shards: int = 1,
):
    return BulkControlOrchestrator(database, lake).run(
        control_source=SOURCE,
        loader=loader,
        selector=_select,
        order_key=lambda record: record.raw["order"],
        max_new_shards=max_new_shards,
    )


def test_cached_receipts_do_not_spend_budget_and_a_prefix_is_never_sealed(
    database: Database,
    tmp_path: Path,
) -> None:
    _seed(
        database,
        (
            _control("control-0", 0, digest=SHA_A),
            _control("control-1", 1, digest=SHA_B),
            _control("control-2", 2, digest=SHA_C),
        ),
    )
    lake = RecordingLake(tmp_path / "lake")
    loader = FakeLoader(cached={"control-0"})

    outcome = _run(database, lake, loader, max_new_shards=1)

    assert loader.calls == ["control-0", "control-1"]
    assert outcome.selected == 3
    assert outcome.examined == 2
    assert outcome.represented == 2
    assert outcome.cached == 1
    assert outcome.new == 1
    assert outcome.rows == 3
    assert outcome.releases == ()
    assert outcome.errors == ()
    assert outcome.control_complete is True
    assert outcome.complete is False
    assert outcome.budget_exhausted is True
    assert lake.calls == []


def test_resume_walks_cached_prefix_for_free_then_seals_the_exact_release(
    database: Database,
    tmp_path: Path,
) -> None:
    _seed(
        database,
        (
            _control("control-0", 0, digest=SHA_A),
            _control("control-1", 1, digest=SHA_B),
            _control("control-2", 2, digest=SHA_C),
        ),
    )
    lake = RecordingLake(tmp_path / "lake")
    loader = FakeLoader(cached={"control-0", "control-1"})

    outcome = _run(database, lake, loader, max_new_shards=1)

    assert loader.calls == ["control-0", "control-1", "control-2"]
    assert (outcome.examined, outcome.cached, outcome.new, outcome.rows) == (3, 2, 1, 6)
    assert outcome.represented == 3
    assert outcome.complete is True
    assert outcome.budget_exhausted is False
    assert len(outcome.releases) == 1
    assert lake.calls == [
        {
            "source": "bulk-data",
            "dataset": "papers",
            "release": "2026-09-01",
            "expected_shards": {
                "shard-0": SHA_A,
                "shard-1": SHA_B,
                "shard-2": SHA_C,
            },
        }
    ]


def test_selector_filters_records_and_receipts_are_grouped_into_releases(
    database: Database,
    tmp_path: Path,
) -> None:
    _seed(
        database,
        (
            _control("ignored", 0, selected=False),
            _control("abstract", 3, dataset="abstracts", digest=SHA_B),
            _control("paper", 2, dataset="papers", digest=SHA_A),
        ),
    )
    lake = RecordingLake(tmp_path / "lake")
    loader = FakeLoader(cached={"abstract", "paper"})

    outcome = _run(database, lake, loader)

    assert loader.calls == ["paper", "abstract"]
    assert outcome.selected == 2
    assert outcome.examined == 2
    assert outcome.new == 0
    assert outcome.cached == 2
    assert outcome.complete is True
    assert {(item.dataset, item.release) for item in outcome.releases} == {
        ("papers", "2026-09-01"),
        ("abstracts", "2026-09-01"),
    }
    assert len(lake.calls) == 2


def test_incomplete_or_failed_control_plane_can_preload_but_never_seal(
    database: Database,
    tmp_path: Path,
) -> None:
    _seed(database, (_control("control-0", 0),), complete=False)
    lake = RecordingLake(tmp_path / "lake")
    loader = FakeLoader()

    incomplete = _run(database, lake, loader)
    assert incomplete.represented == 1
    assert incomplete.control_complete is False
    assert incomplete.complete is False
    assert incomplete.releases == ()
    assert lake.calls == []

    _seed(database, (_control("control-0", 0),), complete=True)
    run_id = database.start_run(SOURCE)
    database.finish_run(run_id, "failed", error="manifest failed")
    failed = _run(
        database,
        lake,
        FakeLoader(cached={"control-0"}),
    )
    assert failed.control_complete is False
    assert failed.releases == ()
    assert lake.calls == []


def test_duplicate_selected_shards_with_conflicting_digests_fail_before_loading(
    database: Database,
    tmp_path: Path,
) -> None:
    _seed(
        database,
        (
            _control("first", 0, shard="same", digest=SHA_A),
            _control("second", 1, shard="same", digest=SHA_B),
        ),
    )
    lake = RecordingLake(tmp_path / "lake")
    loader = FakeLoader()

    outcome = _run(database, lake, loader, max_new_shards=2)

    assert loader.calls == []
    assert outcome.examined == 0
    assert outcome.represented == 0
    assert len(outcome.errors) == 1
    assert outcome.errors[0].stage == "select"
    assert "conflicting digests" in outcome.errors[0].error
    assert outcome.complete is False
    assert lake.calls == []


def test_receipt_digest_conflict_and_loader_error_fail_closed(
    database: Database,
    tmp_path: Path,
) -> None:
    _seed(
        database,
        (
            _control("first", 0, digest=SHA_A),
            _control("second", 1, digest=SHA_B),
        ),
    )
    lake = RecordingLake(tmp_path / "lake")
    conflicting = FakeLoader(digest_overrides={"first": SHA_C})

    mismatch = _run(database, lake, conflicting, max_new_shards=2)
    assert conflicting.calls == ["first"]
    assert mismatch.new == 1
    assert mismatch.represented == 0
    assert len(mismatch.errors) == 1
    assert "digest does not match" in mismatch.errors[0].error
    assert lake.calls == []

    failed_loader = FakeLoader(fail={"first"})
    failed = _run(database, lake, failed_loader, max_new_shards=2)
    assert failed_loader.calls == ["first"]
    assert failed.new == 0
    assert failed.cached == 0
    assert len(failed.errors) == 1
    assert failed.errors[0].stage == "load"
    assert lake.calls == []


def test_seal_errors_are_structured_and_zero_budget_performs_no_transfer(
    database: Database,
    tmp_path: Path,
) -> None:
    _seed(database, (_control("control-0", 0),))
    loader = FakeLoader()
    no_transfer_lake = RecordingLake(tmp_path / "no-transfer")

    stopped = _run(database, no_transfer_lake, loader, max_new_shards=0)
    assert loader.calls == []
    assert stopped.examined == 0
    assert stopped.budget_exhausted is True
    assert stopped.errors == ()

    failing_lake = RecordingLake(tmp_path / "failing", fail=True)
    failed = _run(database, failing_lake, FakeLoader())
    assert len(failed.errors) == 1
    assert failed.errors[0].stage == "seal"
    assert failed.complete is False
    assert failed.releases == ()


def test_real_landing_zone_receipt_is_selected_exactly_and_published(
    database: Database,
    tmp_path: Path,
) -> None:
    control = _control("control-0", 0, digest=SHA_A)
    _seed(database, (control,))
    lake = ParquetLandingZone(tmp_path / "real-lake")

    def load(record: SourceRecord):
        raw = record.raw
        return lake.commit_shard(
            source=str(raw["source"]),
            dataset=str(raw["dataset"]),
            release=str(raw["release"]),
            shard=str(raw["shard"]),
            control_sha256=_control_digest(record.source_record_id),
            upstream_sha256=str(raw["upstream_sha256"]),
            records=(LakeRecord("row-1", {"value": 1}),),
            application_order=ShardApplicationOrder.snapshot(0),
        )

    outcome = BulkControlOrchestrator(database, lake).run(
        control_source=SOURCE,
        loader=load,
        selector=_select,
        max_new_shards=1,
    )

    assert outcome.complete is True
    assert outcome.new == 1
    assert len(outcome.releases) == 1
    batches = list(
        lake.iter_release_batches(
            source="bulk-data",
            dataset="papers",
            release="2026-09-01",
        )
    )
    assert [row["source_record_id"] for batch in batches for row in batch.to_pylist()] == ["row-1"]


@pytest.mark.parametrize("budget", [-1, True, 1.5, None])
def test_new_shard_budget_must_be_a_nonnegative_integer(
    database: Database,
    tmp_path: Path,
    budget: Any,
) -> None:
    with pytest.raises(ValueError, match="max_new_shards"):
        BulkControlOrchestrator(database, RecordingLake(tmp_path / "lake")).run(
            control_source=SOURCE,
            loader=FakeLoader(),
            selector=_select,
            max_new_shards=budget,
        )
