from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from modelome.lake import (
    LakeRecord,
    ParquetLandingZone,
    ShardApplicationOrder,
    canonical_control_sha256,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def control_sha(shard: str, revision: str = "initial") -> str:
    return hashlib.sha256(f"{shard}:{revision}".encode()).hexdigest()


def records(count: int):
    for index in range(count):
        yield LakeRecord(
            source_record_id=f"paper-{index}",
            payload={"paperId": f"paper-{index}", "rank": index},
        )


def commit(
    lake: ParquetLandingZone,
    *,
    shard: str = "papers-000",
    digest: str = SHA_A,
    count: int = 5,
):
    return lake.commit_shard(
        source="semantic-scholar",
        dataset="papers",
        release="2026-09-01",
        shard=shard,
        control_sha256=control_sha(shard),
        upstream_sha256=digest,
        records=records(count),
        expected_rows=count,
        batch_rows=2,
        upstream_url=f"https://example.test/{shard}.jsonl.gz",
    )


def test_shard_streams_to_bounded_parquet_parts_and_scans_only_after_seal(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")

    receipt = commit(lake)

    assert receipt.row_count == 5
    assert receipt.part_count == 3
    assert receipt.path.is_relative_to(lake.root)
    with pytest.raises(ValueError, match="not sealed"):
        list(
            lake.iter_release_batches(
                source="semantic-scholar",
                dataset="papers",
                release="2026-09-01",
            )
        )

    release = lake.seal_release(
        source="semantic-scholar",
        dataset="papers",
        release="2026-09-01",
        expected_shards={"papers-000": SHA_A},
    )
    batches = list(
        lake.iter_release_batches(
            source="semantic-scholar",
            dataset="papers",
            release="2026-09-01",
            batch_size=2,
        )
    )
    rows = [row for batch in batches for row in batch.to_pylist()]

    assert release.shard_count == 1
    assert release.row_count == 5
    assert [row["source_record_id"] for row in rows] == [
        "paper-0",
        "paper-1",
        "paper-2",
        "paper-3",
        "paper-4",
    ]
    assert json.loads(rows[3]["payload_json"])["rank"] == 3
    assert all(row["operation"] == "upsert" for row in rows)
    assert all(len(row["content_sha256"]) == 64 for row in rows)


def test_list_releases_reports_only_verified_seals_and_supports_filters(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    first = commit(lake, count=3)
    lake.seal_release(
        source=first.source,
        dataset=first.dataset,
        release=first.release,
        expected_shards={first.shard: first},
    )
    second = lake.commit_shard(
        source="commoncrawl",
        dataset="wet",
        release="CC-MAIN-2026-30",
        shard="wet-000",
        control_sha256=control_sha("wet-000"),
        upstream_sha256=SHA_B,
        records=(LakeRecord("capture-1", {"text": "one"}),),
        expected_rows=1,
    )
    lake.seal_release(
        source=second.source,
        dataset=second.dataset,
        release=second.release,
        expected_shards={second.shard: second},
    )
    # A committed but unsealed shard must never appear as a complete release.
    lake.commit_shard(
        source="pubmed",
        dataset="citations",
        release="2026-baseline",
        shard="baseline-000",
        control_sha256=control_sha("baseline-000"),
        upstream_sha256="c" * 64,
        records=(LakeRecord("pmid:1", {"pmid": "1"}),),
    )

    releases = lake.list_releases()

    assert [
        (item.source, item.dataset, item.release, item.row_count)
        for item in releases
    ] == [
        ("commoncrawl", "wet", "CC-MAIN-2026-30", 1),
        ("semantic-scholar", "papers", "2026-09-01", 3),
    ]
    assert all(item.already_sealed for item in releases)
    assert lake.list_releases(source="commoncrawl") == (releases[0],)
    assert lake.list_releases(dataset="papers") == (releases[1],)


def test_list_releases_detects_corrupt_sealed_shards(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    shard = commit(lake, count=1)
    lake.seal_release(
        source=shard.source,
        dataset=shard.dataset,
        release=shard.release,
        expected_shards={shard.shard: shard},
    )
    part = next((shard.path / "parts").glob("*.parquet"))
    payload = bytearray(part.read_bytes())
    payload[-1] ^= 0x01
    part.write_bytes(payload)

    with pytest.raises(ValueError, match="checksum mismatch"):
        lake.list_releases()

    # Metadata-only discovery is explicit and does not claim payload verification.
    assert lake.list_releases(verify_shards=False)[0].row_count == 1


def test_identical_upstream_shard_is_idempotent_without_consuming_records(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    first = commit(lake)

    def must_not_iterate():
        raise AssertionError("an already committed shard must not consume input")
        yield LakeRecord("unreachable", {})

    repeated = lake.commit_shard(
        source="semantic-scholar",
        dataset="papers",
        release="2026-09-01",
        shard="papers-000",
        control_sha256=control_sha("papers-000"),
        upstream_sha256=SHA_A,
        upstream_url="https://example.test/papers-000.jsonl.gz",
        records=must_not_iterate(),
    )

    assert repeated.path == first.path
    assert repeated.already_committed is True
    assert repeated.row_count == 5


def test_count_mismatch_never_publishes_partial_shard(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")

    with pytest.raises(ValueError, match="expected 6 rows, received 5"):
        lake.commit_shard(
            source="semantic-scholar",
            dataset="papers",
            release="2026-09-01",
            shard="papers-000",
            control_sha256=control_sha("papers-000"),
            upstream_sha256=SHA_A,
            records=records(5),
            expected_rows=6,
            batch_rows=2,
        )

    assert list(lake.staging_root.iterdir()) == []
    assert not any(lake.shards_root.rglob("manifest.json"))


def test_release_requires_exact_available_checksum_set_and_is_idempotent(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    commit(lake, shard="papers-000", digest=SHA_A, count=2)

    with pytest.raises(ValueError, match="missing"):
        lake.seal_release(
            source="semantic-scholar",
            dataset="papers",
            release="2026-09-01",
            expected_shards={"papers-000": SHA_A, "papers-001": SHA_B},
        )

    first = lake.seal_release(
        source="semantic-scholar",
        dataset="papers",
        release="2026-09-01",
        expected_shards={"papers-000": SHA_A},
    )
    repeated = lake.seal_release(
        source="semantic-scholar",
        dataset="papers",
        release="2026-09-01",
        expected_shards={"papers-000": SHA_A},
    )

    assert first.already_sealed is False
    assert repeated.already_sealed is True
    with pytest.raises(ValueError, match="different shard set"):
        lake.seal_release(
            source="semantic-scholar",
            dataset="papers",
            release="2026-09-01",
            expected_shards={"papers-000": SHA_B},
        )


def test_part_corruption_is_detected_before_reuse_or_scan(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = commit(lake, count=1)
    part = next((receipt.path / "parts").glob("*.parquet"))
    payload = bytearray(part.read_bytes())
    payload[-1] ^= 0x01
    part.write_bytes(payload)

    with pytest.raises(ValueError, match="checksum mismatch"):
        commit(lake, count=1)


def test_verified_control_lookup_hits_exact_descriptor_and_fails_closed(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = commit(lake, count=1)
    exact = lake.lookup_committed_shard(
        source=receipt.source,
        dataset=receipt.dataset,
        release=receipt.release,
        shard=receipt.shard,
        control_sha256=receipt.control_sha256,
        upstream_url=receipt.upstream_url,
        application_order=receipt.application_order,
    )

    assert exact is not None
    assert exact.path == receipt.path
    assert exact.already_committed is True
    assert (
        lake.lookup_committed_shard(
            source=receipt.source,
            dataset=receipt.dataset,
            release=receipt.release,
            shard=receipt.shard,
            control_sha256=control_sha(receipt.shard, "changed"),
            upstream_url=receipt.upstream_url,
            application_order=receipt.application_order,
        )
        is None
    )
    with pytest.raises(ValueError, match="different upstream URL"):
        lake.lookup_committed_shard(
            source=receipt.source,
            dataset=receipt.dataset,
            release=receipt.release,
            shard=receipt.shard,
            control_sha256=receipt.control_sha256,
            upstream_url="https://example.test/different.jsonl.gz",
            application_order=receipt.application_order,
        )


def test_control_lookup_verifies_manifest_checksum_and_canonical_directory(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "first-lake")
    receipt = commit(lake, count=1)
    manifest_path = receipt.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["upstream_url"] = "https://example.test/tampered.jsonl.gz"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        lake.lookup_committed_shard(
            source=receipt.source,
            dataset=receipt.dataset,
            release=receipt.release,
            shard=receipt.shard,
            control_sha256=receipt.control_sha256,
            upstream_url=receipt.upstream_url,
        )

    second_lake = ParquetLandingZone(tmp_path / "second-lake")
    second = commit(second_lake, count=1)
    renamed = second.path.parent / ("0" * 64)
    second.path.rename(renamed)
    with pytest.raises(ValueError, match="path does not match"):
        second_lake.lookup_committed_shard(
            source=second.source,
            dataset=second.dataset,
            release=second.release,
            shard=second.shard,
            control_sha256=second.control_sha256,
            upstream_url=second.upstream_url,
        )


def test_changed_controls_with_same_bytes_are_distinct_and_digest_seal_is_ambiguous(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    first = commit(lake, count=1)
    second = lake.commit_shard(
        source=first.source,
        dataset=first.dataset,
        release=first.release,
        shard=first.shard,
        control_sha256=control_sha(first.shard, "changed"),
        upstream_sha256=first.upstream_sha256,
        upstream_url=first.upstream_url,
        records=records(1),
    )

    assert first.path != second.path
    with pytest.raises(ValueError, match="ambiguous"):
        lake.seal_release(
            source=first.source,
            dataset=first.dataset,
            release=first.release,
            expected_shards={first.shard: first.upstream_sha256},
        )
    sealed = lake.seal_release(
        source=second.source,
        dataset=second.dataset,
        release=second.release,
        expected_shards={second.shard: second},
    )
    assert sealed.shard_count == 1


def test_control_fingerprint_is_canonical_json() -> None:
    assert canonical_control_sha256({"b": 2, "a": [1]}) == (
        canonical_control_sha256({"a": [1], "b": 2})
    )


def test_delete_records_and_hostile_components_remain_inside_lake(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = lake.commit_shard(
        source="../../semantic-scholar",
        dataset="../paper-ids",
        release="../../2026-09-01",
        shard="../../../deletes/000",
        control_sha256=control_sha("../../../deletes/000"),
        upstream_sha256=SHA_A.upper(),
        records=(LakeRecord("paper-1", {"reason": "upstream delete"}, "delete"),),
        expected_rows=1,
    )

    assert receipt.path.is_relative_to(lake.root)
    lake.seal_release(
        source="../../semantic-scholar",
        dataset="../paper-ids",
        release="../../2026-09-01",
        expected_shards={"../../../deletes/000": SHA_A},
    )
    row = next(
        iter(
            lake.iter_release_batches(
                source="../../semantic-scholar",
                dataset="../paper-ids",
                release="../../2026-09-01",
            )
        )
    ).to_pylist()[0]
    assert row["operation"] == "delete"


def test_snapshot_seal_and_scan_follow_manifest_order_not_shard_name(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    later = lake.commit_shard(
        source="semantic-scholar",
        dataset="papers",
        release="2026-09-01",
        shard="000-hash-sorts-first-but-is-second",
        control_sha256=control_sha("000-hash-sorts-first-but-is-second"),
        upstream_sha256=SHA_A,
        records=(LakeRecord("manifest-one", {}),),
        application_order=ShardApplicationOrder.snapshot(1),
    )
    earlier = lake.commit_shard(
        source="semantic-scholar",
        dataset="papers",
        release="2026-09-01",
        shard="fff-hash-sorts-last-but-is-first",
        control_sha256=control_sha("fff-hash-sorts-last-but-is-first"),
        upstream_sha256=SHA_B,
        records=(LakeRecord("manifest-zero", {}),),
        application_order=ShardApplicationOrder.snapshot(0),
    )

    release = lake.seal_release(
        source="semantic-scholar",
        dataset="papers",
        release="2026-09-01",
        expected_shards={later.shard: SHA_A, earlier.shard: SHA_B},
    )
    seal = json.loads(release.path.read_text())
    rows = [
        row
        for batch in lake.iter_release_batches(
            source="semantic-scholar",
            dataset="papers",
            release="2026-09-01",
        )
        for row in batch.to_pylist()
    ]

    assert release.application_mode == "snapshot"
    assert [item["shard"] for item in seal["shards"]] == [
        earlier.shard,
        later.shard,
    ]
    assert [row["source_record_id"] for row in rows] == [
        "manifest-zero",
        "manifest-one",
    ]


def test_two_step_diff_replays_updates_before_deletes_in_each_transition(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    release = "2026-09-01"
    transitions = (
        ("2026-08-18", "2026-08-25"),
        ("2026-08-25", release),
    )
    definitions = (
        (
            "fff-transition-zero-upsert",
            SHA_A,
            ShardApplicationOrder.diff(
                diff_index=0,
                operation="upsert",
                operation_index=0,
                from_release=transitions[0][0],
                to_release=transitions[0][1],
            ),
            LakeRecord("paper", {"version": 0}, "upsert"),
        ),
        (
            "000-transition-zero-delete",
            SHA_B,
            ShardApplicationOrder.diff(
                diff_index=0,
                operation="delete",
                operation_index=0,
                from_release=transitions[0][0],
                to_release=transitions[0][1],
            ),
            LakeRecord("paper", {}, "delete"),
        ),
        (
            "eee-transition-one-upsert",
            "c" * 64,
            ShardApplicationOrder.diff(
                diff_index=1,
                operation="upsert",
                operation_index=0,
                from_release=transitions[1][0],
                to_release=transitions[1][1],
            ),
            LakeRecord("paper", {"version": 1}, "upsert"),
        ),
        (
            "111-transition-one-delete",
            "d" * 64,
            ShardApplicationOrder.diff(
                diff_index=1,
                operation="delete",
                operation_index=0,
                from_release=transitions[1][0],
                to_release=transitions[1][1],
            ),
            LakeRecord("paper", {}, "delete"),
        ),
    )
    expected_shards: dict[str, str] = {}
    for shard, digest, order, record in reversed(definitions):
        receipt = lake.commit_shard(
            source="semantic-scholar",
            dataset="papers",
            release=release,
            shard=shard,
            control_sha256=control_sha(shard),
            upstream_sha256=digest,
            records=(record,),
            application_order=order,
        )
        assert receipt.application_order == order
        expected_shards[receipt.shard] = digest

    sealed = lake.seal_release(
        source="semantic-scholar",
        dataset="papers",
        release=release,
        expected_shards=expected_shards,
    )
    seal = json.loads(sealed.path.read_text())
    rows = [
        row
        for batch in lake.iter_release_batches(
            source="semantic-scholar",
            dataset="papers",
            release=release,
        )
        for row in batch.to_pylist()
    ]

    assert [item["shard"] for item in seal["shards"]] == [
        definition[0] for definition in definitions
    ]
    assert [row["operation"] for row in rows] == [
        "upsert",
        "delete",
        "upsert",
        "delete",
    ]
    materialized: dict[str, dict] = {}
    for row in rows:
        if row["operation"] == "delete":
            materialized.pop(row["source_record_id"], None)
        else:
            materialized[row["source_record_id"]] = json.loads(row["payload_json"])
    assert materialized == {}


@pytest.mark.parametrize(
    "orders",
    [
        (ShardApplicationOrder.snapshot(0), ShardApplicationOrder.snapshot(0)),
        (ShardApplicationOrder.snapshot(0), ShardApplicationOrder.snapshot(2)),
        (ShardApplicationOrder.single(), ShardApplicationOrder.snapshot(1)),
        (
            ShardApplicationOrder.diff(
                diff_index=0,
                operation="upsert",
                operation_index=0,
                from_release="2026-08-18",
                to_release="2026-08-25",
            ),
            ShardApplicationOrder.diff(
                diff_index=0,
                operation="delete",
                operation_index=0,
                from_release="2026-08-19",
                to_release="2026-09-01",
            ),
        ),
    ],
)
def test_release_rejects_duplicate_gapped_or_ambiguous_application_order(
    tmp_path: Path,
    orders: tuple[ShardApplicationOrder, ShardApplicationOrder],
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    expected = {}
    for index, (order, digest) in enumerate(zip(orders, (SHA_A, SHA_B), strict=True)):
        receipt = lake.commit_shard(
            source="semantic-scholar",
            dataset="papers",
            release="2026-09-01",
            shard=f"shard-{index}",
            control_sha256=control_sha(f"shard-{index}"),
            upstream_sha256=digest,
            records=(LakeRecord(f"paper-{index}", {}),),
            application_order=order,
        )
        expected[receipt.shard] = digest

    with pytest.raises(ValueError, match="duplicate|gapped|ambiguous"):
        lake.seal_release(
            source="semantic-scholar",
            dataset="papers",
            release="2026-09-01",
            expected_shards=expected,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upstream_sha256": "not-a-digest"},
        {"batch_rows": 0},
        {"expected_rows": -1},
        {"application_order": {}},
    ],
)
def test_shard_rejects_invalid_integrity_parameters(
    tmp_path: Path,
    kwargs,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    options = {
        "source": "source",
        "dataset": "dataset",
        "release": "release",
        "shard": "shard",
        "control_sha256": control_sha("shard"),
        "upstream_sha256": SHA_A,
        "records": (),
    }
    options.update(kwargs)

    with pytest.raises(ValueError):
        lake.commit_shard(**options)
