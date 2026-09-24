from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from modelome.frontier import FrontierCrawler
from modelome.lake import LakeRecord, ParquetLandingZone, ShardApplicationOrder, ShardReceipt
from modelome.models import ArtifactKind, SourceRecord
from modelome.software_heritage_projection import (
    SoftwareHeritageGitHubProjector,
    run_software_heritage_github_projection,
    run_software_heritage_github_shard_projection,
)
from modelome.storage import Database

RELEASE = "2026-06-04"


def _payload(value: bytes, *, index: int, ordinal: int) -> LakeRecord:
    digest = hashlib.sha256(value).hexdigest()
    object_sha = hashlib.sha256(f"object-{index}".encode()).hexdigest()
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    object_key = f"graph/{RELEASE}/orc/origin/origin-test-{index}.orc"
    return LakeRecord(
        source_record_id=f"software-heritage:origin-url-sha256:{digest}",
        payload={
            "record_type": "software_heritage_origin",
            "origin_url": text,
            "origin_url_base64": base64.b64encode(value).decode(),
            "origin_url_sha256": digest,
            "origin_identifier": {
                "namespace": "software-heritage:origin-url-sha256",
                "value": digest,
            },
            "evidence": {
                "release": RELEASE,
                "table": "origin",
                "object_key": object_key,
                "object_url": f"https://softwareheritage.s3.amazonaws.com/{object_key}",
                "object_sha256": object_sha,
                "manifest_signature": "a" * 64,
                "manifest_index": index,
                "manifest_count": 2,
                "source_row_ordinal": ordinal,
                "locator": (
                    f"swh-export:{RELEASE}:{object_key}:row:{ordinal}:object-sha256:{object_sha}"
                ),
            },
        },
    )


def _land(
    lake: ParquetLandingZone,
    index: int,
    *origins: bytes,
) -> ShardReceipt:
    shard = f"software-heritage:origin-orc:{RELEASE}:origin-test-{index}.orc"
    upstream_sha = hashlib.sha256(f"object-{index}".encode()).hexdigest()
    return lake.commit_shard(
        source="software-heritage",
        dataset="origins",
        release=RELEASE,
        shard=shard,
        control_sha256=hashlib.sha256(f"control-{index}".encode()).hexdigest(),
        upstream_sha256=upstream_sha,
        upstream_url=(
            "https://softwareheritage.s3.amazonaws.com/"
            f"graph/{RELEASE}/orc/origin/origin-test-{index}.orc"
        ),
        upstream_bytes=100 + index,
        application_order=ShardApplicationOrder.snapshot(index),
        records=(
            _payload(value, index=index, ordinal=ordinal) for ordinal, value in enumerate(origins)
        ),
        expected_rows=len(origins),
        batch_rows=1,
    )


def _seal(lake: ParquetLandingZone, *receipts: ShardReceipt) -> None:
    lake.seal_release(
        source="software-heritage",
        dataset="origins",
        release=RELEASE,
        expected_shards={receipt.shard: receipt for receipt in receipts},
    )


def test_unfiltered_lake_projects_only_structurally_exact_github_origins(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = _land(
        lake,
        0,
        b"https://example.org/ordinary-repository",
        b"https://github.com/bio/quiet-network.git/",
        b"git@github.com:physics/silent-solver.git",
        b"https://github.com/not/a/repository",
        b"\xffnot-utf8",
    )
    _seal(lake, receipt)

    page = SoftwareHeritageGitHubProjector(lake).page(RELEASE)

    assert page.complete is True
    assert page.rows_examined == page.total_rows == 5
    assert [candidate.repository_name for candidate in page.candidates] == [
        "bio/quiet-network",
        "physics/silent-solver",
    ]
    assert [candidate.repository_url for candidate in page.candidates] == [
        "https://github.com/bio/quiet-network",
        "https://github.com/physics/silent-solver",
    ]
    record = page.candidates[0].as_source_record()
    assert record.models == ()
    assert record.links[0].crawl is True
    assert record.raw["model_admission_performed"] is False
    assert record.raw["historical_github_census"] is False


def test_committed_shard_projects_and_checkpoints_before_release_seal(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = _land(
        lake,
        0,
        b"https://example.org/non-github",
        b"https://github.com/bio/code-only-network",
        b"git://github.com/chemistry/learned-potential.git",
    )
    database = Database(tmp_path / "store")
    database.initialize()

    first = run_software_heritage_github_shard_projection(
        database,
        lake,
        source_receipt=receipt,
        max_origin_rows=1,
    )
    second = run_software_heritage_github_shard_projection(
        database,
        lake,
        source_receipt=receipt,
        max_origin_rows=1,
    )
    third = run_software_heritage_github_shard_projection(
        database,
        lake,
        source_receipt=receipt,
        max_origin_rows=1,
    )
    repeat = run_software_heritage_github_shard_projection(
        database,
        lake,
        source_receipt=receipt,
        max_origin_rows=1,
    )

    assert first.status == second.status == "partial"
    assert third.status == "complete"
    assert repeat.run_id is None
    assert [item["url"] for item in database.list_frontier()] == [
        "https://github.com/bio/code-only-network",
        "https://github.com/chemistry/learned-potential",
    ]
    assert database.get_source_state("software-heritage-github-shards") == {
        "release": RELEASE,
        "manifest_index": 0,
        "shard": receipt.shard,
        "control_sha256": receipt.control_sha256,
        "upstream_sha256": receipt.upstream_sha256,
        "next_row": 3,
        "total_rows": 3,
        "complete": True,
    }
    # The immediate path did not require a release seal.
    assert lake.list_releases(source="software-heritage", dataset="origins") == ()


def test_shard_runtime_is_monotonic_and_old_completed_shards_are_noops(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    first_receipt = _land(lake, 0, b"https://github.com/lab/first")
    second_receipt = _land(lake, 1, b"https://github.com/lab/second")
    database = Database(tmp_path / "store")
    database.initialize()

    first = run_software_heritage_github_shard_projection(
        database, lake, source_receipt=first_receipt
    )
    second = run_software_heritage_github_shard_projection(
        database, lake, source_receipt=second_receipt
    )
    state = database.get_source_state("software-heritage-github-shards")
    old = run_software_heritage_github_shard_projection(
        database, lake, source_receipt=first_receipt
    )

    assert first.status == second.status == "complete"
    assert old.run_id is None
    assert old.rows_examined == 0
    assert database.get_source_state("software-heritage-github-shards") == state


def test_sealed_release_runtime_is_restart_safe_and_idempotent(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = _land(
        lake,
        0,
        b"https://example.org/skip",
        b"https://github.com/materials/inactive-crystal-code",
    )
    _seal(lake, receipt)
    database = Database(tmp_path / "store")
    database.initialize()

    partial = run_software_heritage_github_projection(
        database,
        lake,
        release=RELEASE,
        max_origin_rows=1,
    )
    complete = run_software_heritage_github_projection(
        database,
        lake,
        release=RELEASE,
        max_origin_rows=1,
    )
    repeat = run_software_heritage_github_projection(
        database,
        lake,
        release=RELEASE,
        max_origin_rows=1,
    )

    assert partial.status == "partial"
    assert complete.status == "complete"
    assert repeat.run_id is None
    assert database.list_frontier()[0]["url"] == (
        "https://github.com/materials/inactive-crystal-code"
    )


def test_tampered_committed_shard_is_rejected_before_projection(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = _land(lake, 0, b"https://github.com/lab/repository")
    part = next((receipt.path / "parts").glob("*.parquet"))
    data = part.read_bytes()
    part.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))

    try:
        SoftwareHeritageGitHubProjector(lake).page_shard(receipt)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("tampered shard was projected")


class CodeOnlyFetcher:
    def accepts(self, url: str) -> bool:
        return url == "https://github.com/bio/code-only-network"

    def fetch(self, url: str) -> SourceRecord:
        return SourceRecord(
            source_record_id="bio/code-only-network",
            kind=ArtifactKind.CODE_REPOSITORY,
            canonical_url=url,
            title="bio/code-only-network",
            text=(
                "# CellSignalArchiveNet-3\n\n"
                "A deep neural network for cell-to-cell signal prediction."
            ),
            raw={"source": "README", "full_name": "bio/code-only-network"},
        )


def test_archived_github_only_origin_reaches_readme_model_admission(
    tmp_path: Path,
) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = _land(lake, 0, b"https://github.com/bio/code-only-network.git")
    database = Database(tmp_path / "store")
    database.initialize()
    projected = run_software_heritage_github_shard_projection(
        database,
        lake,
        source_receipt=receipt,
    )

    crawled = FrontierCrawler(database, [CodeOnlyFetcher()]).crawl(
        limit=10,
        max_depth=0,
    )

    assert projected.status == "complete"
    assert crawled.status == "complete"
    models = database.search_models("CellSignalArchiveNet-3")
    assert len(models) == 1
    assert models[0]["canonical_name"] == "CellSignalArchiveNet-3"
