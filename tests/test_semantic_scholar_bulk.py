from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from modelome.http import HttpResponse
from modelome.lake import ParquetLandingZone, ShardApplicationOrder
from modelome.semantic_scholar_bulk import (
    BulkLoadLimits,
    SemanticScholarBulkLoader,
    _PublicHttpsRedirectHandler,
)
from modelome.sources.semantic_scholar import SemanticScholarDatasetSourceAdapter

API_KEY = "s2-bulk-secret"
TARGET_RELEASE = "2026-08-25"
BASE_RELEASE = "2026-08-18"


class MetadataClient:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.payloads = list(payloads)

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        if not self.payloads:
            raise AssertionError(f"unexpected metadata GET {url}")
        return json_response(self.payloads.pop(0))


class FakeTransport:
    def __init__(
        self,
        manifests: list[Mapping[str, Any]],
        streams: list[bytes],
        *,
        wire_chunk: int = 7,
    ) -> None:
        self.manifests = list(manifests)
        self.streams = list(streams)
        self.wire_chunk = wire_chunk
        self.manifest_calls: list[tuple[str, Mapping[str, str]]] = []
        self.download_calls: list[tuple[str, int]] = []
        self.blocks_yielded = 0

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.manifest_calls.append((url, dict(headers or {})))
        if not self.manifests:
            raise AssertionError(f"unexpected manifest GET {url}")
        return json_response(self.manifests.pop(0))

    def iter_bytes(self, url: str, *, chunk_size: int) -> Iterator[bytes]:
        self.download_calls.append((url, chunk_size))
        if not self.streams:
            raise AssertionError("unexpected shard download")
        payload = self.streams.pop(0)
        step = min(self.wire_chunk, chunk_size)
        for start in range(0, len(payload), step):
            self.blocks_yielded += 1
            yield payload[start : start + step]


def json_response(payload: Any) -> HttpResponse:
    return HttpResponse(
        status=200,
        headers={"content-type": "application/json"},
        body=json.dumps(payload).encode(),
        url="https://api.semanticscholar.org/datasets/v1/fixture",
    )


def compressed_rows(*rows: Mapping[str, Any]) -> bytes:
    body = b"\n".join(
        json.dumps(dict(row), separators=(",", ":")).encode() for row in rows
    )
    return gzip.compress(body + b"\n", mtime=0)


def snapshot_manifest(
    dataset: str,
    *,
    signature: str,
    stable_name: str = "part-000.jsonl.gz",
    description: str | None = None,
) -> dict[str, Any]:
    return {
        "name": dataset,
        "description": description or f"{dataset} description",
        "README": f"{dataset} license",
        "files": [
            f"https://objects.example/s2/{TARGET_RELEASE}/{dataset}/{stable_name}"
            f"?X-Amz-Signature={signature}&X-Amz-Expires=3600"
        ],
    }


def diff_manifest(
    dataset: str,
    *,
    update_signature: str = "update",
    delete_signature: str = "delete",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "start_release": BASE_RELEASE,
        "end_release": TARGET_RELEASE,
        "diffs": [
            {
                "from_release": BASE_RELEASE,
                "to_release": TARGET_RELEASE,
                "update_files": [
                    f"https://objects.example/s2/diffs/{dataset}/update-000.jsonl.gz"
                    f"?X-Amz-Signature={update_signature}"
                ],
                "delete_files": [
                    f"https://objects.example/s2/diffs/{dataset}/delete-000.jsonl.gz"
                    f"?X-Amz-Signature={delete_signature}"
                ],
            }
        ],
    }


def snapshot_control(dataset: str, manifest: Mapping[str, Any]):
    index = ("papers", "abstracts", "paper-ids").index(dataset)
    source = SemanticScholarDatasetSourceAdapter(
        api_key=API_KEY,
        client=MetadataClient(manifest),
    )
    page = source.fetch_page(
        {
            "stage": "snapshot",
            "target_release": TARGET_RELEASE,
            "dataset_index": index,
            "shard_offset": 0,
            "control_records_seen": 1,
            "release_signature": "release-signature",
        }
    )
    assert page.issues == ()
    return page.records[0]


def diff_control(dataset: str, manifest: Mapping[str, Any], operation: str):
    index = ("papers", "abstracts", "paper-ids").index(dataset)
    source = SemanticScholarDatasetSourceAdapter(
        api_key=API_KEY,
        client=MetadataClient(manifest),
    )
    page = source.fetch_page(
        {
            "stage": "diff",
            "target_release": TARGET_RELEASE,
            "base_release": BASE_RELEASE,
            "watermark": BASE_RELEASE,
            "dataset_index": index,
            "shard_offset": 0,
            "control_records_seen": 1,
            "release_signature": "release-signature",
        }
    )
    assert page.issues == ()
    return next(record for record in page.records if record.raw["operation"] == operation)


def landed_rows(lake: ParquetLandingZone, receipt) -> list[dict[str, Any]]:
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


def test_snapshot_loader_refreshes_signed_url_streams_and_commits_digest(
    tmp_path: Path,
) -> None:
    original = snapshot_manifest("papers", signature="expired-secret")
    control = snapshot_control("papers", original)
    refreshed = snapshot_manifest("papers", signature="fresh-secret")
    compressed = compressed_rows(
        {"corpusid": 215416146, "title": "Attention Is All You Need"},
        {"corpusid": 123, "title": "A second paper"},
    )
    transport = FakeTransport([refreshed], [compressed])
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = SemanticScholarBulkLoader(
        lake,
        api_key=API_KEY,
        transport=transport,
        temp_root=tmp_path / "temporary",
    )

    receipt = loader.load_shard(control)
    rows = landed_rows(lake, receipt)

    assert receipt.upstream_sha256 == hashlib.sha256(compressed).hexdigest()
    assert receipt.row_count == 2
    assert receipt.dataset == "papers"
    assert receipt.release == TARGET_RELEASE
    assert receipt.application_order == ShardApplicationOrder.snapshot(0)
    assert [row["source_record_id"] for row in rows] == [
        "semantic-scholar:corpus:215416146",
        "semantic-scholar:corpus:123",
    ]
    assert all(row["operation"] == "upsert" for row in rows)
    assert json.loads(rows[0]["payload_json"])["title"] == "Attention Is All You Need"
    assert transport.manifest_calls[0][1]["x-api-key"] == API_KEY
    assert "fresh-secret" in transport.download_calls[0][0]

    landing_manifest = json.loads((receipt.path / "manifest.json").read_text())
    assert landing_manifest["upstream_url"] == control.raw["stable_object_url"]
    assert landing_manifest["application_order"] == {
        "mode": "snapshot",
        "manifest_index": 0,
    }
    assert "X-Amz" not in landing_manifest["upstream_url"]
    assert "fresh-secret" not in (receipt.path / "manifest.json").read_text()
    assert list((tmp_path / "temporary").iterdir()) == []


def test_loader_exposes_network_free_bulk_plan_and_replay_order(tmp_path: Path) -> None:
    control = snapshot_control(
        "papers", snapshot_manifest("papers", signature="expired")
    )
    loader = SemanticScholarBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        api_key=API_KEY,
        transport=FakeTransport([], []),
    )

    plan = loader.plan_shard(control)

    assert plan is not None
    assert (plan.source, plan.dataset, plan.release, plan.shard) == (
        "semantic-scholar",
        "papers",
        TARGET_RELEASE,
        control.source_record_id,
    )
    assert plan.control_sha256 is not None
    assert plan.upstream_sha256 is None
    assert loader.shard_order(control) == (TARGET_RELEASE, "papers", 0, 0)

    release_record = replace(
        control,
        source_record_id="semantic-scholar:release:fixture",
        raw={"record_type": "release"},
    )
    assert loader.plan_shard(release_record) is None


def test_loader_reuses_the_configured_control_adapter(tmp_path: Path) -> None:
    transport = FakeTransport([], [])
    control_adapter = SemanticScholarDatasetSourceAdapter(
        name="configured-control",
        datasets=("papers",),
        release_id=TARGET_RELEASE,
        api_key=API_KEY,
        client=transport,
    )

    loader = SemanticScholarBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        control_adapter=control_adapter,
        transport=transport,
    )

    assert loader.control is control_adapter


def test_delete_diff_shard_lands_delete_operations(tmp_path: Path) -> None:
    original = diff_manifest("papers", delete_signature="expired-delete")
    control = diff_control("papers", original, "delete")
    refreshed = diff_manifest("papers", delete_signature="fresh-delete")
    compressed = compressed_rows({"corpusid": 215416146})
    transport = FakeTransport([refreshed], [compressed])
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = SemanticScholarBulkLoader(lake, api_key=API_KEY, transport=transport)

    receipt = loader.load_shard(control)
    rows = landed_rows(lake, receipt)

    assert rows[0]["operation"] == "delete"
    assert rows[0]["source_record_id"] == "semantic-scholar:corpus:215416146"
    assert receipt.application_order == ShardApplicationOrder.diff(
        diff_index=0,
        operation="delete",
        operation_index=0,
        from_release=BASE_RELEASE,
        to_release=TARGET_RELEASE,
    )
    assert "fresh-delete" in transport.download_calls[0][0]


def test_paper_id_rows_use_sha_identity_without_collapsing_aliases(tmp_path: Path) -> None:
    original = snapshot_manifest("paper-ids", signature="old")
    control = snapshot_control("paper-ids", original)
    refreshed = snapshot_manifest("paper-ids", signature="new")
    first_sha = "a" * 40
    second_sha = "b" * 40
    compressed = compressed_rows(
        {"sha": first_sha, "corpusid": 42, "primary": True},
        {"sha": second_sha, "corpusid": 42, "primary": False},
    )
    transport = FakeTransport([refreshed], [compressed])
    lake = ParquetLandingZone(tmp_path / "lake")

    receipt = SemanticScholarBulkLoader(
        lake,
        api_key=API_KEY,
        transport=transport,
    ).load_shard(control)
    rows = landed_rows(lake, receipt)

    assert [row["source_record_id"] for row in rows] == [
        f"semantic-scholar:paper-id:{first_sha}",
        f"semantic-scholar:paper-id:{second_sha}",
    ]


def test_identical_shard_is_idempotent(tmp_path: Path) -> None:
    original = snapshot_manifest("abstracts", signature="expired")
    control = snapshot_control("abstracts", original)
    refreshed = snapshot_manifest("abstracts", signature="fresh")
    compressed = compressed_rows({"corpusid": 99, "abstract": "Example"})
    transport = FakeTransport([refreshed], [compressed])
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = SemanticScholarBulkLoader(lake, api_key=API_KEY, transport=transport)

    first = loader.load_shard(control)
    repeated = loader.load_shard(control)

    assert first.path == repeated.path
    assert repeated.already_committed is True
    assert len(transport.manifest_calls) == 1
    assert len(transport.download_calls) == 1
    assert len(list(lake.shards_root.rglob("manifest.json"))) == 1


def test_changed_manifest_signature_is_a_new_control_even_for_identical_bytes(
    tmp_path: Path,
) -> None:
    first_manifest = snapshot_manifest(
        "abstracts",
        signature="first-control",
        description="first description",
    )
    second_manifest = snapshot_manifest(
        "abstracts",
        signature="second-control",
        description="changed description",
    )
    first_control = snapshot_control("abstracts", first_manifest)
    second_control = snapshot_control("abstracts", second_manifest)
    assert first_control.source_record_id == second_control.source_record_id
    assert (
        first_control.raw["manifest_signature"]
        != second_control.raw["manifest_signature"]
    )
    compressed = compressed_rows({"corpusid": 99, "abstract": "Example"})
    transport = FakeTransport(
        [first_manifest, second_manifest],
        [compressed, compressed],
    )
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = SemanticScholarBulkLoader(lake, api_key=API_KEY, transport=transport)

    first = loader.load_shard(first_control)
    second = loader.load_shard(second_control)

    assert first.control_sha256 != second.control_sha256
    assert first.upstream_sha256 == second.upstream_sha256
    assert first.path != second.path
    assert len(transport.manifest_calls) == 2
    assert len(transport.download_calls) == 2


def test_compressed_limit_stops_stream_and_cleans_temp_files(tmp_path: Path) -> None:
    original = snapshot_manifest("papers", signature="old")
    control = snapshot_control("papers", original)
    refreshed = snapshot_manifest("papers", signature="new")
    compressed = compressed_rows({"corpusid": 1, "title": "x" * 500})
    transport = FakeTransport([refreshed], [compressed], wire_chunk=5)
    temp_root = tmp_path / "temporary"
    limits = BulkLoadLimits(
        max_compressed_bytes=10,
        network_chunk_bytes=5,
    )
    loader = SemanticScholarBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        api_key=API_KEY,
        transport=transport,
        limits=limits,
        temp_root=temp_root,
    )

    with pytest.raises(ValueError, match="max_compressed_bytes"):
        loader.load_shard(control)

    assert transport.blocks_yielded == 3
    assert list(temp_root.iterdir()) == []
    assert not list((tmp_path / "lake").rglob("manifest.json"))


@pytest.mark.parametrize(
    ("limits", "row", "message"),
    [
        (
            BulkLoadLimits(
                max_uncompressed_bytes=20,
                decompression_chunk_bytes=10,
            ),
            {"corpusid": 1, "title": "x" * 100},
            "max_uncompressed_bytes",
        ),
        (
            BulkLoadLimits(max_line_bytes=20, decompression_chunk_bytes=10),
            {"corpusid": 1, "title": "x" * 100},
            "max_line_bytes",
        ),
        (
            BulkLoadLimits(max_records=1),
            None,
            "max_records",
        ),
    ],
)
def test_decompression_guards_abort_before_landing(
    tmp_path: Path,
    limits: BulkLoadLimits,
    row: Mapping[str, Any] | None,
    message: str,
) -> None:
    original = snapshot_manifest("papers", signature="old")
    control = snapshot_control("papers", original)
    refreshed = snapshot_manifest("papers", signature="new")
    rows = (
        [row]
        if row is not None
        else [{"corpusid": 1}, {"corpusid": 2}]
    )
    compressed = compressed_rows(*rows)
    transport = FakeTransport([refreshed], [compressed])
    loader = SemanticScholarBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        api_key=API_KEY,
        transport=transport,
        limits=limits,
    )

    with pytest.raises(ValueError, match=message):
        loader.load_shard(control)

    assert not list((tmp_path / "lake").rglob("manifest.json"))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (gzip.compress(b"not-json\n", mtime=0), "not valid JSON"),
        (gzip.compress(b'{"title":"missing ID"}\n', mtime=0), "valid corpusid"),
        (gzip.compress(b"", mtime=0), "no JSONL records"),
        (b"not-gzip", "not valid gzip"),
    ],
)
def test_malformed_rows_and_gzip_never_publish_partial_shards(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    original = snapshot_manifest("papers", signature="old")
    control = snapshot_control("papers", original)
    refreshed = snapshot_manifest("papers", signature="new")
    loader = SemanticScholarBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        api_key=API_KEY,
        transport=FakeTransport([refreshed], [payload]),
    )

    with pytest.raises(ValueError, match=message):
        loader.load_shard(control)

    assert not list((tmp_path / "lake").rglob("manifest.json"))


def test_changed_manifest_and_tampered_manifest_url_fail_before_download(
    tmp_path: Path,
) -> None:
    original = snapshot_manifest("papers", signature="old")
    control = snapshot_control("papers", original)
    changed = snapshot_manifest(
        "papers",
        signature="new",
        stable_name="different-part.jsonl.gz",
    )
    transport = FakeTransport([changed], [])
    loader = SemanticScholarBulkLoader(
        ParquetLandingZone(tmp_path / "first-lake"),
        api_key=API_KEY,
        transport=transport,
    )

    with pytest.raises(ValueError, match="manifest changed"):
        loader.load_shard(control)
    assert transport.download_calls == []

    tampered = replace(
        control,
        raw={
            **control.raw,
            "manifest_url": "https://attacker.example/collect-api-key",
        },
    )
    second_transport = FakeTransport([], [])
    second_loader = SemanticScholarBulkLoader(
        ParquetLandingZone(tmp_path / "second-lake"),
        api_key=API_KEY,
        transport=second_transport,
    )
    with pytest.raises(ValueError, match="manifest URL is not canonical"):
        second_loader.load_shard(tampered)
    assert second_transport.manifest_calls == []


def test_private_redirect_is_rejected_without_echoing_signed_url() -> None:
    request = Request(
        "https://objects.example/shard.gz?X-Amz-Signature=do-not-log"
    )

    with pytest.raises(ValueError, match="not public") as captured:
        _PublicHttpsRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://127.0.0.1/internal?X-Amz-Signature=secret",
        )

    assert "secret" not in str(captured.value)


def test_invalid_limits_and_symlink_temp_root_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        BulkLoadLimits(max_line_bytes=0)

    real = tmp_path / "real"
    real.mkdir()
    symlink = tmp_path / "temp-link"
    symlink.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        SemanticScholarBulkLoader(
            ParquetLandingZone(tmp_path / "lake"),
            api_key=API_KEY,
            transport=FakeTransport([], []),
            temp_root=symlink,
        )
