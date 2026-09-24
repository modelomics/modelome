from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from modelome.commoncrawl_bulk import (
    CommonCrawlBulkLimits,
    CommonCrawlWetBulkLoader,
)
from modelome.lake import ParquetLandingZone, canonical_control_sha256
from modelome.models import ArtifactKind, SourceRecord

COLLECTION = "CC-MAIN-2026-34"
PATH = (
    f"crawl-data/{COLLECTION}/segments/1723456789012.0/wet/"
    "CC-MAIN-20260812010203-20260812040203-00000.warc.wet.gz"
)
URL = f"https://data.commoncrawl.org/{PATH}"
MANIFEST_URL = f"https://data.commoncrawl.org/crawl-data/{COLLECTION}/wet.paths.gz"


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        url: str = URL,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        maximum_read: int = 19,
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
            raise AssertionError("unexpected WET object download")
        body = self.bodies.pop(0)
        response = FakeResponse(
            body,
            url=self.final_url,
            status=self.status,
            headers=self.headers,
        )
        self.responses.append(response)
        return response


def _labelled_sha1(body: bytes) -> str:
    digest = base64.b32encode(hashlib.sha1(body).digest()).decode().rstrip("=")
    return f"sha1:{digest}"


def _record(
    record_type: str,
    body: bytes,
    *,
    record_id: str,
    target_uri: str | None = None,
    date: str = "2026-08-12T01:02:03Z",
    block_digest: str | None = None,
    payload_digest: str | None = None,
    content_type: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> bytes:
    headers = [
        ("WARC-Type", record_type),
        ("WARC-Date", date),
        ("WARC-Record-ID", f"<urn:uuid:{record_id}>"),
    ]
    if target_uri is not None:
        headers.append(("WARC-Target-URI", target_uri))
    if block_digest is not None:
        headers.append(("WARC-Block-Digest", block_digest))
    if payload_digest is not None:
        headers.append(("WARC-Payload-Digest", payload_digest))
    headers.extend((extra_headers or {}).items())
    headers.extend(
        [
            ("Content-Type", content_type or "text/plain"),
            ("Content-Length", str(len(body))),
        ]
    )
    envelope = b"WARC/1.0\r\n" + b"".join(
        f"{name}: {value}\r\n".encode() for name, value in headers
    )
    return envelope + b"\r\n" + body + b"\r\n\r\n"


def _wet_records() -> tuple[bytes, list[bytes]]:
    warcinfo = _record(
        "warcinfo",
        b"software: test crawler\r\n",
        record_id="00000000-0000-0000-0000-000000000000",
        content_type="application/warc-fields",
    )
    first_body = b"A model description.\nCode: https://code.example/repo\n"
    first = _record(
        "conversion",
        first_body,
        record_id="11111111-1111-1111-1111-111111111111",
        target_uri="https://research.example/articles/one",
        block_digest=_labelled_sha1(first_body),
        payload_digest=_labelled_sha1(first_body),
        extra_headers={
            "WARC-Refers-To": "<urn:uuid:aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa>",
            "WARC-Identified-Content-Language": "eng",
        },
    )
    second_body = "Biología profunda".encode()
    second = _record(
        "conversion",
        second_body,
        record_id="22222222-2222-2222-2222-222222222222",
        target_uri="https://another.invalidación.example/página",
        block_digest=_labelled_sha1(second_body),
    )
    return warcinfo + first + second, [warcinfo, first, second]


def _compressed(*, concatenated_members: bool = True) -> bytes:
    complete, records = _wet_records()
    if not concatenated_members:
        return gzip.compress(complete, mtime=0)
    return b"".join(gzip.compress(record, mtime=0) for record in records)


def _control(
    *,
    raw_updates: Mapping[str, Any] | None = None,
    canonical_url: str = URL,
) -> SourceRecord:
    raw: dict[str, Any] = {
        "record_type": "commoncrawl_wet_shard",
        "collection_id": COLLECTION,
        "collection_name": "August 2026 Index",
        "collection_from": "2026-08-07T10:18:45",
        "collection_to": "2026-08-20T01:52:41",
        "manifest_url": MANIFEST_URL,
        "manifest_digest": "a" * 64,
        "manifest_digest_algorithm": "sha256",
        "manifest_etag": '"this-is-the-path-list-etag-not-an-object-hash"',
        "stable_object_url": URL,
        "manifest_index": 0,
        "total_shards": 1,
        "path": PATH,
    }
    raw.update(raw_updates or {})
    shard_key = canonical_control_sha256(
        {"collection_id": COLLECTION, "stable_object_url": URL}
    )
    return SourceRecord(
        source_record_id=f"commoncrawl:wet-shard:{shard_key}",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=canonical_url,
        title="WET shard",
        raw=raw,
    )


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


def test_loader_streams_concatenated_warc_members_and_lands_conversion_text(
    tmp_path: Path,
) -> None:
    compressed = _compressed()
    transport = FakeTransport(
        [compressed],
        headers={
            "Content-Length": str(len(compressed)),
            "ETag": "a" * 64,
        },
    )
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = CommonCrawlWetBulkLoader(
        lake,
        transport=transport,
        limits=replace(CommonCrawlBulkLimits(), download_chunk_bytes=23, parquet_batch_rows=1),
    )

    receipt = loader.load(_control())
    rows = _rows(lake, receipt)

    assert receipt.release == COLLECTION
    assert receipt.dataset == "wet"
    assert receipt.manifest_index == 0
    assert receipt.total_shards == 1
    assert receipt.compressed_bytes == len(compressed)
    assert receipt.uncompressed_bytes == len(_wet_records()[0])
    assert receipt.warc_records == 3
    assert receipt.row_count == 2
    assert receipt.upstream_sha256 == hashlib.sha256(compressed).hexdigest()
    assert receipt.response_etag == "a" * 64
    assert receipt.shard_receipt is not None
    assert receipt.shard_receipt.path == receipt.path
    assert receipt.shard_receipt.row_count == receipt.row_count
    assert transport.calls == [(URL, transport.calls[0][1], transport.calls[0][2])]
    assert transport.calls[0][1]["Accept-Encoding"] == "identity"
    assert transport.responses[0].largest_read_request == 23

    assert len(rows) == 2
    assert len({row["source_record_id"] for row in rows}) == 2
    first = json.loads(rows[0]["payload_json"])
    assert first["uri"] == "https://research.example/articles/one"
    assert first["date"] == "2026-08-12T01:02:03Z"
    assert first["content"] == (
        "A model description.\nCode: https://code.example/repo\n"
    )
    assert first["content_sha256"] == hashlib.sha256(
        first["content"].encode()
    ).hexdigest()
    assert first["warc"]["record_id"] == (
        "urn:uuid:11111111-1111-1111-1111-111111111111"
    )
    assert first["warc"]["verified_digests"]["block"]["verified"] is True
    assert first["warc"]["verified_digests"]["payload"]["algorithm"] == "sha1"
    assert first["bulk"]["object_url"] == URL
    second = json.loads(rows[1]["payload_json"])
    assert second["uri"] == "https://another.invalidación.example/página"
    assert second["content"] == "Biología profunda"


def test_plan_and_replay_order_are_network_free(tmp_path: Path) -> None:
    loader = CommonCrawlWetBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=FakeTransport([]),
    )
    control = _control()

    plan = loader.plan_shard(control)

    assert plan is not None
    assert (plan.source, plan.dataset, plan.release, plan.shard) == (
        "commoncrawl",
        "wet",
        COLLECTION,
        control.source_record_id,
    )
    assert plan.control_sha256 is not None
    assert plan.upstream_sha256 is None
    assert loader.shard_order(control) == (
        "2026-08-07T10:18:45",
        COLLECTION,
        0,
        control.source_record_id,
    )
    unrelated = replace(control, raw={"record_type": "another_control"})
    assert loader.plan_shard(unrelated) is None


def test_exact_control_replay_uses_committed_parquet_without_network(
    tmp_path: Path,
) -> None:
    compressed = _compressed()
    lake = ParquetLandingZone(tmp_path / "lake")
    first_transport = FakeTransport([compressed])
    control = _control()
    first = CommonCrawlWetBulkLoader(lake, transport=first_transport).load(control)
    no_network = FakeTransport([])

    replay = CommonCrawlWetBulkLoader(lake, transport=no_network).load(control)

    assert replay.path == first.path
    assert replay.already_committed is True
    assert replay.row_count == 2
    assert replay.uncompressed_bytes is None
    assert no_network.calls == []


def test_manifest_digest_and_etag_are_not_misrepresented_as_object_sha256(
    tmp_path: Path,
) -> None:
    compressed = _compressed()
    opaque_etag = "0" * 64
    transport = FakeTransport(
        [compressed],
        headers={"Content-Length": str(len(compressed)), "ETag": opaque_etag},
    )
    loader = CommonCrawlWetBulkLoader(
        ParquetLandingZone(tmp_path / "lake"), transport=transport
    )

    receipt = loader.load(
        _control(raw_updates={"manifest_digest": "f" * 64, "manifest_etag": "e" * 64})
    )

    assert receipt.response_etag == opaque_etag
    assert receipt.upstream_sha256 == hashlib.sha256(compressed).hexdigest()
    assert receipt.upstream_sha256 != opaque_etag
    assert receipt.upstream_sha256 != "f" * 64


def test_explicit_compressed_object_digest_is_planned_and_verified(tmp_path: Path) -> None:
    compressed = _compressed()
    digest = hashlib.sha256(compressed).hexdigest()
    control = _control(
        raw_updates={
            "object_digest": {
                "scope": "compressed_object",
                "algorithm": "sha-256",
                "encoding": "hex",
                "value": digest,
            }
        }
    )
    loader = CommonCrawlWetBulkLoader(
        ParquetLandingZone(tmp_path / "lake"), transport=FakeTransport([compressed])
    )

    plan = loader.plan_shard(control)
    receipt = loader.load(control)

    assert plan is not None
    assert plan.upstream_sha256 == digest
    assert receipt.upstream_sha256 == digest

    bad = _control(
        raw_updates={
            "object_digest": {
                "scope": "compressed_object",
                "algorithm": "sha256",
                "encoding": "hex",
                "value": "1" * 64,
            }
        }
    )
    bad_lake = ParquetLandingZone(tmp_path / "bad-lake")
    with pytest.raises(ValueError, match="published compressed-object digest"):
        CommonCrawlWetBulkLoader(
            bad_lake, transport=FakeTransport([compressed])
        ).load(bad)
    assert not any(bad_lake.shards_root.rglob("manifest.json"))


def test_content_md5_is_verified_but_response_etag_remains_opaque(tmp_path: Path) -> None:
    compressed = _compressed()
    invalid_md5 = base64.b64encode(b"x" * 16).decode()
    transport = FakeTransport(
        [compressed],
        headers={
            "Content-Length": str(len(compressed)),
            "Content-MD5": invalid_md5,
            "ETag": hashlib.md5(compressed, usedforsecurity=False).hexdigest(),
        },
    )

    with pytest.raises(ValueError, match="Content-MD5 verification"):
        CommonCrawlWetBulkLoader(
            ParquetLandingZone(tmp_path / "lake"), transport=transport
        ).load(_control())


def test_warc_block_digest_mismatch_aborts_before_parquet_commit(tmp_path: Path) -> None:
    text = b"evidence text"
    bad_record = _record(
        "conversion",
        text,
        record_id="33333333-3333-3333-3333-333333333333",
        target_uri="https://example.test/item",
        block_digest="sha1:" + "A" * 32,
    )
    compressed = gzip.compress(bad_record, mtime=0)
    lake = ParquetLandingZone(tmp_path / "lake")

    with pytest.raises(ValueError, match="WARC-Block-Digest verification failed"):
        CommonCrawlWetBulkLoader(lake, transport=FakeTransport([compressed])).load(
            _control()
        )

    assert not any(lake.shards_root.rglob("manifest.json"))
    assert list(lake.staging_root.iterdir()) == []


@pytest.mark.parametrize(
    ("limits", "error"),
    [
        (
            replace(CommonCrawlBulkLimits(), max_compressed_bytes=64, download_chunk_bytes=16),
            "max_compressed_bytes",
        ),
        (
            replace(CommonCrawlBulkLimits(), max_uncompressed_bytes=64),
            "uncompressed bytes",
        ),
        (
            replace(
                CommonCrawlBulkLimits(),
                max_record_bytes=8,
                record_chunk_bytes=8,
            ),
            "content bytes",
        ),
        (
            replace(CommonCrawlBulkLimits(), max_records=1),
            "WARC records",
        ),
        (
            replace(CommonCrawlBulkLimits(), max_text_chars_per_record=4),
            "max_text_chars_per_record",
        ),
    ],
)
def test_hard_transfer_record_and_text_limits_leave_no_partial_shard(
    tmp_path: Path,
    limits: CommonCrawlBulkLimits,
    error: str,
) -> None:
    compressed = _compressed(concatenated_members=False)
    lake = ParquetLandingZone(tmp_path / "lake")

    with pytest.raises(ValueError, match=error):
        CommonCrawlWetBulkLoader(
            lake,
            transport=FakeTransport([compressed]),
            limits=limits,
        ).load(_control())

    assert not any(lake.shards_root.rglob("manifest.json"))
    assert list(lake.staging_root.iterdir()) == []


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (b"not a gzip stream", "invalid or truncated"),
        (
            gzip.compress(b"WARC/1.0\nContent-Length: 0\n\n\n\n", mtime=0),
            "CRLF",
        ),
        (
            gzip.compress(
                _record(
                    "conversion",
                    b"abc",
                    record_id="44444444-4444-4444-4444-444444444444",
                    target_uri="https://example.test/item",
                )[:-2],
                mtime=0,
            ),
            "content block is truncated|two CRLF",
        ),
    ],
)
def test_malformed_or_truncated_warc_is_rejected(
    tmp_path: Path, body: bytes, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        CommonCrawlWetBulkLoader(
            ParquetLandingZone(tmp_path / "lake"), transport=FakeTransport([body])
        ).load(_control())


def test_exact_object_url_is_required_but_crawled_uri_domains_are_unfiltered(
    tmp_path: Path,
) -> None:
    compressed = _compressed()
    changed = "https://data.commoncrawl.org/crawl-data/other.warc.wet.gz"
    transport = FakeTransport([compressed], final_url=changed)
    with pytest.raises(ValueError, match="changed from its exact control"):
        CommonCrawlWetBulkLoader(
            ParquetLandingZone(tmp_path / "changed"), transport=transport
        ).load(_control())

    bad_control = _control(canonical_url="https://elsewhere.example/object.warc.wet.gz")
    with pytest.raises(ValueError, match="URL, path, and canonical URL disagree"):
        CommonCrawlWetBulkLoader(
            ParquetLandingZone(tmp_path / "control"), transport=FakeTransport([])
        ).plan_shard(bad_control)

    # The successful fixture above includes unrelated, non-allowlisted target hosts.
    assert "research.example" not in URL
    assert "invalidación.example" not in URL
