from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.orc as orc
import pytest

from modelome.lake import ParquetLandingZone
from modelome.models import ArtifactKind, Link, SourceRecord
from modelome.software_heritage_bulk import (
    SOFTWARE_HERITAGE_DEFAULT_NEW_SHARD_BUDGET,
    SoftwareHeritageOriginBulkLimits,
    SoftwareHeritageOriginBulkLoader,
)

RELEASE = "2026-06-04"
KEY = f"graph/{RELEASE}/orc/origin/origin-test-shard.orc"
URL = f"https://softwareheritage.s3.amazonaws.com/{KEY}"
ETAG = '"multipart-etag-16"'


def _orc(*urls: bytes, field: str = "url") -> bytes:
    sink = pa.BufferOutputStream()
    orc.write_table(
        pa.table({field: pa.array(urls, type=pa.binary())}),
        sink,
        stripe_size=64,
    )
    return sink.getvalue().to_pybytes()


def _control(body: bytes, *, index: int = 0, count: int = 1) -> SourceRecord:
    raw = {
        "record_type": "software_heritage_origin_orc_shard",
        "release": RELEASE,
        "table": "origin",
        "object_key": KEY,
        "stable_object_url": URL,
        "expected_bytes": len(body),
        "object_etag": ETAG,
        "object_last_modified": "2026-06-09T07:13:50.000Z",
        "object_checksum_algorithms": ["CRC32"],
        "object_checksum_type": "COMPOSITE",
        "manifest_index": index,
        "manifest_count": count,
        "manifest_signature": "c" * 64,
        "release_approval_url": (
            "https://docs.softwareheritage.org/_sources/devel/swh-export/graph/dataset.rst.txt"
        ),
        "release_approval_document_sha256": "a" * 64,
        "export_metadata_url": (
            f"https://softwareheritage.s3.amazonaws.com/graph/{RELEASE}/meta/export.json"
        ),
        "export_metadata_sha256": "b" * 64,
        "export_flavor": "full",
        "export_formats": ["orc"],
        "export_object_types": ["origin", "revision"],
        "export_start": "2026-06-04T19:11:53.251510+00:00",
        "export_end": "2026-06-09T07:13:50.541733+00:00",
        "export_tool": {"name": "swh.export", "version": "1.11.7"},
        "archive_format": "orc",
        "coverage_scope": "all_software_heritage_origins_in_export",
        "github_name_or_keyword_filter": False,
        "historical_github_census": False,
        "contains_repository_content": False,
        "settlement_age_hours": 168,
    }
    return SourceRecord(
        source_record_id=(f"software-heritage:origin-orc:{RELEASE}:origin-test-shard.orc"),
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=URL,
        title="Software Heritage origin shard",
        raw=raw,
        links=(Link(URL, relation="bulk_payload", crawl=False),),
    )


class Response(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        url: str = URL,
        etag: str = ETAG,
        content_length: int | None = None,
        maximum_read: int = 23,
    ) -> None:
        super().__init__(body)
        self.status = 200
        self.url = url
        self.maximum_read = maximum_read
        self.largest_request = 0
        self.headers = {
            "Content-Length": str(len(body) if content_length is None else content_length),
            "ETag": etag,
            "x-amz-checksum-crc32": "kD/tPA==-16",
            "x-amz-checksum-type": "COMPOSITE",
        }

    def read(self, size: int = -1) -> bytes:
        self.largest_request = max(self.largest_request, size)
        if size < 0:
            size = self.maximum_read
        return super().read(min(size, self.maximum_read))


class Transport:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, Mapping[str, str], Callable[[str], None]]] = []

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> Response:
        self.calls.append((url, dict(headers), redirect_validator))
        if not self.responses:
            raise AssertionError("unexpected ORC download")
        return self.responses.pop(0)


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


def test_streams_every_binary_origin_and_preserves_exact_export_evidence(
    tmp_path: Path,
) -> None:
    body = _orc(
        b"https://github.com/bio/quiet-code",
        b"https://example.org/not-github",
        b"\xffnot-utf8",
    )
    response = Response(body)
    transport = Transport([response])
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = SoftwareHeritageOriginBulkLoader(
        lake,
        transport=transport,
        limits=replace(
            SoftwareHeritageOriginBulkLimits(),
            download_chunk_bytes=31,
            parquet_batch_rows=1,
        ),
    )

    receipt = loader.load(_control(body))
    rows = _rows(lake, receipt)

    assert receipt.release == RELEASE
    assert receipt.dataset == "origins"
    assert receipt.row_count == 3
    assert receipt.upstream_bytes == len(body)
    assert receipt.upstream_sha256 == hashlib.sha256(body).hexdigest()
    assert receipt.response_etag == ETAG
    assert receipt.response_checksum_crc32 == "kD/tPA==-16"
    assert response.largest_request == 31
    assert transport.calls[0][1]["x-amz-checksum-mode"] == "ENABLED"

    payloads = [json.loads(row["payload_json"]) for row in rows]
    assert [payload["origin_url"] for payload in payloads] == [
        "https://github.com/bio/quiet-code",
        "https://example.org/not-github",
        None,
    ]
    assert payloads[2]["origin_url_base64"] == "/25vdC11dGY4"
    assert payloads[0]["evidence"]["release_approval_document_sha256"] == "a" * 64
    assert payloads[0]["evidence"]["export_metadata_sha256"] == "b" * 64
    assert payloads[0]["evidence"]["export_tool"] == {
        "name": "swh.export",
        "version": "1.11.7",
    }
    assert payloads[0]["evidence"]["source_row_ordinal"] == 0
    assert receipt.upstream_sha256 in payloads[0]["evidence"]["locator"]


def test_cached_exact_control_is_an_offline_idempotent_noop(tmp_path: Path) -> None:
    body = _orc(b"https://github.com/lab/inactive")
    transport = Transport([Response(body)])
    loader = SoftwareHeritageOriginBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=transport,
    )

    first = loader.load(_control(body))
    second = loader.load(_control(body))

    assert first.already_committed is False
    assert second.already_committed is True
    assert second.upstream_sha256 == first.upstream_sha256
    assert second.row_count == 1
    assert transport.responses == []
    assert len(transport.calls) == 1


def test_plan_and_source_specific_budget_are_bounded(tmp_path: Path) -> None:
    body = _orc(b"https://example.org/repository")
    loader = SoftwareHeritageOriginBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=Transport([]),
    )

    plan = loader.plan_shard(_control(body))

    assert plan is not None
    assert (plan.source, plan.dataset, plan.release) == (
        "software-heritage",
        "origins",
        RELEASE,
    )
    assert SOFTWARE_HERITAGE_DEFAULT_NEW_SHARD_BUDGET == 4
    assert loader.shard_budget() == 4
    assert loader.shard_budget(9) == 9
    with pytest.raises(ValueError, match="nonnegative"):
        loader.shard_budget(-1)


def test_etag_size_schema_and_orc_integrity_fail_atomically(tmp_path: Path) -> None:
    valid = _orc(b"https://github.com/lab/repo")
    cases = (
        (Response(valid, etag='"changed"'), _control(valid), "ETag changed"),
        (
            Response(valid, content_length=len(valid) + 1),
            _control(valid),
            "Content-Length changed",
        ),
        (
            Response(_orc(b"value", field="other")),
            _control(_orc(b"value", field="other")),
            "schema must contain only binary url",
        ),
        (Response(b"not an ORC file"), _control(b"not an ORC file"), "not a valid ORC"),
    )
    for index, (response, control, error) in enumerate(cases):
        lake = ParquetLandingZone(tmp_path / f"case-{index}")
        loader = SoftwareHeritageOriginBulkLoader(
            lake,
            transport=Transport([response]),
        )
        with pytest.raises((ValueError, pa.ArrowException), match=error):
            loader.load(control)
        assert lake.list_releases(source="software-heritage", dataset="origins") == ()


def test_control_bound_is_checked_before_network(tmp_path: Path) -> None:
    body = _orc(b"https://example.org/repository")
    transport = Transport([])
    loader = SoftwareHeritageOriginBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=transport,
        limits=replace(
            SoftwareHeritageOriginBulkLimits(),
            max_object_bytes=len(body) - 1,
            download_chunk_bytes=min(32, len(body) - 1),
        ),
    )

    with pytest.raises(ValueError, match="max_object_bytes"):
        loader.load(_control(body))
    assert transport.calls == []
