from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from modelome.lake import ParquetLandingZone, ShardApplicationOrder
from modelome.models import ArtifactKind, Link, SourceRecord
from modelome.openaire_bulk import (
    OpenAireBulkLimits,
    OpenAireGraphBulkLoader,
)

RELEASE = "20428976"
BASE_URL = "https://zenodo.org/api/records"


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        url: str,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(body)
        self.status = status
        self.url = url
        self.headers = {"Content-Length": str(len(body))} if headers is None else dict(headers)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class FakeTransport:
    def __init__(
        self,
        *bodies: bytes,
        final_url: str | None = None,
        redirect_url: str | None = None,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.bodies = list(bodies)
        self.final_url = final_url
        self.redirect_url = redirect_url
        self.status = status
        self.headers = dict(headers) if headers is not None else None
        self.calls: list[tuple[str, Mapping[str, str], Any]] = []

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Any,
    ) -> FakeResponse:
        self.calls.append((url, dict(headers), redirect_validator))
        if self.redirect_url is not None:
            redirect_validator(self.redirect_url)
        if not self.bodies:
            raise AssertionError("unexpected OpenAIRE archive download")
        body = self.bodies.pop(0)
        return FakeResponse(
            body,
            url=self.final_url or url,
            status=self.status,
            headers=({"Content-Length": str(len(body))} if self.headers is None else self.headers),
        )


def _gzip_lines(*lines: bytes) -> bytes:
    return gzip.compress(b"".join(line + b"\n" for line in lines), mtime=0)


def _gzip_rows(*rows: Mapping[str, Any]) -> bytes:
    return _gzip_lines(
        *[json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")).encode() for row in rows]
    )


def _tar(
    members: list[tuple[str, bytes, bytes | None, str | None]],
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, body, member_type, linkname in members:
            info = tarfile.TarInfo(name)
            if member_type is not None:
                info.type = member_type
            if linkname is not None:
                info.linkname = linkname
            if info.isreg():
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
            else:
                archive.addfile(info)
    return output.getvalue()


def _archive(*member_rows: tuple[str, tuple[Mapping[str, Any], ...]]) -> bytes:
    return _tar([(name, _gzip_rows(*rows), None, None) for name, rows in member_rows])


def _control(
    body: bytes,
    *,
    file_key: str = "publication_1.tar",
    release: str = RELEASE,
    manifest_index: int = 0,
    manifest_count: int = 1,
    partition: str = "publication",
    checksum: str | None = None,
    size: int | None = None,
    url: str | None = None,
    source_record_id: str | None = None,
) -> SourceRecord:
    encoded = quote(file_key, safe="")
    download_url = url or f"{BASE_URL}/{release}/files/{encoded}/content"
    record_id = source_record_id or (f"openaire-graph:file:{release}:{encoded}")
    raw = {
        "record_type": "dataset_shard",
        "operation": "snapshot",
        "release_id": release,
        "release_version": "11.1.1",
        "manifest_signature": "a" * 64,
        "manifest_index": manifest_index,
        "manifest_count": manifest_count,
        "file_key": file_key,
        "file_id": f"id-{manifest_index}",
        "size": len(body) if size is None else size,
        "checksum": {
            "algorithm": "md5",
            "value": checksum or hashlib.md5(body, usedforsecurity=False).hexdigest(),
        },
        "entity_partition": partition,
        "partition_index": manifest_index + 1,
        "download_url": download_url,
    }
    return SourceRecord(
        source_record_id=record_id,
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=download_url,
        title=file_key,
        raw=raw,
        links=(
            Link(download_url, relation="bulk_payload", crawl=False),
            Link(
                f"{BASE_URL}/{release}",
                relation="release_metadata",
                crawl=False,
            ),
        ),
    )


def _limits(**overrides: int) -> OpenAireBulkLimits:
    values = {
        "max_archive_bytes": 2 * 1024 * 1024,
        "max_archive_members": 100,
        "max_member_name_bytes": 1024,
        "max_member_compressed_bytes": 1024 * 1024,
        "max_member_uncompressed_bytes": 1024 * 1024,
        "max_total_uncompressed_bytes": 2 * 1024 * 1024,
        "max_payload_bytes": 256 * 1024,
        "max_rows": 10_000,
        "download_chunk_bytes": 17,
        "decompression_chunk_bytes": 13,
        "parquet_batch_rows": 2,
    }
    values.update(overrides)
    return OpenAireBulkLimits(**values)


def _landed_rows(
    lake: ParquetLandingZone,
    receipt: Any,
) -> list[dict[str, Any]]:
    lake.seal_release(
        source=receipt.source,
        dataset=receipt.dataset,
        release=receipt.release,
        expected_shards={receipt.shard: receipt},
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


def test_loader_streams_every_json_object_losslessly_into_one_graph_dataset(
    tmp_path: Path,
) -> None:
    neural_biomed = {
        "id": "openair____::synthetic",
        "type": "future-upstream-type",
        "pid": [
            {"scheme": "doi", "value": "10.1000/synthetic"},
            {"scheme": "pmid", "value": "999"},
        ],
        "relations": [
            {
                "type": "isSupplementedBy",
                "target": "https://code.example/repository",
            }
        ],
        "creators": [{"name": "Ada Example", "orcid": "0000-0000-0000-0001"}],
        "host": {"name": "Synthetic Biomedical Journal", "issn": ["1234-5678"]},
        "dates": [{"type": "published", "value": "2026-01-02"}],
        "rights": [{"code": "open", "url": "https://rights.example/open"}],
        "urls": ["https://example.org/article", "https://code.example/repository"],
        "abstract": "A neural system for an unusual biomedical domain.",
        "arbitrary_future_field": {"nested": [True, None, 7, "Δ"]},
    }
    relation = {
        "source": "openair____::synthetic",
        "target": "openair____::software",
        "relation": "Cites",
        "provenance": {"source": "upstream"},
    }
    body = _archive(
        ("publication/part-000.json.gz", (neural_biomed,)),
        ("relations/part-001.json.gz", (relation,)),
    )
    control = _control(body, partition="not-interpreted-by-loader")
    transport = FakeTransport(body)
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = OpenAireGraphBulkLoader(
        lake,
        limits=_limits(),
        transport=transport,
    )

    receipt = loader.load(control)
    rows = _landed_rows(lake, receipt)

    assert receipt.source == "openaire-graph"
    assert receipt.dataset == "graph"
    assert receipt.release == RELEASE
    assert receipt.shard == control.source_record_id
    assert receipt.upstream_sha256 == hashlib.sha256(body).hexdigest()
    assert receipt.upstream_bytes == len(body)
    assert receipt.row_count == 2
    assert receipt.application_order == ShardApplicationOrder.snapshot(0)
    assert [row["operation"] for row in rows] == ["upsert", "upsert"]
    assert json.loads(rows[0]["payload_json"]) == neural_biomed
    assert json.loads(rows[1]["payload_json"]) == relation
    assert len({row["source_record_id"] for row in rows}) == 2
    assert transport.calls[0][0] == control.canonical_url
    assert transport.calls[0][1] == {
        "Accept": "*/*",
        "Accept-Encoding": "identity",
    }
    assert list(lake.staging_root.iterdir()) == []
    assert not list(tmp_path.rglob("part-000.json.gz"))


def test_plans_all_upstream_partitions_as_one_release_group_without_network(
    tmp_path: Path,
) -> None:
    body = _archive(("part.json.gz", ({"id": "one"},)))
    loader = OpenAireGraphBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        limits=_limits(),
        transport=FakeTransport(),
    )
    names = [
        ("publication_1.tar", "publication"),
        ("software.tar", "software"),
        ("dataset_1.tar", "dataset"),
        ("otherresearchproduct.tar", "otherresearchproduct"),
        ("product_Cites_1.tar", "product_Cites"),
        ("future_entity_1.tar", "future_entity"),
    ]
    controls = [
        _control(
            body,
            file_key=file_key,
            partition=partition,
            manifest_index=index,
            manifest_count=len(names),
        )
        for index, (file_key, partition) in enumerate(names)
    ]

    plans = [loader.plan_shard(control) for control in controls]

    assert all(plan is not None for plan in plans)
    assert {(plan.dataset, plan.release) for plan in plans if plan is not None} == {
        ("graph", RELEASE)
    }
    assert [loader.shard_order(control)[1] for control in controls] == list(range(len(names)))
    assert not loader.transport.calls

    release_control = replace(
        controls[0],
        source_record_id=f"openaire-graph:release:{RELEASE}",
        raw={"record_type": "release_manifest"},
    )
    assert loader.plan_shard(release_control) is None


def test_exact_control_replay_is_network_free(tmp_path: Path) -> None:
    body = _archive(("part.json.gz", ({"id": "one"},)))
    control = _control(body)
    transport = FakeTransport(body)
    loader = OpenAireGraphBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        limits=_limits(),
        transport=transport,
    )

    first = loader.load(control)
    repeated = loader.load(control)

    assert first.path == repeated.path
    assert repeated.already_committed is True
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("control_kwargs", "transport_kwargs", "message"),
    [
        ({"checksum": "0" * 32}, {}, "published MD5 checksum"),
        ({"size": 1}, {}, "Content-Length changed"),
        ({}, {"headers": {"Content-Length": "1"}}, "Content-Length changed"),
        ({}, {"headers": {}}, "missing Content-Length"),
        ({}, {"headers": {"Content-Length": "not-a-number"}}, "Content-Length"),
        (
            {},
            {"headers": {"Content-Length": "10240", "Content-Encoding": "gzip"}},
            "content encoding",
        ),
        (
            {},
            {
                "headers": {
                    "Content-Length": "10240",
                    "OC-Checksum": f"MD5:{'0' * 32}",
                }
            },
            "response checksum changed",
        ),
        (
            {},
            {
                "headers": {
                    "Content-Length": "10240",
                    "OC-Checksum": "SHA-256:bad",
                }
            },
            "OC-Checksum header is malformed",
        ),
    ],
)
def test_checksum_and_wire_count_drift_are_rejected(
    tmp_path: Path,
    control_kwargs: Mapping[str, Any],
    transport_kwargs: Mapping[str, Any],
    message: str,
) -> None:
    body = _archive(("part.json.gz", ({"id": "one"},)))
    control = _control(body, **control_kwargs)
    transport = FakeTransport(body, **transport_kwargs)
    loader = OpenAireGraphBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        limits=_limits(),
        transport=transport,
    )

    with pytest.raises(ValueError, match=message):
        loader.load(control)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda record: replace(
                record,
                canonical_url="https://evil.example/payload.tar",
            ),
            "canonical URL",
        ),
        (
            lambda record: replace(
                record,
                links=(Link(record.canonical_url, relation="bulk_payload", crawl=True),),
            ),
            "crawl frontier",
        ),
        (
            lambda record: replace(
                record,
                links=(Link("https://evil.example/file", relation="references", crawl=False),),
            ),
            "bulk_payload",
        ),
        (
            lambda record: replace(
                record,
                raw={**record.raw, "download_url": f"{record.canonical_url}?token=x"},
                canonical_url=f"{record.canonical_url}?token=x",
                links=(
                    Link(
                        f"{record.canonical_url}?token=x",
                        relation="bulk_payload",
                        crawl=False,
                    ),
                ),
            ),
            "query-free",
        ),
        (
            lambda record: replace(
                record,
                raw={**record.raw, "manifest_index": 1},
            ),
            "smaller than manifest_count",
        ),
        (
            lambda record: replace(
                record,
                raw={
                    **record.raw,
                    "checksum": {"algorithm": "sha256", "value": "a" * 64},
                },
            ),
            "algorithm must be md5",
        ),
    ],
)
def test_malformed_or_unsafe_controls_are_rejected_before_network(
    tmp_path: Path,
    change: Any,
    message: str,
) -> None:
    body = _archive(("part.json.gz", ({"id": "one"},)))
    control = change(_control(body))
    transport = FakeTransport(body)
    loader = OpenAireGraphBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        limits=_limits(),
        transport=transport,
    )

    with pytest.raises(ValueError, match=message):
        loader.load(control)
    assert transport.calls == []


@pytest.mark.parametrize(
    ("name", "member_type", "linkname", "message"),
    [
        ("../escape.json.gz", None, None, "unsafe"),
        ("/absolute.json.gz", None, None, "unsafe"),
        ("nested\\escape.json.gz", None, None, "unsafe"),
        ("not-gzip.json", None, None, "unsafe"),
        ("link.json.gz", tarfile.SYMTYPE, "../../escape", "not a regular file"),
        ("hard.json.gz", tarfile.LNKTYPE, "target", "not a regular file"),
        ("device.json.gz", tarfile.CHRTYPE, None, "not a regular file"),
        ("fifo.json.gz", tarfile.FIFOTYPE, None, "not a regular file"),
    ],
)
def test_unsafe_names_links_and_device_members_are_rejected(
    tmp_path: Path,
    name: str,
    member_type: bytes | None,
    linkname: str | None,
    message: str,
) -> None:
    member_body = _gzip_rows({"id": "one"}) if member_type is None else b""
    body = _tar([(name, member_body, member_type, linkname)])
    loader = OpenAireGraphBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        limits=_limits(),
        transport=FakeTransport(body),
    )

    with pytest.raises(ValueError, match=message):
        loader.load(_control(body))


def test_duplicate_archive_file_names_are_rejected(tmp_path: Path) -> None:
    member = _gzip_rows({"id": "one"})
    body = _tar(
        [
            ("part.json.gz", member, None, None),
            ("part.json.gz", member, None, None),
        ]
    )
    loader = OpenAireGraphBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        limits=_limits(),
        transport=FakeTransport(body),
    )

    with pytest.raises(ValueError, match="repeats member"):
        loader.load(_control(body))


def test_truncated_gzip_and_tar_are_rejected_without_committing(
    tmp_path: Path,
) -> None:
    compressed = _gzip_rows({"id": "one"})
    bad_gzip_archive = _tar([("part.json.gz", compressed[:-5], None, None)])
    valid_archive = _tar([("part.json.gz", compressed, None, None)])
    truncated_tar = valid_archive[: 512 + len(compressed) - 4]

    for index, body in enumerate((bad_gzip_archive, truncated_tar)):
        lake = ParquetLandingZone(tmp_path / f"lake-{index}")
        loader = OpenAireGraphBulkLoader(
            lake,
            limits=_limits(),
            transport=FakeTransport(body),
        )
        with pytest.raises(ValueError, match="invalid|truncated"):
            loader.load(_control(body))
        assert not list(lake.shards_root.rglob("manifest.json"))


@pytest.mark.parametrize(
    ("limit_changes", "message"),
    [
        (
            {"max_archive_bytes": 1, "download_chunk_bytes": 1},
            "max_archive_bytes",
        ),
        ({"max_archive_members": 1}, "max_archive_members"),
        ({"max_member_compressed_bytes": 1}, "max_member_compressed_bytes"),
        (
            {
                "max_member_uncompressed_bytes": 1,
                "max_payload_bytes": 1,
                "decompression_chunk_bytes": 1,
            },
            "max_member_uncompressed_bytes",
        ),
        (
            {
                "max_total_uncompressed_bytes": 1,
                "decompression_chunk_bytes": 1,
            },
            "max_total_uncompressed_bytes",
        ),
        ({"max_payload_bytes": 1}, "max_payload_bytes"),
        ({"max_rows": 1}, "max_rows"),
        ({"max_member_name_bytes": 2}, "unsafe"),
    ],
)
def test_every_archive_and_payload_ceiling_is_enforced(
    tmp_path: Path,
    limit_changes: Mapping[str, int],
    message: str,
) -> None:
    body = _archive(
        ("first.json.gz", ({"id": "one", "payload": "long"},)),
        ("second.json.gz", ({"id": "two"},)),
    )
    limits = _limits(**limit_changes)
    loader = OpenAireGraphBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        limits=limits,
        transport=FakeTransport(body),
    )

    with pytest.raises(ValueError, match=message):
        loader.load(_control(body))


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (b"[]", "must contain a JSON object"),
        (b'{"a":1,"a":2}', "not a valid JSON object"),
        (b'{"value":NaN}', "not a valid JSON object"),
        (b"", "empty"),
        (b"{", "not a valid JSON object"),
    ],
)
def test_jsonl_rows_must_be_finite_unique_keyed_objects(
    tmp_path: Path,
    line: bytes,
    message: str,
) -> None:
    body = _tar([("part.json.gz", _gzip_lines(line), None, None)])
    loader = OpenAireGraphBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        limits=_limits(),
        transport=FakeTransport(body),
    )

    with pytest.raises(ValueError, match=message):
        loader.load(_control(body))


def test_unterminated_jsonl_row_is_rejected_as_possible_truncation(
    tmp_path: Path,
) -> None:
    body = _tar(
        [
            (
                "part.json.gz.gz",
                gzip.compress(b'{"id":"one"}', mtime=0),
                None,
                None,
            )
        ]
    )
    loader = OpenAireGraphBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        limits=_limits(),
        transport=FakeTransport(body),
    )

    with pytest.raises(ValueError, match="unterminated JSONL"):
        loader.load(_control(body))


def test_redirect_and_final_url_must_remain_the_exact_control_object(
    tmp_path: Path,
) -> None:
    body = _archive(("part.json.gz", ({"id": "one"},)))
    control = _control(body)

    for index, transport in enumerate(
        (
            FakeTransport(body, redirect_url="https://evil.example/archive.tar"),
            FakeTransport(body, final_url="https://evil.example/archive.tar"),
        )
    ):
        loader = OpenAireGraphBulkLoader(
            ParquetLandingZone(tmp_path / f"lake-{index}"),
            limits=_limits(),
            transport=transport,
        )
        with pytest.raises(ValueError, match="exact object"):
            loader.load(control)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_archive_bytes": 0},
        {"max_archive_members": True},
        {"download_chunk_bytes": 3, "max_archive_bytes": 2},
        {
            "decompression_chunk_bytes": 3,
            "max_member_uncompressed_bytes": 2,
        },
        {"max_payload_bytes": 3, "max_member_uncompressed_bytes": 2},
    ],
)
def test_limit_configuration_is_strict(changes: Mapping[str, Any]) -> None:
    with pytest.raises(ValueError):
        OpenAireBulkLimits(**changes)
