from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.http import HttpResponse
from modelome.sources.arxiv_snapshot import ArxivCompleteSnapshotSourceAdapter

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
REVISION = "a" * 40
DATA_PATH = "metadata/train-00000-of-00001.parquet"
METADATA_URL = "https://hub.example.test/api/datasets/example/arxiv"


class RangeClient:
    def __init__(self, parquet: bytes) -> None:
        self.parquet = parquet
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, headers: dict[str, str] | None = None, **_: Any) -> HttpResponse:
        request_headers = dict(headers or {})
        self.calls.append((url, request_headers))
        if url == METADATA_URL:
            return HttpResponse(
                status=200,
                headers={},
                body=json.dumps(
                    {
                        "id": "example/arxiv",
                        "sha": REVISION,
                        "private": False,
                        "gated": False,
                        "siblings": [{"rfilename": DATA_PATH}],
                    }
                ).encode(),
                url=url,
            )
        range_header = request_headers["Range"]
        assert range_header.startswith("bytes=")
        start, end = (int(value) for value in range_header.removeprefix("bytes=").split("-"))
        body = self.parquet[start : end + 1]
        return HttpResponse(
            status=206,
            headers={"content-range": f"bytes {start}-{end}/{len(self.parquet)}"},
            body=body,
            url=url,
        )


def _parquet() -> bytes:
    table = pa.table(
        {
            "paper_id": ["2401.00001", "hep-th/9901001", "not-an-arxiv-id"],
            "title": ["A Graph Network", "Second Study", "Bad row"],
            "authors": ["Ada Example", "Bo Example", "Cy Example"],
            "abstract": ["A deep neural network model.", "", "ignored"],
            "categories": ["cs.LG cs.AI", "hep-th", "cs.LG"],
            "primary_category": ["cs.LG", "hep-th", "cs.LG"],
            "license": ["https://creativecommons.org/licenses/by/4.0/", None, None],
            "doi": ["10.1000/example", None, None],
            "first_version_date": [
                datetime(2024, 1, 2),
                datetime(1999, 1, 3),
                datetime(2024, 1, 4),
            ],
            "latest_version_date": [
                datetime(2024, 2, 2),
                datetime(1999, 1, 4),
                datetime(2024, 1, 5),
            ],
            "oai_datestamp": ["2024-02-03", "1999-01-05", "2024-01-06"],
            "oai_sets": [["cs:cs:LG"], ["physics:hep-th"], ["cs:cs:LG"]],
            "arxiv_abs_url": [
                "https://arxiv.org/abs/2401.00001",
                "https://arxiv.org/abs/hep-th/9901001",
                "https://arxiv.org/abs/invalid",
            ],
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def test_snapshot_uses_bounded_ranges_and_preserves_paper_provenance() -> None:
    client = RangeClient(_parquet())
    adapter = ArxivCompleteSnapshotSourceAdapter(
        name="arxiv-complete-snapshot",
        artifact_source="arxiv",
        dataset_id="example/arxiv",
        metadata_url=METADATA_URL,
        data_path=DATA_PATH,
        max_range_bytes=64 * 1024,
        client=client,
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is False
    assert page.advance_on_source_issues is True
    assert page.upstream_count == 3
    assert [record.source_record_id for record in page.records] == [
        "2401.00001",
        "hep-th/9901001",
    ]
    first = page.records[0]
    assert first.canonical_url == "https://arxiv.org/abs/2401.00001"
    assert first.published_at == "2024-01-02T00:00:00Z"
    assert first.raw["snapshot_revision"] == REVISION
    assert first.raw["dataset_license"] == "mixed-arxiv-author-licenses"
    assert first.raw["paper_license"] == "https://creativecommons.org/licenses/by/4.0"
    assert [(identifier.namespace, identifier.value) for identifier in first.identifiers] == [
        ("arxiv", "2401.00001"),
        ("doi", "10.1000/example"),
    ]
    assert {link.relation for link in first.links} == {
        "landing_page",
        "license",
        "published_as",
    }
    assert all(link.crawl is False for link in first.links)
    assert len(page.issues) == 1
    assert page.issues[0].source_record_id == "not-an-arxiv-id"
    assert page.next_state["completed_snapshot_revision"] == REVISION
    range_calls = [headers for url, headers in client.calls if url != METADATA_URL]
    assert range_calls
    assert all("Range" in headers for headers in range_calls)


def test_completed_snapshot_checks_manifest_without_reading_the_parquet_again() -> None:
    client = RangeClient(_parquet())
    adapter = ArxivCompleteSnapshotSourceAdapter(
        dataset_id="example/arxiv",
        metadata_url=METADATA_URL,
        data_path=DATA_PATH,
        client=client,
        clock=lambda: NOW,
    )

    page = adapter.fetch_page(
        {"snapshot_revision": REVISION, "completed_snapshot_revision": REVISION}
    )

    assert page.complete is True
    assert page.records == ()
    assert client.calls == [(METADATA_URL, {"Accept": "application/json"})]


def test_snapshot_slices_a_row_group_into_resumable_pages_without_refetching_it() -> None:
    client = RangeClient(_parquet())
    adapter = ArxivCompleteSnapshotSourceAdapter(
        dataset_id="example/arxiv",
        metadata_url=METADATA_URL,
        data_path=DATA_PATH,
        page_size=2,
        client=client,
        clock=lambda: NOW,
    )

    first = adapter.fetch_page({})
    data_calls_after_first = len(client.calls)
    second = adapter.fetch_page(first.next_state)

    assert [record.source_record_id for record in first.records] == [
        "2401.00001",
        "hep-th/9901001",
    ]
    assert first.complete is False
    assert first.next_state["row_group_index"] == 0
    assert first.next_state["row_offset"] == 2
    assert second.records == ()
    assert len(second.issues) == 1
    assert second.complete is True
    # The manifest is only fetched at the start of a frozen snapshot; the
    # decoded row group remains bounded in this adapter instance for page two.
    assert len(client.calls) == data_calls_after_first
