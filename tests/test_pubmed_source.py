from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.pubmed import PubMedBulkSourceAdapter

FIXTURES = Path(__file__).parent / "fixtures"
BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
UPDATE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
NOW = datetime(2026, 9, 2, 18, 0, tzinfo=UTC)


class MappingClient:
    def __init__(self, bodies: Mapping[str, bytes | str]) -> None:
        self.bodies = dict(bodies)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if url not in self.bodies:
            raise AssertionError(f"unexpected GET {url}")
        body = self.bodies[url]
        if isinstance(body, str):
            body = body.encode()
        return HttpResponse(status=200, headers={}, body=body, url=url)


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _bodies() -> dict[str, bytes]:
    result = {
        BASELINE_URL: _fixture("pubmed_baseline_index.html"),
        UPDATE_URL: _fixture("pubmed_update_index.html"),
    }
    for sequence in range(1, 5):
        root = BASELINE_URL if sequence < 3 else UPDATE_URL
        url = f"{root}pubmed26n{sequence:04}.xml.gz.md5"
        result[url] = _fixture(f"pubmed_26n{sequence:04}.md5")
    return result


def _adapter(client: MappingClient, *, page_size: int = 2) -> PubMedBulkSourceAdapter:
    return PubMedBulkSourceAdapter(
        baseline_url=BASELINE_URL,
        update_url=UPDATE_URL,
        page_size=page_size,
        client=client,
        clock=lambda: NOW,
    )


def _finish_scan(
    adapter: PubMedBulkSourceAdapter,
    state: Mapping[str, Any] | None = None,
) -> tuple[list[Any], Mapping[str, Any]]:
    records = []
    current = dict(state or {})
    for _ in range(20):
        page = adapter.fetch_page(current)
        assert not page.issues
        records.extend(page.records)
        current = dict(page.next_state)
        if page.complete:
            return records, current
    raise AssertionError("scan did not complete")


def test_full_manifest_scan_emits_baseline_then_ordered_update_control_records() -> None:
    client = MappingClient(_bodies())
    adapter = _adapter(client)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.upstream_count == 4
    assert first.next_state["scan_mode"] == "reload"
    assert first.next_state["scan_high_sequence"] == 4
    assert first.next_state["cursor"] == 2
    assert [record.raw["sequence"] for record in first.records] == [1, 2]
    assert [record.raw["manifest_kind"] for record in first.records] == [
        "baseline",
        "baseline",
    ]
    baseline_semantics = first.records[0].raw["application_semantics"]
    assert baseline_semantics["shard_action"] == "stage_complete_snapshot_shard"
    assert baseline_semantics["commit_only_after_all_baseline_shards"] is True

    assert second.complete is True
    assert [record.raw["sequence"] for record in second.records] == [3, 4]
    assert [record.raw["manifest_kind"] for record in second.records] == [
        "update",
        "update",
    ]
    assert second.next_state["baseline_cycle"] == "26"
    assert second.next_state["baseline_last_sequence"] == 2
    assert second.next_state["last_applied_sequence"] == 4
    assert second.next_state["baseline_file_count"] == 2
    assert second.next_state["update_file_count"] == 2

    record = second.records[0]
    assert record.source_record_id == "pubmed:pubmed26n0003.xml.gz"
    assert record.kind is ArtifactKind.CATALOG_RECORD
    assert record.identifiers == (
        Identifier("pubmed:bulk-file", "pubmed26n0003.xml.gz"),
    )
    assert record.raw["checksum"]["value"] == "3" * 32
    assert record.raw["production_year"] == 2026
    assert record.raw["application_order"] == 3
    assert record.raw["application_semantics"]["deletion_element"] == "DeleteCitation"
    assert record.raw["application_semantics"]["deletion_action"] == "delete_by_pmid"
    assert {link.relation for link in record.links} == {"bulk_payload", "checksum"}
    assert all(link.crawl is False for link in record.links)
    assert all(not call[0].endswith(".xml.gz") for call in client.calls)


def test_new_update_arriving_mid_scan_waits_for_the_next_incremental_run() -> None:
    bodies = _bodies()
    client = MappingClient(bodies)
    adapter = _adapter(client, page_size=1)

    first = adapter.fetch_page({})
    client.bodies[UPDATE_URL] = _index(
        "updatefiles",
        [
            (3, "2026-01-30 14:02", "43M"),
            (4, "2026-01-31 14:04", "69M"),
            (5, "2026-02-01 14:01", "20M"),
        ],
    ).encode()
    client.bodies[f"{UPDATE_URL}pubmed26n0005.xml.gz.md5"] = (
        b"55555555555555555555555555555555  pubmed26n0005.xml.gz\n"
    )

    records = list(first.records)
    state = dict(first.next_state)
    while True:
        page = adapter.fetch_page(state)
        assert not page.issues
        records.extend(page.records)
        state = dict(page.next_state)
        if page.complete:
            break

    assert [record.raw["sequence"] for record in records] == [1, 2, 3, 4]
    assert state["last_applied_sequence"] == 4

    incremental = adapter.fetch_page(state)

    assert incremental.complete is True
    assert incremental.next_state["last_applied_sequence"] == 5
    assert [record.raw["sequence"] for record in incremental.records] == [5]
    assert incremental.records[0].raw["manifest_kind"] == "update"


def test_unchanged_manifest_completes_without_refetching_sidecars() -> None:
    client = MappingClient(_bodies())
    adapter = _adapter(client, page_size=10)
    _, complete_state = _finish_scan(adapter)
    call_count = len(client.calls)

    page = adapter.fetch_page(complete_state)

    assert page.complete is True
    assert page.records == ()
    assert page.upstream_count == 0
    assert len(client.calls) == call_count + 2
    assert client.calls[-2][0] == BASELINE_URL
    assert client.calls[-1][0] == UPDATE_URL


def test_changed_applied_update_prefix_forces_a_complete_replay() -> None:
    client = MappingClient(_bodies())
    adapter = _adapter(client, page_size=10)
    _, complete_state = _finish_scan(adapter)
    client.bodies[UPDATE_URL] = _index(
        "updatefiles",
        [(3, "2026-01-30 14:02", "43M"), (4, "2026-02-02 09:00", "70M")],
    ).encode()

    replay = adapter.fetch_page(complete_state)

    assert replay.complete is True
    assert [record.raw["sequence"] for record in replay.records] == [1, 2, 3, 4]
    assert [record.raw["manifest_kind"] for record in replay.records[:2]] == [
        "baseline",
        "baseline",
    ]


def test_missing_pair_malformed_name_and_sequence_gap_are_quarantined() -> None:
    baseline = _fixture("pubmed_baseline_index.html").decode().replace(
        '<a href="pubmed26n0002.xml.gz.md5">pubmed26n0002.xml.gz.md5</a> 2026-01-29 14:48   60\n',
        '<a href="pubmed26nBAD.xml.gz">pubmed26nBAD.xml.gz</a> 2026-01-29 14:48  17M\n',
    )
    updates = _index("updatefiles", [(4, "2026-01-31 14:04", "69M")])
    bodies = _bodies()
    bodies[BASELINE_URL] = baseline.encode()
    bodies[UPDATE_URL] = updates.encode()
    client = MappingClient(bodies)

    page = _adapter(client).fetch_page({})

    assert page.complete is False
    assert page.records == ()
    assert page.retry_state == {}
    assert {issue.stage for issue in page.issues} == {"source_manifest"}
    errors = "\n".join(issue.error for issue in page.issues)
    assert "malformed PubMed shard filename" in errors
    assert "has no MD5 sidecar" in errors
    assert "does not continue contiguously" in errors


def test_checksum_target_mismatch_is_quarantined_at_the_same_page_boundary() -> None:
    bodies = _bodies()
    bodies[f"{BASELINE_URL}pubmed26n0001.xml.gz.md5"] = (
        b"MD5 (pubmed26n9999.xml.gz) = 11111111111111111111111111111111\n"
    )
    client = MappingClient(bodies)
    adapter = _adapter(client, page_size=1)

    page = adapter.fetch_page({})

    assert page.complete is False
    assert page.records == ()
    assert page.retry_state is not None
    assert page.retry_state["cursor"] == 0
    assert page.next_state["cursor"] == 1
    assert page.issues[0].stage == "source_normalize"
    assert "expected 'pubmed26n0001.xml.gz'" in page.issues[0].error


def test_frozen_manifest_drift_restarts_from_the_last_completed_checkpoint() -> None:
    client = MappingClient(_bodies())
    adapter = _adapter(client, page_size=1)
    first = adapter.fetch_page({})
    client.bodies[BASELINE_URL] = _index(
        "baseline",
        [(1, "2026-01-29 14:48", "20M"), (2, "2026-01-29 14:48", "17M")],
    ).encode()

    page = adapter.fetch_page(first.next_state)

    assert page.complete is False
    assert page.records == ()
    assert page.retry_state == {}
    assert any("baseline changed" in issue.error for issue in page.issues)


def test_checkpoint_shape_and_configuration_signature_fail_closed() -> None:
    client = MappingClient(_bodies())
    adapter = _adapter(client)

    with pytest.raises(ValueError, match="incomplete PubMed frozen-scan checkpoint"):
        adapter.fetch_page({"cursor": 1})

    same = _adapter(MappingClient(_bodies()))
    different = _adapter(MappingClient(_bodies()), page_size=3)
    assert adapter.checkpoint_signature == same.checkpoint_signature
    assert adapter.checkpoint_signature != different.checkpoint_signature


def _index(
    directory: str,
    shards: list[tuple[int, str, str]],
) -> str:
    rows = [
        "<!doctype html>",
        f"<html><body><h1>Index of /pubmed/{directory}</h1><pre>",
    ]
    for sequence, modified, size in shards:
        filename = f"pubmed26n{sequence:04}.xml.gz"
        rows.extend(
            (
                f'<a href="{filename}">{filename}</a> {modified} {size}',
                f'<a href="{filename}.md5">{filename}.md5</a> {modified} 60',
            )
        )
    rows.append("</pre></body></html>")
    return "\n".join(rows)
