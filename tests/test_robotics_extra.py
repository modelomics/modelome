from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.robotics_extra import ArgusCheckpointInventorySourceAdapter


class _Client:
    def __init__(self, body: str) -> None:
        self.body = body.encode()
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return HttpResponse(200, {}, self.body, url)


def test_argus_inventory_preserves_member_paths_and_exact_archive_link() -> None:
    html = """
    <a href="/downloads/file_stream/4802810">Argus.zip</a>
    <table>
      <tr><th>File path</th><th>Variant</th><th>Task</th><th>Notes</th></tr>
      <tr><td><code>flat_plane.pt</code></td><td>20-leg</td>
          <td>Flat-ground locomotion</td><td>Primary policy</td></tr>
      <tr><td><code>argus_object_tracking/object_tracking_encoder.pt</code></td>
          <td>20-leg</td><td>Object tracking</td><td>Point-cloud encoder</td></tr>
    </table>
    """
    client = _Client(html)
    adapter = ArgusCheckpointInventorySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls[0][0] == "https://datadryad.org/dataset/doi:10.5061/dryad.3j9kd520k"
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["archive_url"] == "https://datadryad.org/downloads/file_stream/4802810"
    assert [record.source_record_id for record in page.records] == [
        "argus:flat_plane.pt",
        "argus:argus_object_tracking/object_tracking_encoder.pt",
    ]

    record = page.records[1]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.identifiers == (
        Identifier("argus:checkpoint", "argus_object_tracking/object_tracking_encoder.pt"),
    )
    assert record.releases[0].metadata["archive_member_path"] == (
        "argus_object_tracking/object_tracking_encoder.pt"
    )
    assert record.releases[0].metadata["robot_variant"] == "20-leg"
    assert (
        "https://datadryad.org/downloads/file_stream/4802810",
        "model_artifact",
        (record.models[0].local_id,),
    ) in {
        (link.url, link.relation, link.model_local_ids) for link in record.links
    }


def test_argus_inventory_rejects_changed_archive_link() -> None:
    client = _Client(
        '<a href="https://datadryad.org/downloads/file_stream/999">Argus.zip</a>'
        "<table><tr><td>flat_plane.pt</td><td>20-leg</td><td>Flat locomotion</td></tr></table>"
    )
    adapter = ArgusCheckpointInventorySourceAdapter(client=client)

    try:
        adapter.fetch_page({})
    except ValueError as error:
        assert "changed unexpectedly" in str(error)
    else:
        raise AssertionError("changed archive URL must be rejected")
