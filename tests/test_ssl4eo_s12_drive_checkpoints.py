from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.ssl4eo_s12_drive_checkpoints import (
    _CHECKPOINTS,
    _README_URL,
    SSL4EOS12DriveCheckpointSourceAdapter,
)


class _Client:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(self.status, {}, self.body, url)


def _readme(file_ids: tuple[str, ...] | None = None) -> bytes:
    ids = file_ids if file_ids is not None else tuple(_CHECKPOINTS)
    return "\n".join(
        f"| {title} | [full ckpt](https://drive.google.com/file/d/{file_id}/view?usp=sharing) |"
        for file_id in ids
        for _, title, _ in [_CHECKPOINTS[file_id]]
    ).encode()


def test_extracts_only_the_first_party_allowlisted_full_checkpoints() -> None:
    client = _Client(_readme())
    page = SSL4EOS12DriveCheckpointSourceAdapter(client=client).fetch_page({})

    assert page.complete and page.upstream_count == len(_CHECKPOINTS) == 11
    assert {record.raw["google_drive_file_id"] for record in page.records} == set(_CHECKPOINTS)
    assert all(record.kind is ArtifactKind.WEIGHTS for record in page.records)
    assert all(
        record.canonical_url.startswith("https://drive.google.com/file/d/")
        for record in page.records
    )
    assert all(record.raw["repository"] == "zhu-xlab/SSL4EO-S12" for record in page.records)
    assert all(
        record.releases[0].metadata["drive_file_id"] in _CHECKPOINTS
        for record in page.records
    )
    assert client.calls == [_README_URL]

    seeds = [
        source_record_to_entry_seed(record, source="ssl4eo-s12-drive-checkpoints")
        for record in page.records
    ]
    entries = build_entries(seeds)
    assert len(entries.entries) == 11
    assert all(
        entry.resources[0].url.startswith("https://drive.google.com/")
        for entry in entries.entries
    )


def test_unchanged_readme_does_not_reemit_records() -> None:
    client = _Client(_readme())
    adapter = SSL4EOS12DriveCheckpointSourceAdapter(client=client)
    first = adapter.fetch_page({})

    second = adapter.fetch_page(first.next_state)

    assert second.records == ()
    assert second.complete and second.upstream_count == 11
    assert len(client.calls) == 2


def test_missing_known_checkpoint_fails_closed() -> None:
    body = _readme(tuple(_CHECKPOINTS)[:-1])
    with pytest.raises(ValueError, match="missing known checkpoint links"):
        SSL4EOS12DriveCheckpointSourceAdapter(client=_Client(body)).fetch_page({})


def test_rejects_non_success_and_oversized_readme() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        SSL4EOS12DriveCheckpointSourceAdapter(client=_Client(b"", status=503)).fetch_page({})
    with pytest.raises(ValueError, match="byte limit"):
        SSL4EOS12DriveCheckpointSourceAdapter(
            max_response_bytes=1,
            client=_Client(_readme()),
        ).fetch_page({})


def test_disabled_proposal_matches_adapter_configuration() -> None:
    path = Path(__file__).parents[1] / "config/proposals/ssl4eo_s12_drive_checkpoints.toml"
    source: dict[str, Any] = tomllib.loads(path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = SSL4EOS12DriveCheckpointSourceAdapter(
        name=source["name"],
        url=source["url"],
        max_response_bytes=source["max_response_bytes"],
    )
    assert adapter.name == source["name"]
    assert adapter.url == source["url"]
    assert adapter.max_response_bytes == source["max_response_bytes"]
