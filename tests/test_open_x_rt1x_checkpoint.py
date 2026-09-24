from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.open_x_rt1x_checkpoint import OpenXRT1XCheckpointSourceAdapter

_PREFIX = "open_x_embodiment_and_rt_x_oss/rt_1_x_jax/"


class _Client:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, Mapping[str, Any] | None]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, params))
        payload = self.payloads[len(self.calls) - 1]
        return HttpResponse(200, {}, json.dumps(payload).encode(), url)


def test_adapter_keeps_only_exact_prefix_objects_and_pins_generation() -> None:
    client = _Client(
        [
            {
                "items": [
                    {
                        "name": _PREFIX + "checkpoint/manifest.ocdbt",
                        "size": "41",
                        "generation": "1700000000000000",
                        "md5Hash": "abc=",
                    },
                    {
                        "name": _PREFIX + "checkpoint/part-0",
                        "size": "99",
                        "generation": "1700000000000001",
                    },
                    {"name": "some-other-model/checkpoint", "size": "8"},
                    {"name": _PREFIX, "size": "0"},
                ],
                "nextPageToken": "page-two",
            },
            {
                "items": [
                    {
                        "name": _PREFIX + "checkpoint/part-1",
                        "size": "101",
                        "generation": "1700000000000002",
                    },
                ]
            },
        ]
    )
    adapter = OpenXRT1XCheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert len(client.calls) == 2
    assert client.calls[0][0] == (
        "https://storage.googleapis.com/storage/v1/b/gdm-robotics-open-x-embodiment/o"
    )
    assert client.calls[0][1]["prefix"] == _PREFIX
    assert client.calls[1][1]["pageToken"] == "page-two"
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == (
        "https://github.com/google-deepmind/open_x_embodiment/blob/main/README.md"
    )
    assert record.identifiers == (Identifier("open-x-embodiment:model", "RT-1-X-JAX"),)
    assert record.releases[0].identifiers == (
        Identifier("open-x-embodiment:checkpoint", "rt_1_x_jax"),
    )
    artifacts = [link.url for link in record.links if link.relation == "model_artifact"]
    assert artifacts == [
        "https://storage.googleapis.com/gdm-robotics-open-x-embodiment/"
        + _PREFIX
        + "checkpoint/manifest.ocdbt?generation=1700000000000000",
        "https://storage.googleapis.com/gdm-robotics-open-x-embodiment/"
        + _PREFIX
        + "checkpoint/part-0?generation=1700000000000001",
        "https://storage.googleapis.com/gdm-robotics-open-x-embodiment/"
        + _PREFIX
        + "checkpoint/part-1?generation=1700000000000002",
    ]


def test_adapter_rejects_lists_that_exceed_configured_file_bound() -> None:
    client = _Client(
        [
            {
                "items": [
                    {"name": _PREFIX + "a", "generation": "1"},
                    {"name": _PREFIX + "b", "generation": "2"},
                ]
            }
        ]
    )
    adapter = OpenXRT1XCheckpointSourceAdapter(client=client, max_files=1)

    with pytest.raises(ValueError, match="exceeds 1 objects"):
        adapter.fetch_page({})


def test_adapter_rejects_repeated_pagination_token() -> None:
    client = _Client(
        [
            {"items": [], "nextPageToken": "loop"},
            {"items": [], "nextPageToken": "loop"},
        ]
    )
    adapter = OpenXRT1XCheckpointSourceAdapter(client=client)

    with pytest.raises(ValueError, match="repeated a page token"):
        adapter.fetch_page({})


def test_adapter_rejects_unsafe_object_names_in_checkpoint_prefix() -> None:
    client = _Client([{"items": [{"name": _PREFIX + "bad name", "generation": "1"}]}])
    adapter = OpenXRT1XCheckpointSourceAdapter(client=client)

    with pytest.raises(ValueError, match="invalid GCS object name"):
        adapter.fetch_page({})
