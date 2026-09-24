from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.ultra_checkpoint_files import UltraCheckpointFilesSourceAdapter

_REVISION = "e" * 40
_FILES = (
    "ultra_3g.pth",
    "ultra_4g.pth",
    "ultra_50g.pth",
    "ultraquery.pth",
)
_PREFIX = f"https://raw.githubusercontent.com/DeepGraphLearning/ULTRA/{_REVISION}/ckpts/"


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: Any) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/response")


def _directory() -> list[dict[str, Any]]:
    return [
        {
            "name": filename,
            "path": f"ckpts/{filename}",
            "type": "file",
            "size": 2_000_000,
            "download_url": _PREFIX + filename,
        }
        for filename in _FILES
    ] + [
        {
            "name": "README.md",
            "path": "ckpts/README.md",
            "type": "file",
            "size": 100,
            "download_url": _PREFIX + "README.md",
        }
    ]


def _adapter(client: _QueuedClient) -> UltraCheckpointFilesSourceAdapter:
    return UltraCheckpointFilesSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_ultra_checkpoint_directory_emits_only_four_pinned_files() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_directory()))

    page = _adapter(client).fetch_page({})

    assert page.authoritative_snapshot is True
    assert page.upstream_count == 4
    assert client.calls[1] == (
        "https://api.github.com/repos/DeepGraphLearning/ULTRA/contents/ckpts"
        f"?ref={_REVISION}"
    )
    for record, filename in zip(page.records, _FILES, strict=True):
        handle = filename.removesuffix(".pth")
        assert record.title == handle
        assert record.models[0].identifiers == (
            Identifier("deepgraphlearning-ultra:checkpoint", filename),
        )
        assert record.releases[0].metadata["weight_url"] == _PREFIX + filename


@pytest.mark.parametrize(
    "mutate",
    [
        lambda entries: [entry for entry in entries if entry["name"] != "ultraquery.pth"],
        lambda entries: [
            {**entries[0], "download_url": "https://example.test/ultra_3g.pth"},
            *entries[1:],
        ],
        lambda entries: [*entries, entries[0]],
        lambda entries: [
            *entries,
            {
                "name": "ultra_next.pth",
                "path": "ckpts/ultra_next.pth",
                "type": "file",
                "size": 100,
                "download_url": _PREFIX + "ultra_next.pth",
            },
        ],
    ],
)
def test_ultra_checkpoint_directory_rejects_incomplete_or_untrusted_inventory(
    mutate: Any,
) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(mutate(_directory())))

    with pytest.raises(ValueError):
        _adapter(client).fetch_page({})
