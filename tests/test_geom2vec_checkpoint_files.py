from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.geom2vec_checkpoint_files import (
    _CHECKPOINTS,
    Geom2VecCheckpointFilesSourceAdapter,
)

_REVISION = "f" * 40
_PREFIX = f"https://raw.githubusercontent.com/dinner-group/geom2vec/{_REVISION}/checkpoints/"


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
            "path": f"checkpoints/{filename}",
            "type": "file",
            "size": 2_000_000,
            "download_url": _PREFIX + filename,
        }
        for filename in _CHECKPOINTS
    ]


def _adapter(client: _QueuedClient) -> Geom2VecCheckpointFilesSourceAdapter:
    return Geom2VecCheckpointFilesSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_geom2vec_checkpoint_inventory_emits_all_published_files() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_directory()))

    page = _adapter(client).fetch_page({})

    assert page.authoritative_snapshot is True
    assert page.upstream_count == 12
    assert client.calls[1] == (
        "https://api.github.com/repos/dinner-group/geom2vec/contents/checkpoints"
        f"?ref={_REVISION}"
    )
    for record, filename in zip(page.records, _CHECKPOINTS, strict=True):
        handle = filename.removesuffix(".pth")
        assert record.title == handle
        assert record.models[0].identifiers == (Identifier("geom2vec:checkpoint", filename),)
        assert record.releases[0].metadata["weight_url"] == _PREFIX + filename


@pytest.mark.parametrize(
    "mutate",
    [
        lambda entries: [item for item in entries if item["name"] != _CHECKPOINTS[-1]],
        lambda entries: [
            {**entries[0], "download_url": "https://example.test/model.pth"},
            *entries[1:],
        ],
        lambda entries: [*entries, entries[0]],
        lambda entries: [
            *entries,
            {
                "name": "unlisted.pth",
                "path": "checkpoints/unlisted.pth",
                "type": "file",
                "size": 1,
                "download_url": _PREFIX + "unlisted.pth",
            },
        ],
    ],
)
def test_geom2vec_rejects_incomplete_or_untrusted_inventory(mutate: Any) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(mutate(_directory())))

    with pytest.raises(ValueError):
        _adapter(client).fetch_page({})
