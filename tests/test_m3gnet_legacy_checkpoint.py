from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.m3gnet_legacy_checkpoint import (
    M3GNetLegacyCheckpointSourceAdapter,
    _parse_registry,
)

_REVISION = "d" * 40
_FILES = (
    "checkpoint",
    "m3gnet.json",
    "m3gnet.index",
    "m3gnet.data-00000-of-00001",
)
_SOURCE = """
MODEL_FILES = {
    "MP-2021.2.8-EFS": [
        "checkpoint", "m3gnet.json", "m3gnet.index", "m3gnet.data-00000-of-00001"
    ],
}
GITHUB_RAW_LINK = "https://raw.githubusercontent.com/materialsvirtuallab/m3gnet/main/pretrained/{model_name}/{filename}"
"""


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


def _response(payload: str | Mapping[str, Any]) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/source")


def _adapter(client: _QueuedClient) -> M3GNetLegacyCheckpointSourceAdapter:
    return M3GNetLegacyCheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_m3gnet_legacy_checkpoint_emits_exact_model_and_file_urls() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_SOURCE))

    page = _adapter(client).fetch_page({})

    assert page.authoritative_snapshot is True
    assert page.upstream_count == 1
    assert len(page.records) == 1
    record = page.records[0]
    assert record.models[0].identifiers == (Identifier("m3gnet:checkpoint", "MP-2021.2.8-EFS"),)
    assert tuple(link.locator for link in record.links[1:]) == _FILES
    assert tuple(link.url for link in record.links[1:]) == tuple(
        "https://raw.githubusercontent.com/materialsvirtuallab/m3gnet/main/"
        f"pretrained/MP-2021.2.8-EFS/{filename}"
        for filename in _FILES
    )
    assert record.releases[0].metadata["model_files"] == list(_FILES)


@pytest.mark.parametrize(
    "source",
    [
        _SOURCE.replace("m3gnet.index", "unlisted.index"),
        _SOURCE.replace("materialsvirtuallab/m3gnet", "attacker/m3gnet"),
        _SOURCE.replace("MP-2021.2.8-EFS", "MP-unknown"),
    ],
)
def test_m3gnet_legacy_checkpoint_rejects_changed_first_party_inventory(source: str) -> None:
    with pytest.raises(ValueError):
        _parse_registry(source, "m3gnet-legacy")


def test_m3gnet_legacy_checkpoint_refresh_skips_unchanged_revision() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}))

    page = _adapter(client).fetch_page({"completed_revision": _REVISION, "model_count": 1})

    assert page.records == ()
    assert page.upstream_count == 1
    assert len(client.calls) == 1
