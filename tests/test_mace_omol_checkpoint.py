from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.mace_omol_checkpoint import MaceOmolCheckpointSourceAdapter

REVISION = "c" * 40
WEIGHT_URL = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/"
    "mace_omol_0/MACE-omol-0-extra-large-1024.model"
)
SOURCE_URL = (
    f"https://raw.githubusercontent.com/ACEsuit/mace/{REVISION}/"
    "mace/calculators/foundations_models.py"
)


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: bytes, url: str = "https://api.github.com") -> HttpResponse:
    return HttpResponse(200, {}, body, url)


def source_text(url: str = WEIGHT_URL) -> bytes:
    return f'''def mace_omol(model=None):
    urls = {{"extra_large": "{url}"}}
'''.encode()


def test_indexes_exact_omol_checkpoint_from_nested_loader_map() -> None:
    api = "https://api.github.com/repos/ACEsuit/mace/commits/develop"
    client = QueueClient(
        response(json.dumps({"sha": REVISION}).encode(), api),
        response(source_text(), SOURCE_URL),
    )
    adapter = MaceOmolCheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 1
    record = page.records[0]
    assert record.source_record_id == "checkpoint:mace-omol:extra_large"
    assert record.models[0].identifiers == (
        Identifier("mace:omol-checkpoint", "extra_large"),
    )
    assert record.raw["weight_url"] == WEIGHT_URL
    assert record.links[0].url == WEIGHT_URL
    assert record.releases[0].metadata["source_function"] == "mace_omol"
    assert client.calls == [api, SOURCE_URL]


def test_skips_source_fetch_when_revision_is_unchanged() -> None:
    api = "https://api.github.com/repos/ACEsuit/mace/commits/develop"
    client = QueueClient(response(json.dumps({"sha": REVISION}).encode(), api))

    page = MaceOmolCheckpointSourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 1}
    )

    assert page.records == ()
    assert page.upstream_count == 1
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "document",
    [
        b'def mace_omol(model=None):\n    urls = get_urls()\n',
        b'def mace_omol(model=None):\n    urls = {"extra_large": URL}\n',
        b'def mace_omol(model=None):\n    urls = {"other": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model"}\n',
        b'def mace_omol(model=None):\n    urls = {"extra_large": "https://example.org/model.model"}\n',
    ],
)
def test_rejects_dynamic_or_unverified_omol_map(document: bytes) -> None:
    api = "https://api.github.com/repos/ACEsuit/mace/commits/develop"
    client = QueueClient(
        response(json.dumps({"sha": REVISION}).encode(), api),
        response(document, SOURCE_URL),
    )
    with pytest.raises(ValueError):
        MaceOmolCheckpointSourceAdapter(client=client).fetch_page({})
