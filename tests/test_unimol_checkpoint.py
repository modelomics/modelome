from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.unimol_checkpoint import (
    _CHECKPOINTS,
    UniMolCheckpointSourceAdapter,
)

REVISION = "f" * 40
SOURCE_URL = (
    f"https://raw.githubusercontent.com/deepmodeling/Uni-Mol/{REVISION}/unimol/README.md"
)
BASE_URL = "https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/"


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: bytes, url: str = "https://api.github.com") -> HttpResponse:
    return HttpResponse(200, {}, body, url)


def source_document(filenames: tuple[str, ...] | None = None) -> bytes:
    items = filenames or tuple(_CHECKPOINTS)
    markdown = [f"- [checkpoint]({BASE_URL}{filename})" for filename in items]
    return ("\n".join(markdown) + "\n").encode()


def test_indexes_six_official_unimol_release_assets() -> None:
    api = "https://api.github.com/repos/deepmodeling/Uni-Mol/commits/main"
    client = QueueClient(
        response(json.dumps({"sha": REVISION}).encode(), api),
        response(source_document(), SOURCE_URL),
    )
    adapter = UniMolCheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 6
    records = {record.raw["checkpoint_filename"]: record for record in page.records}
    assert set(records) == set(_CHECKPOINTS)
    for filename, handle in _CHECKPOINTS.items():
        record = records[filename]
        weight_url = BASE_URL + filename
        assert record.raw["checkpoint_handle"] == handle
        assert record.raw["weight_url"] == weight_url
        assert record.links[0].url == weight_url
        assert record.models[0].identifiers == (Identifier("unimol:checkpoint", handle),)
        assert record.releases[0].version == "v0.1"
    assert client.calls == [api, SOURCE_URL]


def test_skips_source_fetch_when_revision_is_unchanged() -> None:
    api = "https://api.github.com/repos/deepmodeling/Uni-Mol/commits/main"
    client = QueueClient(response(json.dumps({"sha": REVISION}).encode(), api))

    page = UniMolCheckpointSourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 6}
    )

    assert page.records == ()
    assert page.upstream_count == 6
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "document",
    [
        source_document(tuple(_CHECKPOINTS)[:-1]),
        source_document(tuple(_CHECKPOINTS) + ("unknown.pt",)),
        source_document().replace(
            b"mol_pre_no_h_220816.pt",
            b"mol_pre_no_h_220816.pt\n- [duplicate](" + BASE_URL.encode()
            + b"mol_pre_no_h_220816.pt)",
        ),
        source_document().replace(
            BASE_URL.encode(), b"https://example.org/"
        ),
    ],
)
def test_rejects_missing_duplicate_or_noncanonical_release_urls(document: bytes) -> None:
    api = "https://api.github.com/repos/deepmodeling/Uni-Mol/commits/main"
    client = QueueClient(
        response(json.dumps({"sha": REVISION}).encode(), api),
        response(document, SOURCE_URL),
    )
    with pytest.raises(ValueError):
        UniMolCheckpointSourceAdapter(client=client).fetch_page({})
