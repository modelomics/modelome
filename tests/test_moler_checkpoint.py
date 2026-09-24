from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.moler_checkpoint import MoLeRCheckpointSourceAdapter

REVISION = "d" * 40
WEIGHT_URL = "https://figshare.com/ndownloader/files/34642724"
FILENAME = "GNN_Edge_MLP_MoLeR__2022-02-24_07-16-23_best.pkl"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/microsoft/molecule-generation/{REVISION}/README.md"
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


def readme(url: str = WEIGHT_URL) -> bytes:
    return (
        "A MoLeR checkpoint trained using the default hyperparameters is available "
        f"[here]({url}).\n"
        f"Rename the downloaded file to `{FILENAME}`.\n"
    ).encode()


def test_indexes_exact_first_party_moler_checkpoint_link() -> None:
    api = "https://api.github.com/repos/microsoft/molecule-generation/commits/main"
    client = QueueClient(
        response(json.dumps({"sha": REVISION}).encode(), api),
        response(readme(), SOURCE_URL),
    )
    adapter = MoLeRCheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 1
    record = page.records[0]
    assert record.source_record_id == "checkpoint:moler-default-2022-02-24"
    assert record.models[0].identifiers == (
        Identifier("microsoft-moler:checkpoint", "moler-default-2022-02-24"),
    )
    assert record.raw["checkpoint_filename"] == FILENAME
    assert record.raw["weight_url"] == WEIGHT_URL
    assert record.links[0].url == WEIGHT_URL
    assert record.releases[0].metadata["repository"] == "microsoft/molecule-generation"
    assert client.calls == [api, SOURCE_URL]


def test_skips_source_fetch_when_revision_is_unchanged() -> None:
    api = "https://api.github.com/repos/microsoft/molecule-generation/commits/main"
    client = QueueClient(response(json.dumps({"sha": REVISION}).encode(), api))

    page = MoLeRCheckpointSourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 1}
    )

    assert page.records == ()
    assert page.upstream_count == 1
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "document",
    [
        b"A MoLeR checkpoint trained using the default hyperparameters is available [here](https://example.org/model.pkl).",
        b"A MoLeR checkpoint trained using the default hyperparameters is available [here](https://figshare.com/ndownloader/files/1).",
        (
            b"A MoLeR checkpoint trained using the default hyperparameters is available "
            b"[here](https://figshare.com/ndownloader/files/34642724).\n"
            b"A MoLeR checkpoint trained using the default hyperparameters is available "
            b"[here](https://figshare.com/ndownloader/files/34642724)."
        ),
    ],
)
def test_rejects_unverified_or_ambiguous_checkpoint_link(document: bytes) -> None:
    api = "https://api.github.com/repos/microsoft/molecule-generation/commits/main"
    client = QueueClient(
        response(json.dumps({"sha": REVISION}).encode(), api),
        response(document, SOURCE_URL),
    )
    with pytest.raises(ValueError):
        MoLeRCheckpointSourceAdapter(client=client).fetch_page({})
