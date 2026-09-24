from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.normalize import content_hash
from modelome.sources.torchmdnet_aceff_checkpoints import (
    _COLLECTION_URL,
    _MODELS,
    TorchMDNetAceFFCheckpointSourceAdapter,
)

COLLECTION_HTML = """\
<html><body>
{models}
</body></html>
"""


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, _COLLECTION_URL)


def document() -> str:
    links = "\n".join(
        f'<a href="/Acellera/{version}">Acellera/{version}</a>' for version in _MODELS
    )
    return COLLECTION_HTML.format(models=links)


def adapter(client: QueueClient) -> TorchMDNetAceFFCheckpointSourceAdapter:
    return TorchMDNetAceFFCheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )


def test_indexes_exact_torchmdnet_aceff_release_to_file_map() -> None:
    body = document().encode()
    client = QueueClient(response(body))

    page = adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert client.calls == [_COLLECTION_URL]
    records = {record.models[0].name: record for record in page.records}
    record = records["AceFF-1.1 neural network potential"]
    assert record.models[0].identifiers == (
        Identifier("torchmdnet:aceff-checkpoint", "AceFF-1.1"),
    )
    assert record.raw["repository"] == "Acellera/AceFF-1.1"
    assert record.raw["checkpoint_filename"] == "aceff_v1.1.ckpt"
    assert record.raw["checkpoint_url"] == (
        "https://huggingface.co/Acellera/AceFF-1.1/blob/main/aceff_v1.1.ckpt"
    )
    assert records["AceFF-2.0 neural network potential"].raw["checkpoint_filename"] == (
        "aceff_v2.0.ckpt"
    )


def test_skips_records_when_collection_html_is_unchanged() -> None:
    body = document().encode()
    client = QueueClient(response(body))

    page = adapter(client).fetch_page(
        {"source_sha256": content_hash(body), "model_count": len(_MODELS)}
    )

    assert page.records == ()
    assert page.upstream_count == 3


@pytest.mark.parametrize(
    "document_html",
    [
        document().replace("Acellera/AceFF-2.0", "Acellera/AceFF-3.0"),
        document().replace("Acellera/AceFF-1.0", "Acellera/AceFF-1.1", 1),
        document().replace('/Acellera/AceFF-1.1', '/Other/AceFF-1.1'),
    ],
)
def test_rejects_changed_or_ambiguous_collection(document_html: str) -> None:
    client = QueueClient(response(document_html.encode()))

    with pytest.raises(ValueError):
        adapter(client).fetch_page({})
