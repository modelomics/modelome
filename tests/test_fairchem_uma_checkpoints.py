from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.normalize import content_hash
from modelome.sources.fairchem_uma_checkpoints import (
    _CHECKPOINTS,
    FairChemUMACheckpointSourceAdapter,
)

MODEL_URL = "https://huggingface.co/facebook/UMA"
FILE_BASE = "https://huggingface.co/facebook/UMA/blob/main/checkpoints/"
ACCESS_NOTICE = (
    "<p>You need to agree to share your contact information to access this model</p>"
)


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, MODEL_URL)


def document() -> str:
    rows = []
    for handle, (filename, checksum, _archived) in _CHECKPOINTS.items():
        rows.append(
            f'<tr><td>{handle}</td><td><a href="{FILE_BASE}{filename}">'
            f"{filename}</a></td><td>{checksum}</td></tr>"
        )
    return ACCESS_NOTICE + "<table>" + "".join(rows) + "</table>"


def adapter(client: QueueClient) -> FairChemUMACheckpointSourceAdapter:
    return FairChemUMACheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )


def test_indexes_exact_uma_files_checksums_and_archived_status() -> None:
    body = document().encode()
    client = QueueClient(response(body))

    page = adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 5
    assert client.calls == [MODEL_URL]
    records = {record.models[0].name: record for record in page.records}
    record = records["FAIR Chemistry uma-s-1.2.1 checkpoint"]
    assert record.models[0].identifiers == (
        Identifier("fairchem:uma-checkpoint", "uma-s-1.2.1"),
    )
    assert record.raw["checkpoint_filename"] == "uma-s-1p2p1.pt"
    assert record.raw["checkpoint_url"] == f"{FILE_BASE}uma-s-1p2p1.pt"
    assert record.raw["checksum_md5"] == "3497615fd30a24c5b35cd3b41a682e6e"
    assert record.raw["archived"] is False
    assert record.raw["access_restricted"] is True
    assert records["FAIR Chemistry uma-s-1 checkpoint"].raw["archived"] is True
    assert records["FAIR Chemistry uma-s-1 checkpoint"].releases[0].metadata[
        "binary_reachability_checked"
    ] is False


def test_skips_record_emission_when_model_card_is_unchanged() -> None:
    body = document().encode()
    client = QueueClient(response(body))

    page = adapter(client).fetch_page(
        {"source_sha256": content_hash(body), "model_count": len(_CHECKPOINTS)}
    )

    assert page.records == ()
    assert page.upstream_count == 5


@pytest.mark.parametrize(
    "source",
    [
        document().replace("uma-s-1p1.pt", "uma-s-1p1-other.pt"),
        document().replace("36a2f071350be0ee4c15e7ebdd16dde1", "0" * 32),
        document().replace("uma-s-1.2.1", "uma-s-1.2-new", 1),
        document().replace(ACCESS_NOTICE, ""),
    ],
)
def test_rejects_unverified_rows_or_missing_access_gate(source: str) -> None:
    client = QueueClient(response(source.encode()))

    with pytest.raises(ValueError):
        adapter(client).fetch_page({})
