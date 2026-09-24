from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.fairchem_omat24_checkpoints import (
    _CHECKPOINTS,
    FairChemOMat24CheckpointSourceAdapter,
)

DOCS_URL = "https://facebookresearch.github.io/fairchem/models-2/"
HF_BASE = "https://huggingface.co/fairchem/OMAT24/blob/main/"
ACCESS_URL = "https://huggingface.co/facebook/OMAT24"
HTML = """\
<html><body><table><thead><tr><th>Model Name</th><th>Checkpoint</th></tr></thead><tbody>
{rows}
<tr><td>EquiformerV2-153M-OMat-Alex-MP</td><td><a href="{access}">checkpoint</a></td></tr>
</tbody></table></body></html>
"""


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, DOCS_URL)


def document() -> str:
    rows = "\n".join(
        f'<tr><td>{name}</td><td><a href="{HF_BASE}{filename}">checkpoint</a></td></tr>'
        for name, filename in _CHECKPOINTS.items()
    )
    return HTML.format(rows=rows, access=ACCESS_URL)


def adapter(client: QueueClient) -> FairChemOMat24CheckpointSourceAdapter:
    return FairChemOMat24CheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )


def test_indexes_only_exact_fairchem_omat24_checkpoint_rows() -> None:
    client = QueueClient(response(document().encode()))

    page = adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == len(_CHECKPOINTS) == 9
    assert client.calls == [DOCS_URL]
    records = {record.models[0].name: record for record in page.records}
    assert set(records) == set(_CHECKPOINTS)
    record = records["EquiformerV2-31M-OMat"]
    assert record.models[0].identifiers == (
        Identifier("fairchem:omat24-checkpoint", "equiformerv2-31m-omat"),
    )
    assert record.raw["checkpoint_filename"] == "eqV2_31M_omat.pt"
    assert record.raw["checkpoint_url"] == f"{HF_BASE}eqV2_31M_omat.pt"
    assert record.raw["access_restricted"] is True
    assert record.releases[0].metadata["binary_reachability_checked"] is False
    assert record.links[1].url == ACCESS_URL


def test_skips_record_emission_when_source_content_is_unchanged() -> None:
    body = document().encode()
    from modelome.normalize import content_hash

    client = QueueClient(response(body))
    page = adapter(client).fetch_page({"source_sha256": content_hash(body), "model_count": 9})

    assert page.records == ()
    assert page.upstream_count == 9


@pytest.mark.parametrize(
    "source",
    [
        document().replace("eqV2_31M_omat.pt", "unknown.pt"),
        document().replace("EquiformerV2-31M-OMat", "Unknown-31M-OMat", 1),
        document().replace("eqV2_86M_omat.pt", "eqV2_31M_omat.pt"),
        document().replace(
            f'<a href="{HF_BASE}eqV2_153M_omat.pt">',
            '<a href="https://example.org/eqV2_153M_omat.pt">',
        ),
    ],
)
def test_rejects_unexpected_or_incomplete_checkpoint_map(source: str) -> None:
    client = QueueClient(response(source.encode()))

    with pytest.raises(ValueError):
        adapter(client).fetch_page({})
