from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.line_checkpoint_card_catalog import (
    LineCheckpointCardCatalogSourceAdapter,
)

_REVISION = "e" * 40


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: str | Mapping[str, Any]) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/cards.yaml")


_CARDS = """\
name: tokenizer
tokenizer: https://download.example.test/tokenizer.model
name: omniASR_CTC_300M_v2
model_family: wav2vec2_asr
model_arch: 300m_v2
checkpoint: https://download.example.test/omniASR_CTC_300M_v2.pt
name: omniASR_LLM_1B_v2
model_family: llm_asr
model_arch: 1b_v2
checkpoint: https://download.example.test/omniASR_LLM_1B_v2.pth
"""


def _adapter(client: _QueuedClient) -> LineCheckpointCardCatalogSourceAdapter:
    return LineCheckpointCardCatalogSourceAdapter(
        name="fixture-line-cards",
        repository="example-org/cards",
        branch="main",
        source_path="cards/models.yaml",
        provider_namespace="fixture:card-model",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_line_card_catalog_reads_only_named_direct_checkpoint_cards() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_CARDS))

    page = _adapter(client).fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert client.calls[1][0].endswith(f"/{_REVISION}/cards/models.yaml")
    ctc, llm = page.records
    assert ctc.kind is ArtifactKind.MODEL_CARD
    assert ctc.models[0].identifiers == (
        Identifier("fixture:card-model", "omniASR_CTC_300M_v2"),
    )
    assert ctc.raw["model_family"] == "wav2vec2_asr"
    assert ctc.releases[0].metadata["model_family"] == "wav2vec2_asr"
    assert ctc.releases[0].metadata["model_arch"] == "300m_v2"
    assert llm.title == "omniASR_LLM_1B_v2"
    assert {
        (link.url, link.relation, link.crawl, link.model_local_ids)
        for link in ctc.links
    } == {
        (
            f"https://github.com/example-org/cards/blob/{_REVISION}/cards/models.yaml",
            "model_card",
            False,
            ("model:omniASR_CTC_300M_v2",),
        ),
        (
            "https://github.com/example-org/cards",
            "source_implementation",
            False,
            ("model:omniASR_CTC_300M_v2",),
        ),
        (
            "https://download.example.test/omniASR_CTC_300M_v2.pt",
            "weights",
            False,
            ("model:omniASR_CTC_300M_v2",),
        ),
    }


def test_line_card_catalog_skips_an_unchanged_source_file() -> None:
    adapter = _adapter(_QueuedClient(_response({"sha": _REVISION}), _response(_CARDS)))
    first = adapter.fetch_page({})
    adapter.client = _QueuedClient(_response({"sha": _REVISION}))

    second = adapter.fetch_page(first.next_state)

    assert second.records == ()
    assert second.upstream_count == 2


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ("checkpoint: https://example.test/a.pt", "precedes a top-level name"),
        (
            "name: model\ncheckpoint: https://example.test/paper.pdf",
            "recognised checkpoint",
        ),
        (
            "name: model\ncheckpoint: https://example.test/a.pt\ncheckpoint: https://example.test/b.pt",
            "duplicate checkpoint",
        ),
        (
            "name: model\ncheckpoint: https://example.test/a.pt\nname: model\ncheckpoint: https://example.test/b.pt",
            "duplicate checkpoint handle",
        ),
        (
            "name: ../escape\ncheckpoint: https://example.test/a.pt",
            "invalid checkpoint handle",
        ),
        ("name: tokenizer", "contains no named direct checkpoints"),
    ],
)
def test_line_card_catalog_rejects_ambiguous_or_non_checkpoint_cards(
    document: str,
    match: str,
) -> None:
    adapter = _adapter(_QueuedClient(_response({"sha": _REVISION}), _response(document)))

    with pytest.raises(ValueError, match=match):
        adapter.fetch_page({})


def test_line_card_catalog_bounds_the_final_card() -> None:
    adapter = LineCheckpointCardCatalogSourceAdapter(
        name="fixture-line-cards",
        repository="example-org/cards",
        branch="main",
        source_path="cards/models.yaml",
        provider_namespace="fixture:card-model",
        max_entries=1,
        client=_QueuedClient(_response({"sha": _REVISION}), _response(_CARDS)),
    )

    with pytest.raises(ValueError, match="exceeds 1 entries"):
        adapter.fetch_page({})
