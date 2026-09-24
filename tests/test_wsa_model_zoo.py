from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.wsa_model_zoo import WSAModelZooSourceAdapter, _parse_model_zoo

_SHA = "b" * 40
_ROWS = (
    ("WSA-Base", "Pretrained policy", "Base pretrained model"),
    ("WSA-Base-RoboTwin", "RoboTwin finetuned model", "Fine-tuned from WSA-Base"),
    ("WSA-Base-LIBERO", "LIBERO finetuned model", "Fine-tuned from WSA-Base"),
    ("WSA-Large", "Pretrained policy", "Large pretrained model"),
    ("WSA-Large-RoboTwin", "RoboTwin finetuned model", "Fine-tuned from WSA-Large"),
    ("WSA-Large-LIBERO", "LIBERO finetuned model", "Fine-tuned from WSA-Large"),
)
_README = "\n".join(
    (
        "# WSA",
        "## Model Zoo",
        "<table><thead><tr><th>Name</th><th>Type</th><th>Usage</th></tr></thead><tbody>",
        *[
            f'<tr><td><a href="https://huggingface.co/zaleni/{slug}">{slug}</a></td>'
            f"<td>{kind}</td><td>{usage}</td></tr>"
            for slug, kind, usage in _ROWS
        ],
        "</tbody></table>",
        "## Choosing a Model",
        "[unlisted](https://huggingface.co/zaleni/WSA-Other)",
    )
)


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        del params, headers
        self.calls.append(url)
        body = (
            json.dumps({"sha": _SHA}).encode()
            if url.endswith("/commits/main")
            else _README.encode()
        )
        return HttpResponse(200, {}, body, url)


def test_parser_extracts_only_the_six_exact_model_zoo_links() -> None:
    entries = _parse_model_zoo(_README, source="test", maximum=8)

    assert [(entry[0], entry[1], entry[2]) for entry in entries] == list(_ROWS)


def test_adapter_emits_exact_models_and_finetune_lineage() -> None:
    client = _Client()
    adapter = WSAModelZooSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/zaleni/WSA/commits/main",
        f"https://raw.githubusercontent.com/zaleni/WSA/{_SHA}/README.md",
    ]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 6
    libero = next(item for item in page.records if item.title == "WSA-Base-LIBERO checkpoint")
    assert libero.kind is ArtifactKind.WEIGHTS
    assert libero.identifiers == (Identifier("huggingface:model", "zaleni/WSA-Base-LIBERO"),)
    assert libero.releases[0].metadata["checkpoint_type"] == "LIBERO finetuned model"
    assert libero.model_relations[0].predicate == "fine_tuned_from"
    assert libero.model_relations[0].target.identifiers == (
        Identifier("huggingface:model", "zaleni/WSA-Base"),
    )


@pytest.mark.parametrize(
    "document,maximum",
    [
        (_README, 5),
        (
            '## Model Zoo\n<table><tr><td><a href="https://huggingface.co/other/WSA-Other">'
            "WSA-Other</a></td><td>type</td><td>usage</td></tr></table>",
            8,
        ),
    ],
)
def test_parser_enforces_inventory_bound_and_first_party_owner(document: str, maximum: int) -> None:
    if maximum == 5:
        with pytest.raises(ValueError, match="exceeds 5"):
            _parse_model_zoo(document, source="test", maximum=maximum)
    else:
        assert _parse_model_zoo(document, source="test", maximum=maximum) == ()
