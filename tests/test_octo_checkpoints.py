from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.octo_checkpoints import (
    OctoCheckpointSourceAdapter,
    _parse_checkpoints,
)

_SHA = "e" * 40
_README = "\n".join(
    (
        "# Octo",
        "## Get Started",
        'OctoModel.load_pretrained("hf://rail-berkeley/octo-base-1.5")',
        "[not in inventory](https://huggingface.co/rail-berkeley/octo-other)",
        "## Checkpoints",
        "Model | Inference | Size",
        "--- | --- | ---",
        "[Octo-Base](https://huggingface.co/rail-berkeley/octo-base) | "
        "13 it/sec | 93M Params",
        "[Octo-Small](https://huggingface.co/rail-berkeley/octo-small) | "
        "17 it/sec | 27M Params",
        "python scripts/finetune.py --config.pretrained_path=hf://rail-berkeley/octo-small-1.5",
        "## Examples",
        "[unlisted](https://huggingface.co/rail-berkeley/octo-other)",
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
        self.calls.append(url)
        body = (
            json.dumps({"sha": _SHA}).encode()
            if url.endswith("/commits/main")
            else _README.encode()
        )
        return HttpResponse(200, {}, body, url)


def test_parser_selects_only_the_two_first_party_checkpoint_rows() -> None:
    entries = _parse_checkpoints(_README, source="test", maximum=5)

    assert {entry[0] for entry in entries} == {
        "Octo-Base",
        "Octo-Small",
        "Octo-Base 1.5",
        "Octo-Small 1.5",
    }
    assert {entry[1] for entry in entries} == {"93M Params", "27M Params"}
    assert {entry[2] for entry in entries} == {
        "https://huggingface.co/rail-berkeley/octo-base",
        "https://huggingface.co/rail-berkeley/octo-small",
        "https://huggingface.co/rail-berkeley/octo-base-1.5",
        "https://huggingface.co/rail-berkeley/octo-small-1.5",
    }


def test_adapter_emits_exact_checkpoint_repo_refs() -> None:
    client = _Client()
    adapter = OctoCheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/octo-models/octo/commits/main",
        f"https://raw.githubusercontent.com/octo-models/octo/{_SHA}/README.md",
    ]
    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 4
    record = next(record for record in page.records if record.title == "Octo Octo-Base")
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == "https://huggingface.co/rail-berkeley/octo-base"
    assert record.identifiers == (
        Identifier("huggingface:model", "rail-berkeley/octo-base"),
    )
    assert record.releases[0].metadata["parameter_count"] == "93M Params"
    assert record.releases[0].metadata["revision"] == _SHA


@pytest.mark.parametrize(
    "document,maximum",
    [
        (
            "## Checkpoints\n| A | B | C |\n| --- | --- | --- |\n"
            "[wrong](https://huggingface.co/other/octo-base) | x | y",
            10,
        ),
        (_README, 1),
        (
            "## Checkpoints\n| A | B | C |\n| --- | --- | --- |\n"
            "[wrong](https://huggingface.co/rail-berkeley/octo-medium) | x | y",
            10,
        ),
    ],
)
def test_parser_rejects_unknown_repositories_and_oversized_tables(
    document: str,
    maximum: int,
) -> None:
    if document == _README and maximum == 1:
        with pytest.raises(ValueError):
            _parse_checkpoints(document, source="test", maximum=maximum)
    else:
        assert _parse_checkpoints(document, source="test", maximum=maximum) == ()
