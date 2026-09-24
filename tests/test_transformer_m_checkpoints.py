from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.transformer_m_checkpoints import TransformerMCheckpointSourceAdapter

_REVISION = "d" * 40
_README = "\n".join(
    (
        "## Checkpoints",
        "| Model | File Size | Update Date | Valid MAE | Download Link |",
        "| ----- | --------- | ----------- | --------- | ------------- |",
        "| L12 | 189MB | Oct 04, 2022 | 0.0785 | "
        "https://1drv.ms/u/s!AgZyC7AzHtDBdWUZttg6N2TsOxw?e=sUOhox |",
        "| L18 | 270MB | Oct 04, 2022 | 0.0772 | "
        "https://1drv.ms/u/s!AgZyC7AzHtDBdrY59-_mP38jsCg?e=URoyUK |",
        "| L12_old | 189MB | Mar 31, 2023 | 0.0787 | "
        "https://1drv.ms/u/s!AgZyC7AzHtDBesDk9tZK1yvbtzE?e=5H91Zq |",
        "```shell",
        "# download the above model weights (L12.pt, L18.pt) to ./",
        "```",
        "## Datasets",
        "Other section.",
    )
)


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
    return HttpResponse(200, {}, body, "https://fixtures.test/README.md")


def _adapter(client: _QueuedClient) -> TransformerMCheckpointSourceAdapter:
    return TransformerMCheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )


def test_transformer_m_checkpoint_table_emits_exact_model_rows() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_README))

    page = _adapter(client).fetch_page({})

    assert page.authoritative_snapshot is True
    assert page.upstream_count == 3
    expected = (
        ("L12", "https://1drv.ms/u/s!AgZyC7AzHtDBdWUZttg6N2TsOxw?e=sUOhox"),
        ("L18", "https://1drv.ms/u/s!AgZyC7AzHtDBdrY59-_mP38jsCg?e=URoyUK"),
        ("L12_old", "https://1drv.ms/u/s!AgZyC7AzHtDBesDk9tZK1yvbtzE?e=5H91Zq"),
    )
    for record, (handle, url) in zip(page.records, expected, strict=True):
        assert record.title == handle
        assert record.models[0].identifiers == (Identifier("transformer-m:checkpoint", handle),)
        assert record.releases[0].metadata["weight_url"] == url


@pytest.mark.parametrize(
    "document",
    [
        _README.replace("## Checkpoints", "## Missing"),
        _README.replace("L12_old", "L12_revision"),
        _README.replace("https://1drv.ms/", "https://example.test/"),
    ],
)
def test_transformer_m_rejects_missing_or_untrusted_checkpoint_rows(document: str) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(document))

    with pytest.raises(ValueError):
        _adapter(client).fetch_page({})
