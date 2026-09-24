from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier, ModelStatus
from modelome.sources.jax_registry import JaxRegistrySourceAdapter

_SHA = "a" * 40


class _Client:
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
        return self.responses.pop(0)


def _response(body: bytes | Mapping[str, Any]) -> HttpResponse:
    if isinstance(body, Mapping):
        body = json.dumps(body).encode()
    return HttpResponse(200, {}, body, "https://fixture.test")


def test_t5x_registry_reads_explicit_rows_and_never_fetches_checkpoint() -> None:
    document = (
        b"# Models\nModel | Gin File Location | Checkpoint Location\n"
        b"--- | --- | ---\nT5 Small | [small.gin]("
        b"https://github.com/google-research/t5x/blob/main/small.gin) | "
        b"[`gs://t5-data/pretrained_models/t5x/t5_small/checkpoint_1000000`]("
        b"https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/"
        b"t5x/t5_small/checkpoint_1000000) |\n"
        b"T5 1.1 LM-100K Small | [t5_1_1_small.gin]("
        b"https://github.com/google-research/t5x/blob/main/t5_1_1_small.gin) | "
        b"[t5_1_1_lm100k_small/checkpoint_1100000]("
        b"https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/"
        b"t5x/t5_1_1_lm100k_small)\n"
    )
    client = _Client(_response({"sha": _SHA}), _response(document))
    page = JaxRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    ).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    record = page.records[0]
    assert record.models[0].name == "T5 Small"
    assert record.models[0].status is ModelStatus.RELEASED
    assert record.identifiers == (Identifier("t5x:model", "T5 Small"),)
    assert record.raw["gin_config"] == "small.gin"
    assert record.raw["checkpoint_location"].startswith("gs://")
    assert record.releases[0].metadata["checkpoint_location"] == record.raw["checkpoint_location"]
    lm_adapted = page.records[1]
    assert lm_adapted.models[0].name == "T5 1.1 LM-100K Small"
    assert lm_adapted.raw["checkpoint_location"] == (
        "gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_small/checkpoint_1100000"
    )
    assert len(client.calls) == 2
    assert not any("storage.googleapis.com" in url for url in client.calls)


def test_t5x_registry_checkpointed_by_revision() -> None:
    client = _Client(_response({"sha": _SHA}), _response(b"# Models\n"))
    adapter = JaxRegistrySourceAdapter(client=client)
    # A successful first fetch needs an admitted row.
    client.responses[1] = _response(
        b"Model | Gin | Checkpoint\n--- | --- | ---\nT5 Base | "
        b"[base.gin](https://github.com/google-research/t5x/blob/main/base.gin) | "
        b"[`gs://t5-data/pretrained_models/t5x/base/checkpoint_1`]("
        b"https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/"
        b"t5x/base/checkpoint_1) |\n"
    )
    first = adapter.fetch_page({})
    adapter.client = _Client(_response({"sha": _SHA}))
    second = adapter.fetch_page(first.next_state)
    assert second.records == ()
    assert second.upstream_count == 1
