from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.esm1v_variants import ESM1vVariantRegistryAdapter, _parse_names

REVISION = "e" * 40
BASE = "esm1v_t33_650M_UR90S_"
NAMES = tuple(f"{BASE}{index}" for index in range(2, 6))


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Any = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(value: Any) -> HttpResponse:
    body = json.dumps(value).encode() if isinstance(value, dict) else value.encode()
    return HttpResponse(200, {}, body, "https://example.test")


def test_emits_only_exact_missing_esm1v_variants_and_direct_urls() -> None:
    source = "\n".join(
        f"def {name}():\n    return load_model_and_alphabet_hub({name!r})"
        for name in (f"{BASE}{i}" for i in range(1, 6))
    )
    client = QueuedClient(response({"sha": REVISION}), response(source))
    adapter = ESM1vVariantRegistryAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    assert tuple(record.raw["checkpoint_name"] for record in page.records) == NAMES
    assert all(
        record.canonical_url.endswith(f"/{name}.pt")
        for record, name in zip(page.records, NAMES, strict=True)
    )
    assert [record.raw["ensemble_member"] for record in page.records] == [2, 3, 4, 5]
    assert page.records[0].models[0].identifiers[0].value == NAMES[0]
    assert len(client.calls) == 2


def test_noop_when_source_revision_is_unchanged() -> None:
    client = QueuedClient(response({"sha": REVISION}))
    page = ESM1vVariantRegistryAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 4}
    )
    assert page.records == ()
    assert page.upstream_count == 4
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "source",
    [
        "def esm1v_t33_650M_UR90S_2(): return load_model_and_alphabet_hub(NAME)",
        "def esm1v_t33_650M_UR90S_2(): return other('esm1v_t33_650M_UR90S_2')",
        "def esm1v_t33_650M_UR90S_2(): return load_model_and_alphabet_hub("
        "'esm1v_t33_650M_UR90S_3')",
    ],
)
def test_rejects_nonliteral_or_disagreeing_loaders(source: str) -> None:
    with pytest.raises(ValueError):
        _parse_names(source, max_entries=8)
