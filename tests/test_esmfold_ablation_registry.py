from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.esmfold_ablation_registry import (
    ESMFoldAblationRegistryAdapter,
    _parse_names,
)

REVISION = "d" * 40
NAMES = (
    "esmfold_structure_module_only_8M",
    "esmfold_structure_module_only_8M_270K",
    "esmfold_structure_module_only_35M",
    "esmfold_structure_module_only_35M_270K",
    "esmfold_structure_module_only_150M",
    "esmfold_structure_module_only_150M_270K",
    "esmfold_structure_module_only_650M",
    "esmfold_structure_module_only_650M_270K",
    "esmfold_structure_module_only_3B",
    "esmfold_structure_module_only_3B_270K",
    "esmfold_structure_module_only_15B",
)


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


def test_parses_literal_first_party_loader_inventory_and_direct_weights() -> None:
    source = "\n".join(
        f"def {name}():\n    return _load_model({name!r})" for name in NAMES
    ) + "\ndef esmfold_v1():\n    return _load_model('esmfold_3B_v1')"
    client = QueuedClient(response({"sha": REVISION}), response(source))
    adapter = ESMFoldAblationRegistryAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == len(NAMES)
    assert tuple(record.raw["checkpoint_name"] for record in page.records) == tuple(sorted(NAMES))
    record = next(r for r in page.records if r.raw["checkpoint_name"] == NAMES[0])
    assert record.canonical_url.endswith(f"/{NAMES[0]}.pt")
    assert record.links[0].url == record.canonical_url
    assert record.models[0].identifiers[0].value == NAMES[0]
    assert record.releases[0].revision == REVISION
    assert len(client.calls) == 2


def test_noop_when_loader_revision_is_unchanged() -> None:
    client = QueuedClient(response({"sha": REVISION}))
    page = ESMFoldAblationRegistryAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 11}
    )
    assert page.records == ()
    assert page.upstream_count == 11
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "source",
    [
        "def esmfold_structure_module_only_8M(): return _load_model(NAME)",
        "def esmfold_structure_module_only_8M(): return other('esmfold_structure_module_only_8M')",
        "def esmfold_structure_module_only_8M(): return _load_model("
        "'esmfold_structure_module_only_35M')",
        "def esmfold_structure_module_only_999M(): return _load_model("
        "'esmfold_structure_module_only_999M')",
    ],
)
def test_rejects_nonliteral_or_ambiguous_loaders(source: str) -> None:
    with pytest.raises(ValueError):
        _parse_names(source, max_entries=32)
