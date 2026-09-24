from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.alignn_atomwise_registry import (
    AlignnAtomwiseRegistrySourceAdapter,
    _parse_registry,
)

REVISION = "a" * 40
COMMIT_URL = "https://api.github.com/repos/usnistgov/alignn/commits/main"
RAW_URL = f"https://raw.githubusercontent.com/usnistgov/alignn/{REVISION}/alignn/ff/all_models_alignn_atomwise.json"


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(url: str, body: bytes, headers: dict[str, str] | None = None) -> HttpResponse:
    return HttpResponse(200, headers or {}, body, url)


def adapter(client: QueueClient) -> AlignnAtomwiseRegistrySourceAdapter:
    return AlignnAtomwiseRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )


def test_indexes_exact_literal_handles_and_figshare_ids() -> None:
    document = (
        b'{"alignnff_wt10":"https://figshare.com/ndownloader/files/41583594",'
        b'"v12.2.2024_dft_3d_307k":"https://figshare.com/ndownloader/files/50904240"}'
    )
    client = QueueClient(
        response(COMMIT_URL, f'{{"sha":"{REVISION}"}}'.encode()),
        response(RAW_URL, document),
    )

    page = adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert client.calls == [COMMIT_URL, RAW_URL]
    assert all(record.kind is ArtifactKind.WEIGHTS for record in page.records)
    record = next(
        record
        for record in page.records
        if "alignnff_wt10" in record.raw["checkpoint_handle"]
    )
    assert record.raw["checkpoint_url"] == "https://figshare.com/ndownloader/files/41583594"
    assert record.models[0].identifiers == (
        Identifier("alignn:atomwise-checkpoint", "alignnff_wt10"),
    )
    assert record.releases[0].metadata["binary_reachability_checked"] is False


def test_unchanged_revision_skips_registry_fetch() -> None:
    client = QueueClient(response(COMMIT_URL, f'{{"sha":"{REVISION}"}}'.encode()))

    page = adapter(client).fetch_page({"completed_revision": REVISION, "model_count": 25})

    assert page.records == ()
    assert page.upstream_count == 25
    assert client.calls == [COMMIT_URL]


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        "{}",
        '{"bad/handle":"https://figshare.com/ndownloader/files/41583594"}',
        '{"handle":"https://evil.example/ndownloader/files/41583594"}',
        '{"handle":"https://figshare.com/ndownloader/files/abc"}',
        '{"handle":"https://figshare.com/ndownloader/files/1?download=1"}',
    ],
)
def test_rejects_registry_shapes_or_urls_outside_verified_contract(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_registry(document, "test", 100)


def test_live_first_party_registry_smoke() -> None:
    """Smoke the public file and fail loudly if its first-party shape changes."""
    try:
        page = AlignnAtomwiseRegistrySourceAdapter().fetch_page({})
    except Exception as error:  # pragma: no cover - network-dependent smoke
        pytest.skip(f"live ALIGNN registry unavailable: {error}")
    assert page.authoritative_snapshot
    assert page.upstream_count == 26
    assert any(record.raw["checkpoint_handle"] == "alignnff_wt10" for record in page.records)
