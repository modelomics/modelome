from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.alphafold_registry import AlphaFoldParameterArchiveSourceAdapter

REVISION = "d" * 40
ARCHIVE = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"


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


def response(value: Any, status: int = 200) -> HttpResponse:
    body = json.dumps(value).encode() if isinstance(value, dict) else value.encode()
    return HttpResponse(status, {}, body, "https://example.test")


def test_reads_exact_official_archive_url_without_fetching_archive() -> None:
    client = QueuedClient(
        response({"sha": REVISION}),
        response(f'SOURCE_URL="{ARCHIVE}"\n'),
    )
    source = AlphaFoldParameterArchiveSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = source.fetch_page({})

    assert page.complete and not page.authoritative_snapshot
    assert page.upstream_count == 1
    archive = page.records[0]
    assert archive.canonical_url == ARCHIVE
    assert archive.releases[0].version == "2022-12-06"
    assert archive.releases[0].metadata["archive_url"] == ARCHIVE
    assert archive.raw["revision"] == REVISION
    assert len(client.calls) == 2
    assert client.calls[-1].endswith("/scripts/download_alphafold_params.sh")


def test_skips_script_when_official_repo_revision_is_unchanged() -> None:
    client = QueuedClient(response({"sha": REVISION}))
    page = AlphaFoldParameterArchiveSourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "archive_count": 1}
    )
    assert page.records == ()
    assert page.upstream_count == 1
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "script",
    [
        'SOURCE_URL="https://example.org/alphafold_params_2022-12-06.tar"',
        'SOURCE_URL="${BASE}/alphafold_params_2022-12-06.tar"',
        'SOURCE_URL="https://storage.googleapis.com/alphafold/other.tar"',
    ],
)
def test_rejects_unverified_or_ambiguous_archive_sources(script: str) -> None:
    client = QueuedClient(response({"sha": REVISION}), response(script))
    with pytest.raises(ValueError):
        AlphaFoldParameterArchiveSourceAdapter(client=client).fetch_page({})
