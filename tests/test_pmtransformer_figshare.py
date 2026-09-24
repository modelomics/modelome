from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.pmtransformer_figshare import (
    _API_URL,
    PMTransformerFigshareAdapter,
    _parse_file,
)

FILES = [
    {
        "id": 40298269,
        "name": "moftransformer.ckpt",
        "size": 1045762691,
        "download_url": "https://ndownloader.figshare.com/files/40298269",
        "computed_md5": "56d21835c95dc3c17fd2a97184a1465a",
    },
    {
        "id": 40298992,
        "name": "pmtransformer.ckpt",
        "size": 1045762627,
        "download_url": "https://ndownloader.figshare.com/files/40298992",
        "computed_md5": "dfea8141311cb39eb52801f4791d0fe0",
    },
    {
        "id": 52910984,
        "name": "jsons.zip",
        "size": 354168030,
        "download_url": "https://ndownloader.figshare.com/files/52910984",
        "computed_md5": "c47fa991e324365a04c024c6f1fbefb6",
    },
]


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, _API_URL)


def article(files: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": 22698655,
        "title": "PMTransformer pre-trained model",
        "is_public": True,
        "download_disabled": False,
        "version": 3,
        "files": files if files is not None else FILES,
    }


def test_selects_exact_pmtransformer_file_metadata() -> None:
    row = _parse_file(FILES, 3, "test")

    assert row["id"] == 40298992
    assert row["article_version"] == 3
    assert row["size"] == 1045762627
    assert row["computed_md5"] == "dfea8141311cb39eb52801f4791d0fe0"


def test_rejects_missing_duplicate_and_non_figshare_file_url() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _parse_file(FILES[:1], 3, "test")
    with pytest.raises(ValueError, match="exactly one"):
        _parse_file(FILES + [FILES[1]], 3, "test")
    wrong = [dict(row) for row in FILES]
    wrong[1]["download_url"] = "https://example.org/files/40298992"
    with pytest.raises(ValueError, match="invalid Figshare download URL"):
        _parse_file(wrong, 3, "test")


def test_fetches_metadata_only_and_records_exact_checkpoint_identity() -> None:
    client = QueueClient(response(json.dumps(article()).encode()))

    page = PMTransformerFigshareAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    assert client.calls == [_API_URL]
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.identifiers == (Identifier("pmtransformer:figshare-file", "40298992"),)
    assert record.raw["filename"] == "pmtransformer.ckpt"
    assert record.raw["download_url"] == "https://ndownloader.figshare.com/files/40298992"
    assert record.releases[0].metadata["size_bytes"] == 1045762627
    assert record.releases[0].metadata["md5"] == "dfea8141311cb39eb52801f4791d0fe0"
    assert record.releases[0].metadata["binary_reachability_checked"] is False


def test_live_pmtransformer_figshare_metadata_smoke() -> None:
    """Fetch the public Figshare article JSON, never checkpoint file bytes."""
    try:
        page = PMTransformerFigshareAdapter().fetch_page({})
    except Exception as error:  # pragma: no cover - network-dependent smoke
        pytest.skip(f"live PMTransformer metadata unavailable: {error}")
    assert page.authoritative_snapshot
    assert page.upstream_count == 1
