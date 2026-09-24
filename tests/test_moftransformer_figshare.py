from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.moftransformer_figshare import (
    _API_URL,
    MOFTransformerFigshareAdapter,
    _parse_files,
)

FILES = [
    {
        "id": 37511767,
        "name": "best_mtp_moc_vfp.ckpt",
        "size": 1042971511,
        "download_url": "https://ndownloader.figshare.com/files/37511767",
        "computed_md5": "fdd517e48ac3c547d237a17af2387483",
    },
    {
        "id": 37621520,
        "name": "finetuned_bandgap.ckpt",
        "size": 1032780001,
        "download_url": "https://ndownloader.figshare.com/files/37621520",
        "computed_md5": "027921dcf911ae1241daf87e06a391c4",
    },
    {
        "id": 37622693,
        "name": "finetuned_h2_uptake.ckpt",
        "size": 1032779873,
        "download_url": "https://ndownloader.figshare.com/files/37622693",
        "computed_md5": "67ef83c1bc961ebf1fd7ea79cebf8147",
    },
]


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(url: str, body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, url)


def article(files: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": 21155506,
        "title": "MOFTransformer",
        "is_public": True,
        "download_disabled": False,
        "version": 2,
        "files": files if files is not None else FILES,
    }


def test_parses_only_the_three_literal_checkpoint_rows_and_retains_asset_metadata() -> None:
    rows = _parse_files(FILES, 2, "test")

    assert [row["name"] for row in rows] == [
        "best_mtp_moc_vfp.ckpt",
        "finetuned_bandgap.ckpt",
        "finetuned_h2_uptake.ckpt",
    ]
    assert rows[0]["id"] == 37511767
    assert rows[0]["article_version"] == 2
    assert rows[0]["computed_md5"] == "fdd517e48ac3c547d237a17af2387483"


def test_rejects_missing_duplicate_and_wrong_provider_checkpoint_rows() -> None:
    with pytest.raises(ValueError, match="missing checkpoint"):
        _parse_files(FILES[:-1], 2, "test")
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        _parse_files(FILES + [FILES[0]], 2, "test")
    wrong = [dict(row) for row in FILES]
    wrong[0]["download_url"] = "https://example.org/files/37511767"
    with pytest.raises(ValueError, match="invalid Figshare download URL"):
        _parse_files(wrong, 2, "test")


def test_fetches_one_metadata_page_and_records_exact_figshare_files() -> None:
    client = QueueClient(response(_API_URL, json.dumps(article()).encode()))

    page = MOFTransformerFigshareAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert client.calls == [_API_URL]
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.raw["file_id"] == 37511767
    assert record.identifiers == (Identifier("moftransformer:figshare-file", "37511767"),)
    assert record.raw["download_url"] == "https://ndownloader.figshare.com/files/37511767"
    assert record.releases[0].metadata["size_bytes"] == 1042971511
    assert record.releases[0].metadata["binary_reachability_checked"] is False


def test_live_moftransformer_figshare_metadata_smoke() -> None:
    """Fetch the public Figshare article JSON, never the checkpoint files."""
    try:
        page = MOFTransformerFigshareAdapter().fetch_page({})
    except Exception as error:  # pragma: no cover - network-dependent smoke
        pytest.skip(f"live MOFTransformer metadata unavailable: {error}")
    assert page.authoritative_snapshot
    assert page.upstream_count == 3
