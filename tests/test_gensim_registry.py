from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.gensim_registry import GensimDownloaderModelRegistrySourceAdapter

REVISION = "f" * 40
BASE = "https://github.com/RaRe-Technologies/gensim-data/releases/download"


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None,
            headers: dict[str, str] | None = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, "https://api.github.com")


def model_row(handle: str, **overrides: Any) -> dict[str, Any]:
    return {
        "num_records": 100,
        "file_size": 2048,
        "base_dataset": "Example training corpus",
        "reader_code": f"{BASE}/{handle}/__init__.py",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "parameters": {"dimension": 50},
        "description": f"Pretrained embedding model {handle}",
        "checksum": "a" * 32,
        "file_name": f"{handle}.gz",
        "parts": 1,
        **overrides,
    }


def test_enumerates_source_native_model_handles_and_exact_downloader_archives() -> None:
    manifest = json.dumps({
        "corpora": {"text8": {"file_name": "text8.gz"}},
        "models": {
            "glove-twitter-25": model_row("glove-twitter-25"),
            "word2vec-google-news-300": model_row("word2vec-google-news-300"),
            "multipart-model": model_row("multipart-model", parts=2),
        },
    }).encode()
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()), response(manifest))
    adapter = GensimDownloaderModelRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    records = {record.source_record_id: record for record in page.records}
    assert set(records) == {
        "model:glove-twitter-25",
        "model:word2vec-google-news-300",
    }
    glove = records["model:glove-twitter-25"]
    assert glove.models[0].identifiers[0].value == "glove-twitter-25"
    assert glove.releases[0].version == "glove-twitter-25"
    assert glove.raw["weight_url"] == f"{BASE}/glove-twitter-25/glove-twitter-25.gz"
    assert glove.raw["md5"] == "a" * 32
    assert len(client.calls) == 2


def test_fails_closed_for_model_rows_without_exact_downloader_contract() -> None:
    manifest = json.dumps({
        "models": {
            "wrong-name": model_row("wrong-name", file_name="other.gz"),
            "bad-reader": model_row(
                "bad-reader", reader_code="https://example.org/model/__init__.py"
            ),
        }
    }).encode()
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()), response(manifest))
    with pytest.raises(ValueError, match="no supported"):
        GensimDownloaderModelRegistrySourceAdapter(client=client).fetch_page({})


def test_rejects_invalid_checksum_and_nonpositive_size_rows() -> None:
    manifest = json.dumps({
        "models": {
            "bad-checksum": model_row("bad-checksum", checksum="not-md5"),
            "no-size": model_row("no-size", file_size=0),
        }
    }).encode()
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()), response(manifest))
    with pytest.raises(ValueError, match="no supported"):
        GensimDownloaderModelRegistrySourceAdapter(client=client).fetch_page({})


def test_rejects_invalid_git_revision() -> None:
    client = QueuedClient(response(json.dumps({"sha": "main"}).encode()))
    with pytest.raises(ValueError, match="invalid commit"):
        GensimDownloaderModelRegistrySourceAdapter(client=client).fetch_page({})


def test_skips_manifest_fetch_when_revision_is_unchanged() -> None:
    client = QueuedClient(response(json.dumps({"sha": REVISION}).encode()))
    page = GensimDownloaderModelRegistrySourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 18}
    )
    assert page.records == ()
    assert page.upstream_count == 18
    assert len(client.calls) == 1
