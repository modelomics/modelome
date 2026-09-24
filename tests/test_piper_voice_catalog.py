from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.piper_voice_catalog import PiperVoiceCatalogSourceAdapter

_REVISION = "a" * 40
_COMMIT_URL = "https://api.github.com/repos/rhasspy/piper/commits/master"
_RAW_URL = f"https://raw.githubusercontent.com/rhasspy/piper/{_REVISION}/VOICES.md"
_MODEL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true"
_CONFIG = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true.json"
_LOW_MODEL = _MODEL.replace("/medium/", "/low/").replace("-medium.", "-low.")
_LOW_CONFIG = _CONFIG.replace("/medium/", "/low/").replace("-medium.", "-low.")
_DOCUMENT = f"""# Voices
  * English (en_US)
    * lessac
        * low - [[model]({_LOW_MODEL})] [[config]({_LOW_CONFIG})]
        * medium - [[model]({_MODEL})] [[config]({_CONFIG})]
"""


class _Client:
    def __init__(self, document: str = _DOCUMENT, status: int = 200) -> None:
        self.document = document.encode()
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        if url == _COMMIT_URL:
            body = json.dumps({"sha": _REVISION}).encode()
            return HttpResponse(200, {}, body, url)
        return HttpResponse(self.status, {}, self.document, url)


def test_indexes_exact_language_voice_quality_rows_with_pinned_asset_pairs() -> None:
    client = _Client()
    adapter = PiperVoiceCatalogSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [_COMMIT_URL, _RAW_URL]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    low, medium = page.records
    assert low.models[0].name == "Piper en_US lessac (low)"
    assert medium.releases[0].metadata["language"] == "en_US"
    assert medium.releases[0].metadata["quality"] == "medium"
    assert medium.links[-2].url == _MODEL
    assert medium.links[-1].url == _CONFIG
    assert all(link.crawl is False for link in medium.links)


def test_skips_unchanged_revision_without_fetching_voice_document() -> None:
    client = _Client()
    page = PiperVoiceCatalogSourceAdapter(client=client).fetch_page(
        {"completed_revision": _REVISION, "model_count": 2}
    )

    assert client.calls == [_COMMIT_URL]
    assert page.records == ()
    assert page.upstream_count == 2


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            _DOCUMENT.replace("?download=true", "?download=false", 1),
            "invalid pinned Piper asset URL",
        ),
        (
            _DOCUMENT.replace("huggingface.co/rhasspy", "example.com/rhasspy", 1),
            "invalid pinned Piper asset URL",
        ),
        (_DOCUMENT.replace("v1.0.0", "main", 1), "invalid pinned Piper asset URL"),
        (_DOCUMENT + _DOCUMENT, "duplicate language/voice/quality identity"),
    ],
)
def test_fails_closed_on_nonpinned_or_duplicate_voice_rows(document: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PiperVoiceCatalogSourceAdapter(client=_Client(document)).fetch_page({})


def test_rejects_http_error_and_missing_rows() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        PiperVoiceCatalogSourceAdapter(client=_Client(status=503)).fetch_page({})
    with pytest.raises(ValueError, match="no exact model/config voice rows"):
        PiperVoiceCatalogSourceAdapter(client=_Client("# Voices\n")).fetch_page({})
