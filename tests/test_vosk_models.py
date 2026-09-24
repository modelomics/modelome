from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.vosk_models import VoskModelsSourceAdapter

_INDEX = "https://alphacephei.com/vosk/models"


class _Client:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(self.status, {"content-type": "text/html"}, self.body, url)


def _html(*rows: str) -> bytes:
    return (
        "<html><body><table>"
        + "".join(rows)
        + "</table><table><tr><td>Vosk</td></tr></table></body></html>"
    ).encode()


def _row(name: str, notes: str = "Lightweight model", license_name: str = "Apache 2.0") -> str:
    return (
        f'<tr><td><a href="/vosk/models/{name}.zip">{name}</a></td>'
        f"<td>40M</td><td></td><td>{notes}</td><td>{license_name}</td></tr>"
    )


def test_parses_asr_speaker_and_punctuation_archives_with_versioned_urls() -> None:
    client = _Client(
        _html(
            _row("vosk-model-small-en-us-0.15"),
            _row("vosk-model-spk-0.4", "Speaker identification model"),
            _row("vosk-recasepunc-en-0.22", "Punctuation model"),
        )
    )
    adapter = VoskModelsSourceAdapter(
        client=client,
        min_models=1,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [_INDEX]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    by_title = {record.title: record for record in page.records}
    assert by_title["vosk-model-small-en-us-0.15"].releases[0].metadata["category"] == (
        "automatic_speech_recognition"
    )
    assert by_title["vosk-model-spk-0.4"].releases[0].metadata["category"] == (
        "speaker_identification"
    )
    assert by_title["vosk-recasepunc-en-0.22"].releases[0].metadata["category"] == (
        "punctuation_and_case_restoration"
    )
    assert by_title["vosk-model-small-en-us-0.15"].canonical_url == (
        "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    )
    assert all(record.links[-1].crawl is False for record in page.records)


def test_skips_unlinked_table_headers_and_non_zip_links() -> None:
    html = (
        "<html><body><table><tr><th>English</th></tr>"
        '<tr><td><a href="https://github.com/alphacep/vosk-api">repo</a></td></tr>'
        + _row("vosk-model-small-en-us-0.15")
        + "</table></body></html>"
    ).encode()
    page = VoskModelsSourceAdapter(client=_Client(html), min_models=1).fetch_page({})
    assert page.upstream_count == 1


@pytest.mark.parametrize(
    ("html", "message"),
    [
        (b"<html><body><a href='/vosk/models/vosk-model-en-0.22.zip'>model</a>", "incomplete"),
        (
            _html('<tr><td><a href="https://example.com/model.zip">bad</a></td></tr>'),
            "invalid Vosk model archive URL",
        ),
        (
            _html(
                _row("vosk-model-en-us-0.22"),
                _row("vosk-model-en-us-0.22"),
            ),
            "duplicate Vosk model archive",
        ),
    ],
)
def test_fails_closed_on_incomplete_invalid_or_duplicate_index_rows(
    html: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        VoskModelsSourceAdapter(client=_Client(html), min_models=1).fetch_page({})


def test_enforces_minimum_count_and_response_status() -> None:
    with pytest.raises(ValueError, match="only 1 archives"):
        VoskModelsSourceAdapter(
            client=_Client(_html(_row("vosk-model-en-us-0.22"))), min_models=2
        ).fetch_page({})
    with pytest.raises(ValueError, match="HTTP 503"):
        VoskModelsSourceAdapter(client=_Client(b"", status=503), min_models=1).fetch_page({})
