from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.kaldi_model_index import KaldiModelIndexSourceAdapter


class _QueuedClient:
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
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(body: str) -> HttpResponse:
    return HttpResponse(200, {}, body.encode(), "https://fixtures.test/kaldi")


def test_kaldi_index_collects_archive_variants_from_official_detail_pages() -> None:
    index = """<table>
      <tr><th>Resource</th><th>Name</th><th>Category</th><th>Summary</th></tr>
      <tr><td>M1</td><td><a href="/models/m1">ASpIRE Chain Model</a></td>
      <td>ASR</td><td>Fisher English</td></tr>
      <tr><td>M14</td><td><a href="/models/m14">GigaSpeech ASR model</a></td>
      <td>ASR</td><td>GigaSpeech</td></tr>
    </table>"""
    m1 = """<h1>ASpIRE Chain Model</h1>
      <h2>ASpIRE Chain Model</h2>
      <a href="/models/1/0001_aspire_chain_model.tar.gz">Download 452M</a>
      <h2>ASpIRE Chain Model with already compiled HCLG</h2>
      <a href="https://kaldi-asr.org/models/1/0001_aspire_chain_model_with_hclg.tar.bz2">
      Download 793M</a>"""
    m14 = """<h1>GigaSpeech ASR model</h1>
      <h2>GigaSpeech ASR S</h2><a href="/models/14/0014_gigaspeech_v1_S.tar.xz">Download 370M</a>
      <h2>GigaSpeech ASR M</h2><a href="/models/14/0014_gigaspeech_v1_M.tar.xz">Download 546M</a>
      <h2>GigaSpeech ASR L</h2><a href="/models/14/0014_gigaspeech_v1_L.tar.xz">Download 594M</a>
      <h2>GigaSpeech ASR XL</h2>
      <a href="/models/14/0014_gigaspeech_v1_XL.tar.xz">Download 1.1G</a>"""
    client = _QueuedClient(_response(index), _response(m1), _response(m14))
    adapter = KaldiModelIndexSourceAdapter(
        client=client, clock=lambda: datetime(2026, 1, 2, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == page.next_state["archive_count"] == 6
    assert page.next_state["resource_count"] == 2
    assert [r.releases[0].version for r in page.records[:2]] == [
        "0001_aspire_chain_model.tar.gz",
        "0001_aspire_chain_model_with_hclg.tar.bz2",
    ]
    assert page.records[0].models[0].name == "ASpIRE Chain Model"
    assert page.records[1].releases[0].metadata["resource_id"] == "M1"
    assert page.records[1].links[-1].url == (
        "https://kaldi-asr.org/models/1/0001_aspire_chain_model_with_hclg.tar.bz2"
    )
    assert client.calls == [
        "https://www.kaldi-asr.org/models.html",
        "https://www.kaldi-asr.org/models/m1",
        "https://www.kaldi-asr.org/models/m14",
    ]


def test_kaldi_adapter_rejects_downloads_outside_resource_archive_path() -> None:
    index = """<table><tr><td>M1</td><td><a href="/models/m1">ASpIRE</a></td>
      <td>ASR</td><td>Fisher</td></tr></table>"""
    detail = '<h1>ASpIRE</h1><h2>Model</h2><a href="https://example.org/model.tar.gz">Download</a>'
    adapter = KaldiModelIndexSourceAdapter(
        client=_QueuedClient(_response(index), _response(detail))
    )

    with pytest.raises(ValueError, match="has no archive links"):
        adapter.fetch_page({})


def test_kaldi_adapter_is_bound_to_official_index() -> None:
    with pytest.raises(ValueError, match="official Kaldi"):
        KaldiModelIndexSourceAdapter(index_url="https://example.org/models.html")
