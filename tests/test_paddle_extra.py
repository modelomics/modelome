from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier, ModelStatus
from modelome.sources.paddle_extra import (
    PaddleNlpTokenEmbeddingSourceAdapter,
    _parse_model_rows,
)

_HTML = """\
<html><body>
<h3>English Word Vectors</h3><table>
<tr><th>Corpus</th><th>Name</th></tr><tr><td>Wiki</td><td>fasttext.wiki-news.target.word-word.dim300.en</td></tr>
</table>
<h3>Model Information</h3><table>
<tr><th>Model</th><th>File Size</th><th>Vocabulary Size</th></tr>
<tr><td>w2v.baidu_encyclopedia.target.word-word.dim300</td><td>678.21 MB</td><td>635965</td></tr>
<tr><td>glove.twitter.target.word-word.dim50.en</td><td>221.08 MB</td><td>1,193,516</td></tr>
<tr><td>Navigation</td><td>n/a</td><td>n/a</td></tr>
</table>
</body></html>
"""


class _Client:
    def __init__(self, html: str) -> None:
        self.html = html
        self.calls = 0

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls += 1
        return HttpResponse(200, {}, self.html.encode(), url)


def test_parser_reads_only_exact_rows_from_model_information_section() -> None:
    assert _parse_model_rows(_HTML, max_rows=20) == (
        ("glove.twitter.target.word-word.dim50.en", "221.08 MB", "1193516"),
        ("w2v.baidu_encyclopedia.target.word-word.dim300", "678.21 MB", "635965"),
    )


def test_parser_requires_registry_table_and_validates_metadata() -> None:
    with pytest.raises(ValueError, match="Model Information table is missing"):
        _parse_model_rows("<h3>Other</h3><table></table>", max_rows=20)


def test_adapter_emits_documented_models_and_skips_unchanged_document() -> None:
    client = _Client(_HTML)
    adapter = PaddleNlpTokenEmbeddingSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    first = adapter.fetch_page({})

    assert first.complete is True
    assert first.authoritative_snapshot is True
    assert first.upstream_count == 2
    assert first.next_state["model_count"] == 2
    model = first.records[0].models[0]
    assert model.status is ModelStatus.DOCUMENTED
    assert model.identifiers == (
        Identifier("paddlenlp:token-embedding", "glove.twitter.target.word-word.dim50.en"),
    )
    assert first.records[0].raw["vocabulary_size"] == 1_193_516

    second = adapter.fetch_page(first.next_state)
    assert second.records == ()
    assert second.upstream_count == 2
    assert client.calls == 2
