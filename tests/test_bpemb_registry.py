from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.pipeline import SyncEngine
from modelome.sources.bpemb_registry import BPEmbPretrainedVectorRegistrySourceAdapter
from modelome.storage import Database

_REVISION = "c" * 40


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if "/commits/" in url:
            body = f'{{"sha":"{_REVISION}"}}'.encode()
        elif url.endswith("/README.md"):
            body = (
                b"[English](http://nlp.h-its.org/bpemb/en) "
                b"[Japanese](http://nlp.h-its.org/bpemb/ja)"
            )
        elif url == "https://nlp.h-its.org/bpemb/en/":
            body = b"""<table>
              <tr><td>10000</td><td><a href="en.wiki.bpe.vs10000.model">model</a></td>
              <td><a href="en.wiki.bpe.vs10000.d50.w2v.bin.tar.gz">bin</a></td>
              <td><a href="en.wiki.bpe.vs10000.d50.w2v.txt.tar.gz">txt</a></td></tr>
              <tr><td>10000</td>
              <td><a href="en.wiki.bpe.vs10000.d100.w2v.bin.tar.gz">bin</a></td></tr>
              <a href="https://example.test/en.wiki.bpe.vs10000.d300.w2v.bin.tar.gz">bad</a>
            </table>"""
        elif url == "https://nlp.h-its.org/bpemb/ja/":
            body = b"""<a href="ja.wiki.bpe.vs5000.d100.w2v.bin.tar.gz">bin</a>
              <a href="ja.wiki.bpe.vs5000.d100.w2v.txt.tar.gz">txt</a>"""
        elif url == "https://bpemb.h-its.org/multi/multi/":
            body = b"""<a href="multi.wiki.bpe.vs100000.d300.w2v.bin.tar.gz">bin</a>"""
        else:
            raise AssertionError(f"unexpected URL {url}")
        return HttpResponse(200, {"content-type": "text/html"}, body, url)


def test_bpemb_registry_groups_exact_public_vector_formats_and_paginates() -> None:
    config = tomllib.loads(
        (Path(__file__).parents[1] / "config/proposals/bpemb_pretrained_vectors.toml").read_text()
    )["source"][0]
    assert config["enabled"] is False
    client = _FakeClient()
    adapter = BPEmbPretrainedVectorRegistrySourceAdapter(
        name=config["name"],
        repository=config["repository"],
        branch=config["branch"],
        provider_namespace=config["provider_namespace"],
        languages_per_page=2,
        client=client,
    )

    page = adapter.fetch_page({})
    assert not page.complete
    final_page = adapter.fetch_page(page.next_state)

    assert final_page.complete and not final_page.authoritative_snapshot
    assert final_page.upstream_count == 4
    entries = {record.source_record_id: record for record in (*page.records, *final_page.records)}
    assert set(entries) == {
        "model:en:vs10000:d50",
        "model:en:vs10000:d100",
        "model:ja:vs5000:d100",
        "model:multi:vs100000:d300",
    }
    english = entries["model:en:vs10000:d50"]
    assert english.canonical_url.endswith("/en.wiki.bpe.vs10000.d50.w2v.bin.tar.gz")
    assert set(english.raw["weight_urls"]) == {
        "https://nlp.h-its.org/bpemb/en/en.wiki.bpe.vs10000.d50.w2v.bin.tar.gz",
        "https://nlp.h-its.org/bpemb/en/en.wiki.bpe.vs10000.d50.w2v.txt.tar.gz",
    }
    assert english.models[0].identifiers[0].value == "en:vs10000:d50"
    assert english.releases[0].metadata["dimension"] == 50
    assert not any(".model" in url or ".bin.tar.gz" in url for url in client.calls)


def test_bpemb_registry_rejects_malformed_terminal_cursors() -> None:
    adapter = BPEmbPretrainedVectorRegistrySourceAdapter(
        client=_FakeClient(),
    )
    base_state = {
        "catalog_revision": _REVISION,
        "languages": ["en", "ja", "multi"],
    }

    with pytest.raises(ValueError, match="cursor exceeds"):
        adapter.fetch_page({**base_state, "language_offset": 4, "model_count": 4})
    with pytest.raises(ValueError, match="invalid terminal model count"):
        adapter.fetch_page({**base_state, "language_offset": 3})
    with pytest.raises(ValueError, match="invalid language index"):
        adapter.fetch_page({**base_state, "languages": ["en", "../../host", "multi"]})
    with pytest.raises(ValueError, match="invalid terminal model count"):
        adapter.fetch_page({**base_state, "language_offset": 3, "model_count": True})


def test_bpemb_registry_enforces_cumulative_entry_limit_across_pages() -> None:
    adapter = BPEmbPretrainedVectorRegistrySourceAdapter(
        max_entries=3,
        languages_per_page=2,
        client=_FakeClient(),
    )

    first_page = adapter.fetch_page({})
    assert first_page.next_state["model_count"] == 3

    with pytest.raises(ValueError, match="vector count exceeds"):
        adapter.fetch_page(first_page.next_state)


def test_bpemb_partial_runs_keep_prior_page_artifacts_active(tmp_path: Path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    adapter = BPEmbPretrainedVectorRegistrySourceAdapter(
        languages_per_page=1,
        client=_FakeClient(),
    )
    engine = SyncEngine(database, {adapter.name: adapter}, extractor="fixture")

    outcomes = [engine.sync(max_pages=1)[0] for _ in range(3)]

    assert [outcome.status for outcome in outcomes] == ["partial", "partial", "complete"]
    artifacts = [
        row for row in database.table_rows("artifacts") if row["source"] == adapter.name
    ]
    assert len(artifacts) == 4
    assert all(row["active"] == 1 for row in artifacts)
