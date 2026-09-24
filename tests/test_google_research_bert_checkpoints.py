from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.google_bert_checkpoints import (
    GoogleResearchBertCheckpointSourceAdapter,
)

_REVISION = "a" * 40
_BASE = "https://storage.googleapis.com/bert_models"


class FakeClient:
    def __init__(self, document: str) -> None:
        self.document = document.encode()
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
            return HttpResponse(200, {}, body, url)
        if url.endswith("/README.md"):
            return HttpResponse(200, {}, self.document, url)
        raise AssertionError(f"unexpected URL {url}")


def test_original_bert_readme_checkpoint_links_preserve_exact_gcs_archives() -> None:
    document = """# BERT
| H=128 | H=256 |
| --- | --- |
| L=2 | 128 | 2 | `uncased_L-2_H-128_A-2.zip` |
| L=12 | 768 | 12 | `uncased_L-12_H-768_A-12.zip` |

## Pre-trained models
* [`BERT-Large, Uncased (Whole Word Masking)`](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip)
* [`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)
* [`BERT-Base, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)
* [`BERT-Base, Multilingual Cased`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)
* [`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
* [all models](https://storage.googleapis.com/bert_models/2020_02_20/all_bert_models.zip)
* [dataset](https://storage.googleapis.com/bert_models/2020_02_20/squad.zip)
* [unsafe host](https://example.test/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)
"""
    adapter = GoogleResearchBertCheckpointSourceAdapter(client=FakeClient(document))

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 5
    by_handle = {record.raw["checkpoint_handle"]: record for record in page.records}
    assert set(by_handle) == {
        "2018_10_18/cased_L-12_H-768_A-12",
        "2018_10_18/uncased_L-12_H-768_A-12",
        "2018_11_03/chinese_L-12_H-768_A-12",
        "2018_11_23/multi_cased_L-12_H-768_A-12",
        "2019_05_30/wwm_uncased_L-24_H-1024_A-16",
    }
    assert by_handle["2018_10_18/uncased_L-12_H-768_A-12"].links[-1].url == (
        f"{_BASE}/2018_10_18/uncased_L-12_H-768_A-12.zip"
    )
    assert all(record.links[-1].relation == "weights" for record in page.records)
    assert not any(url.endswith(".zip") for url in adapter.client.calls)


def test_adapter_is_only_in_a_disabled_proposal() -> None:
    path = Path(__file__).parents[1] / "config/proposals/google_research_bert_checkpoints.toml"
    proposal = tomllib.loads(path.read_text())["source"][0]

    assert proposal["adapter"] == "google_research_bert_checkpoints"
    assert proposal["enabled"] is False
    assert proposal["repository"] == "google-research/bert"
