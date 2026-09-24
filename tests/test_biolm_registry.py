from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.biolm_registry import BioLMRegistrySourceAdapter, _parse_model_table

README_URL = "https://raw.githubusercontent.com/facebookresearch/bio-lm/main/README.md"


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def response(body: str) -> HttpResponse:
    return HttpResponse(200, {}, body.encode(), README_URL)


def table_row(name: str, size: str, description: str) -> str:
    return (
        f"| {name} | {size} | {description} "
        f"| [download](https://dl.fbaipublicfiles.com/biolm/{name}-hf.tar.gz) "
        f"| [download](https://dl.fbaipublicfiles.com/biolm/{name}-fairseq.tar.gz) |"
    )


def test_biolm_emits_official_names_and_both_exact_archive_formats() -> None:
    readme = "\n".join([
        "## Models",
        "Model | Size | Description | 🤗 Transformers Link | fairseq link |",
        "| --- | --- | --- | --- | --- |",
        table_row("RoBERTa-base-PM", "base", "PubMed and PMC"),
        table_row("RoBERTa-large-PM-M3-Voc", "large", "PubMed PMC MIMIC-III"),
        "## Code",
    ])
    client = QueueClient(response(readme))
    source = BioLMRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 2
    assert [record.source_record_id for record in page.records] == [
        "facebook-bio-lm-first-party-archives:model:RoBERTa-base-PM",
        "facebook-bio-lm-first-party-archives:model:RoBERTa-large-PM-M3-Voc",
    ]
    base = page.records[0]
    assert base.models[0].identifiers[0].value == "RoBERTa-base-PM"
    assert {r.metadata["format"] for r in base.releases} == {"transformers", "fairseq"}
    assert {r.metadata["archive_url"] for r in base.releases} == {
        "https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-hf.tar.gz",
        "https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-fairseq.tar.gz",
    }
    assert all(r.metadata["archive_format"] == "tar.gz" for r in base.releases)
    assert all(not link.crawl for record in page.records for link in record.links)
    assert client.calls == [README_URL]
    assert page.next_state["completed_revision"] == hashlib.sha256(readme.encode()).hexdigest()


def test_biolm_unchanged_readme_is_noop() -> None:
    readme = "\n".join([
        "## Models", table_row("RoBERTa-base-PM", "base", "PubMed and PMC"), "## Code",
    ])
    source = BioLMRegistrySourceAdapter(client=QueueClient(response(readme)))
    digest = hashlib.sha256(readme.encode()).hexdigest()

    page = source.fetch_page({"completed_revision": digest, "checkpoint_count": 9})

    assert page.complete and page.records == () and page.upstream_count == 9


def test_biolm_rejects_untrusted_mismatched_duplicate_or_unbounded_rows() -> None:
    unsafe = "## Models\n" + table_row("Fake", "base", "test").replace(
        "https://dl.fbaipublicfiles.com/biolm/Fake-hf.tar.gz",
        "https://example.org/Fake-hf.tar.gz",
    )
    mismatch = "## Models\n" + table_row("Fake", "base", "test").replace(
        "Fake-fairseq.tar.gz", "different-fairseq.tar.gz",
    )
    duplicate = "## Models\n" + table_row("Fake", "base", "test") + "\n" + table_row(
        "Fake", "base", "test",
    )
    with pytest.raises(ValueError, match="non-admitted archive URL"):
        _parse_model_table(unsafe, 10)
    with pytest.raises(ValueError, match="do not match model"):
        _parse_model_table(mismatch, 10)
    with pytest.raises(ValueError, match="repeats model"):
        _parse_model_table(duplicate, 10)
    with pytest.raises(ValueError, match="exceeds entry limit"):
        _parse_model_table(
            "## Models\n" + table_row("A", "base", "a") + "\n" + table_row("B", "base", "b"),
            1,
        )
