from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.gpt4all_model_catalog import Gpt4AllModelCatalogSourceAdapter

_SHA = "a" * 40


class FixtureClient:
    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        if url == "https://api.github.com/repos/nomic-ai/gpt4all/commits/main":
            body = json.dumps({"sha": _SHA}).encode()
        elif url.endswith(f"/{_SHA}/gpt4all-chat/metadata/models3.json"):
            body = json.dumps(self.payload).encode()
        else:
            raise AssertionError(f"unexpected fetch; artifact bytes must not be downloaded: {url}")
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=body,
            url=url,
        )


def _adapter(
    payload: list[dict[str, Any]],
) -> tuple[Gpt4AllModelCatalogSourceAdapter, FixtureClient]:
    client = FixtureClient(payload)
    adapter = Gpt4AllModelCatalogSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )
    return adapter, client


def test_manifest_emits_exact_variant_file_and_artifact_reference_without_download() -> None:
    adapter, client = _adapter(
        [
            {
                "order": "a",
                "name": "Llama 3 Instruct",
                "filename": "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
                "url": "https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
                "filesize": "4661724384",
                "requires": "2.7.1",
                "quant": "q4_0",
                "md5sum": "c" * 32,
                "sha256sum": "d" * 64,
            },
            {
                "order": "b",
                "name": "Reasoner v1",
                "filename": "qwen2.5-coder-7b-instruct-q4_0.gguf",
                "url": "https://gpt4all.io/models/gguf/qwen2.5-coder-7b-instruct-q4_0.gguf",
                "removedIn": "3.9.0",
                "quant": "q4_0",
            },
        ]
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert len(page.records) == 2
    llama = page.records[0]
    assert llama.kind.value == "weights"
    assert llama.source_record_id == "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    assert llama.identifiers[0].value == "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    assert llama.models[0].name == "Llama 3 Instruct"
    assert llama.releases[0].version == "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    assert llama.releases[0].revision == _SHA
    assert llama.releases[0].metadata["md5sum"] == "c" * 32
    assert llama.links[0].url == (
        "https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    )
    assert llama.links[0].relation == "model_artifact"
    assert llama.links[0].crawl is False
    assert llama.raw["gpt4all_manifest_revision"] == _SHA
    assert page.records[1].releases[0].metadata["removed_in_gpt4all_version"] == "3.9.0"
    assert len(client.calls) == 2
    assert not any("models/gguf/" in url or "/resolve/" in url for url in client.calls)


def test_manifest_quarantines_unsupported_or_incomplete_download_rows() -> None:
    adapter, _ = _adapter(
        [
            {
                "order": "a",
                "name": "Good model",
                "filename": "good-model-q4.gguf",
                "url": "https://gpt4all.io/models/gguf/good-model-q4.gguf",
            },
            {
                "order": "bad-host",
                "name": "Bad host",
                "filename": "bad-host.gguf",
                "url": "https://example.invalid/bad-host.gguf",
            },
            {
                "order": "hub-host",
                "name": "Already indexed from the hub",
                "filename": "hub-host.gguf",
                "url": "https://huggingface.co/org/model/resolve/main/hub-host.gguf",
            },
            {"order": "missing-url", "name": "No exact download", "filename": "missing.gguf"},
        ]
    )

    page = adapter.fetch_page({})

    assert [record.source_record_id for record in page.records] == ["good-model-q4.gguf"]
    assert page.advance_on_source_issues
    assert [issue.source_record_id for issue in page.issues] == [
        "bad-host",
        "hub-host",
        "missing-url",
    ]


def test_unchanged_manifest_revision_skips_catalog_refetch() -> None:
    adapter, client = _adapter([])

    page = adapter.fetch_page({"completed_revision": _SHA, "model_count": 7})

    assert page.records == ()
    assert page.complete
    assert page.upstream_count == 7
    assert len(client.calls) == 1


def test_gpt4all_proposal_is_disabled_and_pins_first_party_manifest_location() -> None:
    path = Path(__file__).parents[1] / "config/proposals/gpt4all_model_catalog.toml"
    with path.open("rb") as handle:
        config = tomllib.load(handle)["source"][0]

    assert config["enabled"] is False
    assert config["adapter"] == "gpt4all_model_catalog"
    assert config["repository"] == "nomic-ai/gpt4all"
    assert config["source_path"] == "gpt4all-chat/metadata/models3.json"


def test_duplicate_filenames_fail_closed() -> None:
    row = {
        "name": "Duplicate",
        "filename": "duplicate.gguf",
        "url": "https://gpt4all.io/models/gguf/duplicate.gguf",
    }
    adapter, _ = _adapter([row, row])

    with pytest.raises(ValueError, match="duplicate model filename"):
        adapter.fetch_page({})
