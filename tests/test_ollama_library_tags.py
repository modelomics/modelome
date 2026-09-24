from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.ollama_library_tags import OllamaLibraryTagCatalogAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/ollama_library_tags.toml"
_INDEX = "https://ollama.com/library"
_FAMILY = "https://ollama.com/library/mistral"
_TAGS_1 = "https://ollama.com/library/mistral/tags"
_TAGS_2 = "https://ollama.com/library/mistral/tags?cursor=cursor-from-page"


class _FixtureClient:
    def __init__(self, responses: Mapping[str, str]) -> None:
        self.responses = dict(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        assert headers == {"Accept": "text/html"}
        self.calls.append(url)
        if url not in self.responses:
            raise AssertionError(f"unexpected Ollama request: {url}")
        return HttpResponse(
            200,
            {"Content-Type": "text/html"},
            self.responses[url].encode(),
            url,
        )


def _adapter(client: _FixtureClient) -> OllamaLibraryTagCatalogAdapter:
    config = tomllib.loads(_PROPOSAL.read_text())["source"][0]
    return OllamaLibraryTagCatalogAdapter(
        name=config["name"],
        url=config["url"],
        client=client,
        max_response_bytes=config["max_response_bytes"],
        max_families=config["max_families"],
        max_tags_per_family=config["max_tags_per_family"],
        max_pages_per_family=config["max_pages_per_family"],
        max_anchors_per_page=config["max_anchors_per_page"],
    )


def test_ollama_catalog_follows_declared_family_tags_and_pagination_links() -> None:
    client = _FixtureClient(
        {
            _INDEX: '<ul><li><a href="/library/mistral"><div title="mistral"></div></a></li></ul>',
            _FAMILY: (
                '<h1>Mistral</h1><a href="/library/mistral/tags">View all →</a>'
            ),
            _TAGS_1: """<main>
              <a href="/library/mistral%3Alatest"
                aria-label="mistral:latest 9.1GB 32K context window">
                mistral:latest 54a0e45 9.1GB 32K context window</a>
              <a href="/library/mistral%3A7b-instruct-v0.3-q4_K_M"
                title="mistral:7b-instruct-v0.3-q4_K_M 4.1GB">
                mistral:7b-instruct-v0.3-q4_K_M 0123456789ab · 4.1GB</a>
              <a rel="next" href="/library/mistral/tags?cursor=cursor-from-page"
                aria-label="Next page"></a>
            </main>""",
            _TAGS_2: """<main>
              <a href="/library/mistral%3A7b-instruct-v0.3-q8_0">
                mistral:7b-instruct-v0.3-q8_0 0123456789abcdef · 7.7GB</a>
            </main>""",
        }
    )

    page = _adapter(client).fetch_page({})

    assert client.calls == [_INDEX, _FAMILY, _TAGS_1, _TAGS_2]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    record = page.records[0]
    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.models[0].identifiers == (Identifier("ollama:model-family", "mistral"),)
    assert [release.version for release in record.releases] == [
        "7b-instruct-v0.3-q4_K_M",
        "7b-instruct-v0.3-q8_0",
        "latest",
    ]
    first = record.releases[0]
    assert first.identifiers == (
        Identifier("ollama:model-tag", "mistral:7b-instruct-v0.3-q4_K_M"),
    )
    assert first.metadata["size_label"] == "4.1GB"
    assert first.metadata["size_bytes"] == 4_100_000_000
    assert first.metadata["digest"] == "0123456789ab"
    assert record.releases[-1].metadata["context_window"] == "32K"
    assert record.releases[-1].metadata["tag_url"] == (
        "https://ollama.com/library/mistral%3Alatest"
    )
    assert all(link.crawl is False for link in record.links)


def test_ollama_adapter_fails_closed_when_family_does_not_declare_tag_page() -> None:
    client = _FixtureClient(
        {
            _INDEX: '<a href="/library/mistral">mistral</a>',
            _FAMILY: "<h1>Mistral</h1><p>No tag listing link.</p>",
        }
    )
    with pytest.raises(ValueError, match="did not declare its tag-list link"):
        _adapter(client).fetch_page({})


def test_ollama_adapter_does_not_accept_pagination_state_or_fetch_manifests() -> None:
    adapter = _adapter(_FixtureClient({}))
    with pytest.raises(ValueError, match="does not accept pagination state"):
        adapter.fetch_page({"cursor": "invented"})
