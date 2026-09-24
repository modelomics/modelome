from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        self.calls.append((url, dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def html_response(
    body: str,
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
    url: str = "https://docs.example.test/library/catalog/",
) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers=dict(headers or {}),
        body=body.encode(),
        url=url,
    )


def test_link_rule_enumerates_structural_catalog_without_model_vocabulary() -> None:
    body = """
    <html>
      <head><title>Official model inventory</title></head>
      <body>
        <nav><a href="/unrelated/start/">Start</a></nav>
        <main>
          <a class="reference" href="../generated/copper.html">Copper Network</a>
          <a class="mobile-copy" href="../generated/copper.html">Copper Network</a>
          <a title="Argon Encoder" href="../generated/argon.html"><span></span></a>
          <a href="https://elsewhere.example.test/generated/external.html">External</a>
        </main>
        <script><a href="../generated/not-html.html">Not parsed</a></script>
      </body>
    </html>
    """
    client = QueuedClient(
        html_response(
            body,
            headers={
                "ETag": '"catalog-v3"',
                "Last-Modified": "Fri, 04 Sep 2026 09:30:00 GMT",
            },
        )
    )
    adapter = HtmlCatalogSourceAdapter(
        name="library-docs",
        url="https://docs.example.test/library/catalog/",
        provider_namespace="library:model",
        rules=(
            {
                "kind": "link",
                "href_pattern": r"^https://docs\.example\.test/library/generated/[^/]+\.html$",
                "identity_source": "url",
                "identity_pattern": r"/generated/(?P<id>[^/.]+)\.html$",
            },
        ),
        client=client,
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state == {
        "checked_at": "2026-09-04T12:00:00Z",
        "content_hash": page.records[0].raw["content_hash"],
        "entry_count": 2,
        "etag": '"catalog-v3"',
        "http_last_modified": "Fri, 04 Sep 2026 09:30:00 GMT",
    }

    record = page.records[0]
    assert record.kind is ArtifactKind.CATALOG_RECORD
    assert record.source_record_id == "library-docs:catalog"
    assert record.canonical_url == "https://docs.example.test/library/catalog"
    assert record.title == "Official model inventory"
    assert record.modified_at == "Fri, 04 Sep 2026 09:30:00 GMT"
    assert [model.name for model in record.models] == ["Copper Network", "Argon Encoder"]
    assert [model.identifiers for model in record.models] == [
        (Identifier("library:model", "copper"),),
        (Identifier("library:model", "argon"),),
    ]
    assert all(model.status is ModelStatus.DOCUMENTED for model in record.models)
    assert {link.url for link in record.links} == {
        "https://docs.example.test/library/generated/copper.html",
        "https://docs.example.test/library/generated/argon.html",
    }
    first_evidence = record.raw["entries"][0]
    assert record.raw["document_text"] == body
    assert record.raw["content_bytes"] == len(body.encode())
    assert len(first_evidence["occurrences"]) == 2
    assert first_evidence["occurrences"][0] == {
        "kind": "link",
        "locator": "html:a[1]",
        "href": "../generated/copper.html",
        "url": "https://docs.example.test/library/generated/copper.html",
        "text": "Copper Network",
        "title": "",
        "attributes": {
            "class": "reference",
            "href": "../generated/copper.html",
        },
    }


def test_link_rule_preserves_nested_table_anchors_without_corrupting_outer_rows() -> None:
    body = """
    <html><body>
      <table>
        <tr><th>Catalog</th></tr>
        <tr><td>
          <table><tr><td><a href="/models/inner.html">Inner Model</a></td></tr></table>
        </td></tr>
      </table>
      <a href="/models/outer.html">Outer Model</a>
    </body></html>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="nested-table-catalog",
        url="https://docs.example.test/catalog/",
        provider_namespace="library:model",
        rules=(
            {
                "kind": "link",
                "href_pattern": r"^https://docs\.example\.test/models/[a-z]+\.html$",
                "identity_source": "url",
                "identity_pattern": r"/models/(?P<id>[a-z]+)\.html$",
            },
        ),
        client=QueuedClient(html_response(body)),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert [model.name for model in page.records[0].models] == [
        "Inner Model",
        "Outer Model",
    ]
    assert [model.identifiers for model in page.records[0].models] == [
        (Identifier("library:model", "inner"),),
        (Identifier("library:model", "outer"),),
    ]


def test_link_rule_can_use_exact_url_identity_as_its_model_name() -> None:
    body = """
    <html><body>
      <a href="https://hub.example.test/research/argon-encoder#training">215M pairs</a>
    </body></html>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="identity-named-links",
        url="https://docs.example.test/catalog/",
        provider_namespace="hub:model",
        rules=(
            {
                "kind": "link",
                "href_pattern": r"^https://hub\.example\.test/[a-z-]+/[a-z-]+$",
                "identity_source": "url",
                "identity_pattern": r"^https://hub\.example\.test/(?P<id>[a-z-]+/[a-z-]+)$",
                "name_source": "identity",
            },
        ),
        allowed_origins=("https://hub.example.test",),
        client=QueuedClient(html_response(body)),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert [model.name for model in page.records[0].models] == ["research/argon-encoder"]
    assert page.records[0].models[0].identifiers == (
        Identifier("hub:model", "research/argon-encoder"),
    )


def test_link_rule_can_use_documented_href_fragment_as_exact_identity() -> None:
    body = """
    <html><body>
      <a href="/api/components/forecast/#sktime.forecasting.arima.ARIMA">ARIMA</a>
      <a href="/api/components/forecast/#citation-1">Citation</a>
    </body></html>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="fragment-identities",
        url="https://docs.example.test/catalog/",
        provider_namespace="sktime:component",
        rules=(
            {
                "kind": "link",
                "href_pattern": r"^https://docs\.example\.test/api/components/forecast$",
                "raw_href_pattern": r"#sktime\.",
                "identity_source": "href",
                "identity_pattern": r"#(?P<id>sktime\.[A-Za-z0-9_.]+)$",
            },
        ),
        client=QueuedClient(html_response(body)),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert [model.name for model in page.records[0].models] == ["ARIMA"]
    assert page.records[0].models[0].identifiers == (
        Identifier("sktime:component", "sktime.forecasting.arima.ARIMA"),
    )
    assert page.records[0].links[0].url == (
        "https://docs.example.test/api/components/forecast"
    )


def test_table_rule_enumerates_every_row_and_retains_row_evidence() -> None:
    body = """
    <html><head><title>Component matrix</title></head><body>
      <table>
        <thead><tr><th>Dataset</th><th>Rows</th></tr></thead>
        <tbody><tr><td>Example data</td><td>12</td></tr></tbody>
      </table>
      <table>
        <thead>
          <tr><th>Architecture</th><th>Task</th><th>Reference</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Quartz Generator</td><td>imaging</td>
            <td><a href="/library/model-doc/quartz">details</a></td>
          </tr>
          <tr>
            <td>Neon Predictor</td><td>molecules</td>
            <td><a href="/library/model-doc/neon">details</a></td>
          </tr>
        </tbody>
      </table>
    </body></html>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="matrix",
        url="https://docs.example.test/library/catalog/",
        provider_namespace="matrix:model",
        rules=(
            {
                "kind": "table",
                "header_pattern": r"^Architecture \| Task \| Reference$",
                "name_column": 0,
                "link_column": 2,
                "href_pattern": r"^https://docs\.example\.test/library/model-doc/",
                "require_url": True,
            },
        ),
        model_status="released",
        client=QueuedClient(html_response(body)),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    record = page.records[0]
    assert [model.name for model in record.models] == [
        "Quartz Generator",
        "Neon Predictor",
    ]
    assert [model.status for model in record.models] == [
        ModelStatus.RELEASED,
        ModelStatus.RELEASED,
    ]
    assert record.models[0].identifiers == (
        Identifier("matrix:model", "Quartz Generator"),
    )
    assert record.raw["entries"][0]["occurrences"] == [
        {
            "kind": "table_row",
            "locator": "html:table[1]/tr[1]",
            "cells": ["Quartz Generator", "imaging", "details"],
            "cell_locators": [
                "html:table[1]/tr[1]/td[0]",
                "html:table[1]/tr[1]/td[1]",
                "html:table[1]/tr[1]/td[2]",
            ],
            "url": "https://docs.example.test/library/model-doc/quartz",
            "scoped_links": [],
        }
    ]
    assert record.text == "Quartz Generator\nNeon Predictor"


def test_table_rule_can_use_model_text_before_a_citation_anchor() -> None:
    body = """
    <table>
      <tr><th>Model</th><th>Type</th></tr>
      <tr>
        <td>Argon Network<br><a href="https://papers.example.test/argon">Doe et al. 2026</a></td>
        <td>classifier</td>
      </tr>
    </table>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="citation-names",
        url="https://docs.example.test/catalog",
        provider_namespace="example:model",
        allowed_origins=("https://docs.example.test", "https://papers.example.test"),
        rules=(
            {
                "kind": "table",
                "header_pattern": r"^Model \| Type$",
                "name_column": 0,
                "link_column": 1,
                "name_source": "before_first_anchor",
                "require_url": False,
                "row_link_rules": (
                    {
                        "href_pattern": r"^https://papers\.example\.test/argon$",
                        "relation": "paper_reference",
                    },
                ),
            },
        ),
        client=QueuedClient(html_response(body)),
        clock=lambda: NOW,
    )

    record = adapter.fetch_page({}).records[0]

    assert [model.name for model in record.models] == ["Argon Network"]
    assert record.models[0].identifiers == (Identifier("example:model", "Argon Network"),)
    assert {(link.url, link.relation) for link in record.links} == {
        ("https://papers.example.test/argon", "paper_reference"),
    }


def test_table_row_links_are_attached_only_to_their_declared_model() -> None:
    body = """
    <table>
      <tr><th>Model</th><th>Documentation</th><th>Resources</th></tr>
      <tr>
        <td>Argon Encoder</td>
        <td><a href="/models/argon">docs</a></td>
        <td>
          <a href="https://papers.example.test/argon">paper</a>
          <a href="https://github.com/example/argon">code</a>
        </td>
      </tr>
      <tr>
        <td>Boron Decoder</td>
        <td><a href="/models/boron">docs</a></td>
        <td>
          <a href="https://github.com/example/boron">code</a>
          <a href="https://artifacts.example.test/boron.safetensors">weights</a>
        </td>
      </tr>
    </table>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="row-resources",
        url="https://docs.example.test/catalog",
        provider_namespace="example:model",
        allowed_origins=(
            "https://docs.example.test",
            "https://papers.example.test",
            "https://github.com",
            "https://artifacts.example.test",
        ),
        rules=(
            {
                "kind": "table",
                "header_pattern": r"^Model \| Documentation \| Resources$",
                "name_column": 0,
                "link_column": 1,
                "href_pattern": r"^https://docs\.example\.test/models/[a-z]+$",
                "require_url": True,
                "row_link_rules": (
                    {
                        "href_pattern": r"^https://papers\.example\.test/[a-z]+$",
                        "relation": "paper",
                    },
                    {
                        "href_pattern": r"^https://github\.com/example/[a-z]+$",
                        "relation": "official_implementation",
                    },
                    {
                        "href_pattern": r"^https://artifacts\.example\.test/[a-z]+\.safetensors$",
                        "relation": "weights",
                        "crawl": False,
                    },
                ),
            },
        ),
        client=QueuedClient(html_response(body)),
        clock=lambda: NOW,
    )

    record = adapter.fetch_page({}).records[0]
    model_ids = {model.name: model.local_id for model in record.models}
    resources_by_model = {
        local_id: {
            (link.url, link.relation, link.crawl)
            for link in record.links
            if link.model_local_ids == (local_id,)
        }
        for local_id in model_ids.values()
    }

    assert resources_by_model[model_ids["Argon Encoder"]] == {
        ("https://docs.example.test/models/argon", "model_page", True),
        ("https://papers.example.test/argon", "paper", True),
        ("https://github.com/example/argon", "official_implementation", True),
    }
    assert resources_by_model[model_ids["Boron Decoder"]] == {
        ("https://docs.example.test/models/boron", "model_page", True),
        ("https://github.com/example/boron", "official_implementation", True),
        ("https://artifacts.example.test/boron.safetensors", "weights", False),
    }
    occurrence = record.raw["entries"][0]["occurrences"][0]
    assert occurrence["scoped_links"] == [
        {
            "url": "https://papers.example.test/argon",
            "relation": "paper",
            "locator": "html:a[1]",
            "crawl": True,
        },
        {
            "url": "https://github.com/example/argon",
            "relation": "official_implementation",
            "locator": "html:a[2]",
            "crawl": True,
        },
    ]


def test_table_rule_can_preserve_display_name_and_separate_exact_id_column() -> None:
    body = """
    <table>
      <tr><th>Provider</th><th>Model Name</th><th>Model ID</th><th>Task</th></tr>
      <tr><td>Example Lab</td><td>Crystal Reasoner</td><td>example-crystal-v2</td><td>Text</td></tr>
      <tr><td>Example Lab</td><td>Crystal Reasoner Mini</td>
          <td>example-crystal-v2-mini</td><td>Text</td></tr>
    </table>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="separate-id-column",
        url="https://docs.example.test/models",
        provider_namespace="example:runtime-model",
        rules=(
            {
                "kind": "table",
                "header_pattern": r"^Provider \| Model Name \| Model ID \| Task$",
                "name_column": 1,
                "identity_source": "id",
                "identity_column": 2,
            },
        ),
        client=QueuedClient(html_response(body, url="https://docs.example.test/models")),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    assert [model.name for model in page.records[0].models] == [
        "Crystal Reasoner",
        "Crystal Reasoner Mini",
    ]
    assert [model.identifiers for model in page.records[0].models] == [
        (Identifier("example:runtime-model", "example-crystal-v2"),),
        (Identifier("example:runtime-model", "example-crystal-v2-mini"),),
    ]


def test_table_rule_can_strip_structural_footnote_markers_from_model_names() -> None:
    body = """
    <table>
      <tr><th>Model</th><th>Training command</th></tr>
      <tr><td>RedwoodNet 1</td><td><a href="/recipes/redwood.sh">recipe</a></td></tr>
      <tr><td>AuroraNet 12</td><td><a href="/recipes/aurora.sh">recipe</a></td></tr>
    </table>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="footnotes",
        url="https://docs.example.test/library/catalog/",
        provider_namespace="footnotes:model",
        rules=(
            {
                "kind": "table",
                "header_pattern": r"^Model \| Training command$",
                "name_column": 0,
                "name_pattern": r"^(?P<name>[A-Za-z]+Net)(?: \d+)?$",
                "link_column": 1,
                "identity_source": "name",
                "require_url": True,
            },
        ),
        client=QueuedClient(html_response(body)),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert [model.name for model in page.records[0].models] == [
        "RedwoodNet",
        "AuroraNet",
    ]


def test_table_rule_can_skip_source_declared_ineligible_cells() -> None:
    body = """
    <table>
      <tr><th>Corpus</th><th>Embedding A</th><th>Embedding B</th></tr>
      <tr><td>Corpus one</td><td>vector.alpha.300</td><td>N/A</td></tr>
      <tr><td>Corpus two</td><td>vector.beta.300</td><td>vector.gamma.300</td></tr>
    </table>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="embedding-matrix",
        url="https://docs.example.test/library/catalog/",
        provider_namespace="embedding:model",
        rules=(
            {
                "kind": "table",
                "header_pattern": r"^Corpus \| Embedding A \| Embedding B$",
                "name_column": 1,
                "name_pattern": r"^(?P<name>vector\.[a-z]+\.300)$",
                "skip_nonmatching_names": True,
                "identity_source": "name",
                "require_url": False,
            },
            {
                "kind": "table",
                "header_pattern": r"^Corpus \| Embedding A \| Embedding B$",
                "name_column": 2,
                "name_pattern": r"^(?P<name>vector\.[a-z]+\.300)$",
                "skip_nonmatching_names": True,
                "identity_source": "name",
                "require_url": False,
            },
        ),
        client=QueuedClient(html_response(body)),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert [model.name for model in page.records[0].models] == [
        "vector.alpha.300",
        "vector.beta.300",
        "vector.gamma.300",
    ]
    assert page.upstream_count == 3
    assert page.records[0].links == ()


def test_text_rule_enumerates_machine_readable_document_manifest() -> None:
    body = """
    - sections:
      - local: setup
        title: Setup
      - sections:
        - local: model_doc/boron
          title: Boron Mixer
        - local: model_doc/xenon
          title: Xenon Decoder
        title: Architectures
    """
    adapter = HtmlCatalogSourceAdapter(
        name="manifest",
        url="https://raw.example.test/project/main/docs/navigation.yml",
        provider_namespace="manifest:model",
        rules=(
            {
                "kind": "text",
                "entry_pattern": (
                    r"(?m)^\s{8}- local: model_doc/(?P<id>[a-z0-9_-]+)\s*$"
                    r"\n\s{10}title: (?P<name>[^\n]+)$"
                ),
                "url_template": (
                    "https://docs.example.test/project/model-doc/{id}"
                ),
                "require_url": True,
            },
        ),
        allowed_origins=("https://docs.example.test",),
        client=QueuedClient(
            html_response(
                body,
                url="https://raw.example.test/project/main/docs/navigation.yml",
            )
        ),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    record = page.records[0]
    assert [model.name for model in record.models] == [
        "Boron Mixer",
        "Xenon Decoder",
    ]
    assert [model.identifiers for model in record.models] == [
        (Identifier("manifest:model", "boron"),),
        (Identifier("manifest:model", "xenon"),),
    ]
    assert {link.url for link in record.links} == {
        "https://docs.example.test/project/model-doc/boron",
        "https://docs.example.test/project/model-doc/xenon",
    }
    evidence = record.raw["entries"][0]["occurrences"][0]
    assert evidence["kind"] == "text_match"
    assert evidence["groups"] == {"id": "boron", "name": "Boron Mixer"}
    assert evidence["locator"].startswith("text:")
    assert "local: model_doc/boron" in evidence["match"]


def test_text_rule_can_retain_clean_card_titles_from_a_model_library() -> None:
    body = """
    <ul>
      <li class="card">
        <a data-kind="model" href="/library/llama3.1">
          <div title="llama3.1"><span>Llama 3.1</span></div>
          <p>Display text does not become the model name.</p>
        </a>
      </li>
      <li class="card">
        <a href="/library/nomic-embed-text"><div title="nomic-embed-text"></div></a>
      </li>
    </ul>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="ollama-library",
        url="https://ollama.com/library",
        provider_namespace="ollama:model-family",
        artifact_kind="model_card",
        model_status="released",
        rules=(
            {
                "kind": "text",
                "entry_pattern": (
                    r'(?s)<li[^>]*>.*?<a(?=[^>]*href="/library/'
                    r'(?P<id>[a-z0-9][a-z0-9_.-]*)")[^>]*>.*?<div'
                    r'(?=[^>]*title="(?P<name>[^"]+)")[^>]*>'
                ),
                "url_template": "https://ollama.com/library/{id}",
                "require_url": True,
            },
        ),
        client=QueuedClient(html_response(body, url="https://ollama.com/library")),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    record = page.records[0]
    assert [model.name for model in record.models] == ["llama3.1", "nomic-embed-text"]
    assert [model.identifiers for model in record.models] == [
        (Identifier("ollama:model-family", "llama3.1"),),
        (Identifier("ollama:model-family", "nomic-embed-text"),),
    ]
    assert all(model.status is ModelStatus.RELEASED for model in record.models)
    assert {link.url for link in record.links} == {
        "https://ollama.com/library/llama3.1",
        "https://ollama.com/library/nomic-embed-text",
    }
    assert record.raw["entries"][0]["occurrences"][0]["groups"] == {
        "id": "llama3.1",
        "name": "llama3.1",
    }


def test_text_rule_extracts_current_provider_model_cards_with_native_ids() -> None:
    body = """
    <div data-models-cell data-name="@cf/openai/gpt-oss-120b"
         data-model-id="@cf/openai/gpt-oss-120b"
         data-model-label="gpt-oss-120b"
         data-model-href="/workers-ai/models/gpt-oss-120b/"
         data-model-task="Text Generation"></div>
    <div data-models-cell data-name="@cf/black-forest-labs/flux-1-schnell"
         data-model-id="@cf/black-forest-labs/flux-1-schnell"
         data-model-label="flux-1-schnell"
         data-model-href="/workers-ai/models/flux-1-schnell/"
         data-model-task="Text to Image"></div>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="cloudflare-workers-ai",
        url="https://developers.cloudflare.com/workers-ai/models/",
        provider_namespace="cloudflare-workers-ai:model",
        artifact_kind="model_card",
        rules=(
            {
                "kind": "text",
                "entry_pattern": (
                    r'(?s)<div(?=[^>]*\bdata-models-cell\b)'
                    r'(?=[^>]*\bdata-model-id="(?P<id>@cf/[A-Za-z0-9_./:-]+)")'
                    r'(?=[^>]*\bdata-model-label="(?P<name>[^"]+)")'
                    r'(?=[^>]*\bdata-model-href="(?P<url>/workers-ai/models/[A-Za-z0-9_.:-]+/)")[^>]*>'
                ),
                "require_url": True,
            },
        ),
        client=QueuedClient(
            html_response(
                body,
                url="https://developers.cloudflare.com/workers-ai/models/",
            )
        ),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    record = page.records[0]
    assert record.kind is ArtifactKind.MODEL_CARD
    assert [model.name for model in record.models] == ["gpt-oss-120b", "flux-1-schnell"]
    assert [model.identifiers for model in record.models] == [
        (Identifier("cloudflare-workers-ai:model", "@cf/openai/gpt-oss-120b"),),
        (
            Identifier(
                "cloudflare-workers-ai:model",
                "@cf/black-forest-labs/flux-1-schnell",
            ),
        ),
    ]
    assert {link.url for link in record.links} == {
        "https://developers.cloudflare.com/workers-ai/models/gpt-oss-120b",
        "https://developers.cloudflare.com/workers-ai/models/flux-1-schnell",
    }
    assert record.raw["entries"][0]["occurrences"][0]["groups"] == {
        "id": "@cf/openai/gpt-oss-120b",
        "name": "gpt-oss-120b",
        "url": "/workers-ai/models/gpt-oss-120b/",
    }


def test_mistral_catalog_links_current_and_retired_public_model_pages() -> None:
    body = """
    <html><head><title>Models Overview</title></head><body>
      <section><h2>All models</h2>
        <a href="/models/mistral-small-4-0-26-03">Mistral Small 4</a>
      </section>
      <section><h3>Deprecated &amp; retired models</h3>
        <table><tr><th>Model</th><th>Version</th><th>API</th></tr>
          <tr><td><a href="/models/mistral-7b-0-3">Mistral 7B</a></td>
              <td>0.3</td><td>open-mistral-7b</td></tr>
        </table>
      </section>
      <nav><a href="/models/mistral-7b-0-3">Mistral 7B</a></nav>
    </body></html>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="mistral-model-documentation",
        url="https://docs.mistral.ai/models",
        provider_namespace="mistral:model-documentation",
        artifact_kind="model_card",
        model_status="documented",
        rules=(
            {
                "kind": "link",
                "href_pattern": r"^https://docs\.mistral\.ai/models/[a-z0-9][a-z0-9.-]*$",
                "identity_source": "url",
                "identity_pattern": (
                    r"^https://docs\.mistral\.ai/models/"
                    r"(?P<id>[a-z0-9][a-z0-9.-]*)$"
                ),
                "relation": "model_documentation",
                "require_url": True,
            },
        ),
        client=QueuedClient(
            html_response(body, url="https://docs.mistral.ai/models")
        ),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    record = page.records[0]
    assert {model.name for model in record.models} == {"Mistral Small 4", "Mistral 7B"}
    assert {
        identifier.value for model in record.models for identifier in model.identifiers
    } == {"mistral-small-4-0-26-03", "mistral-7b-0-3"}
    assert {link.url for link in record.links} == {
        "https://docs.mistral.ai/models/mistral-small-4-0-26-03",
        "https://docs.mistral.ai/models/mistral-7b-0-3",
    }
    assert all(link.relation == "model_documentation" for link in record.links)
    assert all(link.crawl is True for link in record.links)
    assert all(link.model_local_ids for link in record.links)


def test_cohere_catalog_reads_exact_model_names_from_multiple_documented_tables() -> None:
    body = """
    <html><head><title>Cohere models</title></head><body>
      <table>
        <tr><th>Model Name</th><th>Status</th><th>Description</th><th>Endpoints</th></tr>
        <tr><td><code>command-a-03-2025</code></td><td>Live</td>
            <td>Command A</td><td>Chat</td></tr>
        <tr><td><code>command-r-03-2024</code></td><td>Deprecated</td>
            <td>Command R</td><td>Chat</td></tr>
      </table>
      <table>
        <tr><th>Model Name</th><th>Description</th><th>Dimensions</th></tr>
        <tr><td><code>embed-v4.0</code></td><td>Embedding model</td><td>1536</td></tr>
      </table>
      <table>
        <tr><th>Model family</th><th>Type</th></tr>
        <tr><td>Command</td><td>Generative</td></tr>
      </table>
    </body></html>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="cohere-model-documentation",
        url="https://docs.cohere.com/docs/models",
        provider_namespace="cohere:model-documentation",
        artifact_kind="model_card",
        model_status="documented",
        rules=(
            {
                "kind": "table",
                "header_pattern": r"^Model Name \| (?:Status \| )?Description \|",
                "name_column": 0,
                "identity_source": "name",
                "require_url": False,
            },
        ),
        client=QueuedClient(
            html_response(body, url="https://docs.cohere.com/docs/models")
        ),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 3
    record = page.records[0]
    assert [model.name for model in record.models] == [
        "command-a-03-2025",
        "command-r-03-2024",
        "embed-v4.0",
    ]
    assert {identifier.value for model in record.models for identifier in model.identifiers} == {
        "command-a-03-2025",
        "command-r-03-2024",
        "embed-v4.0",
    }
    assert record.links == ()


def test_aws_bedrock_catalog_retains_each_first_party_model_card() -> None:
    body = """
    <html><head><title>Models at a glance</title></head><body>
      <table>
        <tr><th>Provider</th><th>Supported models</th></tr>
        <tr>
          <td>Example provider</td>
          <td>
            <a href="/bedrock/latest/userguide/model-card-example-aurora-net-2.html">
              AuroraNet 2
            </a>
            <a href="/bedrock/latest/userguide/model-card-example-lumen-embed.html">
              Lumen Embed
            </a>
          </td>
        </tr>
      </table>
      <a href="/bedrock/latest/userguide/model-parameters.html">Not a card</a>
    </body></html>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="aws-bedrock-model-cards",
        url="https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
        provider_namespace="aws:bedrock-model-card",
        artifact_kind="model_card",
        model_status="documented",
        rules=(
            {
                "kind": "link",
                "href_pattern": (
                    r"^https://docs\.aws\.amazon\.com/bedrock/latest/userguide/"
                    r"model-card-[a-z0-9][a-z0-9-]*\.html$"
                ),
                "identity_source": "url",
                "identity_pattern": (
                    r"^https://docs\.aws\.amazon\.com/bedrock/latest/userguide/"
                    r"model-card-(?P<id>[a-z0-9][a-z0-9-]*)\.html$"
                ),
                "relation": "model_card",
                "require_url": True,
            },
        ),
        client=QueuedClient(
            html_response(
                body,
                url="https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
            )
        ),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    record = page.records[0]
    assert record.kind is ArtifactKind.MODEL_CARD
    assert [model.name for model in record.models] == ["AuroraNet 2", "Lumen Embed"]
    assert [model.identifiers for model in record.models] == [
        (Identifier("aws:bedrock-model-card", "example-aurora-net-2"),),
        (Identifier("aws:bedrock-model-card", "example-lumen-embed"),),
    ]
    assert {(link.url, link.relation) for link in record.links} == {
        (
            "https://docs.aws.amazon.com/bedrock/latest/userguide/"
            "model-card-example-aurora-net-2.html",
            "model_card",
        ),
        (
            "https://docs.aws.amazon.com/bedrock/latest/userguide/"
            "model-card-example-lumen-embed.html",
            "model_card",
        ),
    }


def test_openai_model_catalog_uses_exact_path_ids_and_excludes_index_pages() -> None:
    body = """
    <html><head><title>All models | OpenAI API</title></head><body>
      <a href="/api/docs/models/gpt-4">GPT-4 Deprecated</a>
      <a href="/api/docs/models/text-embedding-ada-002">text-embedding-ada-002</a>
      <a href="/api/docs/models/all">All models</a>
      <a href="/api/docs/models/compare">Compare models</a>
    </body></html>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="openai-model-documentation",
        url="https://developers.openai.com/api/docs/models/all",
        provider_namespace="openai:model",
        artifact_kind="model_card",
        model_status="documented",
        rules=(
            {
                "kind": "link",
                "href_pattern": (
                    r"^https://developers\.openai\.com/api/docs/models/"
                    r"(?!all$|compare$)[a-z0-9][a-z0-9.-]*$"
                ),
                "identity_source": "url",
                "identity_pattern": (
                    r"^https://developers\.openai\.com/api/docs/models/"
                    r"(?P<id>[a-z0-9][a-z0-9.-]*)$"
                ),
                "name_source": "identity",
                "relation": "model_documentation",
                "require_url": True,
            },
        ),
        client=QueuedClient(
            html_response(
                body,
                url="https://developers.openai.com/api/docs/models/all",
            )
        ),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    record = page.records[0]
    assert [model.name for model in record.models] == [
        "gpt-4",
        "text-embedding-ada-002",
    ]
    assert [model.identifiers for model in record.models] == [
        (Identifier("openai:model", "gpt-4"),),
        (Identifier("openai:model", "text-embedding-ada-002"),),
    ]
    assert {(link.url, link.relation) for link in record.links} == {
        (
            "https://developers.openai.com/api/docs/models/gpt-4",
            "model_documentation",
        ),
        (
            "https://developers.openai.com/api/docs/models/text-embedding-ada-002",
            "model_documentation",
        ),
    }


def test_html_catalog_prefers_html_to_avoid_documentation_markdown_redirects() -> None:
    client = QueuedClient(
        html_response(
            "<a href=\"/models/example\">Example Model</a>",
            url="https://docs.example.test/models",
        )
    )
    adapter = HtmlCatalogSourceAdapter(
        name="html-preference",
        url="https://docs.example.test/models",
        provider_namespace="html-preference:model",
        rules=(
            {
                "kind": "link",
                "href_pattern": r"^https://docs\.example\.test/models/[a-z]+$",
            },
        ),
        client=client,
        clock=lambda: NOW,
    )

    adapter.fetch_page({})

    assert client.calls[0][1]["Accept"] == (
        "text/html,application/xhtml+xml"
    )


def test_detail_resource_rules_emit_resumable_model_scoped_records() -> None:
    """A model page's selected resources never become catalog-wide links."""

    index = """
    <html><body>
      <a href="/models/amber">Amber Network</a>
      <a href="/models/blue">Blue Network</a>
    </body></html>
    """
    amber = """
    <html><head><title>Amber docs</title></head><body>
      <nav><a href="https://github.com/example/navigation">Navigation</a></nav>
      <a href="https://arxiv.org/abs/2601.00001">Paper</a>
      <a href="https://github.com/example/amber">Implementation</a>
    </body></html>
    """
    blue = """
    <html><head><title>Blue docs</title></head><body>
      <a href="https://github.com/example/blue">Implementation</a>
    </body></html>
    """
    client = QueuedClient(
        html_response(index),
        html_response(amber, url="https://docs.example.test/models/amber"),
        html_response(blue, url="https://docs.example.test/models/blue"),
    )
    adapter = HtmlCatalogSourceAdapter(
        name="detail-catalog",
        url="https://docs.example.test/library/catalog/",
        provider_namespace="detail:model",
        rules=(
            {
                "kind": "link",
                "href_pattern": r"^https://docs\.example\.test/models/[a-z]+$",
                "identity_source": "url",
                "identity_pattern": r"/models/(?P<id>[a-z]+)$",
                "relation": "model_documentation",
            },
        ),
        detail_resource_rules=(
            {
                "href_pattern": r"^https://arxiv\.org/abs/\d{4}\.\d{5}$",
                "relation": "paper_reference",
            },
            {
                "href_pattern": r"^https://github\.com/example/(?:amber|blue)$",
                "relation": "official_implementation",
            },
        ),
        detail_batch_size=1,
        allowed_origins=(
            "https://docs.example.test",
            "https://arxiv.org",
            "https://github.com",
        ),
        client=client,
        clock=lambda: NOW,
    )

    index_page = adapter.fetch_page({})

    assert index_page.complete is False
    assert index_page.authoritative_snapshot is False
    assert index_page.upstream_count == 2
    assert len(index_page.next_state["detail_tasks"]) == 2
    assert index_page.next_state["detail_offset"] == 0
    assert index_page.records[0].source_record_id == "detail-catalog:catalog"

    amber_page = adapter.fetch_page(index_page.next_state)

    assert amber_page.complete is False
    assert amber_page.authoritative_snapshot is False
    amber_record = amber_page.records[0]
    assert amber_record.title == "Amber docs"
    assert [model.name for model in amber_record.models] == ["Amber Network"]
    amber_local_id = amber_record.models[0].local_id
    assert {
        (link.url, link.relation, link.model_local_ids)
        for link in amber_record.links
    } == {
        (
            "https://docs.example.test/models/amber",
            "model_documentation",
            (amber_local_id,),
        ),
        (
            "https://arxiv.org/abs/2601.00001",
            "paper_reference",
            (amber_local_id,),
        ),
        (
            "https://github.com/example/amber",
            "official_implementation",
            (amber_local_id,),
        ),
    }
    assert all("navigation" not in link.url for link in amber_record.links)
    assert amber_page.next_state["detail_offset"] == 1
    assert amber_page.next_state["detail_rule_matches"] == [1, 1]

    blue_page = adapter.fetch_page(amber_page.next_state)

    assert blue_page.complete is True
    assert blue_page.authoritative_snapshot is True
    assert "detail_tasks" not in blue_page.next_state
    blue_record = blue_page.records[0]
    assert [model.name for model in blue_record.models] == ["Blue Network"]
    blue_local_id = blue_record.models[0].local_id
    assert {
        (link.url, link.relation, link.model_local_ids)
        for link in blue_record.links
    } == {
        (
            "https://docs.example.test/models/blue",
            "model_documentation",
            (blue_local_id,),
        ),
        (
            "https://github.com/example/blue",
            "official_implementation",
            (blue_local_id,),
        ),
    }


def test_text_rules_can_group_multiple_declared_weight_artifacts_by_native_id() -> None:
    body = """
    <table><tbody><tr><td>English:
      <a href="https://artifacts.example.test/cc.en.300.bin.gz">bin</a>,
      <a href="https://artifacts.example.test/cc.en.300.vec.gz">text</a>
    </td></tr></tbody></table>
    <p>Reference: <a href="https://papers.example.test/vectors">Vector paper</a></p>
    """
    adapter = HtmlCatalogSourceAdapter(
        name="vectors",
        url="https://docs.example.test/vectors",
        provider_namespace="vectors:language-model",
        artifact_kind="model_card",
        model_status="released",
        allowed_origins=(
            "https://docs.example.test",
            "https://artifacts.example.test",
            "https://papers.example.test",
        ),
        shared_link_rules=(
            {
                "href_pattern": r"^https://papers\.example\.test/vectors$",
                "text_pattern": r"^Vector paper$",
                "relation": "paper_reference",
            },
        ),
        rules=(
            {
                "kind": "text",
                "entry_pattern": (
                    r'(?s)<td>(?P<name>[^:<]+):\s*<a'
                    r'(?=[^>]*href="(?P<url>https://artifacts\.example\.test/'
                    r'cc\.(?P<id>[a-z]+)\.300\.bin\.gz)")[^>]*>bin</a>'
                ),
                "name_template": "fastText CC 300 {name}",
                "relation": "weights",
                "crawl": False,
                "require_url": True,
            },
            {
                "kind": "text",
                "entry_pattern": (
                    r'(?s)<td>(?P<name>[^:<]+):\s*<a'
                    r'(?=[^>]*href="https://artifacts\.example\.test/'
                    r'cc\.(?P<id>[a-z]+)\.300\.bin\.gz")[^>]*>bin</a>,\s*<a'
                    r'(?=[^>]*href="(?P<url>https://artifacts\.example\.test/'
                    r'cc\.(?P=id)\.300\.vec\.gz)")[^>]*>text</a>'
                ),
                "name_template": "fastText CC 300 {name}",
                "relation": "weights",
                "crawl": False,
                "require_url": True,
            },
        ),
        client=QueuedClient(html_response(body)),
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert [model.name for model in record.models] == ["fastText CC 300 English"]
    assert record.models[0].identifiers == (
        Identifier("vectors:language-model", "en"),
    )
    assert {
        (link.url, link.relation, link.crawl, link.model_local_ids)
        for link in record.links
    } == {
        ("https://artifacts.example.test/cc.en.300.bin.gz", "weights", False, (
            record.models[0].local_id,
        )),
        ("https://artifacts.example.test/cc.en.300.vec.gz", "weights", False, (
            record.models[0].local_id,
        )),
        ("https://papers.example.test/vectors", "paper_reference", True, ()),
    }
    assert len(record.raw["entries"][0]["occurrences"]) == 2
    assert record.raw["shared_links"] == [
        {
            "url": "https://papers.example.test/vectors",
            "relation": "paper_reference",
            "locator": "html:a[2]",
            "crawl": True,
        }
    ]


def test_not_modified_uses_conditional_headers_without_claiming_zero_entries() -> None:
    client = QueuedClient(html_response("", status=304))
    adapter = HtmlCatalogSourceAdapter(
        name="conditional",
        url="https://docs.example.test/library/catalog/",
        provider_namespace="conditional:model",
        rules=({"kind": "link", "href_pattern": r"/model-doc/"},),
        client=client,
        clock=lambda: NOW,
    )
    state = {
        "etag": '"old"',
        "http_last_modified": "Thu, 03 Sep 2026 09:30:00 GMT",
        "entry_count": 42,
    }

    page = adapter.fetch_page(state)

    assert page.records == ()
    assert page.complete is True
    assert page.authoritative_snapshot is False
    assert page.upstream_count is None
    assert page.next_state == {**state, "checked_at": "2026-09-04T12:00:00Z"}
    assert client.calls == [
        (
            "https://docs.example.test/library/catalog",
            {
                "Accept": "text/html,application/xhtml+xml",
                "If-None-Match": '"old"',
                "If-Modified-Since": "Thu, 03 Sep 2026 09:30:00 GMT",
            },
        )
    ]


def test_selector_drift_fails_closed_instead_of_emitting_empty_snapshot() -> None:
    adapter = HtmlCatalogSourceAdapter(
        name="drift",
        url="https://docs.example.test/library/catalog/",
        provider_namespace="drift:model",
        rules=(
            {
                "kind": "table",
                "header_pattern": r"^Architecture \| Task$",
            },
        ),
        client=QueuedClient(
            html_response("<table><tr><th>Renamed column</th></tr></table>")
        ),
    )

    with pytest.raises(ValueError, match="rule 0 matched no catalog entries"):
        adapter.fetch_page({})


def test_selected_cross_origin_model_url_fails_closed_by_default() -> None:
    adapter = HtmlCatalogSourceAdapter(
        name="origin",
        url="https://docs.example.test/library/catalog/",
        provider_namespace="origin:model",
        rules=(
            {
                "kind": "link",
                "href_pattern": r"/model-doc/",
            },
        ),
        client=QueuedClient(
            html_response(
                '<a href="https://other.example.test/model-doc/item">Item</a>'
            )
        ),
    )

    with pytest.raises(ValueError, match="outside allowed origins"):
        adapter.fetch_page({})


@pytest.mark.parametrize(
    "rules",
    [
        (),
        ({"kind": "link"},),
        ({"kind": "table"},),
        ({"kind": "semantic_guess", "href_pattern": ".*"},),
        ({"kind": "link", "href_pattern": ".*", "model_names": ["x"]},),
        ({"kind": "link", "href_pattern": ".*", "require_url": "yes"},),
    ],
)
def test_rule_configuration_fails_closed(rules: tuple[Mapping[str, Any], ...]) -> None:
    with pytest.raises(ValueError):
        HtmlCatalogSourceAdapter(
            name="invalid",
            url="https://docs.example.test/library/catalog/",
            provider_namespace="invalid:model",
            rules=rules,
            client=QueuedClient(),
        )


def test_catalog_bounds_matching_entries() -> None:
    adapter = HtmlCatalogSourceAdapter(
        name="bounded",
        url="https://docs.example.test/library/catalog/",
        provider_namespace="bounded:model",
        rules=({"kind": "link", "href_pattern": r"/model-doc/"},),
        max_entries=1,
        client=QueuedClient(
            html_response(
                """
                <a href="/model-doc/one">One</a>
                <a href="/model-doc/two">Two</a>
                """
            )
        ),
    )

    with pytest.raises(ValueError, match="exceeds 1 matched entries"):
        adapter.fetch_page({})
