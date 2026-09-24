from __future__ import annotations

from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.cloud_extra import DEFAULT_URL, OciGenerativeAIModelCatalog


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self, url: str, *, params: Any = None, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        assert params is None
        assert headers == {"Accept": "text/html"}
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: str, url: str = DEFAULT_URL, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=body.encode(), url=url)


def test_oci_public_catalog_resolves_exact_model_name_from_first_party_card() -> None:
    card_url = "https://docs.oracle.com/en-us/iaas/Content/generative-ai/google-gemini-2-5-pro.htm"
    catalog = (
        '<h2>Chat Models</h2><ul><li><a href="google-gemini-2-5-pro.htm">'
        "Google Gemini 2.5 Pro</a></li></ul>"
    )
    card = (
        "<h1>Google Gemini 2.5 Pro</h1><p>The model name is "
        "<code>google.gemini-2.5-pro</code>.</p>"
        "<p>Model Name in OCI Generative AI: <code>google.gemini-2.5-pro</code></p>"
    )
    client = Client(response(catalog), response(card, card_url))

    page = OciGenerativeAIModelCatalog(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    assert client.calls == [DEFAULT_URL, card_url]
    record = page.records[0]
    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.canonical_url == card_url
    assert record.models[0].name == "Google Gemini 2.5 Pro"
    assert record.models[0].identifiers == (
        Identifier("oci:generative-ai-model", "google.gemini-2.5-pro"),
    )
    assert record.models[0].status is ModelStatus.RELEASED


def test_oci_catalog_rejects_empty_and_over_limit_catalogs() -> None:
    with pytest.raises(ValueError, match="no linked OCI model cards"):
        OciGenerativeAIModelCatalog(
            client=Client(response("<html>no models</html>"))
        ).fetch_page({})

    two_cards = (
        '<a href="model-one.htm">Model One</a>'
        '<a href="model-two.htm">Model Two</a>'
    )
    with pytest.raises(ValueError, match="exceeds 1 model entries"):
        OciGenerativeAIModelCatalog(
            client=Client(response(two_cards)), max_entries=1
        ).fetch_page({})


def test_oci_catalog_rejects_bad_card_status() -> None:
    client = Client(
        response('<a href="google-gemini-2-5-pro.htm">Google Gemini 2.5 Pro</a>'),
        response(
            "<html>details unavailable</html>",
            "https://docs.oracle.com/en-us/iaas/Content/generative-ai/google-gemini-2-5-pro.htm",
            404,
        ),
    )
    with pytest.raises(ValueError, match="model card returned HTTP 404"):
        OciGenerativeAIModelCatalog(client=client).fetch_page({})
