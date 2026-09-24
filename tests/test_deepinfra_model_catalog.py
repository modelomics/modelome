from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.deepinfra_model_catalog import DeepInfraModelCatalogAdapter

_URL = "https://deepinfra.com/models"


class _Client:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.body = (
            '<html><script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(payload)
            + "</script></html>"
        ).encode()
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
        return HttpResponse(200, {"content-type": "text/html"}, self.body, url)


def _item(
    full_name: str = "deepseek-ai/DeepSeek-V4-Flash",
    *,
    deprecated: bool = False,
) -> dict[str, Any]:
    owner, name = full_name.split("/", 1)
    return {
        "owner": owner,
        "name": name,
        "full_name": full_name,
        "type": "text-generation",
        "description": "Hosted text generation endpoint.",
        "tags": ["open-weight", "featured"],
        "deprecated": deprecated,
        "replaced_by": None,
        "quantization": "fp8",
        "private": False,
    }


def _payload(*items: Mapping[str, Any], page: int = 1) -> dict[str, Any]:
    return {"props": {"pageProps": {"page": page, "models": list(items)}}}


def test_catalog_extracts_exact_public_model_ids_without_weight_claims() -> None:
    private_item = _item("internal/not-public")
    private_item["private"] = True
    client = _Client(
        _payload(_item(), _item("BAAI/bge-m3", deprecated=True), private_item)
    )

    page = DeepInfraModelCatalogAdapter(client=client).fetch_page({})

    assert client.calls == [_URL]
    assert page.complete and not page.authoritative_snapshot
    assert page.upstream_count == 2
    assert [record.title for record in page.records] == [
        "deepseek-ai/DeepSeek-V4-Flash",
        "BAAI/bge-m3",
    ]
    record = page.records[0]
    assert record.kind is ArtifactKind.PROVIDER_PAGE
    assert record.canonical_url == "https://deepinfra.com/deepseek-ai/DeepSeek-V4-Flash"
    assert record.identifiers == (
        Identifier("deepinfra:model", "deepseek-ai/DeepSeek-V4-Flash"),
    )
    assert record.models[0].identifiers == record.identifiers
    assert record.releases == ()
    assert page.records[1].raw["deprecated"] is True


def test_catalog_rejects_duplicate_or_invalid_exact_ids() -> None:
    client = _Client(_payload(_item(), _item()))
    with pytest.raises(ValueError, match="duplicate model ID"):
        DeepInfraModelCatalogAdapter(client=client).fetch_page({})

    malformed = _item()
    malformed["full_name"] = "DeepSeek-V4-Flash"
    client = _Client(_payload(malformed))
    with pytest.raises(ValueError, match="invalid exact ID"):
        DeepInfraModelCatalogAdapter(client=client).fetch_page({})


def test_catalog_requires_complete_root_payload_and_rejects_paging_state() -> None:
    client = _Client(_payload(_item(), page=2))
    with pytest.raises(ValueError, match="first page"):
        DeepInfraModelCatalogAdapter(client=client).fetch_page({})

    client = _Client(_payload(_item()))
    with pytest.raises(ValueError, match="one complete catalog page"):
        DeepInfraModelCatalogAdapter(client=client).fetch_page({"page": 2})


def test_catalog_fails_closed_when_next_data_is_missing() -> None:
    class _NoPayloadClient(_Client):
        def __init__(self) -> None:
            self.body = b"<html><body>catalog changed</body></html>"
            self.calls = []

    with pytest.raises(ValueError, match="exactly one Next.js data payload"):
        DeepInfraModelCatalogAdapter(client=_NoPayloadClient()).fetch_page({})
