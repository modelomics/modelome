from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.fengwu_checkpoint_registry import (
    FengWuCheckpointRegistrySourceAdapter,
    _parse_readme,
)

REV = "e" * 40
V1 = (
    "https://pjlab-my.sharepoint.cn/:u:/g/personal/chenkang_pjlab_org_cn/"
    "EVA6V_Qkp6JHgXwAKxXIzDsBPIddo5RgDtGCBQ-sQbMmwg"
)
V2 = (
    "https://pjlab-my.sharepoint.cn/:u:/g/personal/chenkang_pjlab_org_cn/"
    "EZkFM7nQcEtBve6MsqlWaeIB_lmpa__hX0I8QYOPzf-X6A"
)
README = "\n".join(
    (
        "# FengWu",
        "",
        "## Downloading trained models",
        "",
        f"Fengwu without transfer learning (fengwu_v1.onnx): [Onedrive({V1})]",
        "",
        "Fengwu with transfer learning (fengwu_v2.onnx, "
        f"finetune the model with analysis data up to 2021): [Onedrive({V2})]",
        "",
        "## Data Format",
        "Coordinates and variables.",
    )
)


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.urls.append(url)
        return self.responses.pop(0)


def response(value: Any) -> HttpResponse:
    body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
    return HttpResponse(200, {}, body, "https://fixture.test")


def test_official_fengwu_readme_emits_two_exact_share_pages() -> None:
    client = Client(response({"sha": REV}), response(README))
    adapter = FengWuCheckpointRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot and page.upstream_count == 2
    assert page.records[0].identifiers == (Identifier("fengwu:checkpoint", "fengwu_v1.onnx"),)
    assert page.records[0].models[0].name == "FengWu v1"
    assert any(
        link.relation == "checkpoint_download_page" and link.url == V1
        for link in page.records[0].links
    )
    assert page.records[1].releases[0].metadata["checkpoint_share_url"] == V2
    assert not any(link.relation == "weights" for link in page.records[0].links)
    assert client.urls[1].endswith(f"/{REV}/README.md")


@pytest.mark.parametrize(
    "document",
    [
        README.replace("fengwu_v2.onnx", "fengwu_v3.onnx"),
        README.replace(V1, "https://example.org/fengwu_v1.onnx"),
        README.replace("## Data Format", "## Downloading trained models\n\n## Data Format"),
    ],
)
def test_fengwu_readme_rejects_unexpected_or_ambiguous_inventory(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_readme(document, "test", "README.md")
