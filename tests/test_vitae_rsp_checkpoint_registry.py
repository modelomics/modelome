from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.vitae_rsp_checkpoint_registry import (
    VitaeRSPCheckpointRegistrySourceAdapter,
    _parse_readme,
)

ROWS = {
    "RSP-ResNet-50-E300": "1K3P4_fDfcBRGqpKoSdSa6OXS4xC1xLC9",
    "RSP-Swin-T-E300": "1G5wjbjIHepmT6VVOuW03bWmyvrhcfe1F",
    "RSP-ViTAEv2-S-E100": "1cDB69frN-NxCyoy8lghjx6NiH1JriYUc",
}
README = "\n".join(
    (
        "# An Empirical Study of Remote Sensing Pretraining",
        "### MillionAID",
        "| Backbone | Input size | Pretrained model |",
        "| --- | --- | --- |",
        *(
            f"| {model_id} | 224 × 224 | "
            f"[google](https://drive.google.com/file/d/{file_id}/view?usp=sharing) & "
            "[baidu](https://pan.baidu.com/s/declared) |"
            for model_id, file_id in ROWS.items()
        ),
        "### Usage",
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


def response(body: str, status: int = 200) -> HttpResponse:
    return HttpResponse(status, {"content-type": "text/plain"}, body.encode(), "https://fixture.test")


def test_adapter_emits_exact_three_google_drive_checkpoint_urls() -> None:
    client = Client(response(README))
    page = VitaeRSPCheckpointRegistrySourceAdapter(client=client).fetch_page({})
    observed = {
        link.url
        for record in page.records
        for link in record.links
        if link.relation == "weights"
    }
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 3
    assert observed == {
        f"https://drive.google.com/file/d/{file_id}/view" for file_id in ROWS.values()
    }
    assert client.urls == [
        "https://raw.githubusercontent.com/ViTAE-Transformer/RSP/main/README.md"
    ]
    assert len({record.models[0].local_id for record in page.records}) == 3


def test_adapter_skips_unchanged_readme_after_fetching_it() -> None:
    client = Client(response(README), response(README))
    adapter = VitaeRSPCheckpointRegistrySourceAdapter(client=client)
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    assert len(first.records) == 3
    assert second.records == ()
    assert second.next_state == first.next_state
    assert len(client.urls) == 2


@pytest.mark.parametrize(
    "document",
    [
        README.replace(ROWS["RSP-ResNet-50-E300"], "wrong-id"),
        README.replace("RSP-ViTAEv2-S-E100", "RSP-ViTAEv2-S-E200"),
        README.replace("### Usage", f"{README.splitlines()[4]}\n### Usage"),
        README + "\n### MillionAID\n",
    ],
)
def test_parser_rejects_changed_or_ambiguous_checkpoint_table(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_readme(document, "test")


def test_adapter_rejects_http_errors_and_oversized_readme() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        VitaeRSPCheckpointRegistrySourceAdapter(client=Client(response("", 503))).fetch_page({})
    with pytest.raises(ValueError, match="byte limit"):
        VitaeRSPCheckpointRegistrySourceAdapter(
            client=Client(response(README)), max_response_bytes=len(README) - 1
        ).fetch_page({})
