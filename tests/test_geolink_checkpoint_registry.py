from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.geolink_checkpoint_registry import (
    GeoLinkCheckpointRegistrySourceAdapter,
    _parse_readme,
)

FILES = {
    "unimodal": ("geolink_vit_large_patch16_224.pth", "12u0goOohBYHjlkKIs11bVeTYyscd2Z9g"),
    "multimodal": (
        "geolink_mutimodal_vit_large_patch16_224.pth",
        "1bAeNurdrH9nEI7qzwNWyLBWBSZnfwj_9",
    ),
}
README = "\n".join(
    (
        "# GeoLink",
        "## 🏋️‍♂️ Pre-trained Models",
        "We provide two pretrained models.",
        "### 1. Unimodal GeoLink(ViT)",
        "- **Download**: [PKU Disk Link](https://disk.pku.edu.cn/link/example) or "
        f"[Google Drive](https://drive.google.com/file/d/{FILES['unimodal'][1]}/view?usp=drive_link)",
        "### 2. Multimodal GeoLink",
        "- **Download**: [PKU Disk Link](https://disk.pku.edu.cn/link/example) or "
        f"[Google Drive](https://drive.google.com/file/d/{FILES['multimodal'][1]}/view?usp=drive_link)",
        "## Installation",
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


def test_adapter_emits_two_exact_first_party_google_drive_checkpoints() -> None:
    client = Client(response(README))
    page = GeoLinkCheckpointRegistrySourceAdapter(client=client).fetch_page({})
    observed = {
        record.raw["modality"]: (record.raw["filename"], record.canonical_url)
        for record in page.records
    }
    assert page.complete and page.authoritative_snapshot and page.upstream_count == 2
    assert observed == {
        kind: (filename, f"https://drive.google.com/file/d/{file_id}/view")
        for kind, (filename, file_id) in FILES.items()
    }
    assert client.urls == [
        "https://raw.githubusercontent.com/bailubin/GeoLink_NeurIPS2025/main/README.md"
    ]


def test_adapter_skips_unchanged_readme_after_fetching() -> None:
    client = Client(response(README), response(README))
    adapter = GeoLinkCheckpointRegistrySourceAdapter(client=client)
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    assert len(first.records) == 2
    assert second.records == ()
    assert second.next_state == first.next_state
    assert len(client.urls) == 2


@pytest.mark.parametrize(
    "document",
    [
        README.replace(FILES["unimodal"][1], "wrong-id"),
        README.replace("### 2. Multimodal GeoLink", "### 2. Other model"),
        README.replace("### 1. Unimodal GeoLink(ViT)", "### 1. Other model"),
        README + "\n## 🏋️‍♂️ Pre-trained Models\n",
    ],
)
def test_parser_rejects_changed_or_ambiguous_checkpoint_links(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_readme(document, "test")


def test_adapter_rejects_http_errors_and_oversized_readme() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        GeoLinkCheckpointRegistrySourceAdapter(client=Client(response("", 503))).fetch_page({})
    with pytest.raises(ValueError, match="byte limit"):
        GeoLinkCheckpointRegistrySourceAdapter(
            client=Client(response(README)), max_response_bytes=len(README) - 1
        ).fetch_page({})
