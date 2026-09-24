from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.seco_checkpoint_registry import (
    SeCoCheckpointRegistrySourceAdapter,
    _parse_readme,
)

FILES = (
    "seco_resnet18_100k.ckpt",
    "seco_resnet18_1m.ckpt",
    "seco_resnet50_100k.ckpt",
    "seco_resnet50_1m.ckpt",
)
README = "\n".join(
    (
        "# Seasonal Contrast",
        "### Pre-trained Models",
        "dataset | architecture | link | md5",
        "--- | --- | --- | ---",
        *(
            f"SeCo | ResNet | https://zenodo.org/record/4728033/files/{name}?download=1 | checksum"
            for name in FILES
        ),
        "## About",
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


def test_adapter_emits_exact_four_readme_zenodo_checkpoint_links() -> None:
    client = Client(response(README))
    page = SeCoCheckpointRegistrySourceAdapter(client=client).fetch_page({})
    observed = {
        link.url
        for record in page.records
        for link in record.links
        if link.relation == "weights"
    }
    assert page.authoritative_snapshot and page.complete and page.upstream_count == 4
    assert observed == {
        f"https://zenodo.org/record/4728033/files/{name}?download=1" for name in FILES
    }
    assert {record.raw["filename"] for record in page.records} == set(FILES)
    assert {record.raw["md5"] for record in page.records} == {
        "dcf336be31f6c6b0e77dcb6cc958fca8",
        "53d5c41d0f479bdfd31d6746ad4126db",
        "9672c303f6334ef816494c13b9d05753",
        "7b09c54aed33c0c988b425c54f4ef948",
    }
    assert client.urls == [
        "https://raw.githubusercontent.com/ServiceNow/seasonal-contrast/main/README.md"
    ]


def test_adapter_uses_readme_hash_for_freshness() -> None:
    client = Client(response(README), response(README))
    adapter = SeCoCheckpointRegistrySourceAdapter(client=client)
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    assert len(first.records) == 4
    assert second.records == ()
    assert second.next_state == first.next_state


@pytest.mark.parametrize(
    "document",
    [
        README.replace("record/4728033", "record/9999999", 1),
        README.replace(FILES[0], "unexpected.ckpt", 1),
        README.replace(FILES[0], FILES[1], 1),
        README.replace(FILES[3] + "?download=1", "other text"),
        README + "\n### Pre-trained Models\n",
    ],
)
def test_parser_rejects_wrong_missing_or_ambiguous_links(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_readme(document, "test")


def test_adapter_rejects_failed_readme_response() -> None:
    adapter = SeCoCheckpointRegistrySourceAdapter(client=Client(response(README, 503)))
    with pytest.raises(ValueError, match="HTTP 503"):
        adapter.fetch_page({})
