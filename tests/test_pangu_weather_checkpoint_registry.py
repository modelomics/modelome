from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.pangu_weather_checkpoint_registry import (
    PanguWeatherCheckpointRegistrySourceAdapter,
    _parse_readme,
)

REV = "c" * 40
README = "\n".join(
    (
        "## Pangu-Weather",
        "#### Downloading trained models",
        "The four models are linked below:",
        "The 1-hour model (pangu_weather_1.onnx): [Google drive]("
        "https://drive.google.com/file/d/one_1/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1)",
        "The 3-hour model (pangu_weather_3.onnx): [Google drive]("
        "https://drive.google.com/file/d/three_3/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/3)",
        "The 6-hour model (pangu_weather_6.onnx): [Google drive]("
        "https://drive.google.com/file/d/six_6/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/6)",
        "The 24-hour model (pangu_weather_24.onnx): [Google drive]("
        "https://drive.google.com/file/d/twenty_four/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/24)",
        "#### Input data preparation",
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


def test_official_pangu_readme_emits_four_exact_share_page_locations() -> None:
    client = Client(response({"sha": REV}), response(README))
    adapter = PanguWeatherCheckpointRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot and page.upstream_count == 4
    record = page.records[0]
    assert record.identifiers == (
        Identifier("pangu-weather:checkpoint", "pangu_weather_1.onnx"),
    )
    assert any(
        link.relation == "checkpoint_download_page"
        and link.url == "https://drive.google.com/file/d/one_1/view?usp=share_link"
        for link in record.links
    )
    assert record.releases[0].metadata["checkpoint_share_url"].endswith("one_1/view?usp=share_link")
    assert not any(link.relation == "weights" for link in record.links)
    assert client.urls[1].endswith(f"/{REV}/README.md")


@pytest.mark.parametrize(
    "document",
    [
        "#### Downloading trained models\nThe 1-hour model (pangu_weather_1.onnx): [Google drive](https://drive.google.com/file/d/id/view?usp=share_link)",
        README.replace("pangu_weather_24.onnx", "pangu_weather_12.onnx"),
        README.replace("https://drive.google.com/file/d/six_6/view?usp=share_link", "https://example.org/six.onnx"),
    ],
)
def test_pangu_readme_rejects_incomplete_or_unexpected_inventory(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_readme(document, "test", "README.md")
