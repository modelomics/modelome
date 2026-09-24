from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import unquote

import pytest

from modelome.http import HttpResponse
from modelome.sources.weathernext2_checkpoint_registry import (
    _BUCKET_PREFIX,
    WeatherNext2CheckpointRegistrySourceAdapter,
    _parse_readme,
)

REV = "b" * 40
PATTERNS = (
    "WeatherNext2_<2025_model{1,2,3,4}.npz",
    "WeatherNextCyclones_<2025_model{1,2,3,4}.npz",
    "WeatherNextCyclones_<2024_model{1,2,3,4}.npz",
    "WeatherNextCyclones_<2023_model{1,2,3,4}.npz",
    "WeatherNextCyclones_Mini_<2024.npz",
    "WeatherNextCyclones_Mini_<2023.npz",
)
README = "\n".join(f"Weights: `{pattern}`" for pattern in PATTERNS)


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


def test_weathernext_readme_expands_only_its_18_declared_checkpoint_paths() -> None:
    checkpoints = _parse_readme(README, "test", "README.md")
    filenames = {unquote(item.url.rsplit("/", 1)[-1]) for item in checkpoints}

    expected = {
        *(f"WeatherNext2_<2025_model{seed}.npz" for seed in range(1, 5)),
        *(
            f"WeatherNextCyclones_<{year}_model{seed}.npz"
            for year in (2023, 2024, 2025)
            for seed in range(1, 5)
        ),
        "WeatherNextCyclones_Mini_<2024.npz",
        "WeatherNextCyclones_Mini_<2023.npz",
    }
    assert filenames == expected
    assert len({item.handle for item in checkpoints}) == 18
    assert all(item.url.startswith(_BUCKET_PREFIX) for item in checkpoints)


def test_adapter_emits_file_urls_without_listing_or_fetching_checkpoint_bytes() -> None:
    client = Client(response({"sha": REV}), response(README))
    adapter = WeatherNext2CheckpointRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot and page.upstream_count == 18
    assert len({record.models[0].local_id for record in page.records}) == 18
    assert all(
        link.url.startswith(_BUCKET_PREFIX)
        for record in page.records
        for link in record.links
        if link.relation == "weights"
    )
    assert len(client.urls) == 2
    assert client.urls[1].endswith(f"/{REV}/README.md")


@pytest.mark.parametrize(
    "document",
    [
        README.replace(PATTERNS[0], "WeatherNext2_<2025_model{1,2,3}.npz"),
        README.replace(PATTERNS[1], PATTERNS[0]),
        README.replace(PATTERNS[2], "WeatherNextCyclones_<2022_model{1,2,3,4}.npz"),
        README.replace(PATTERNS[5], "WeatherNextCyclones_Mini_<2023.zip"),
    ],
)
def test_weathernext_parser_rejects_unexpected_or_incomplete_patterns(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_readme(document, "test", "README.md")
