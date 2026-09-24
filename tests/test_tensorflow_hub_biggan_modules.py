from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter


class QueuedClient:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        self.calls.append(url)
        return self.response


def _adapter(body: str) -> HtmlCatalogSourceAdapter:
    proposal_path = (
        Path(__file__).parents[1]
        / "config"
        / "proposals"
        / "tensorflow_hub_biggan_modules.toml"
    )
    source = tomllib.loads(proposal_path.read_text())["source"][0]
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={"Content-Type": "text/html"},
            body=body.encode(),
            url=source["url"],
        )
    )
    return HtmlCatalogSourceAdapter(
        name=source["name"],
        url=source["url"],
        provider_namespace=source["provider_namespace"],
        rules=source["rules"],
        artifact_kind=source["artifact_kind"],
        model_status=source["model_status"],
        allowed_origins=source["allowed_origins"],
        max_response_bytes=source["max_response_bytes"],
        max_entries=source["max_entries"],
        client=client,
    )


def test_official_biggan_index_captures_six_exact_versioned_handles() -> None:
    body = """<html><head><title>Generating Images with BigGAN</title></head><body>
    # module_path = 'https://tfhub.dev/deepmind/biggan-deep-128/1'
    module_path = 'https://tfhub.dev/deepmind/biggan-deep-256/1'
    # module_path = 'https://tfhub.dev/deepmind/biggan-deep-512/1'
    # module_path = 'https://tfhub.dev/deepmind/biggan-128/2'
    # module_path = 'https://tfhub.dev/deepmind/biggan-256/2'
    # module_path = 'https://tfhub.dev/deepmind/biggan-512/2'
    # Similar, but not an exact declared handle:
    # https://tfhub.dev/deepmind/biggan-512/2-preview
    # https://tfhub.dev/deepmind/biggan-512/2/extra
    <a href="https://tfhub.dev/deepmind/biggan-deep-256/1">active module</a>
    </body></html>"""

    page = _adapter(body).fetch_page({})

    assert page.complete is True
    assert page.upstream_count == 6
    assert len(page.records) == 1
    record = page.records[0]
    assert {model.name for model in record.models} == {
        "DeepMind biggan-deep-128",
        "DeepMind biggan-deep-256",
        "DeepMind biggan-deep-512",
        "DeepMind biggan-128",
        "DeepMind biggan-256",
        "DeepMind biggan-512",
    }
    assert {link.url for link in record.links} == {
        f"https://tfhub.dev/deepmind/{model}/{version}"
        for model, version in (
            ("biggan-deep-128", "1"),
            ("biggan-deep-256", "1"),
            ("biggan-deep-512", "1"),
            ("biggan-128", "2"),
            ("biggan-256", "2"),
            ("biggan-512", "2"),
        )
    }
    assert len({model.identifiers for model in record.models}) == 6


def test_official_biggan_index_does_not_infer_neighboring_module_names() -> None:
    body = """<html><body>
    https://tfhub.dev/deepmind/bigbigan-resnet50/1
    https://tfhub.dev/deepmind/biggan-deep-1024/1
    https://tfhub.dev/other/biggan-deep-128/1
    https://tfhub.dev/deepmind/biggan-128/3
    </body></html>"""

    with pytest.raises(ValueError, match="matched no catalog entries"):
        _adapter(body).fetch_page({})
