from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.paddlex_model_list import PaddleXModelListSourceAdapter

_REVISION = "b" * 40
_INFERENCE = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/Foo_infer.tar"
_TRAINING = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/Foo_pretrained.pdparams"
_DOCUMENT = f"""\
## Image Classification Module

Model Name | Top-1 Acc (%) | Model Download Link
--- | --- | ---
Foo | 91.2 | [Inference Model]({_INFERENCE}) / [Training Model]({_TRAINING})
Bar | 80.1 | N/A
"""


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if "/commits/" in url:
            body = json.dumps({"sha": _REVISION}).encode()
        elif url.endswith("/docs/support_list/models_list.en.md"):
            body = _DOCUMENT.encode()
        else:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(200, {}, body, url)


def test_paddlex_model_list_extracts_only_exact_row_declared_artifact_urls() -> None:
    client = _Client()
    adapter = PaddleXModelListSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.models[0].identifiers == (
        Identifier("paddlex:model", "Image Classification Module / Foo"),
    )
    artifact_urls = {
        link.url
        for link in record.links
        if link.relation in {"weights", "inference_artifact"}
    }
    assert artifact_urls == {_INFERENCE, _TRAINING}
    assert record.releases[0].metadata["artifacts"] == [
        {"url": _INFERENCE, "relation": "inference_artifact"},
        {"url": _TRAINING, "relation": "weights"},
    ]
    assert client.calls[1].endswith(f"/{_REVISION}/docs/support_list/models_list.en.md")


def test_paddlex_model_list_disabled_proposal_matches_adapter() -> None:
    proposal_path = Path(__file__).parents[1] / "config/proposals/paddlex_model_list.toml"
    source = tomllib.loads(proposal_path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = PaddleXModelListSourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        source_path=source["source_path"],
        max_response_bytes=source["max_response_bytes"],
        max_rows=source["max_rows"],
    )
    assert adapter.name == source["name"]
    assert adapter.repository == source["repository"]
    assert adapter.branch == source["branch"]
    assert adapter.document_path == source["source_path"]
