from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.paddleocr_ppstructure_model_list import (
    PaddleOcrPPStructureModelListSourceAdapter,
)

_REVISION = "c" * 40
_INFERENCE = "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/Foo_infer.tar"
_TRAINED = "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/Foo.pdparams"
_KIE_SER = "https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_pretrained.tar"
_KIE_RE = "https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/re_vi_layoutxlm_xfund_pretrained.tar"
_DOCUMENT = (
    """\
## 1. Layout Analysis

|model name| description | inference model size |download|dict path|
| --- |----| --- | --- | --- |
"""
    + f"| Foo | layout model | 9.7M | [inference model]({_INFERENCE}) / "
    + f"[trained model]({_TRAINED}) | dictionary |\n"
    + f"""

## 3. KIE

|Model|Backbone|Task|Config|Hmean|Time cost(ms)|Download link|
| --- | --- | --- | --- | --- | --- |--- |
|VI-LayoutXLM|base|SER|ser.yml|93.19%|15.49|[trained model]({_KIE_SER})|
|VI-LayoutXLM|base|RE|re.yml|83.92%|15.49|[trained model]({_KIE_RE})|
"""
)


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
        elif url.endswith("/docs/version2.x/ppstructure/models_list.en.md"):
            body = _DOCUMENT.encode()
        else:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(200, {}, body, url)


def test_ppstructure_table_keeps_layout_and_distinct_kie_checkpoint_rows() -> None:
    client = _Client()
    adapter = PaddleOcrPPStructureModelListSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 3
    assert all(record.kind is ArtifactKind.MODEL_CARD for record in page.records)
    identities = {record.models[0].identifiers[0] for record in page.records}
    assert identities == {
        Identifier("paddleocr:ppstructure-model", "1. Layout Analysis / Foo / 9.7M"),
        Identifier("paddleocr:ppstructure-model", "3. KIE / VI-LayoutXLM / SER"),
        Identifier("paddleocr:ppstructure-model", "3. KIE / VI-LayoutXLM / RE"),
    }
    assert {
        link.url
        for record in page.records
        for link in record.links
        if link.relation in {"weights", "inference_artifact"}
    } == {_INFERENCE, _TRAINED, _KIE_SER, _KIE_RE}
    assert client.calls[1].endswith(f"/{_REVISION}/docs/version2.x/ppstructure/models_list.en.md")


def test_ppstructure_disabled_proposal_matches_adapter() -> None:
    proposal_path = (
        Path(__file__).parents[1] / "config/proposals/paddleocr_ppstructure_model_list.toml"
    )
    source = tomllib.loads(proposal_path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = PaddleOcrPPStructureModelListSourceAdapter(
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
