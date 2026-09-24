from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.markdown_model_table import MarkdownModelTableSourceAdapter

_PROPOSAL = (
    Path(__file__).parents[1] / "config/proposals/tensorflow_slim_mobilenet_v2_checkpoints.toml"
)
_REVISION = "1" * 40


class _Client:
    def __init__(self, document: str) -> None:
        self.document = document

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        body = (
            f'{{"sha":"{_REVISION}"}}'.encode()
            if "/commits/" in url
            else self.document.encode()
        )
        return HttpResponse(200, {}, body, url)


def test_proposal_maps_inline_mobile_v2_float_checkpoints_only() -> None:
    config = tomllib.loads(_PROPOSAL.read_text())
    assert len(config["source"]) == 1
    source = config["source"][0]
    assert source["enabled"] is False
    markdown = "\n".join(
        (
            "### Mobilenet V2 Imagenet Checkpoints",
            "Classification Checkpoint | Quantized | MACs (M) | Top 1 Accuracy",
            "--- | --- | --- | ---",
            "[float_v2_1.4_224](https://storage.googleapis.com/mobilenet_v2/"
            "checkpoints/mobilenet_v2_1.4_224.tgz) | [uint8][quantized_v2_1.4_224] | 582 | 75.0",
            "[float_v2_1.0_224](https://storage.googleapis.com/mobilenet_v2/"
            "checkpoints/mobilenet_v2_1.0_224.tgz) | [uint8][quantized_v2_1.0_224] | 300 | 71.8",
            "[quantized_v2_1.4_224]: https://storage.googleapis.com/mobilenet_v2/"
            "checkpoints/quantized_v2_224_140.tgz",
            "[quantized_v2_1.0_224]: https://storage.googleapis.com/mobilenet_v2/"
            "checkpoints/quantized_v2_224_100.tgz",
        )
    )
    adapter = MarkdownModelTableSourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        document_path=source["document_path"],
        provider_namespace=source["provider_namespace"],
        model_column=source["model_column"],
        model_header_pattern=source["model_header_pattern"],
        identity_include_heading=source["identity_include_heading"],
        max_response_bytes=source["max_response_bytes"],
        max_rows=source["max_rows"],
        client=_Client(markdown),
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    records = {record.title: record for record in page.records}
    assert set(records) == {"float_v2_1.4_224", "float_v2_1.0_224"}
    assert records["float_v2_1.4_224"].releases[0].metadata["artifacts"] == [
        {
            "url": "https://storage.googleapis.com/mobilenet_v2/checkpoints/"
            "mobilenet_v2_1.4_224.tgz",
            "relation": "weights",
        }
    ]
