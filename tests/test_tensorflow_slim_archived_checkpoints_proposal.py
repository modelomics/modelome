from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.markdown_model_table import MarkdownModelTableSourceAdapter

_PROPOSAL = Path(__file__).parents[1] / "config/proposals/tensorflow_slim_archived_checkpoints.toml"
_REVISION = "f" * 40


class _Client:
    def __init__(self, document: str) -> None:
        self.document = document
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        body = (
            f'{{"sha":"{_REVISION}"}}'.encode()
            if "/commits/" in url
            else self.document.encode()
        )
        return HttpResponse(200, {}, body, url)


def _proposal_sources() -> list[dict[str, Any]]:
    sources = tomllib.loads(_PROPOSAL.read_text())["source"]
    assert len(sources) == 2
    assert all(source["enabled"] is False for source in sources)
    return sources


def _adapter(source: dict[str, Any], document: str, client: _Client):
    return MarkdownModelTableSourceAdapter(
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
        client=client,
    )


def test_slim_mobile_v1_table_emits_float_and_quantized_archive_checkpoints() -> None:
    source = _proposal_sources()[0]
    document = "\n".join(
        (
            "# MobileNet V1",
            "## Pre-trained Models",
            "| Model | Million MACs | Top-1 |",
            "| :----: | :---------: | :---: |",
            "| [MobileNet_v1_1.0_224](http://download.tensorflow.org/models/"
            "mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz) | 569 | 70.9 |",
            "| [MobileNet_v1_1.0_224_quant](http://download.tensorflow.org/models/"
            "mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) | 569 | 70.1 |",
        )
    )
    client = _Client(document)

    page = _adapter(source, document, client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    records = {record.title: record for record in page.records}
    assert set(records) == {"MobileNet_v1_1.0_224", "MobileNet_v1_1.0_224_quant"}
    assert records["MobileNet_v1_1.0_224"].releases[0].metadata["artifacts"] == [
        {
            "url": (
                "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/"
                "mobilenet_v1_1.0_224.tgz"
            ),
            "relation": "weights",
        }
    ]
    assert len(client.calls) == 2


def test_slim_nasnet_table_emits_exact_first_party_checkpoint_archives() -> None:
    source = _proposal_sources()[1]
    document = "\n".join(
        (
            "# Pre-Trained Models",
            "| Model Checkpoint | Million MACs | Top-1 |",
            "| :----: | :---------: | :---: |",
            "| [NASNet-A_Mobile_224](https://storage.googleapis.com/download.tensorflow.org/"
            "models/nasnet-a_mobile_04_10_2017.tar.gz) | 564 | 74.0 |",
            "| [NASNet-A_Large_331](https://storage.googleapis.com/download.tensorflow.org/"
            "models/nasnet-a_large_04_10_2017.tar.gz) | 23800 | 82.7 |",
        )
    )
    client = _Client(document)

    page = _adapter(source, document, client).fetch_page({})

    assert page.complete and page.upstream_count == 2
    assert {record.title for record in page.records} == {
        "NASNet-A_Mobile_224",
        "NASNet-A_Large_331",
    }
    assert all(
        record.releases[0].metadata["artifacts"][0]["relation"] == "weights"
        for record in page.records
    )
