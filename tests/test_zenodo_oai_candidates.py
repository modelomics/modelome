from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.zenodo_oai_candidates import ZenodoOaiModelCandidatesSourceAdapter

URL = "https://zenodo.org/oai2d"


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def response(xml: str) -> HttpResponse:
    return HttpResponse(200, {"content-type": "application/xml"}, xml.encode(), URL)


def harvest_page(content: str, token: str = "") -> str:
    token_xml = f"<resumptionToken>{token}</resumptionToken>" if token else ""
    return f"""<?xml version="1.0"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>{content}{token_xml}</ListRecords>
</OAI-PMH>"""


def oai_record(
    record_id: int,
    *,
    title: str,
    description: str,
    filename: str,
    url: str,
    url_shape: str = "download_resource",
) -> str:
    if url_shape == "distribution_about":
        distribution = f"""<rdf:Description rdf:about="{url}">
        <rdf:type rdf:resource="http://www.w3.org/ns/dcat#Distribution"/>
        <dct:title>{filename}</dct:title>
      </rdf:Description>"""
    elif url_shape == "distribution_resource":
        distribution = f'<dcat:distribution rdf:resource="{url}"/>'
    elif url_shape == "download_text":
        distribution = f"""<dcat:Distribution><dct:title>{filename}</dct:title>
        <dcat:downloadURL>{url}</dcat:downloadURL>
      </dcat:Distribution>"""
    else:
        distribution = f"""<dcat:Distribution><dct:title>{filename}</dct:title>
        <dcat:downloadURL rdf:resource="{url}"/>
      </dcat:Distribution>"""
    return f"""<record>
  <header><identifier>oai:zenodo.org:{record_id}</identifier>
    <datestamp>2026-09-22T10:00:00Z</datestamp><setSpec>openaire_data</setSpec>
  </header>
  <metadata><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      xmlns:dcat="http://www.w3.org/ns/dcat#" xmlns:dct="http://purl.org/dc/terms/">
    <dcat:Dataset><dct:title>{title}</dct:title><dct:description>{description}</dct:description>
      <dct:type>Dataset</dct:type>
      <dcat:distribution>{distribution}</dcat:distribution>
    </dcat:Dataset>
  </rdf:RDF></metadata>
</record>"""


def test_oai_dcat_discovers_candidate_checkpoint_and_resumes_with_opaque_token() -> None:
    candidate = oai_record(
        123,
        title="OceanNet dataset",
        description="A dataset and evaluation of a neural network model.",
        filename="oceannet_checkpoint.safetensors",
        url="https://zenodo.org/api/files/abc/oceannet_checkpoint.safetensors?download=1",
    )
    non_candidate = oai_record(
        124,
        title="OceanNet data",
        description="A dataset used to train neural network models.",
        filename="training_examples.csv",
        url="https://zenodo.org/api/files/abc/training_examples.csv",
    )
    client = Client(
        response(harvest_page(candidate + non_candidate, "opaque+token/2")),
        response(harvest_page("")),
    )
    now = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)
    adapter = ZenodoOaiModelCandidatesSourceAdapter(client=client, clock=lambda: now)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert client.calls == [
        (URL, {"verb": "ListRecords", "metadataPrefix": "dcat"}),
        (URL, {"verb": "ListRecords", "resumptionToken": "opaque+token/2"}),
    ]
    assert len(first.records) == 1
    record = first.records[0]
    assert record.kind is ArtifactKind.CATALOG_RECORD
    assert record.source_record_id == "record:123"
    assert len(record.models) == 1
    assert record.models[0].name == "OceanNet"
    assert record.models[0].status.value == "candidate"
    assert record.raw["candidate_signal"] == (
        "neural metadata plus named model/checkpoint/weights distribution"
    )
    assert any(
        link.relation == "checkpoint"
        and urlsplit(link.url).path.endswith("oceannet_checkpoint.safetensors")
        and not link.crawl
        and link.model_local_ids == (record.models[0].local_id,)
        for link in record.links
    )
    assert first.next_state["pages_seen"] == 1
    assert second.complete is True
    assert second.next_state["records_seen"] == 2


def test_punctuation_distinct_checkpoint_stems_keep_distinct_candidate_ids() -> None:
    xml = harvest_page(
        """<record>
  <header><identifier>oai:zenodo.org:125</identifier>
    <datestamp>2026-09-22T10:00:00Z</datestamp><setSpec>openaire_data</setSpec>
  </header>
  <metadata><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      xmlns:dcat="http://www.w3.org/ns/dcat#" xmlns:dct="http://purl.org/dc/terms/">
    <dcat:Dataset><dct:title>Alpha-Net and Alpha.Net neural network models</dct:title>
      <dct:description>Separate neural network checkpoint models Alpha-Net
        and Alpha.Net.</dct:description>
      <dct:type>Dataset</dct:type>
      <dcat:distribution><dcat:Distribution><dct:title>Alpha-Net checkpoint</dct:title>
        <dcat:downloadURL rdf:resource="https://zenodo.org/api/files/abc/Alpha-Net_checkpoint.safetensors"/>
      </dcat:Distribution></dcat:distribution>
      <dcat:distribution><dcat:Distribution><dct:title>Alpha.Net weights</dct:title>
        <dcat:downloadURL rdf:resource="https://zenodo.org/api/files/abc/Alpha.Net_weights.ckpt"/>
      </dcat:Distribution></dcat:distribution>
    </dcat:Dataset>
  </rdf:RDF></metadata>
</record>"""
    )
    adapter = ZenodoOaiModelCandidatesSourceAdapter(
        client=Client(response(xml)),
        clock=lambda: datetime(2026, 9, 22, 12, 0, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert len(page.records[0].models) == 2
    # Both context matches can resolve to the same display spelling, but the
    # source's distinct file stems must remain distinct candidate identities.
    assert len({model.local_id for model in page.records[0].models}) == 2


@pytest.mark.parametrize(
    ("filename", "url_shape"),
    [
        ("OceanNet_model_weights.bin", "distribution_about"),
        ("OceanNet_weights.gguf", "download_text"),
        ("OceanNet_model_weights.bin", "distribution_resource"),
        ("OceanNet_checkpoint.keras", "download_resource"),
        ("OceanNet_checkpoint.msgpack", "download_resource"),
        ("OceanNet_checkpoint.pth.tar", "download_resource"),
        ("OceanNet.tflite", "download_resource"),
        ("OceanNet.mlmodel", "download_resource"),
        ("OceanNet.pt2", "download_resource"),
        ("OceanNet.gguf", "download_resource"),
        ("OceanNet.keras", "download_resource"),
    ],
)
def test_dcat_distribution_shapes_and_documented_weight_formats(
    filename: str, url_shape: str
) -> None:
    url = f"https://zenodo.org/records/126/files/{filename}?download=1"
    xml = harvest_page(
        oai_record(
            126,
            title="OceanNet model",
            description="OceanNet is a deep learning neural network model.",
            filename=filename,
            url=url,
            url_shape=url_shape,
        )
    )
    adapter = ZenodoOaiModelCandidatesSourceAdapter(
        client=Client(response(xml)),
        clock=lambda: datetime(2026, 9, 22, 12, 0, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert len(page.records) == 1
    assert [model.name for model in page.records[0].models] == ["OceanNet"]
    assert any(link.relation == "checkpoint" for link in page.records[0].links)


def test_dcat_distribution_subject_requires_zenodo_file_path_and_model_suffix() -> None:
    unrelated = oai_record(
        127,
        title="OceanNet neural network model",
        description="OceanNet uses a neural network.",
        filename="OceanNet_weights.bin",
        url="https://zenodo.org/records/127",
        url_shape="distribution_about",
    )
    no_model_extension = oai_record(
        128,
        title="OceanNet neural network model",
        description="OceanNet uses a neural network.",
        filename="OceanNet_weights.csv",
        url="https://zenodo.org/records/128/files/OceanNet_weights.csv",
        url_shape="distribution_about",
    )
    unmarked_generic_binary = oai_record(
        129,
        title="OceanNet neural network model",
        description="OceanNet uses a neural network.",
        filename="OceanNet.bin",
        url="https://zenodo.org/records/129/files/OceanNet.bin",
        url_shape="distribution_about",
    )
    page_xml = harvest_page(unrelated + no_model_extension + unmarked_generic_binary)
    adapter = ZenodoOaiModelCandidatesSourceAdapter(
        client=Client(response(page_xml)),
        clock=lambda: datetime(2026, 9, 22, 12, 0, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.records == ()


def test_oai_dcat_rejects_expired_token_checkpoint() -> None:
    now = datetime(2026, 9, 22, 12, 2, tzinfo=UTC)
    adapter = ZenodoOaiModelCandidatesSourceAdapter(clock=lambda: now)

    try:
        adapter.fetch_page(
            {
                "resumption_token": "old-token",
                "token_issued_at": "2026-09-22T12:00:00Z",
            }
        )
    except ValueError as error:
        assert "restart the harvest" in str(error)
    else:
        raise AssertionError("expected stale OAI-PMH token to be rejected")
