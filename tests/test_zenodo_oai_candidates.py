from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

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


def oai_record(record_id: int, *, title: str, description: str, filename: str, url: str) -> str:
    return f"""<record>
  <header><identifier>oai:zenodo.org:{record_id}</identifier>
    <datestamp>2026-09-22T10:00:00Z</datestamp><setSpec>openaire_data</setSpec>
  </header>
  <metadata><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      xmlns:dcat="http://www.w3.org/ns/dcat#" xmlns:dct="http://purl.org/dc/terms/">
    <dcat:Dataset><dct:title>{title}</dct:title><dct:description>{description}</dct:description>
      <dct:type>Dataset</dct:type>
      <dcat:distribution><dcat:Distribution><dct:title>{filename}</dct:title>
        <dcat:downloadURL rdf:resource="{url}"/>
      </dcat:Distribution></dcat:distribution>
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
        "neural metadata plus named checkpoint/weights distribution"
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
