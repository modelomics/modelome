from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.figshare_model_candidates import (
    FigshareModelCandidatesSourceAdapter,
)

OAI = "https://api.figshare.com/v2/oai"
DETAIL = "https://api.figshare.com/v2/articles/19105067"
VERSIONS = f"{DETAIL}/versions"


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def response(
    body: str | bytes,
    url: str = OAI,
    content_type: str = "application/xml",
) -> HttpResponse:
    encoded = body.encode() if isinstance(body, str) else body
    return HttpResponse(200, {"content-type": content_type}, encoded, url)


def wssnet_record() -> str:
    return """<record>
      <header><identifier>oai:figshare.com:article/19105067</identifier></header>
      <metadata><mets:mets xmlns:mets="http://www.loc.gov/METS/"
          xmlns:dc="http://purl.org/dc/elements/1.1/"
          xmlns:dcterms="http://purl.org/dc/terms/">
        <mets:dmdSec><mets:mdWrap><mets:xmlData>
          <dc:title>WSSNet: aortic 4D Flow MRI wall shear stress
            estimation neural network</dc:title>
          <dc:description>&lt;p&gt;A deep learning neural network. Pre-trained weights:&lt;/p&gt;
            &lt;p&gt;1, wssnet: contains the network weights and optimisation
            parameters to restore/continue training&lt;/p&gt;
            &lt;p&gt;2. wssnet_dataset: contains the dataset files used for
            training the network&lt;/p&gt;</dc:description>
          <dc:relation>https://figshare.com/articles/software/WSSNet/19105067</dc:relation>
          <dcterms:hasVersion>2</dcterms:hasVersion>
          <dcterms:hasPart>https://ndownloader.figshare.com/files/33947396</dcterms:hasPart>
          <dcterms:hasPart>https://ndownloader.figshare.com/files/33947432</dcterms:hasPart>
          <dcterms:hasPart>https://ndownloader.figshare.com/files/34002611</dcterms:hasPart>
        </mets:xmlData></mets:mdWrap></mets:dmdSec>
        <mets:fileSec><mets:fileGrp>
          <mets:file><mets:FLocat xlink:href="https://ndownloader.figshare.com/files/33947396" xmlns:xlink="http://www.w3.org/1999/xlink"/></mets:file>
          <mets:file><mets:FLocat xlink:href="https://ndownloader.figshare.com/files/33947432" xmlns:xlink="http://www.w3.org/1999/xlink"/></mets:file>
          <mets:file><mets:FLocat xlink:href="https://ndownloader.figshare.com/files/34002611" xmlns:xlink="http://www.w3.org/1999/xlink"/></mets:file>
        </mets:fileGrp></mets:fileSec>
      </mets:mets></metadata>
    </record>"""


def list_records(records: str, token: str = "") -> str:
    token_xml = (
        f'<resumptionToken expirationDate="2026-09-24T11:00:00Z">{token}</resumptionToken>'
        if token
        else ""
    )
    return f"""<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>{records}{token_xml}</ListRecords></OAI-PMH>"""


ARTICLE = {
    "id": 19105067,
    "version": 2,
    "title": "WSSNet: aortic 4D Flow MRI wall shear stress estimation neural network",
    "description": (
        "<p>A deep learning neural network. Pre-trained weights:</p>"
        "<p>1, wssnet: contains the network weights and optimisation parameters "
        "to restore/continue training</p>"
        "<p>2. wssnet_dataset: contains the dataset files used for training "
        "the network</p>"
    ),
    "url_public_html": "https://auckland.figshare.com/articles/software/WSSNet/19105067/2",
    "doi": "10.17608/k6.auckland.19105067.v2",
    "defined_type_name": "software",
    "files": [
        {
            "id": 33947396,
            "name": "wssnet_dataset.zip",
            "download_url": "https://ndownloader.figshare.com/files/33947396",
        },
        {
            "id": 33947432,
            "name": "wssnet.zip",
            "download_url": "https://ndownloader.figshare.com/files/33947432",
        },
        {
            "id": 34002611,
            "name": "examples_csv_wssnet.zip",
            "download_url": "https://ndownloader.figshare.com/files/34002611",
        },
    ],
}

ARTICLE_V1 = {
    **ARTICLE,
    "version": 1,
    "url_public_html": (
        "https://auckland.figshare.com/articles/software/"
        "WSSNet_aortic_4D_Flow_MRI_wall_shear_stress_estimation_neural_network/19105067/1"
    ),
    "doi": "10.17608/k6.auckland.19105067.v1",
    "description": (
        "We developed WSSNet, a deep learning method to estimate aortic wall shear "
        "stress based on geometry and velocity from 4D Flow MRI.<div>The network was "
        "trained on synthetic data, generated from CFD simulations. </div><div><br></div>"
        "<div>The input and output of the network consists of flattened representation "
        "of the aortic vessel surface, translated into velocity sheets and coordinate "
        "flatmaps. </div><div><br></div><div>Training dataset and pre-trained weights "
        "are provided here.</div><div>The network was trained using Tensorflow 2.3 "
        "with Keras backend.</div><div><br></div><div>Files:</div><div>1, wssnet: "
        "contains the network weights and optimisation parameters to restore/continue "
        "training</div><div>2. wssnet_dataset: contains the dataset files used for "
        "training the network</div>"
    ),
    "files": [
        {
            "id": 33947396,
            "name": "wssnet_dataset.zip",
            "download_url": "https://ndownloader.figshare.com/files/33947396",
        },
        {
            "id": 33947432,
            "name": "wssnet.zip",
            "download_url": "https://ndownloader.figshare.com/files/33947432",
        },
    ],
}


def test_oai_traversal_preserves_article_file_relation_and_resumes() -> None:
    class DetailClient(Client):
        def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
            self.calls.append((url, dict(params or {})))
            if url == VERSIONS:
                body = [
                    {"version": 1, "url": f"{VERSIONS}/1"},
                    {"version": 2, "url": f"{VERSIONS}/2"},
                ]
            elif url == f"{VERSIONS}/1":
                body = ARTICLE_V1
            elif url == f"{VERSIONS}/2":
                body = ARTICLE
            else:
                return self.responses.pop(0)
            return response(json.dumps(body), url, "application/json")

    client = DetailClient(
        response(list_records(wssnet_record(), "opaque-token")),
        response(list_records("")),
    )
    adapter = FigshareModelCandidatesSourceAdapter(
        from_date="2022-02-07",
        until_date="2022-02-08",
        client=client,
        clock=lambda: datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert client.calls == [
        (
            OAI,
            {
                "verb": "ListRecords",
                "metadataPrefix": "mets",
                "from": "2022-02-07",
                "until": "2022-02-08",
            },
        ),
        (VERSIONS, {}),
        (f"{VERSIONS}/1", {}),
        (f"{VERSIONS}/2", {}),
        (OAI, {"verb": "ListRecords", "resumptionToken": "opaque-token"}),
    ]
    assert first.complete is False
    assert first.next_state["records_seen"] == 1
    assert second.complete is True
    assert len(first.records) == 2
    historical, record = first.records
    assert historical.source_record_id == "article:19105067:version:1"
    assert historical.raw["article_version"] == "1"
    assert historical.raw["matched_files"] == [
        {
            "id": "33947432",
            "name": "wssnet.zip",
            "url": "https://ndownloader.figshare.com/files/33947432",
        }
    ]
    assert historical.identifiers[1].namespace == "figshare:article-version"
    assert historical.identifiers[1].value == "19105067:v1"
    assert historical.links[0].url.endswith("/19105067/1")
    assert record.source_record_id == "article:19105067:version:2"
    assert record.raw["article_id"] == "19105067"
    assert record.raw["article_version"] == "2"
    assert record.raw["matched_files"] == [
        {
            "id": "33947432",
            "name": "wssnet.zip",
            "url": "https://ndownloader.figshare.com/files/33947432",
        }
    ]
    assert [link.url for link in record.links if link.relation == "checkpoint"] == [
        "https://ndownloader.figshare.com/files/33947432"
    ]
    assert record.models[0].status.value == "candidate"


def test_half_open_window_and_token_expiry_are_validated() -> None:
    import pytest

    with pytest.raises(ValueError, match="later"):
        FigshareModelCandidatesSourceAdapter(from_date="2022-02-08", until_date="2022-02-08")

    client = Client(response(list_records("", "expired-token")))
    adapter = FigshareModelCandidatesSourceAdapter(
        from_date="2022-02-07",
        until_date="2022-02-08",
        client=client,
        clock=lambda: datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
    )
    first = adapter.fetch_page({})
    assert first.next_state["token_expires_at"] == "2026-09-24T11:00:00Z"
    client.responses = []
    with pytest.raises(ValueError, match="restart this date window"):
        adapter.fetch_page({**first.next_state, "token_expires_at": "2026-09-24T10:04:00Z"})


def test_no_records_match_completes_empty_window() -> None:
    client = Client(
        response(
            '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            '<error code="noRecordsMatch">empty</error></OAI-PMH>'
        )
    )
    adapter = FigshareModelCandidatesSourceAdapter(
        from_date="2022-02-07", until_date="2022-02-08", client=client
    )
    page = adapter.fetch_page({})
    assert page.complete is True
    assert page.records == ()
