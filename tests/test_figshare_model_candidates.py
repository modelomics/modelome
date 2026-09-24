from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.figshare_model_candidates import (
    FigshareModelCandidatesSourceAdapter,
)

OAI = "https://api.figshare.com/v2/oai"
DETAIL = "https://api.figshare.com/v2/articles/19105067"


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def response(body: str | bytes, url: str = OAI) -> HttpResponse:
    encoded = body.encode() if isinstance(body, str) else body
    return HttpResponse(200, {"content-type": "application/xml"}, encoded, url)


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
    "figshare_url": "https://figshare.com/articles/software/WSSNet/19105067",
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


def test_oai_traversal_preserves_article_file_relation_and_resumes() -> None:
    client = Client(
        response(list_records(wssnet_record(), "opaque-token")),
        HttpResponse(200, {"content-type": "application/json"}, b'{"id":19105067}', DETAIL),
        response(list_records("")),
    )

    # Supply the captured REST detail through the minimal response interface.
    class DetailClient(Client):
        def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
            self.calls.append((url, dict(params or {})))
            result = self.responses.pop(0)
            if url == DETAIL:
                result = HttpResponse(
                    200,
                    {"content-type": "application/json"},
                    __import__("json").dumps(ARTICLE).encode(),
                    url,
                )
            return result

    client = DetailClient(
        response(list_records(wssnet_record(), "opaque-token")),
        response("<ignored/>", DETAIL),
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
        (DETAIL, {}),
        (OAI, {"verb": "ListRecords", "resumptionToken": "opaque-token"}),
    ]
    assert first.complete is False
    assert first.next_state["records_seen"] == 1
    assert second.complete is True
    assert len(first.records) == 1
    record = first.records[0]
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
