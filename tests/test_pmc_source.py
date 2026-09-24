from __future__ import annotations

import gzip
from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO
from typing import Any
from urllib.error import HTTPError

import pytest

from modelome.http import HttpFailure, HttpResponse
from modelome.models import Identifier
from modelome.normalize import content_hash
from modelome.sources.pmc import PmcSourceAdapter

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
OAI = "http://www.openarchives.org/OAI/2.0/"
JATS = "https://jats.nlm.nih.gov/ns/archiving/1.4/"
XLINK = "http://www.w3.org/1999/xlink"
ALI = "http://www.niso.org/schemas/ali/1.0/"


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


class HttpErrorClient:
    def __init__(self, body: str) -> None:
        self.body = body

    def get(self, url: str, **_: Any) -> HttpResponse:
        compressed = gzip.compress(self.body.encode())
        upstream = HTTPError(
            url,
            404,
            "Not Found",
            {"Content-Encoding": "gzip", "Content-Type": "application/xml"},
            BytesIO(compressed),
        )
        raise HttpFailure("GET failed") from upstream


def response(body: str, *, status: int = 200, **headers: str) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers={"content-type": "application/xml", **headers},
        body=body.encode(),
        url="https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/",
    )


def compressed_response(body: str) -> HttpResponse:
    return HttpResponse(
        status=200,
        headers={"content-encoding": "gzip", "content-type": "application/xml"},
        body=gzip.compress(body.encode()),
        url="https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/",
    )


def oai(body: str, *, verb: str = "ListRecords") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="{OAI}">
  <responseDate>2026-09-01T12:00:00Z</responseDate>
  <request verb="{verb}">https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/</request>
  {body}
</OAI-PMH>"""


def identify(
    *,
    base_url: str = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/",
    protocol: str = "2.0",
    earliest: str = "1999-01-01",
    deleted: str = "no",
    granularity: str = "YYYY-MM-DD",
) -> str:
    return oai(
        f"""
<Identify>
  <repositoryName>PubMed Central</repositoryName>
  <baseURL>{base_url}</baseURL>
  <protocolVersion>{protocol}</protocolVersion>
  <adminEmail>example@example.test</adminEmail>
  <earliestDatestamp>{earliest}</earliestDatestamp>
  <deletedRecord>{deleted}</deletedRecord>
  <granularity>{granularity}</granularity>
</Identify>""",
        verb="Identify",
    )


def record(
    pmc_number: str = "1234567",
    *,
    datestamp: str = "2026-08-31",
    metadata_pmc_number: str | None = None,
) -> str:
    metadata_number = metadata_pmc_number or pmc_number
    return f"""
<record>
  <header>
    <identifier>oai:pubmedcentral.nih.gov:{pmc_number}</identifier>
    <datestamp>{datestamp}</datestamp>
    <setSpec>example-journal</setSpec>
    <setSpec>pmc-open</setSpec>
  </header>
  <metadata>
    <article xmlns="{JATS}" xmlns:xlink="{XLINK}" xmlns:ali="{ALI}"
             article-type="research-article" xml:lang="en">
      <front>
        <journal-meta>
          <journal-id journal-id-type="nlm-ta">Example Biomed</journal-id>
          <journal-title-group>
            <journal-title>Example Biomedical Journal</journal-title>
          </journal-title-group>
          <issn pub-type="epub">1234-5678</issn>
          <publisher><publisher-name>Example Publisher</publisher-name></publisher>
        </journal-meta>
        <article-meta>
          <article-id pub-id-type="pmcid">PMC{metadata_number}</article-id>
          <article-id pub-id-type="pmcid-ver">PMC{metadata_number}.2</article-id>
          <article-id pub-id-type="pmid">9876543</article-id>
          <article-id pub-id-type="doi">10.1234/example.42</article-id>
          <title-group>
            <article-title>A data-driven tissue risk system</article-title>
          </title-group>
          <contrib-group>
            <contrib contrib-type="author" corresp="yes">
              <name><surname>Researcher</surname><given-names>Ada</given-names></name>
              <contrib-id contrib-id-type="orcid">0000-0002-1825-0097</contrib-id>
              <xref ref-type="aff" rid="aff1" />
            </contrib>
          </contrib-group>
          <pub-date pub-type="epub" iso-8601-date="2024-02-03">
            <day>3</day><month>2</month><year>2024</year>
          </pub-date>
          <pub-history>
            <event event-type="pmc-last-change">
              <date iso-8601-date="2026-08-31 09:15:30">
                <day>31</day><month>8</month><year>2026</year>
              </date>
            </event>
          </pub-history>
          <volume>12</volume><issue>4</issue>
          <permissions>
            <copyright-statement>Copyright the authors.</copyright-statement>
            <license license-type="open-access">
              <ali:license_ref>https://creativecommons.org/licenses/by/4.0/</ali:license_ref>
              <license-p>Reusable under an attribution license.</license-p>
            </license>
          </permissions>
          <abstract>
            <p>We introduce a deep neural architecture for clinical tissue measurements.</p>
          </abstract>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Methods</title>
          <p>The system learns directly from histology images.</p>
          <p>Code is available at
            <ext-link ext-link-type="uri"
              xlink:href="https://github.com/example/tissue-risk">the repository</ext-link>.
          </p>
        </sec>
      </body>
    </article>
  </metadata>
</record>"""


def deleted_record(pmc_number: str = "7654321") -> str:
    return f"""
<record>
  <header status="deleted">
    <identifier>oai:pubmedcentral.nih.gov:{pmc_number}</identifier>
    <datestamp>2026-08-31</datestamp>
    <setSpec>pmc-open</setSpec>
  </header>
</record>"""


def list_records(
    records: str,
    *,
    token: str = "",
    total: int | None = None,
    cursor: int | None = None,
) -> str:
    attributes = ""
    if total is not None:
        attributes += f' completeListSize="{total}"'
    if cursor is not None:
        attributes += f' cursor="{cursor}"'
    return oai(
        f"<ListRecords>{records}"
        f"<resumptionToken{attributes}>{token}</resumptionToken>"
        "</ListRecords>"
    )


def adapter(client: QueueClient, **overrides: Any) -> PmcSourceAdapter:
    values: dict[str, Any] = {"client": client, "clock": lambda: NOW}
    values.update(overrides)
    return PmcSourceAdapter(**values)


def test_full_text_source_resumes_and_preserves_jats_evidence() -> None:
    client = QueueClient(
        response(list_records(record(), token="opaque-page-two", total=2, cursor=0)),
        response(list_records(deleted_record(), total=2, cursor=1)),
    )
    source = adapter(client, initial_lookback_days=2, overlap_days=2)

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    assert first.complete is False
    assert first.upstream_count == 2
    assert first.next_state["raw_items_seen"] == 1
    assert first.next_state["scan_total"] == 2
    assert first.next_state["window_start"] == "2026-08-30"
    assert first.next_state["window_end"] == "2026-08-31"
    assert client.calls[0][1] == {
        "verb": "ListRecords",
        "metadataPrefix": "pmc",
        "set": "pmc-open",
        "from": "2026-08-30",
        "until": "2026-08-31",
    }
    assert client.calls[1][1] == {
        "verb": "ListRecords",
        "resumptionToken": "opaque-page-two",
    }
    assert client.calls[0][2]["Accept-Encoding"] == "gzip, deflate"

    item = first.records[0]
    assert item.source_record_id == "PMC1234567"
    assert item.canonical_url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC1234567"
    assert item.title == "A data-driven tissue risk system"
    assert item.published_at == "2024-02-03"
    assert item.modified_at == "2026-08-31"
    assert item.identifiers == (
        Identifier("pmcid", "PMC1234567"),
        Identifier("pmid", "9876543"),
        Identifier("doi", "10.1234/example.42"),
        Identifier("pmcid:version", "PMC1234567.2"),
    )
    assert "deep neural architecture" in item.text
    assert "The system learns directly from histology images" in item.text
    assert item.raw["jats"]["abstract"].startswith("We introduce")
    assert item.raw["jats"]["body_text"].startswith("Methods")
    assert item.raw["jats"]["authors"] == [
        {
            "name": "Ada Researcher",
            "given_names": "Ada",
            "surname": "Researcher",
            "corresponding": True,
            "affiliation_refs": ["aff1"],
            "orcid": "0000-0002-1825-0097",
        }
    ]
    assert item.raw["jats"]["journal"]["title"] == "Example Biomedical Journal"
    assert item.raw["jats"]["licenses"][0]["urls"] == [
        "https://creativecommons.org/licenses/by/4.0"
    ]
    assert "<" in item.raw["jats_xml"]
    assert any(
        link.url == "https://github.com/example/tissue-risk"
        and link.relation == "implementation"
        and link.crawl
        for link in item.links
    )
    assert all(
        not link.crawl
        for link in item.links
        if link.relation in {"landing_page", "open_full_text", "license"}
    )

    assert second.complete is True
    assert second.next_state["watermark"] == "2026-08-31"
    tombstone = second.records[0]
    assert tombstone.source_record_id == "PMC7654321"
    assert tombstone.deleted is True
    assert tombstone.text == ""
    assert tombstone.identifiers == (Identifier("pmcid", "PMC7654321"),)


def test_identify_validates_and_returns_repository_contract() -> None:
    client = QueueClient(
        compressed_response(
            identify(
                earliest="1999-01-01T00:00:00Z",
                deleted="persistent",
                granularity="YYYY-MM-DDThh:mm:ssZ",
            )
        )
    )

    identity = adapter(client).identify()

    assert client.calls[0][1] == {"verb": "Identify"}
    assert identity.repository_name == "PubMed Central"
    assert identity.base_url.endswith("/api/oai/v1/mh/")
    assert identity.protocol_version == "2.0"
    assert identity.earliest_datestamp == "1999-01-01T00:00:00Z"
    assert identity.deleted_record == "persistent"
    assert identity.granularity == "YYYY-MM-DDThh:mm:ssZ"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (identify(base_url="https://example.test/oai"), "baseURL does not match"),
        (identify(protocol="1.1"), "unsupported OAI-PMH protocol version"),
        (identify(earliest="not-a-date"), "invalid OAI datestamp"),
        (identify(deleted="sometimes"), "invalid OAI-PMH deletedRecord policy"),
        (identify(granularity="YYYY"), "unsupported OAI-PMH datestamp granularity"),
        (
            oai('<error code="badArgument">bad request</error>', verb="Identify"),
            "OAI-PMH Identify error",
        ),
    ],
)
def test_identify_rejects_inconsistent_repository_contract(
    body: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        adapter(QueueClient(response(body))).identify()


def test_no_records_match_completes_closed_window_and_overlap_replays() -> None:
    body = oai('<error code="noRecordsMatch">No matching records</error>')
    client = QueueClient(response(body, status=404))
    source = adapter(client, overlap_days=3)

    page = source.fetch_page({"watermark": "2026-08-31"})

    assert page.complete is True
    assert page.records == ()
    assert page.upstream_count == 0
    assert page.next_state["watermark"] == "2026-08-31"
    assert client.calls[0][1]["from"] == "2026-08-29"
    assert client.calls[0][1]["until"] == "2026-08-31"
    assert client.calls[0][1]["set"] == "pmc-open"


def test_transport_preserves_pmcs_http_404_oai_error_body() -> None:
    body = oai('<error code="noRecordsMatch">No matching records</error>')
    source = PmcSourceAdapter(client=HttpErrorClient(body), clock=lambda: NOW)

    page = source.fetch_page({})

    assert page.complete is True
    assert page.records == ()
    assert page.upstream_count == 0


def test_malformed_jats_is_quarantined_without_advancing_page() -> None:
    bad_record = record(metadata_pmc_number="9999999")
    source = adapter(
        QueueClient(response(list_records(bad_record, total=1, cursor=0)))
    )

    page = source.fetch_page({})

    assert page.records == ()
    assert page.complete is False
    assert page.next_state == page.retry_state
    assert page.retry_state["raw_items_seen"] == 0
    assert len(page.issues) == 1
    assert page.issues[0].source_record_id == "PMC1234567"
    assert "does not match JATS PMCID" in page.issues[0].error


def test_deleted_record_cannot_contain_metadata() -> None:
    malformed = deleted_record().replace("</record>", "<metadata /></record>")
    source = adapter(
        QueueClient(response(list_records(malformed, total=1, cursor=0)))
    )

    page = source.fetch_page({})

    assert page.records == ()
    assert page.issues
    assert "deleted record unexpectedly contains metadata" in page.issues[0].error


def test_truncation_cursor_drift_and_token_cycles_hold_checkpoint() -> None:
    truncated = list_records(record(), total=2, cursor=0)
    truncated_page = adapter(QueueClient(response(truncated))).fetch_page({})
    assert truncated_page.complete is False
    assert any("before completeListSize 2" in issue.error for issue in truncated_page.issues)

    drifted = list_records(record(), total=2, cursor=1)
    drifted_page = adapter(QueueClient(response(drifted))).fetch_page({})
    assert any("response cursor 1" in issue.error for issue in drifted_page.issues)

    token = "same-token"
    cycling = list_records(record("2222222"), token=token, total=4, cursor=2)
    cycling_page = adapter(QueueClient(response(cycling))).fetch_page(
        {
            "window_start": "2026-08-31",
            "window_end": "2026-08-31",
            "resumption_token": token,
            "seen_token_hashes": [content_hash(token)],
            "raw_items_seen": 2,
            "scan_total": 4,
        }
    )
    assert any("resumption token repeated" in issue.error for issue in cycling_page.issues)
    assert cycling_page.next_state == cycling_page.retry_state


def test_bad_resumption_token_restarts_the_same_frozen_window() -> None:
    source = adapter(
        QueueClient(
            response(
                oai('<error code="badResumptionToken">expired token</error>')
            )
        )
    )

    page = source.fetch_page(
        {
            "window_start": "2026-08-30",
            "window_end": "2026-08-31",
            "resumption_token": "expired",
            "raw_items_seen": 100,
            "scan_total": 110,
        }
    )

    assert page.complete is False
    assert page.retry_state["window_start"] == "2026-08-30"
    assert page.retry_state["window_end"] == "2026-08-31"
    assert page.retry_state["raw_items_seen"] == 0
    assert "resumption_token" not in page.retry_state
    assert "restarting the frozen window" in page.issues[0].error


def test_rejects_unsafe_xml_and_enforces_decoded_response_bound() -> None:
    unsafe = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY y "boom">]><x>&y;</x>'
    unsafe_response = HttpResponse(200, {}, unsafe, "https://example.test/oai")
    with pytest.raises(ValueError, match="unsafe XML"):
        adapter(QueueClient(unsafe_response)).fetch_page({})

    malformed_response = HttpResponse(
        200,
        {},
        f'<OAI-PMH xmlns="{OAI}"><ListRecords>'.encode(),
        "https://example.test/oai",
    )
    with pytest.raises(ValueError, match="malformed XML"):
        adapter(QueueClient(malformed_response)).fetch_page({})

    oversized = response(oai("<ListRecords />"))
    with pytest.raises(ValueError, match="exceeds 32 decoded bytes"):
        adapter(QueueClient(oversized), max_response_bytes=32).fetch_page({})


def test_requires_closed_complete_frozen_windows_and_bounded_jats() -> None:
    source = adapter(QueueClient())
    with pytest.raises(ValueError, match="requires both boundaries"):
        source.fetch_page({"window_start": "2026-08-30"})
    with pytest.raises(ValueError, match="closed UTC day"):
        source.fetch_page(
            {"window_start": "2026-08-31", "window_end": "2026-09-01"}
        )
    with pytest.raises(ValueError, match="missing its frozen window"):
        source.fetch_page({"resumption_token": "opaque"})

    bounded = adapter(
        QueueClient(response(list_records(record(), total=1, cursor=0))),
        max_text_chars_per_record=20,
    ).fetch_page({})
    assert bounded.records == ()
    assert "extracted record text exceeds" in bounded.issues[0].error


def test_active_record_must_remain_in_reusable_full_text_collection() -> None:
    outside_collection = record().replace("<setSpec>pmc-open</setSpec>", "")
    page = adapter(
        QueueClient(response(list_records(outside_collection, total=1, cursor=0)))
    ).fetch_page({})

    assert page.records == ()
    assert "not declared in 'pmc-open'" in page.issues[0].error
