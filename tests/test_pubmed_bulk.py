from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.lake import ParquetLandingZone
from modelome.models import ArtifactKind, SourceRecord
from modelome.pubmed_bulk import PubMedBulkLoader

FILENAME = "pubmed26n0003.xml.gz"
URL = f"https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/{FILENAME}"

XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2025//EN"
  "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_250101.dtd">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM" IndexingMethod="Automated">
      <PMID Version="2">12345</PMID>
      <DateCompleted><Year>2026</Year><Month>04</Month><Day>03</Day></DateCompleted>
      <DateRevised><Year>2026</Year><Month>Apr</Month><Day>04</Day></DateRevised>
      <Article PubModel="Electronic">
        <Journal>
          <ISSN IssnType="Electronic">1234-5678</ISSN>
          <JournalIssue CitedMedium="Internet">
            <Volume>12</Volume><Issue>4</Issue>
            <PubDate><Year>2026</Year><Month>Mar</Month><Day>14</Day></PubDate>
          </JournalIssue>
          <Title>Journal of Neural Medicine</Title>
          <ISOAbbreviation>J Neural Med</ISOAbbreviation>
        </Journal>
        <ArticleTitle>A neural <i>foundation model</i> for medicine</ArticleTitle>
        <Pagination><MedlinePgn>10-20</MedlinePgn></Pagination>
        <ELocationID EIdType="doi" ValidYN="Y">10.5555/Example.1</ELocationID>
        <Abstract>
          <AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">Background text.</AbstractText>
          <AbstractText Label="CODE" NlmCategory="METHODS">Code is at https://github.com/lab/model.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author ValidYN="Y">
            <LastName>Lovelace</LastName><ForeName>Ada</ForeName><Initials>A</Initials>
            <Identifier Source="ORCID">0000-0001-2345-6789</Identifier>
            <AffiliationInfo><Affiliation>Example Institute</Affiliation></AffiliationInfo>
          </Author>
        </AuthorList>
        <Language>eng</Language>
        <PublicationTypeList>
          <PublicationType UI="D016428">Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
      <MedlineJournalInfo>
        <Country>United States</Country><MedlineTA>J Neural Med</MedlineTA>
        <NlmUniqueID>1234567</NlmUniqueID><ISSNLinking>1234-5678</ISSNLinking>
      </MedlineJournalInfo>
    </MedlineCitation>
    <PubmedData>
      <History>
        <PubMedPubDate PubStatus="received">
          <Year>2025</Year><Month>12</Month><Day>01</Day>
        </PubMedPubDate>
      </History>
      <PublicationStatus>epublish</PublicationStatus>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345</ArticleId>
        <ArticleId IdType="doi">10.5555/Example.1</ArticleId>
        <ArticleId IdType="pmc">PMC1234567</ArticleId>
      </ArticleIdList>
      <ReferenceList><Reference><Citation>Implementation https://gitlab.com/lab/related</Citation></Reference></ReferenceList>
    </PubmedData>
  </PubmedArticle>
  <PubmedBookArticle>
    <BookDocument>
      <PMID Version="1">23456</PMID>
      <ArticleTitle>Deep learning in a medical handbook</ArticleTitle>
      <Abstract><AbstractText>Book abstract.</AbstractText></Abstract>
    </BookDocument>
    <PubmedBookData>
      <ArticleIdList><ArticleId IdType="pubmed">23456</ArticleId></ArticleIdList>
    </PubmedBookData>
  </PubmedBookArticle>
  <DeleteCitation><PMID Version="1">34567</PMID><PMID>45678</PMID></DeleteCitation>
</PubmedArticleSet>
"""


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        url: str = URL,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        maximum_read: int = 17,
    ) -> None:
        super().__init__(body)
        self.url = url
        self.status = status
        self.headers = dict(headers or {"Content-Length": str(len(body))})
        self.maximum_read = maximum_read

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.maximum_read
        return super().read(min(size, self.maximum_read))


class FakeTransport:
    def __init__(
        self,
        body: bytes,
        *,
        final_url: str = URL,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.body = body
        self.final_url = final_url
        self.status = status
        self.headers = headers
        self.calls: list[tuple[str, Mapping[str, str], Callable[[str], None]]] = []

    def open(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None],
    ) -> FakeResponse:
        self.calls.append((url, dict(headers), redirect_validator))
        return FakeResponse(
            self.body,
            url=self.final_url,
            status=self.status,
            headers=self.headers,
        )


def _compressed(xml: bytes = XML) -> bytes:
    return gzip.compress(xml, mtime=0)


def _control(
    body: bytes,
    *,
    url: str = URL,
    filename: str = FILENAME,
    kind: str = "update",
    sequence: int = 3,
    checksum: str | None = None,
) -> SourceRecord:
    checksum = checksum or hashlib.md5(body, usedforsecurity=False).hexdigest()
    return SourceRecord(
        source_record_id=f"pubmed:{filename}",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=url,
        title=f"PubMed {kind} shard {filename}",
        raw={
            "application_order": sequence,
            "checksum": {"algorithm": "md5", "value": checksum},
            "filename": filename,
            "manifest_kind": kind,
            "production_cycle": "26",
            "production_year": 2026,
            "sequence": sequence,
            "url": url,
        },
    )


def _rows(lake: ParquetLandingZone, receipt: Any) -> list[dict[str, Any]]:
    lake.seal_release(
        source=receipt.source,
        dataset=receipt.dataset,
        release=receipt.release,
        expected_shards={receipt.shard: receipt.upstream_sha256},
    )
    return [
        row
        for batch in lake.iter_release_batches(
            source=receipt.source,
            dataset=receipt.dataset,
            release=receipt.release,
        )
        for row in batch.to_pylist()
    ]


def test_loader_verifies_streams_and_lands_articles_books_and_deletes(
    tmp_path: Path,
) -> None:
    body = _compressed()
    lake = ParquetLandingZone(tmp_path / "lake")
    transport = FakeTransport(body)
    loader = PubMedBulkLoader(
        lake,
        transport=transport,
        download_chunk_bytes=32,
        parquet_batch_rows=2,
    )

    receipt = loader.load(_control(body))
    rows = _rows(lake, receipt)

    assert receipt.release == "2026-update-00000003"
    assert receipt.manifest_kind == "update"
    assert receipt.compressed_bytes == len(body)
    assert receipt.uncompressed_bytes == len(XML)
    assert receipt.row_count == 4
    assert receipt.upsert_count == 2
    assert receipt.delete_count == 2
    assert receipt.published_md5 == hashlib.md5(
        body, usedforsecurity=False
    ).hexdigest()
    assert receipt.upstream_sha256 == hashlib.sha256(body).hexdigest()
    manifest = json.loads((receipt.path / "manifest.json").read_text())
    assert manifest["application_order"] == {"mode": "single"}
    assert transport.calls[0][0] == URL
    assert transport.calls[0][1]["Accept-Encoding"] == "identity"

    assert [row["source_record_id"] for row in rows] == [
        "pmid:12345",
        "pmid:23456",
        "pmid:34567",
        "pmid:45678",
    ]
    assert [row["operation"] for row in rows] == [
        "upsert",
        "upsert",
        "delete",
        "delete",
    ]
    article = json.loads(rows[0]["payload_json"])
    assert article["title"] == "A neural foundation model for medicine"
    assert article["abstract"] == (
        "Background text.\n\nCode is at https://github.com/lab/model."
    )
    assert article["doi"] == "10.5555/example.1"
    assert article["pmc"] == "PMC1234567"
    assert article["pmid_version"] == "2"
    assert article["authors"][0]["last_name"] == "Lovelace"
    assert article["publication"]["publication_date"]["display"] == "2026-03-14"
    assert article["publication"]["completed_date"]["display"] == "2026-04-03"
    assert article["publication"]["revised_date"]["display"] == "2026-04-04"
    assert article["publication"]["journal"] == "Journal of Neural Medicine"
    assert article["publication"]["publication_status"] == "epublish"
    assert article["repository_urls"] == [
        "https://github.com/lab/model",
        "https://gitlab.com/lab/related",
    ]
    assert article["bulk"]["operation_index"] == 1
    deletion = json.loads(rows[2]["payload_json"])
    assert deletion == {
        "bulk": {
            "application_order": 3,
            "filename": FILENAME,
            "manifest_kind": "update",
            "operation_index": 3,
            "production_cycle": "26",
            "production_year": 2026,
            "sequence": 3,
        },
        "pmid": "34567",
        "pmid_version": "1",
    }


def test_loader_exposes_network_free_bulk_plan_and_order(tmp_path: Path) -> None:
    compressed = _compressed()
    control = _control(
        compressed,
        filename="pubmed26n0007.xml.gz",
        kind="update",
        sequence=7,
    )
    loader = PubMedBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=FakeTransport(b""),
    )

    plan = loader.plan_shard(control)

    assert plan is not None
    assert (plan.source, plan.dataset, plan.release, plan.shard) == (
        "pubmed",
        "citations",
        "2026-update-00000007",
        "pubmed26n0007.xml.gz",
    )
    assert plan.control_sha256 is not None
    assert plan.upstream_sha256 is None
    assert loader.shard_order(control) == (
        2026,
        1,
        7,
        "pubmed26n0007.xml.gz",
    )


def test_baseline_shards_share_an_unsealed_staging_release(tmp_path: Path) -> None:
    xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
    <PMID>1</PMID><Article><ArticleTitle>Title</ArticleTitle></Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    body = _compressed(xml)
    filename = "pubmed26n0001.xml.gz"
    url = f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{filename}"
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = PubMedBulkLoader(lake, transport=FakeTransport(body, final_url=url)).load(
        _control(
            body,
            url=url,
            filename=filename,
            kind="baseline",
            sequence=1,
        )
    )

    assert receipt.release == "2026-baseline"
    manifest = json.loads((receipt.path / "manifest.json").read_text())
    assert manifest["application_order"] == {
        "manifest_index": 0,
        "mode": "snapshot",
    }
    with pytest.raises(ValueError, match="not sealed"):
        list(
            lake.iter_release_batches(
                source=receipt.source,
                dataset=receipt.dataset,
                release=receipt.release,
            )
        )


def test_identical_verified_shard_is_idempotent_without_reparsing(tmp_path: Path) -> None:
    body = _compressed()
    lake = ParquetLandingZone(tmp_path / "lake")
    transport = FakeTransport(body)
    loader = PubMedBulkLoader(lake, transport=transport)
    control = _control(body)

    first = loader.load(control)
    repeated = loader.load(control)

    assert first.already_committed is False
    assert repeated.already_committed is True
    assert repeated.row_count == first.row_count
    assert repeated.uncompressed_bytes is None
    assert repeated.upsert_count is None
    assert repeated.delete_count is None
    assert len(transport.calls) == 1


def test_changed_published_md5_is_a_new_control_and_does_not_skip_download(
    tmp_path: Path,
) -> None:
    first_body = _compressed()
    changed_xml = XML.replace(
        b"A neural <i>foundation model</i> for medicine",
        b"A changed neural <i>foundation model</i> for medicine",
    )
    second_body = _compressed(changed_xml)
    lake = ParquetLandingZone(tmp_path / "lake")
    first_transport = FakeTransport(first_body)
    second_transport = FakeTransport(second_body)

    first = PubMedBulkLoader(lake, transport=first_transport).load(
        _control(first_body)
    )
    second = PubMedBulkLoader(lake, transport=second_transport).load(
        _control(second_body)
    )

    assert len(first_transport.calls) == 1
    assert len(second_transport.calls) == 1
    assert first.upstream_sha256 != second.upstream_sha256
    assert first.control_sha256 != second.control_sha256
    assert first.path != second.path
    assert second.already_committed is False


def test_md5_failure_never_creates_a_parquet_shard(tmp_path: Path) -> None:
    body = _compressed()
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = PubMedBulkLoader(lake, transport=FakeTransport(body))

    with pytest.raises(ValueError, match="MD5 mismatch"):
        loader.load(_control(body, checksum="0" * 32))

    assert list(lake.staging_root.iterdir()) == []
    assert not any(lake.shards_root.rglob("manifest.json"))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_compressed_bytes": 10}, "compressed payload exceeds"),
        ({"max_uncompressed_bytes": 100}, "uncompressed payload exceeds"),
        ({"max_records": 1}, "exceeds 1 records"),
        ({"max_text_chars_per_record": 20}, "record text exceeds"),
    ],
)
def test_resource_limits_fail_before_visibility(
    tmp_path: Path,
    options: Mapping[str, int],
    message: str,
) -> None:
    body = _compressed()
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = PubMedBulkLoader(lake, transport=FakeTransport(body), **options)

    with pytest.raises(ValueError, match=message):
        loader.load(_control(body))

    assert list(lake.staging_root.iterdir()) == []
    assert not any(lake.shards_root.rglob("manifest.json"))


def test_content_length_and_http_encoding_must_describe_published_bytes(
    tmp_path: Path,
) -> None:
    body = _compressed()
    for headers, message in (
        ({"Content-Length": str(len(body) + 1)}, "length mismatch"),
        (
            {"Content-Length": str(len(body)), "Content-Encoding": "gzip"},
            "without HTTP content encoding",
        ),
    ):
        lake = ParquetLandingZone(tmp_path / hashlib.sha256(message.encode()).hexdigest())
        loader = PubMedBulkLoader(
            lake,
            transport=FakeTransport(body, headers=headers),
        )
        with pytest.raises(ValueError, match=message):
            loader.load(_control(body))
        assert not any(lake.shards_root.rglob("manifest.json"))


@pytest.mark.parametrize(
    "url",
    [
        f"http://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/{FILENAME}",
        f"https://evil.example/pubmed/updatefiles/{FILENAME}",
        f"https://ftp.ncbi.nlm.nih.gov:444/pubmed/updatefiles/{FILENAME}",
        f"https://user@ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/{FILENAME}",
        f"https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/{FILENAME}?token=secret",
        f"https://ftp.ncbi.nlm.nih.gov/pubmed/../updatefiles/{FILENAME}",
    ],
)
def test_unsafe_control_urls_are_rejected_before_transport(
    tmp_path: Path,
    url: str,
) -> None:
    body = _compressed()
    transport = FakeTransport(body)
    loader = PubMedBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=transport,
    )

    with pytest.raises(ValueError, match="allowed HTTPS origins|unsafe path"):
        loader.load(_control(body, url=url))

    assert transport.calls == []


def test_untrusted_redirect_target_is_rejected_even_by_injected_transport(
    tmp_path: Path,
) -> None:
    body = _compressed()
    transport = FakeTransport(
        body,
        final_url=f"https://evil.example/pubmed/updatefiles/{FILENAME}",
    )
    loader = PubMedBulkLoader(
        ParquetLandingZone(tmp_path / "lake"),
        transport=transport,
    )

    with pytest.raises(ValueError, match="allowed HTTPS origins"):
        loader.load(_control(body))


def test_invalid_gzip_xml_and_custom_entities_never_become_visible(
    tmp_path: Path,
) -> None:
    bad_payloads = (
        b"not a gzip stream",
        _compressed(b"<PubmedArticleSet><PubmedArticle>"),
        _compressed(
            b'<!DOCTYPE PubmedArticleSet [<!ENTITY x "boom">]>'
            b"<PubmedArticleSet>&x;</PubmedArticleSet>"
        ),
    )
    for index, body in enumerate(bad_payloads):
        lake = ParquetLandingZone(tmp_path / f"lake-{index}")
        loader = PubMedBulkLoader(lake, transport=FakeTransport(body))
        with pytest.raises(ValueError):
            loader.load(_control(body))
        assert list(lake.staging_root.iterdir()) == []
        assert not any(lake.shards_root.rglob("manifest.json"))


def test_external_dtd_character_entities_do_not_require_network_resolution(
    tmp_path: Path,
) -> None:
    xml = b"""<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle//EN"
    "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_250101.dtd">
    <PubmedArticleSet><PubmedArticle><MedlineCitation>
    <PMID>1</PMID><Article><ArticleTitle>Known &alpha; preserved &agr;</ArticleTitle></Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    body = _compressed(xml)
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = PubMedBulkLoader(lake, transport=FakeTransport(body)).load(_control(body))

    payload = json.loads(_rows(lake, receipt)[0]["payload_json"])

    assert payload["title"] == "Known α preserved &agr;"


def test_delete_citation_in_baseline_is_rejected(tmp_path: Path) -> None:
    xml = b"<PubmedArticleSet><DeleteCitation><PMID>1</PMID></DeleteCitation></PubmedArticleSet>"
    body = _compressed(xml)
    filename = "pubmed26n0001.xml.gz"
    url = f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{filename}"
    lake = ParquetLandingZone(tmp_path / "lake")
    loader = PubMedBulkLoader(lake, transport=FakeTransport(body, final_url=url))

    with pytest.raises(ValueError, match="not valid in a baseline"):
        loader.load(
            _control(
                body,
                url=url,
                filename=filename,
                kind="baseline",
                sequence=1,
            )
        )

    assert not any(lake.shards_root.rglob("manifest.json"))
