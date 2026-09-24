from __future__ import annotations

from pathlib import Path

import pytest

from modelome.storage import Database
from modelome.text_import import (
    extract_doi,
    extract_text_from_pdf_bytes,
    fetch_oa_pdf_url,
    ingest_locally_authorized_text,
    inventory_text_artifacts,
    read_text_file,
    resolve_lawful_access_routes,
)


def test_extract_doi() -> None:
    assert extract_doi("See https://doi.org/10.1038/nature12373 .") == "10.1038/nature12373"
    assert extract_doi("(doi:10.1126/science.1234567)") == "10.1126/science.1234567"
    assert extract_doi("10.5281/zenodo.123456,") == "10.5281/zenodo.123456"
    assert extract_doi("No DOI here.") is None


def test_inventory_text_artifacts() -> None:
    text = (
        "Code at https://github.com/lab/model. Data at https://zenodo.org/record/123. "
        "Landing at https://doi.org/10.1038/xyz. Orphan link: https://google.com/about."
    )
    inv = inventory_text_artifacts(text)
    assert inv.doi == "10.1038/xyz"
    github = tuple(u for u in inv.artifact_urls if "github.com" in u)
    zenodo = tuple(u for u in inv.artifact_urls if "zenodo.org" in u)
    assert len(github) == 1
    assert len(zenodo) == 1
    generic = tuple(u for u in inv.artifact_urls if "google.com" in u)
    assert generic == ()


def test_fetch_oa_pdf_url_resolves() -> None:
    class FakeResponse:
        def json(self):
            return {
                "locations": [
                    {"pdf_url": None, "version": "submittedVersion"},
                    {
                        "pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12345/pdf/x.pdf",
                        "version": "publishedVersion",
                    },
                ]
            }

    class FakeClient:
        def get(self, url: str, **kw):
            return FakeResponse()

    out = fetch_oa_pdf_url("10.1038/example", client=FakeClient())
    assert out is not None and out[0].endswith(".pdf")


def test_extract_text_from_pdf_bytes() -> None:
    text = extract_text_from_pdf_bytes(
        b"%PDF-1.4\n1 0 obj\n<<>>\nstream\nBT /F1 12 Tf (Hello World) Tj\nendstream\nendobj"
    )
    assert "Hello World" in text


def test_read_text_file_rejects_binary_and_empty(tmp_path: Path) -> None:
    good = tmp_path / "paper.txt"
    good.write_text("Hello, UTF-8 world.", encoding="utf-8")
    text = read_text_file(good)
    assert text == "Hello, UTF-8 world."

    bad = tmp_path / "paper.pdf"
    bad.write_bytes(b"\xff\xfe\x00\x01not utf-8")
    with pytest.raises(ValueError, match="UTF-8"):
        read_text_file(bad)

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        read_text_file(empty)


def test_resolve_lawful_access_routes_reports_oa_and_institutional() -> None:
    class FakeResponse:
        def json(self):
            return {
                "locations": [
                    {
                        "pdf_url": None,
                        "landing_page_url": "https://nature.com/articles/x1",
                        "version": "submittedVersion",
                    },
                ]
            }

    class FakeClient:
        def get(self, url: str, **kw):
            return FakeResponse()

    out = resolve_lawful_access_routes("10.1038/example", client=FakeClient())
    assert out["doi"] == "10.1038/example"
    assert out["open_access"] is False
    assert "landing_url" in out
    assert "advice" in out
    assert "institutional" in out


def test_resolve_lawful_access_routes_find_oa_pdf() -> None:
    class FakeResponse:
        def json(self):
            return {
                "locations": [
                    {
                        "pdf_url": "https://pmc.ncbi.nlm.nih.gov/blob.pdf",
                        "landing_page_url": None,
                        "version": "publishedVersion",
                    },
                ]
            }

    class FakeClient:
        def get(self, url: str, **kw):
            return FakeResponse()

    out = resolve_lawful_access_routes("10.1038/oa1", client=FakeClient())
    assert out["open_access"] is True
    assert out["oa_pdf_url"] == "https://pmc.ncbi.nlm.nih.gov/blob.pdf"


def test_ingest_locally_authorized_text_integrates_with_store(tmp_path: Path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "We present DeepCell 2.0, a convolutional neural network for image segmentation. "
        "Source code: https://github.com/example/DeepCell2.",
        encoding="utf-8",
    )
    original = read_text_file(text_path)
    outcome = ingest_locally_authorized_text(
        database,
        doi="10.1038/deepcell2",
        title="DeepCell 2.0: an institutionally accessible paper",
        text=original,
        access_confirmed=True,
    )
    assert outcome.doi == "10.1038/deepcell2"
    inv = inventory_text_artifacts(original)
    assert any("github.com" in u for u in inv.artifact_urls)
    stats = outcome.stats
    assert stats["new_artifacts"] == 1
    assert stats["models_touched"] == 1
    assert outcome.access_scope == "institutional-restricted"
    assert outcome.text_bytes == len(original.encode("utf-8"))
