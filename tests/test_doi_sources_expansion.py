from __future__ import annotations

from modelome.sources.crossref import CrossrefSourceAdapter
from modelome.sources.datacite import DataCiteSourceAdapter


def test_crossref_subject_terms_expand_searchable_text_without_changing_evidence() -> None:
    source = CrossrefSourceAdapter()
    raw = {
        "DOI": "10.5555/model-paper",
        "title": ["A general result"],
        "subject": ["Foundation models", "medical imaging", "Foundation models"],
    }

    record = source._record(raw)

    assert "Subjects: Foundation models; medical imaging" in record.text
    assert record.raw == raw
    assert record.source_record_id == "10.5555/model-paper"


def test_datacite_subject_terms_expand_searchable_text_without_changing_evidence() -> None:
    source = DataCiteSourceAdapter()
    raw = {
        "id": "10.5281/zenodo.123",
        "type": "dois",
        "attributes": {
            "doi": "10.5281/zenodo.123",
            "state": "findable",
            "isActive": True,
            "titles": [{"title": "Model release record"}],
            "subjects": [
                {"subject": "Vision transformer"},
                {"subject": "medical imaging"},
                {"subject": "Vision transformer"},
            ],
        },
    }

    record = source._record(raw)

    assert "Subjects: Vision transformer; medical imaging" in record.text
    assert record.raw == raw
    assert record.source_record_id == "10.5281/zenodo.123"


def test_crossref_pdf_link_with_mime_parameters_remains_full_text() -> None:
    source = CrossrefSourceAdapter()
    record = source._record(
        {
            "DOI": "10.5555/pdf-paper",
            "link": [
                {
                    "URL": "https://publisher.example/paper.pdf",
                    "content-type": " Application/PDF ; version=1.7 ",
                }
            ],
        }
    )

    link = next(link for link in record.links if link.url.endswith("paper.pdf"))
    assert link.relation == "full_text"
    assert link.locator == "$.link[0].URL"
