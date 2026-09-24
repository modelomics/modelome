from __future__ import annotations

from modelome.sources.europe_pmc import EuropePmcSourceAdapter


def test_europe_pmc_exposes_keyword_and_mesh_discovery_terms_with_raw_evidence() -> None:
    source = EuropePmcSourceAdapter()
    raw = {
        "source": "MED",
        "id": "12345",
        "title": "Neural methods for image analysis",
        "abstractText": "A study of a biomedical model.",
        "keywordList": {"keyword": ["machine learning", "foundation model"]},
        "meshHeadingList": {
            "meshHeading": [
                {
                    "descriptorName": "Artificial Intelligence",
                    "qualifierName": ["methods", "standards"],
                },
                {"descriptorName": "Artificial Intelligence"},
            ]
        },
    }

    record = source._record(raw)

    assert "Keywords: machine learning; foundation model" in record.text
    assert "MeSH: Artificial Intelligence; methods; standards" in record.text
    assert record.raw == raw
    assert record.source_record_id == "MED:12345"
