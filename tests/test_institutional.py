from __future__ import annotations

import json

import pytest

from modelome.institutional import ingest_institutional_text
from modelome.storage import Database


def test_authorized_manual_text_import_extracts_models_and_preserves_restricted_provenance(
    tmp_path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "We introduce TissueNet-X, a deep neural network for spatial transcriptomics. "
        "Code is available at https://github.com/lab/tissuenet-x.",
        encoding="utf-8",
    )

    outcome = ingest_institutional_text(
        database,
        doi="https://doi.org/10.1038/TissueNet.1",
        title="TissueNet-X: an institutionally accessible paper",
        text_path=text_path,
        landing_url="https://www.nature.com/articles/tissuenet?utm_source=library",
        access_confirmed=True,
    )

    assert outcome.source == "institutional-text"
    assert outcome.doi == "10.1038/tissuenet.1"
    assert outcome.access_scope == "institutional-restricted"
    assert outcome.stats["new_artifacts"] == 1
    assert outcome.stats["models_touched"] == 1
    artifact = database.table_rows("artifacts")[0]
    revision = database.table_rows("artifact_revisions")[0]
    assert artifact["source"] == "institutional-text"
    assert artifact["canonical_url"] == "https://doi.org/10.1038/tissuenet.1"
    assert revision["text"].startswith("We introduce TissueNet-X")
    assert json.loads(revision["raw_json"]) == {
        "access_method": "authorized-user-manual-import",
        "access_scope": "institutional-restricted",
        "export_policy": "metadata-only; source text excluded",
        "text_bytes": outcome.text_bytes,
        "text_encoding": "utf-8",
    }
    assert database.search_models("TissueNet-X")
    frontier = database.list_frontier(status="observed")
    assert [item["url"] for item in frontier] == [
        "https://www.nature.com/articles/tissuenet"
    ]
    assert [item["url"] for item in database.list_frontier()] == [
        "https://github.com/lab/tissuenet-x"
    ]


def test_manual_text_import_requires_confirmed_access_and_extracted_utf8_text(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    text_path = tmp_path / "paper.pdf"
    text_path.write_bytes(b"%PDF-1.7\x00not extracted text")

    with pytest.raises(ValueError, match="--access-confirmed"):
        ingest_institutional_text(
            database,
            doi="10.1126/example.1",
            title="Example",
            text_path=text_path,
            access_confirmed=False,
        )
    with pytest.raises(ValueError, match="extracted UTF-8 text"):
        ingest_institutional_text(
            database,
            doi="10.1126/example.1",
            title="Example",
            text_path=text_path,
            access_confirmed=True,
        )
