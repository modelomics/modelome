from __future__ import annotations

from modelome.models import ArtifactKind, Identifier, ModelHint, SourceRecord
from modelome.storage import Database


def test_artifact_identifier_index_is_rebuilt_and_deduplicates_reingestion(tmp_path) -> None:
    root = tmp_path / "store"
    record = SourceRecord(
        source_record_id="paper",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.example/paper",
        title="Paper",
        raw={},
        identifiers=(Identifier("doi", "10.1000/example"),),
    )
    first = Database(root)
    first.initialize()
    first.ingest_page("papers", (record,), {}, extractor="fixture")

    reopened = Database(root)
    reopened.initialize()
    reopened.ingest_page("papers", (record,), {}, extractor="fixture")

    assert reopened.table_rows("artifact_identifiers") == [
        {
            "artifact_id": reopened.table_rows("artifacts")[0]["id"],
            "namespace": "doi",
            "value": "10.1000/example",
            "first_seen_revision_id": reopened.table_rows("artifacts")[0]["current_revision_id"],
        }
    ]


def test_model_identifier_and_alias_indexes_are_rebuilt_and_updated(tmp_path) -> None:
    root = tmp_path / "store"
    identifier = Identifier("provider:model", "example-model")
    record = SourceRecord(
        source_record_id="model-card",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://models.example/example-model",
        title="Example model card",
        raw={},
        models=(
            ModelHint(
                local_id="example-model",
                name="Example model",
                identifiers=(identifier,),
                aliases=("Example",),
            ),
        ),
    )
    database = Database(root)
    database.initialize()
    database.ingest_page("models", (record,), {}, extractor="fixture")

    reopened = Database(root)
    reopened.initialize()
    key = (identifier.namespace, identifier.value)
    model_id = reopened.table_rows("models")[0]["id"]
    assert [row["model_id"] for row in reopened._model_identifier_claims_by_key[key]] == [
        model_id
    ]
    assert [row["model_id"] for row in reopened._model_external_identifiers_by_key[key]] == [
        model_id
    ]
    assert (model_id, "Example model") in reopened._model_alias_keys
    assert (model_id, "Example") in reopened._model_alias_keys

    reopened.ingest_page("models", (record,), {}, extractor="fixture")
    assert len(reopened.table_rows("model_aliases")) == 2
    assert len(reopened.table_rows("model_identifier_claims")) == 1
