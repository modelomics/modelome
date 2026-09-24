from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace

import pytest

from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelRelationHint,
    ReleaseHint,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.storage import (
    _MUTABLE_ROW_TABLES,
    _TABLES,
    Database,
    _state_and_table_digests,
)


@pytest.fixture
def database(tmp_path) -> Database:
    result = Database(tmp_path / "store")
    result.initialize()
    return result


def _record(
    record_id: str,
    model: ModelHint,
    *,
    raw: dict | None = None,
    links: tuple[Link, ...] = (),
    relations: tuple[ModelRelationHint, ...] = (),
) -> SourceRecord:
    return SourceRecord(
        source_record_id=record_id,
        kind=ArtifactKind.MODEL_CARD,
        canonical_url=f"https://source.example/models/{record_id}",
        title=model.name,
        raw=raw or {"id": record_id},
        models=(model,),
        links=links,
        model_relations=relations,
    )


def test_streaming_state_digest_is_byte_compatible_with_legacy_encoding() -> None:
    tables = {name: [] for name in _TABLES}
    tables["artifacts"] = [
        {"id": "second", "nested": {"name": "B", "number": 2}},
        {"id": "first", "nested": {"name": "A", "number": 1}},
    ]
    tables["models"] = [{"id": "model", "aliases": ["one", "two"]}]

    legacy = hashlib.sha256()
    legacy_table_digests = {}
    for name in _TABLES:
        payload = json.dumps(
            list(tables[name]), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        legacy_table_digests[name] = hashlib.sha256(payload).hexdigest()
        legacy.update(len(name).to_bytes(4, "big"))
        legacy.update(name.encode("utf-8"))
        legacy.update(len(payload).to_bytes(8, "big"))
        legacy.update(payload)

    state_digest, table_digests = _state_and_table_digests(tables)

    assert state_digest == legacy.hexdigest()
    assert table_digests == legacy_table_digests


def test_transaction_rolls_back_in_place_updates_without_copying_append_only_rows(
    database,
) -> None:
    run_id = database.start_run("catalog-a")
    database.ingest_page(
        "catalog-a",
        (_record("one", ModelHint("primary", "Example Model")),),
        {"cursor": "1"},
        run_id,
        "fixture",
    )
    with pytest.raises(RuntimeError, match="rollback"), database._write_transaction(
        "fixture"
    ):
        database._tables["artifacts"][0]["title"] = "Changed"
        database._tables["artifact_revisions"].append({"id": "discarded"})
        raise RuntimeError("rollback")

    assert database._tables["artifacts"][0]["title"] == "Example Model"
    assert len(database._tables["artifact_revisions"]) == 1
    assert "artifacts" in _MUTABLE_ROW_TABLES
    assert "artifact_revisions" not in _MUTABLE_ROW_TABLES


def test_ingest_batch_persists_page_checkpoints_in_one_commit(database) -> None:
    before = set((database.root / "commits").iterdir())
    first = SourcePage(
        records=(_record("one", ModelHint("primary", "First")),),
        next_state={"cursor": "two"},
        complete=False,
    )
    second = SourcePage(
        records=(_record("two", ModelHint("primary", "Second")),),
        next_state={"cursor": "done"},
        complete=True,
    )

    with database.ingest_batch():
        first_result = database.ingest_page("catalog", first, extractor="fixture")
        second_result = database.ingest_page("catalog", second, extractor="fixture")

    after = set((database.root / "commits").iterdir())
    assert len(after - before) == 1
    assert first_result["new_artifacts"] == 1
    assert second_result["new_artifacts"] == 1
    assert database.get_source_state("catalog") == {"cursor": "done"}
    assert database.stats()["models"] == 2


def test_exact_identifiers_resolve_but_equal_names_do_not_merge(database, tmp_path):
    identifier = Identifier("provider:model", "team/model-1")
    first = _record(
        "one",
        ModelHint("primary", "Shared Name", identifiers=(identifier,), aliases=("One",)),
    )
    second = _record(
        "two",
        ModelHint("primary", "Provider Display Name", identifiers=(identifier,)),
    )
    same_name_without_identifier = _record("three", ModelHint("primary", "Shared Name"))

    first_run = database.start_run("catalog-a")
    initial = database.ingest_page(
        "catalog-a", (first,), {"cursor": "1"}, first_run, "fixture"
    )
    replay = database.ingest_page(
        "catalog-a", (first,), {"cursor": "1"}, first_run, "fixture"
    )
    database.ingest_page("catalog-b", (second,), {"cursor": "2"}, extractor="fixture")
    database.ingest_page(
        "catalog-c", (same_name_without_identifier,), {"cursor": "3"}, extractor="fixture"
    )

    assert initial["new_artifacts"] == 1
    assert initial["new_revisions"] == 1
    assert replay["new_artifacts"] == 0
    assert replay["new_revisions"] == 0
    assert database.stats()["models"] == 2

    exact = database.search_models("provider:model:team/model-1")
    assert len(exact) == 1
    detail = database.model_detail(exact[0]["id"])
    assert detail is not None
    assert {artifact["source"] for artifact in detail["artifacts"]} == {
        "catalog-a",
        "catalog-b",
    }
    assert {"Shared Name", "One", "Provider Display Name"} <= set(detail["aliases"])

    same_name_results = database.search_models("Shared Name")
    assert len({item["id"] for item in same_name_results}) == 2

    other = Database(tmp_path / "second-store")
    other.initialize()
    other.ingest_page("catalog-a", (first,), {}, extractor="fixture")
    assert other.search_models("provider:model:team/model-1")[0]["id"] == exact[0]["id"]


def test_corrected_away_model_identifier_does_not_poison_future_resolution(database):
    identifier_x = Identifier("provider:model", "model-x")
    identifier_y = Identifier("provider:model", "model-y")

    def assertion(version: int, identifier: Identifier) -> SourceRecord:
        return SourceRecord(
            source_record_id="alpha-card",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url="https://provider.example/alpha",
            title="Alpha",
            raw={"version": version},
            models=(ModelHint("model", "Alpha", identifiers=(identifier,)),),
        )

    database.ingest_page("provider", (assertion(1, identifier_x),), {}, extractor="fixture")
    original_id = database.search_models("provider:model:model-x")[0]["id"]

    # A source revision that still asserts X must retain its model identity.
    database.ingest_page("provider", (assertion(2, identifier_x),), {}, extractor="fixture")
    assert database.search_models("provider:model:model-x")[0]["id"] == original_id

    # Once the current source revision corrects X to Y, a later independent X
    # assertion must not inherit Alpha's historical aliases or identity.
    database.ingest_page("provider", (assertion(3, identifier_y),), {}, extractor="fixture")
    beta = SourceRecord(
        source_record_id="beta-card",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://other.example/beta",
        title="Beta",
        raw={"id": "beta"},
        models=(ModelHint("model", "Beta", identifiers=(identifier_x,)),),
    )
    result = database.ingest_page("other", (beta,), {}, extractor="fixture")

    rebound = database.search_models("provider:model:model-x")[0]
    assert result["record_errors"] == 0
    assert rebound["id"] != original_id
    assert rebound["canonical_name"] == "Beta"
    assert rebound["aliases"] == ["Beta"]


def test_corrected_away_release_identifier_can_be_rebound_to_another_model(database):
    weight_x = Identifier("weights:sha256", "digest-x")
    weight_y = Identifier("weights:sha256", "digest-y")

    def release_card(
        record_id: str,
        model_name: str,
        model_identifier: str,
        revision: str,
        weight_identifier: Identifier,
    ) -> SourceRecord:
        return SourceRecord(
            source_record_id=record_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=f"https://provider.example/{record_id}",
            title=model_name,
            raw={"revision": revision},
            models=(
                ModelHint(
                    "model",
                    model_name,
                    identifiers=(Identifier("provider:model", model_identifier),),
                ),
            ),
            releases=(
                ReleaseHint(
                    "release",
                    "model",
                    revision=revision,
                    identifiers=(weight_identifier,),
                ),
            ),
        )

    alpha_x = release_card("alpha", "Alpha", "alpha", "r1", weight_x)
    database.ingest_page("provider", (alpha_x,), {}, extractor="fixture")
    alpha_id = database.search_models("provider:model:alpha")[0]["id"]
    old_release_id = database.model_detail(alpha_id)["releases"][0]["id"]

    alpha_y = release_card("alpha", "Alpha", "alpha", "r2", weight_y)
    database.ingest_page("provider", (alpha_y,), {}, extractor="fixture")
    beta_x = release_card("beta", "Beta", "beta", "b1", weight_x)
    result = database.ingest_page("other", (beta_x,), {}, extractor="fixture")

    beta_id = database.search_models("provider:model:beta")[0]["id"]
    beta_release = database.model_detail(beta_id)["releases"][0]
    assert result["record_errors"] == 0
    assert beta_release["id"] != old_release_id
    assert beta_release["identifiers"] == [
        {"namespace": "weights:sha256", "value": "digest-x"}
    ]
    old_release = next(
        release
        for release in database.model_detail(alpha_id)["releases"]
        if release["id"] == old_release_id
    )
    assert old_release["identifiers"] == []


def test_artifact_revisions_are_content_addressed_and_immutable(database):
    hint = ModelHint("model", "Revision Model")
    original = _record("changing", hint, raw={"version": 1})
    changed = replace(original, raw={"version": 2}, modified_at="2026-09-01")

    database.ingest_page("changing-source", (original,), {"page": 1}, extractor="fixture")
    database.ingest_page("changing-source", (changed,), {"page": 2}, extractor="fixture")

    stats = database.stats()
    assert stats["artifacts"] == 1
    assert stats["artifact_revisions"] == 2
    existing_revisions = {
        row["id"]: row for row in database.table_rows("artifact_revisions")
    }

    third = replace(original, raw={"version": 3}, modified_at="2026-09-02")
    database.ingest_page("changing-source", (third,), {"page": 3}, extractor="fixture")

    all_revisions = {
        row["id"]: row for row in database.table_rows("artifact_revisions")
    }
    assert database.stats()["artifact_revisions"] == 3
    assert len(all_revisions.keys() - existing_revisions.keys()) == 1
    assert {key: all_revisions[key] for key in existing_revisions} == existing_revisions


def test_derived_extraction_is_pruned_only_when_replaying_the_same_revision(database):
    class MutableExtractor:
        name = "mutable-derived"

        def __init__(self) -> None:
            self.model_name = "First Derived Model"

        def extract(self, _record: SourceRecord) -> tuple[ModelHint, ...]:
            return (ModelHint("derived", self.model_name),)

    record = SourceRecord(
        source_record_id="derived-paper",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.example/derived-paper",
        title="Derived paper",
        raw={"version": 1},
    )
    extractor = MutableExtractor()
    database.ingest_page("derived", (record,), {"page": 1}, extractor=extractor)
    extractor.model_name = "Second Derived Model"
    database.ingest_page("derived", (record,), {"page": 2}, extractor=extractor)

    links = database.table_rows("artifact_model_links")
    assert len(links) == 1
    derived_alias_evidence = [
        row["value_json"]
        for row in database.table_rows("evidence_provenance")
        if row["extractor"] == "mutable-derived" and row["predicate"] == "alias"
    ]
    assert derived_alias_evidence == ['"Second Derived Model"']


def test_trained_releases_use_exact_checkpoint_identity_and_are_idempotent(database):
    model_identifier = Identifier("provider:model", "lab/base-model")
    first_checkpoint = Identifier("weights:sha256", "111111")
    second_checkpoint = Identifier("weights:sha256", "222222")
    record = SourceRecord(
        source_record_id="release-card",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://provider.example/lab/base-model",
        title="Base Model",
        raw={"model": "lab/base-model", "revisions": ["a1", "b2"]},
        models=(
            ModelHint("model", "Base Model", identifiers=(model_identifier,)),
        ),
        releases=(
            ReleaseHint(
                "checkpoint-a",
                "model",
                version="1.0",
                revision="a1",
                identifiers=(first_checkpoint,),
                released_at="2025-01-01",
                metadata={"format": "safetensors", "parameters": 100},
                locator="raw:revisions:0",
            ),
            ReleaseHint(
                "checkpoint-b",
                "model",
                version="1.1",
                revision="b2",
                identifiers=(second_checkpoint,),
                released_at="2025-02-01",
                metadata={"format": "safetensors", "parameters": 100},
                locator="raw:revisions:1",
            ),
        ),
    )
    page = SourcePage(
        records=(record,),
        next_state={"complete": True},
        complete=True,
        upstream_count=1,
    )

    database.ingest_page("provider", page, extractor="fixture")
    database.ingest_page("provider", page, extractor="fixture")
    stats = database.stats()
    assert stats["model_releases"] == 2
    assert stats["artifact_release_links"] == 2

    mirrored = SourceRecord(
        source_record_id="release-mirror",
        kind=ArtifactKind.WEIGHTS,
        canonical_url="https://mirror.example/base-model-a1",
        title="Base Model checkpoint a1",
        raw={"digest": first_checkpoint.value},
        models=(ModelHint("model", "Base Model Mirror", identifiers=(model_identifier,)),),
        releases=(
            ReleaseHint(
                "mirror-a",
                "model",
                version="release-1",
                revision="a1",
                identifiers=(first_checkpoint,),
                metadata={"mirror": True},
            ),
        ),
    )
    database.ingest_page("mirror", (mirrored,), {}, extractor="fixture")

    assert database.stats()["model_releases"] == 2
    assert database.stats()["artifact_release_links"] == 3
    model = database.search_models("provider:model:lab/base-model")[0]
    detail = database.model_detail(model["id"])
    assert detail is not None
    assert detail["status"] == "released"
    releases = {release["revision"]: release for release in detail["releases"]}
    assert set(releases) == {"a1", "b2"}
    assert releases["a1"]["metadata"]["format"] == "safetensors"
    assert releases["a1"]["identifiers"] == [
        {"namespace": "weights:sha256", "value": "111111"}
    ]
    assert len(releases["a1"]["evidence"]) == 2
    assert len(releases["b2"]["evidence"]) == 1

    coverage = database.coverage_metrics()
    assert coverage["global"]["models_with_exact_identifiers"] == 1
    assert coverage["global"]["model_statuses"]["released"] == 1
    provider_coverage = next(
        source for source in coverage["sources"] if source["source"] == "provider"
    )
    assert provider_coverage == {
        "source": "provider",
        "upstream_count": 1,
        "complete": True,
        "artifacts": 1,
        "active_artifacts": 1,
        "current_revisions": 1,
        "model_links": 1,
        "release_links": 2,
    }


def test_authoritative_snapshot_tombstones_unseen_artifacts_without_deleting(database):
    first = _record("snapshot-a", ModelHint("model", "Snapshot A"))
    second = _record("snapshot-b", ModelHint("model", "Snapshot B"))
    first_run = database.start_run("snapshot")
    database.ingest_page(
        "snapshot",
        SourcePage(
            records=(first, second),
            next_state={"snapshot": 1},
            complete=True,
            upstream_count=2,
            authoritative_snapshot=True,
        ),
        run_id=first_run,
        extractor="fixture",
    )
    database.finish_run(first_run, "complete")

    second_run = database.start_run("snapshot")
    database.ingest_page(
        "snapshot",
        SourcePage(
            records=(first,),
            next_state={"snapshot": 2},
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        ),
        run_id=second_run,
        extractor="fixture",
        allow_large_snapshot_shrink=True,
    )

    assert database.stats()["artifacts"] == 2
    assert database.stats()["active_artifacts"] == 1
    rows = {
        row["source_record_id"]: row
        for row in database.table_rows("artifacts")
        if row["source"] == "snapshot"
    }
    assert rows["snapshot-a"]["active"] == 1
    assert rows["snapshot-a"]["last_seen_run_id"] == second_run
    assert rows["snapshot-b"]["active"] == 0
    assert rows["snapshot-b"]["tombstoned_at"] is not None


def test_explicit_source_deletion_tombstones_and_reappearance_reactivates(database):
    original = _record("mutable-paper", ModelHint("model", "Mutable Neural Model"))
    deletion = SourceRecord(
        source_record_id=original.source_record_id,
        kind=original.kind,
        canonical_url=original.canonical_url,
        title="[deleted record] mutable-paper",
        raw={"deleted": True},
        modified_at="2026-09-01",
        identifiers=(Identifier("upstream", "mutable-paper"),),
        deleted=True,
    )

    database.ingest_page("mutable-corpus", (original,), {}, extractor="fixture")
    result = database.ingest_page(
        "mutable-corpus", (deletion,), {}, extractor="fixture"
    )

    artifact = next(
        row
        for row in database.table_rows("artifacts")
        if row["source"] == "mutable-corpus"
        and row["source_record_id"] == "mutable-paper"
    )
    assert result["new_artifacts"] == 0
    assert result["new_revisions"] == 1
    assert artifact["active"] == 0
    assert artifact["tombstoned_at"] is not None
    assert database.stats()["active_artifacts"] == 0
    model = database.search_models("Mutable Neural Model")[0]
    assert model["current_artifact_count"] == 0
    assert model["active_current_artifact_count"] == 0

    database.ingest_page("mutable-corpus", (original,), {}, extractor="fixture")

    reactivated = next(
        row
        for row in database.table_rows("artifacts")
        if row["source"] == "mutable-corpus"
        and row["source_record_id"] == "mutable-paper"
    )
    assert reactivated["active"] == 1
    assert reactivated["tombstoned_at"] is None
    assert database.search_models("Mutable Neural Model")[0][
        "active_current_artifact_count"
    ] == 1


def test_explicit_source_deletion_cannot_introduce_model_claims(database):
    deletion = SourceRecord(
        source_record_id="deleted-with-claim",
        kind=ArtifactKind.PAPER,
        canonical_url="https://example.test/deleted-with-claim",
        title="Deleted",
        raw={"deleted": True},
        models=(ModelHint("model", "Invalid Deleted Model"),),
        deleted=True,
    )

    result = database.ingest_page(
        "deleting-source", (deletion,), {}, extractor="fixture"
    )

    assert result["errors"] == 1
    assert result["checkpoint_advanced"] == 0
    assert database.stats()["artifacts"] == 0


def test_provenance_only_link_is_observed_but_never_queued(database):
    record = SourceRecord(
        source_record_id="bulk-control",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url="https://data.example.test/shard.xml.gz",
        title="Bulk shard",
        raw={},
        links=(
            Link(
                "https://data.example.test/shard.xml.gz.md5",
                relation="checksum",
                crawl=False,
            ),
        ),
    )

    database.ingest_page("bulk-source", (record,), {}, extractor="fixture")

    frontier = database.table_rows("url_frontier")
    assert len(frontier) == 1
    assert frontier[0]["url"] == "https://data.example.test/shard.xml.gz.md5"
    assert frontier[0]["status"] == "observed"


def test_empty_authoritative_snapshot_cannot_erase_a_nonempty_catalog(database):
    record = _record("survivor", ModelHint("model", "Surviving Model"))
    first_run = database.start_run("guarded-snapshot")
    database.ingest_page(
        "guarded-snapshot",
        SourcePage(
            records=(record,),
            next_state={"snapshot": 1},
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        ),
        run_id=first_run,
        extractor="fixture",
    )
    database.finish_run(first_run, "complete")

    empty_run = database.start_run("guarded-snapshot")
    with pytest.raises(ValueError, match="refusing an empty authoritative snapshot"):
        database.ingest_page(
            "guarded-snapshot",
            SourcePage(
                records=(),
                next_state={"snapshot": 2},
                complete=True,
                upstream_count=0,
                authoritative_snapshot=True,
            ),
            run_id=empty_run,
            extractor="fixture",
        )

    assert database.stats()["active_artifacts"] == 1
    assert database.get_source_state("guarded-snapshot") == {"snapshot": 1}


def test_large_authoritative_snapshot_shrink_requires_operator_override(database):
    original = _record("catalog-original", ModelHint("model", "Catalog Original"))
    first_run = database.start_run("catalog")
    database.ingest_page(
        "catalog",
        SourcePage(
            records=(original,),
            next_state={"snapshot": 1},
            complete=True,
            upstream_count=3594,
            authoritative_snapshot=True,
        ),
        run_id=first_run,
        extractor="fixture",
    )
    database.finish_run(first_run, "complete")

    suspicious = _record("catalog-one", ModelHint("model", "Suspicious Lone Row"))
    shrink_run = database.start_run("catalog")
    with pytest.raises(ValueError, match="count drop from 3594 to 1"):
        database.ingest_page(
            "catalog",
            SourcePage(
                records=(suspicious,),
                next_state={"snapshot": 2},
                complete=True,
                upstream_count=1,
                authoritative_snapshot=True,
            ),
            run_id=shrink_run,
            extractor="fixture",
        )

    assert database.get_source_state("catalog") == {"snapshot": 1}
    assert database.stats()["active_artifacts"] == 1


def test_page_quarantines_bad_records_without_advancing_checkpoint(database):
    good = _record("good", ModelHint("model", "Good Model"))
    bad = _record("bad", ModelHint("model", "Bad Model"), raw={"bad": object()})
    database.ingest_page(
        "resilient",
        (),
        {"cursor": "retry-this-page"},
        complete=False,
        upstream_count=7,
        extractor="fixture",
    )
    run_id = database.start_run("resilient")

    result = database.ingest_page(
        "resilient",
        SourcePage(
            records=(good, bad),
            next_state={"cursor": "next"},
            complete=True,
            upstream_count=9,
        ),
        run_id=run_id,
        extractor="fixture",
    )

    assert result == {
        "records_seen": 2,
        "new_artifacts": 1,
        "new_revisions": 1,
        "models_touched": 1,
        "links_discovered": 0,
        "errors": 1,
        "record_errors": 1,
        "checkpoint_advanced": 0,
    }
    assert database.get_source_state("resilient") == {"cursor": "retry-this-page"}
    status = database.source_status()[0]
    assert status["complete"] is False
    assert status["upstream_count"] == 7
    assert database.stats()["dead_letters"] == 1
    letter = database.list_dead_letters("resilient")[0]
    assert letter["source_record_id"] == "bad"
    assert letter["stage"] == "record_ingest"
    assert letter["attempts"] == 1

    database.ingest_page(
        "resilient", (bad,), {"cursor": "after-bad"}, run_id, "fixture"
    )
    assert database.list_dead_letters("resilient")[0]["attempts"] == 2

    finished = database.finish_run(run_id, "partial", result, "one quarantined record")
    assert finished["status"] == "partial"
    assert database.source_status()[0]["last_run"]["status"] == "partial"


def test_source_issue_also_holds_checkpoint_for_direct_storage_call(database):
    database.ingest_page(
        "issue-source",
        (),
        {"cursor": "prior"},
        complete=False,
        upstream_count=4,
        extractor="fixture",
    )
    run_id = database.start_run("issue-source")
    result = database.ingest_page(
        "issue-source",
        SourcePage(
            records=(_record("good-issue-page", ModelHint("model", "Good")),),
            next_state={"cursor": "next"},
            complete=True,
            upstream_count=5,
            issues=(
                SourceIssue(
                    source_record_id="malformed",
                    stage="source_normalize",
                    error="missing required field",
                ),
            ),
        ),
        run_id=run_id,
        extractor="fixture",
    )

    assert result["errors"] == 1
    assert result["record_errors"] == 0
    assert result["checkpoint_advanced"] == 0
    assert database.get_source_state("issue-source") == {"cursor": "prior"}
    status = database.source_status()[0]
    assert status["complete"] is False
    assert status["upstream_count"] == 4


def test_strict_page_failure_rolls_back_records_and_checkpoint(database):
    good = _record("strict-good", ModelHint("model", "Good Model"))
    bad = _record("strict-bad", ModelHint("model", "Bad Model"), raw={"bad": object()})
    run_id = database.start_run("strict")

    with pytest.raises(TypeError):
        database.ingest_page(
            "strict",
            (good, bad),
            {"cursor": "must-not-commit"},
            run_id,
            "fixture",
            quarantine_errors=False,
        )

    assert database.get_source_state("strict") == {}
    stats = database.stats()
    assert stats["artifacts"] == 0
    assert stats["models"] == 0
    assert stats["dead_letters"] == 0


def test_only_text_and_explicit_urls_enter_the_frontier(database):
    openalex_record = SourceRecord(
        source_record_id="W1",
        kind=ArtifactKind.PAPER,
        canonical_url="https://api.openalex.org/works/W1",
        title="A work with many structured references",
        raw={
            "authorships": [
                {"author": {"id": "https://openalex.org/A1"}},
                {"author": {"id": "https://openalex.org/A2"}},
            ]
        },
        text="Institution metadata: https://openalex.org/I1",
    )

    result = database.ingest_page(
        "openalex",
        (openalex_record,),
        {},
        extractor="fixture",
        enqueue_links=True,
        link_depth=4,
    )

    assert result["links_discovered"] == 1
    assert database.list_frontier() == []
    observed = database.list_frontier("observed")
    assert {item["url"] for item in observed} == {"https://openalex.org/I1"}
    assert {item["depth"] for item in observed} == {4}
    assert len(database.table_rows("url_discoveries")) == 1
    assert sum(
        row["subject_type"] == "url"
        for row in database.table_rows("evidence_provenance")
    ) == 1

    explicit_record = _record(
        "explicit",
        ModelHint("model", "Explicit Link Model"),
        links=(Link("https://openalex.org/A1", "references"),),
    )
    database.ingest_page(
        "catalog",
        (explicit_record,),
        {},
        extractor="fixture",
        enqueue_links=True,
        link_depth=2,
    )

    pending = database.list_frontier()
    assert [(item["url"], item["depth"]) for item in pending] == [
        ("https://openalex.org/A1", 2)
    ]
    assert {item["url"] for item in database.list_frontier("observed")} == {
        "https://openalex.org/I1"
    }


def test_frontier_claim_is_atomic_and_recovers_an_expired_lease(database, monkeypatch):
    clock = {"now": "2026-01-01T00:00:00.000000+00:00"}
    monkeypatch.setattr("modelome.storage._now", lambda: clock["now"])
    database.ingest_page(
        "seed",
        (
            _record(
                "lease",
                ModelHint("model", "Lease Model"),
                links=(Link("https://crawl.example/model-card"),),
            ),
        ),
        {},
        extractor="fixture",
    )

    first = database.claim_frontier(limit=1, lease_seconds=60)
    assert first[0]["status"] == "claimed"
    assert first[0]["attempts"] == 1
    assert database.claim_frontier(limit=1, lease_seconds=60) == []

    clock["now"] = "2026-01-02T00:00:00.000000+00:00"

    recovered = database.claim_frontier(limit=1, lease_seconds=60)
    assert recovered[0]["status"] == "claimed"
    assert recovered[0]["attempts"] == 2


def test_frontier_refreshes_stale_done_and_failed_urls_but_not_ignored(
    database, monkeypatch
):
    clock = {"now": "2026-01-01T00:00:00.000000+00:00"}
    monkeypatch.setattr("modelome.storage._now", lambda: clock["now"])
    urls = (
        "https://crawl.example/done",
        "https://crawl.example/failed",
        "https://crawl.example/ignored",
    )
    database.ingest_page(
        "seed",
        (
            _record(
                "refresh",
                ModelHint("model", "Refresh Model"),
                links=tuple(Link(url) for url in urls),
            ),
        ),
        {},
        extractor="fixture",
    )
    assert len(database.claim_frontier(limit=3)) == 3
    database.update_frontier(urls[0], "done")
    database.update_frontier(urls[1], "failed", error="temporary failure")
    database.update_frontier(urls[2], "ignored", error="policy")
    assert database.claim_frontier(limit=3, refresh_after_seconds=3600) == []

    clock["now"] = "2026-01-02T00:00:00.000000+00:00"

    never_fetched = "https://crawl.example/new"
    database.ingest_page(
        "seed",
        (
            _record(
                "new-pending",
                ModelHint("model", "New Pending Model"),
                links=(Link(never_fetched),),
            ),
        ),
        {},
        extractor="fixture",
    )
    first = database.claim_frontier(limit=1, refresh_after_seconds=60)
    assert first[0]["url"] == never_fetched

    refreshed = database.claim_frontier(limit=3, refresh_after_seconds=60)
    assert {item["url"] for item in refreshed} == set(urls[:2])
    assert {item["attempts"] for item in refreshed} == {1}
    assert database.list_frontier("ignored")[0]["url"] == urls[2]


def test_source_run_lease_blocks_overlap_and_recovers_abandoned_run(database, monkeypatch):
    clock = {"now": "2026-01-01T00:00:00.000000+00:00"}
    monkeypatch.setattr("modelome.storage._now", lambda: clock["now"])
    first_run = database.start_run("exclusive-source")

    with pytest.raises(RuntimeError, match="another sync run is active"):
        database.start_run("exclusive-source")
    clock["now"] = "2026-01-02T00:00:00.000000+00:00"

    replacement = database.start_run("exclusive-source")
    old_status = next(
        row["status"]
        for row in database.table_rows("sync_runs")
        if row["id"] == first_run
    )

    assert replacement != first_run
    assert old_status == "abandoned"


def test_structured_hint_change_creates_revision_and_marks_old_link_historical(database):
    documented = _record("mutable", ModelHint("model", "Retracted Model Claim"))
    corrected = SourceRecord(
        source_record_id=documented.source_record_id,
        kind=documented.kind,
        canonical_url=documented.canonical_url,
        title=documented.title,
        raw=documented.raw,
    )

    first = database.ingest_page("mutable-source", (documented,), {}, extractor="fixture")
    second = database.ingest_page("mutable-source", (corrected,), {}, extractor="fixture")

    assert first["new_revisions"] == 1
    assert second["new_revisions"] == 1
    result = database.search_models("Retracted Model Claim")[0]
    assert result["artifact_count"] == 1
    assert result["current_artifact_count"] == 0
    assert result["active_current_artifact_count"] == 0
    detail = database.model_detail(result["id"])
    assert detail is not None
    assert detail["artifacts"][0]["active"] == 1
    assert detail["artifacts"][0]["is_current"] == 0


def test_declared_and_derived_provenance_are_separate_and_reextractable(database):
    class MutableExtractor:
        name = "derived-fixture"

        def __init__(self) -> None:
            self.hints = (ModelHint("derived", "Derived Candidate"),)

        def extract(self, _record):
            return self.hints

    declared = ModelHint(
        "declared",
        "Declared Model",
        identifiers=(Identifier("provider:model", "declared"),),
    )
    record = SourceRecord(
        source_record_id="provenance",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://source.example/provenance",
        title="Declared Model",
        raw={"id": "provenance"},
        identifiers=(Identifier("doi", "10.1234/provenance"),),
        links=(Link("https://repository.example/provenance", "implementation"),),
        models=(declared,),
        releases=(ReleaseHint("release", "declared", version="1"),),
        model_relations=(
            ModelRelationHint(
                "declared",
                "related_to",
                ModelHint("named-target", "Named Target"),
            ),
        ),
    )
    extractor = MutableExtractor()
    database.ingest_page("source", (record,), {}, extractor=extractor)

    model_provenance = Counter(
        row["extractor"] for row in database.table_rows("artifact_model_links")
    )
    assert dict(model_provenance) == {"derived-fixture": 1, "source-declared": 1}
    assert {
        row["extractor"] for row in database.table_rows("artifact_release_links")
    } == {"source-declared"}
    assert {
        row["extractor"] for row in database.table_rows("model_relation_claims")
    } == {"source-declared"}
    assert {
        row["extractor"]
        for row in database.table_rows("evidence_provenance")
        if row["subject_type"] == "url"
    } == {"source-declared"}

    extractor.hints = ()
    replay = database.ingest_page("source", (record,), {}, extractor=extractor)
    assert replay["new_revisions"] == 0
    assert database.search_models("Derived Candidate")[0]["current_artifact_count"] == 0
    assert database.search_models("Declared Model")[0]["current_artifact_count"] == 1

    assert not any(
        row["extractor"] == "derived-fixture"
        for row in database.table_rows("artifact_model_links")
    )
    assert not any(
        row["extractor"] == "derived-fixture"
        for row in database.table_rows("evidence_provenance")
    )


def test_model_detail_projects_cross_source_artifact_connections(database):
    doi = Identifier("doi", "10.1234/example-model")
    catalog_record = SourceRecord(
        source_record_id="catalog-entry",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url="https://catalog.example/models/example",
        title="Example Model",
        raw={"model": "Example Model"},
        identifiers=(doi,),
        links=(
            Link(
                "https://github.com/example/model?utm_source=catalog",
                "implementation",
                "raw:repository",
            ),
        ),
        models=(ModelHint("catalog-model", "Example Model"),),
    )
    paper_record = SourceRecord(
        source_record_id="paper-entry",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.example/10.1234/example-model",
        title="The Example Model Paper",
        raw={"doi": doi.value},
        identifiers=(doi,),
        # The same name is a separate source-scoped model identity.
        models=(ModelHint("paper-model", "Example Model"),),
    )
    repository_record = SourceRecord(
        source_record_id="repository-entry",
        kind=ArtifactKind.CODE_REPOSITORY,
        canonical_url="https://github.com/example/model/",
        title="Example model implementation",
        raw={"repository": "example/model"},
    )

    database.ingest_page("catalog", (catalog_record,), {}, extractor="fixture")
    database.ingest_page("papers", (paper_record,), {}, extractor="fixture")
    database.ingest_page("repositories", (repository_record,), {}, extractor="fixture")
    # Replaying the direct artifact must not duplicate projected connections.
    database.ingest_page("catalog", (catalog_record,), {}, extractor="fixture")

    catalog_model = None
    for result in database.search_models("Example Model"):
        candidate_detail = database.model_detail(result["id"])
        assert candidate_detail is not None
        if any(artifact["source"] == "catalog" for artifact in candidate_detail["artifacts"]):
            catalog_model = result
            break
    assert catalog_model is not None
    detail = database.model_detail(catalog_model["id"])
    assert detail is not None
    assert len(detail["artifacts"]) == 1
    assert detail["artifacts"][0]["source"] == "catalog"
    assert database.stats()["models"] == 2

    connected = {item["source"]: item for item in detail["connected_artifacts"]}
    assert set(connected) == {"papers", "repositories"}
    paper_connection = connected["papers"]["connections"]
    assert len(paper_connection) == 1
    assert paper_connection[0]["connection_type"] == "shared_identifier"
    assert paper_connection[0]["namespace"] == "doi"
    assert paper_connection[0]["value"] == doi.value
    assert paper_connection[0]["evidence_source"]["source"] == "catalog"
    assert paper_connection[0]["matched_evidence"]["source"] == "papers"

    repository_connection = connected["repositories"]["connections"]
    assert len(repository_connection) == 1
    assert repository_connection[0]["connection_type"] == "discovered_url"
    assert repository_connection[0]["url"] == "https://github.com/example/model"
    assert repository_connection[0]["relation"] == "implementation"
    assert repository_connection[0]["evidence_source"]["source"] == "catalog"


def test_connected_artifacts_use_only_current_claims_and_include_reverse_links(database):
    doi = Identifier("doi", "10.1234/corrected-away")
    direct_url = "https://catalog.example/models/current"
    initial = SourceRecord(
        source_record_id="catalog-entry",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=direct_url,
        title="Current Model",
        raw={"version": 1},
        identifiers=(doi,),
        links=(Link("https://repository.example/current", "implementation"),),
        models=(ModelHint("model", "Current Model"),),
    )
    paper = SourceRecord(
        source_record_id="paper",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.example/corrected-away",
        title="Historical paper match",
        raw={"doi": doi.value},
        identifiers=(doi,),
    )
    repository = SourceRecord(
        source_record_id="repository",
        kind=ArtifactKind.CODE_REPOSITORY,
        canonical_url="https://repository.example/current",
        title="Historical repository match",
        raw={},
    )
    database.ingest_page("catalog", (initial,), {}, extractor="fixture")
    database.ingest_page("papers", (paper,), {}, extractor="fixture")
    database.ingest_page("repositories", (repository,), {}, extractor="fixture")
    model_id = database.search_models("Current Model")[0]["id"]
    assert {item["source"] for item in database.model_detail(model_id)["connected_artifacts"]} == {
        "papers",
        "repositories",
    }

    corrected = SourceRecord(
        source_record_id="catalog-entry",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=direct_url,
        title="Current Model",
        raw={"version": 2},
        models=(ModelHint("model", "Current Model"),),
    )
    database.ingest_page("catalog", (corrected,), {}, extractor="fixture")
    assert database.model_detail(model_id)["connected_artifacts"] == []

    reverse = SourceRecord(
        source_record_id="reverse-link",
        kind=ArtifactKind.CODE_REPOSITORY,
        canonical_url="https://reverse.example/repository",
        title="Repository linking back to the catalog",
        raw={},
        links=(Link(direct_url, "documents"),),
    )
    database.ingest_page("reverse", (reverse,), {}, extractor="fixture")
    connected = database.model_detail(model_id)["connected_artifacts"]
    assert len(connected) == 1
    assert connected[0]["source"] == "reverse"
    connection = connected[0]["connections"][0]
    assert connection["connection_type"] == "discovered_url"
    assert connection["direction"] == "incoming"
    assert connection["url"] == direct_url
    assert connection["relation"] == "documents"
    assert connection["evidence_source"]["artifact_id"] == connected[0]["artifact_id"]
    assert connection["evidence_source"]["source"] == "reverse"
    assert connection["evidence_source"]["source_record_id"] == "reverse-link"


def test_connected_artifact_uses_requested_url_alias_after_redirect(database):
    requested = "https://provider.example/redirect/model-card"
    origin = _record(
        "origin",
        ModelHint("model", "Redirected Card Model"),
        links=(Link(requested, "model_card"),),
    )
    fetched = SourceRecord(
        source_record_id="fetched-card",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://provider.example/cards/canonical",
        title="Canonical model card",
        raw={"canonical": True},
        identifiers=(Identifier("url", requested),),
    )
    database.ingest_page("origin", (origin,), {}, extractor="fixture")
    database.ingest_page("frontier", (fetched,), {}, extractor="fixture")

    model = database.search_models("Redirected Card Model")[0]
    detail = database.model_detail(model["id"])

    assert detail is not None
    assert detail["connected_artifacts"][0]["canonical_url"] == (
        "https://provider.example/cards/canonical"
    )
    assert detail["connected_artifacts"][0]["connections"][0]["url"] == requested


def test_relations_frontier_depth_and_conflicting_identifier_claims(database):
    first_identifier = Identifier("registry", "first")
    second_identifier = Identifier("registry", "second")
    database.ingest_page(
        "ids",
        (_record("first", ModelHint("first", "First", identifiers=(first_identifier,))),),
        {},
        extractor="fixture",
    )
    database.ingest_page(
        "ids",
        (_record("second", ModelHint("second", "Second", identifiers=(second_identifier,))),),
        {},
        extractor="fixture",
    )
    database.ingest_page(
        "ids",
        (
            _record(
                "conflict",
                ModelHint(
                    "conflict",
                    "Conflicting assertion",
                    identifiers=(first_identifier, second_identifier),
                ),
            ),
        ),
        {},
        extractor="fixture",
    )
    assert database.stats()["models"] == 3
    assert database.stats()["identifier_conflicts"] == 2

    base = ModelHint("base", "Base")
    child = ModelHint("child", "Child")
    relations = (
        ModelRelationHint("child", "derived_from", base, locator="raw:relation:0"),
        ModelRelationHint(
            "child",
            "related_to",
            ModelHint("unseen", "Only Named Target"),
            locator="raw:relation:1",
        ),
    )
    relation_record = SourceRecord(
        source_record_id="relations",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url="https://source.example/relations",
        title="Child",
        raw={"reference": "https://target.example/model?utm_source=test"},
        links=(
            Link(
                "https://target.example/model?utm_source=test",
                "implementation",
                "raw:reference",
            ),
        ),
        models=(base, child),
        model_relations=relations,
    )
    result = database.ingest_page(
        "relations",
        (relation_record,),
        {},
        extractor="fixture",
        enqueue_links=False,
        link_depth=2,
    )
    assert result["links_discovered"] == 1
    stats = database.stats()
    assert stats["resolved_relation_claims"] == 1
    assert stats["unresolved_relation_claims"] == 1
    assert stats["pending_urls"] == 0
    observed = database.list_frontier("observed")
    assert observed[0]["url"] == "https://target.example/model"
    assert observed[0]["depth"] == 2

    database.ingest_page(
        "relations-2",
        (
            _record(
                "queue-it",
                ModelHint("model", "Queue It"),
                links=(Link("https://target.example/model"),),
            ),
        ),
        {},
        extractor="fixture",
        enqueue_links=True,
        link_depth=0,
    )
    pending = database.list_frontier()
    assert pending[0]["depth"] == 0

    child_result = next(item for item in database.search_models("Child") if item["name"] == "Child")
    child_detail = database.model_detail(child_result["id"])
    assert child_detail is not None
    assert {relation["resolution_status"] for relation in child_detail["relations"]} == {
        "resolved",
        "unresolved_target",
    }


def test_current_bulk_controls_round_trip_from_evidence_without_crawl(database):
    shard = SourceRecord(
        source_record_id="shard-b",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url="https://objects.example.test/shard-b.xml.gz",
        title="Bulk shard B",
        raw={"record_type": "dataset_shard", "sequence": 2},
        modified_at="2026-09-01T00:00:00Z",
        identifiers=(Identifier("bulk:shard", "b"),),
        links=(
            Link(
                "https://objects.example.test/shard-b.xml.gz",
                relation="bulk_payload",
                locator="$.url",
                crawl=False,
            ),
        ),
    )
    release = SourceRecord(
        source_record_id="release-a",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url="https://api.example.test/releases/a",
        title="Release A",
        raw={"record_type": "release_manifest"},
    )
    database.ingest_page(
        "bulk-control",
        (shard, release),
        {},
        extractor="fixture",
    )

    records = database.list_control_records(
        "bulk-control",
        record_type="dataset_shard",
    )

    assert records == [shard]
    assert records[0].links[0].crawl is False
    assert database.list_frontier("pending") == []


def test_bulk_control_listing_excludes_tombstones_and_bounds_results(database):
    records = tuple(
        SourceRecord(
            source_record_id=f"shard-{index}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=f"https://objects.example.test/shard-{index}",
            title=f"Shard {index}",
            raw={"record_type": "dataset_shard", "sequence": index},
        )
        for index in range(3)
    )
    database.ingest_page("bulk-control", records, {}, extractor="fixture")
    database.ingest_page(
        "bulk-control",
        (
            SourceRecord(
                source_record_id="shard-1",
                kind=ArtifactKind.CATALOG_RECORD,
                canonical_url="https://objects.example.test/shard-1",
                title="[deleted] shard-1",
                raw={"record_type": "dataset_shard", "sequence": 1},
                deleted=True,
            ),
        ),
        {},
        extractor="fixture",
    )

    controls = database.list_control_records("bulk-control", limit=1)

    assert [record.source_record_id for record in controls] == ["shard-0"]
    with pytest.raises(ValueError, match="limit must be positive"):
        database.list_control_records("bulk-control", limit=0)
