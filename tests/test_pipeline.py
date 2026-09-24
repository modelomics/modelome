from modelome.models import (
    ArtifactKind,
    ModelHint,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.pipeline import SyncEngine
from modelome.storage import Database


def record(record_id: str, name: str) -> SourceRecord:
    return SourceRecord(
        source_record_id=record_id,
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=f"https://catalog.test/{record_id}",
        title=name,
        raw={"id": record_id, "name": name},
        models=(ModelHint(local_id="model", name=name),),
    )


class PagedSource:
    name = "paged"

    def __init__(self):
        self.states = []

    def fetch_page(self, state):
        self.states.append(dict(state))
        if not state:
            return SourcePage(
                records=(record("one", "First Unseen Model"),),
                next_state={"cursor": "page-two"},
                complete=False,
                upstream_count=2,
            )
        return SourcePage(
            records=(record("two", "Second Unseen Model"),),
            next_state={"watermark": "done"},
            complete=True,
            upstream_count=2,
        )


def test_sync_commits_each_page_and_is_idempotent(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    source = PagedSource()
    engine = SyncEngine(database, {source.name: source})

    first = engine.sync()
    second = engine.sync()

    assert first[0].status == "complete"
    assert first[0].stats["new_artifacts"] == 2
    assert second[0].stats["new_artifacts"] == 0
    assert second[0].stats["new_revisions"] == 0
    assert database.stats()["models"] == 2


def test_page_budget_persists_cursor_for_next_run(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    source = PagedSource()
    engine = SyncEngine(database, {source.name: source})

    partial = engine.sync(max_pages=1)
    resumed = engine.sync(max_pages=1)

    assert partial[0].status == "partial"
    assert database.get_source_state("paged") == {"watermark": "done"}
    assert resumed[0].status == "complete"
    assert source.states[1] == {"cursor": "page-two"}


def test_source_can_batch_local_commits_without_skipping_page_checkpoints(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    source = PagedSource()
    source.commit_pages = 2
    before = set((database.root / "commits").iterdir())

    outcome = SyncEngine(database, {source.name: source}).sync()[0]

    after = set((database.root / "commits").iterdir())
    # The run start, two-page ingest batch, and run finish each create one
    # durable snapshot; the two source cursors are still observed in order.
    assert len(after - before) == 3
    assert outcome.status == "complete"
    assert outcome.stats["pages"] == 2
    assert database.get_source_state(source.name) == {"watermark": "done"}
    assert source.states == [{}, {"cursor": "page-two"}]


class PartiallyMalformedSource:
    name = "partially-malformed"
    checkpoint_signature = "partially-malformed-v1"

    def __init__(self):
        self.states = []
        self.calls = 0

    def fetch_page(self, state):
        self.states.append(dict(state))
        self.calls += 1
        return SourcePage(
            records=(record("valid", "Valid Unseen Model"),),
            next_state={"watermark": "done"},
            complete=True,
            upstream_count=2,
            issues=(
                SourceIssue(
                    source_record_id="bad-record",
                    stage="source_normalize",
                    error="ValueError: missing source ID",
                    summary={"raw": {"title": "Malformed"}},
                ),
            )
            if self.calls == 1
            else (),
        )


def test_malformed_item_holds_checkpoint_and_fails_until_page_is_clean(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    source = PartiallyMalformedSource()
    engine = SyncEngine(database, {source.name: source})

    failed = engine.sync()[0]

    assert failed.status == "failed"
    assert failed.stats["records_seen"] == 2
    assert failed.stats["new_artifacts"] == 1
    assert failed.stats["complete"] is False
    assert failed.error is not None and "checkpoint retained for retry" in failed.error
    assert database.get_source_state(source.name) == {
        "_modelome_source_signature": source.checkpoint_signature
    }
    assert database.stats()["dead_letters"] == 1
    assert database.list_dead_letters()[0]["source_record_id"] == "bad-record"

    retried = engine.sync()[0]

    assert retried.status == "complete"
    assert retried.stats["new_artifacts"] == 0
    assert source.states == [
        {},
        {"_modelome_source_signature": source.checkpoint_signature},
    ]
    assert database.get_source_state(source.name) == {
        "watermark": "done",
        "_modelome_source_signature": source.checkpoint_signature,
    }


class ImmutableMalformedSource:
    name = "immutable-malformed"

    def fetch_page(self, state):
        return SourcePage(
            records=(record("valid-immutable", "Valid Immutable Model"),),
            next_state={"snapshot": "sealed"},
            complete=True,
            upstream_count=2,
            issues=(
                SourceIssue(
                    source_record_id="permanently-malformed",
                    stage="source_normalize",
                    error="ValueError: immutable row omits a canonical URL",
                ),
            ),
            advance_on_source_issues=True,
        )


def test_immutable_source_issue_is_quarantined_and_advances_checkpoint(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    source = ImmutableMalformedSource()

    outcome = SyncEngine(database, {source.name: source}).sync()[0]

    assert outcome.status == "complete"
    assert outcome.stats["new_artifacts"] == 1
    assert outcome.stats["complete"] is True
    assert "checkpoint advanced" in outcome.stats["errors"][0]
    assert database.get_source_state(source.name) == {"snapshot": "sealed"}
    assert database.list_dead_letters(source.name)[0]["source_record_id"] == "permanently-malformed"


class SignedPagedSource(PagedSource):
    name = "signed"

    def __init__(self, signature: str):
        super().__init__()
        self.checkpoint_signature = signature


def test_checkpoint_refuses_to_resume_under_changed_source_config(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    first = SignedPagedSource("configuration-a")
    partial = SyncEngine(database, {first.name: first}).sync(max_pages=1)[0]

    changed = SignedPagedSource("configuration-b")
    refused = SyncEngine(database, {changed.name: changed}).sync(max_pages=1)[0]

    assert partial.status == "partial"
    assert database.get_source_state(first.name) == {
        "cursor": "page-two",
        "_modelome_source_signature": "configuration-a",
    }
    assert refused.status == "failed"
    assert "different source configuration" in (refused.error or "")
    assert changed.states == []


class InvalidSignatureSource:
    name = "invalid-signature"
    checkpoint_signature = ""

    def fetch_page(self, state):
        raise AssertionError("an invalid checkpoint signature must fail before fetching")


def test_failure_after_run_start_releases_source_lease(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    source = InvalidSignatureSource()

    outcome = SyncEngine(database, {source.name: source}).sync()[0]

    assert outcome.status == "failed"
    assert outcome.run_id > 0
    assert "checkpoint signature must not be empty" in (outcome.error or "")
    source_status = database.source_status()[0]
    assert source_status["last_run"]["status"] == "failed"

    # A finalized failure must not leave the six-hour source lease behind.
    next_run = database.start_run(source.name)
    database.finish_run(next_run, "complete")


class NonAdvancingSource:
    name = "nonadvancing"

    def __init__(self):
        self.calls = 0

    def fetch_page(self, state):
        self.calls += 1
        return SourcePage(
            records=(record("repeated", "Repeated Model"),),
            next_state=dict(state),
            complete=False,
        )


def test_nonempty_incomplete_page_must_advance_checkpoint(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    source = NonAdvancingSource()

    outcome = SyncEngine(database, {source.name: source}).sync(max_pages=10)[0]

    assert outcome.status == "failed"
    assert "did not advance checkpoint state" in (outcome.error or "")
    assert source.calls == 1
    assert outcome.stats["records_seen"] == 1


class StatefulExtractor:
    name = "stateful-extractor"

    def __init__(self):
        self.fail = True

    def extract(self, source_record):
        if self.fail:
            raise ValueError("temporary extraction failure")
        return ()


class SinglePageSource:
    name = "single-page"

    def fetch_page(self, state):
        return SourcePage(
            records=(record("one", "One Model"),),
            next_state={"watermark": "done"},
            complete=True,
        )


class DerivedExtractionOptOutSource(SinglePageSource):
    name = "derived-extraction-opt-out"
    disable_derived_extraction = True


class FailingExtractor:
    name = "must-not-run"

    def extract(self, _record):
        raise AssertionError("source opted out of derived extraction")


def test_source_can_opt_out_of_derived_model_extraction(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    source = DerivedExtractionOptOutSource()

    outcome = SyncEngine(
        database,
        {source.name: source},
        extractor=FailingExtractor(),
    ).sync()[0]

    assert outcome.status == "complete"
    assert database.stats()["models"] == 1


def test_record_quarantine_holds_checkpoint_and_fails_run(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    source = SinglePageSource()
    extractor = StatefulExtractor()
    engine = SyncEngine(database, {source.name: source}, extractor=extractor)

    failed = engine.sync()[0]

    assert failed.status == "failed"
    assert "checkpoint retained for retry" in (failed.error or "")
    assert database.get_source_state(source.name) == {}
    assert database.stats()["dead_letters"] == 1

    extractor.fail = False
    retried = engine.sync()[0]

    assert retried.status == "complete"
    assert database.get_source_state(source.name) == {"watermark": "done"}
