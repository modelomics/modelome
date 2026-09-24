import json

from modelome.frontier import FrontierCrawler
from modelome.models import ArtifactKind, Link, SourcePage, SourceRecord
from modelome.storage import Database


def _seed_link(database: Database, url: str, *, relation: str, crawl: bool) -> None:
    database.ingest_page(
        "fixture-catalog",
        SourcePage(
            records=(
                SourceRecord(
                    source_record_id=f"seed:{url}",
                    kind=ArtifactKind.CATALOG_RECORD,
                    canonical_url="https://catalog.example/model",
                    title="Model catalog entry",
                    raw={},
                    links=(Link(url, relation=relation, crawl=crawl),),
                ),
            ),
            next_state={},
            complete=True,
        ),
    )


def test_explicit_crawl_false_model_artifact_file_is_reference_only(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://example.org/models/candidate.safetensors"
    _seed_link(database, url, relation="model_artifact", crawl=False)

    assert [row["url"] for row in database.list_frontier(status="observed")] == [url]

    outcome = FrontierCrawler(database).crawl(limit=10)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 1
    assert [row["url"] for row in database.list_frontier(status="done")] == [url]
    checkpoint = next(
        row
        for row in database.table_rows("artifacts")
        if row.get("source") == "frontier"
    )
    revision = next(
        row
        for row in database.table_rows("artifact_revisions")
        if row["id"] == checkpoint["current_revision_id"]
    )
    assert (checkpoint["kind"], checkpoint["canonical_url"]) == ("weights", url)
    assert json.loads(revision["raw_json"])["reference_only"] is True


def test_paperswithcode_declared_model_file_reaches_reference_frontier(tmp_path) -> None:
    from modelome.sources.paperswithcode import _evaluation_records

    url = "https://example.org/models/candidate.safetensors"
    records, rejected, _ = _evaluation_records(
        [
            {
                "task": "Classification",
                "datasets": [
                    {
                        "dataset": "Example",
                        "sota": {
                            "rows": [
                                {
                                    "model_name": "Candidate Model",
                                    "paper_url": "https://arxiv.org/abs/2401.12345",
                                    "paper_title": "Candidate Model",
                                    "model_links": [
                                        {"url": url, "title": "Candidate checkpoint"}
                                    ],
                                }
                            ]
                        },
                    }
                ],
            }
        ],
        revision="c" * 40,
        data_path="data/train.parquet",
        dataset_id="pwc-archive/evaluation-tables",
        license="CC-BY-SA-4.0",
        max_model_rows=10,
    )
    assert rejected == {}
    assert any(
        link.url == url and link.relation == "model_artifact" and not link.crawl
        for link in records[0].links
    )

    database = Database(tmp_path / "store")
    database.initialize()
    database.ingest_page(
        "paperswithcode-fixture",
        SourcePage(records=tuple(records), next_state={}, complete=True),
    )

    class CatchAllBodyFetcher:
        """Record URLs routed to a fetcher that could retrieve response bodies."""

        def __init__(self) -> None:
            self.urls: list[str] = []

        def accepts(self, candidate_url: str) -> bool:
            return True

        def fetch(self, candidate_url: str) -> SourceRecord:
            self.urls.append(candidate_url)
            raise AssertionError("declared checkpoint reached a body-fetch fallback")

    body_fetcher = CatchAllBodyFetcher()
    outcome = FrontierCrawler(database, [body_fetcher]).crawl(limit=10)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 1
    assert any(row["url"] == url for row in database.list_frontier(status="done"))
    assert body_fetcher.urls == []
    checkpoint = next(
        row
        for row in database.table_rows("artifacts")
        if row.get("source") == "frontier"
    )
    revision = next(
        row
        for row in database.table_rows("artifact_revisions")
        if row["id"] == checkpoint["current_revision_id"]
    )
    assert json.loads(revision["raw_json"])["reference_only"] is True


def test_model_artifact_bundle_and_text_mention_are_not_promoted(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    bundle_url = "https://example.org/models/training-data.zip"
    checkpoint_url = "https://example.org/models/from-text.safetensors"
    database.ingest_page(
        "fixture-catalog",
        SourcePage(
            records=(
                SourceRecord(
                    source_record_id="structured-bundle",
                    kind=ArtifactKind.CATALOG_RECORD,
                    canonical_url="https://catalog.example/bundle",
                    title="Bundle entry",
                    raw={},
                    links=(Link(bundle_url, relation="model_artifact", crawl=False),),
                ),
                SourceRecord(
                    source_record_id="text-mention",
                    kind=ArtifactKind.PAPER,
                    canonical_url="https://papers.example/model",
                    title="Paper",
                    text=f"Weights may be found at {checkpoint_url}",
                    raw={},
                ),
            ),
            next_state={},
            complete=True,
        ),
    )

    outcome = FrontierCrawler(database).crawl(limit=10)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 0
    assert database.list_frontier(status="done") == []
    assert {row["url"] for row in database.list_frontier(status="observed")} == {
        bundle_url,
        checkpoint_url,
    }


def test_crawl_limit_bounds_promoted_observed_checkpoint_files(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    urls = [
        "https://example.org/models/first.safetensors",
        "https://example.org/models/second.safetensors",
    ]
    for index, url in enumerate(urls):
        database.ingest_page(
            f"fixture-catalog-{index}",
            SourcePage(
                records=(
                    SourceRecord(
                        source_record_id=f"seed-{index}",
                        kind=ArtifactKind.CATALOG_RECORD,
                        canonical_url=f"https://catalog.example/model-{index}",
                        title=f"Model {index}",
                        raw={},
                        links=(Link(url, relation="model_artifact", crawl=False),),
                    ),
                ),
                next_state={},
                complete=True,
            ),
        )

    outcome = FrontierCrawler(database).crawl(limit=1)

    assert outcome.status == "partial"
    assert outcome.stats["records_seen"] == 1
    assert len(database.list_frontier(status="done")) == 1
    assert len(database.list_frontier(status="observed")) == 1


def test_declared_model_file_promotion_uses_indexes_not_table_materialization(
    tmp_path, monkeypatch
) -> None:
    from modelome.frontier import _promote_declared_model_file_urls

    database = Database(tmp_path / "store")
    database.initialize()
    urls = [
        "https://example.org/models/first.safetensors",
        "https://example.org/models/second.safetensors",
    ]
    for url in urls:
        _seed_link(database, url, relation="model_artifact", crawl=False)

    original_table_rows = database.table_rows

    def reject_frontier_table_materialization(table: str):
        if table in {"url_frontier", "url_discoveries", "artifacts"}:
            raise AssertionError(f"unexpected full-table read: {table}")
        return original_table_rows(table)

    monkeypatch.setattr(database, "table_rows", reject_frontier_table_materialization)

    assert _promote_declared_model_file_urls(database, limit=1) == [urls[0]]
