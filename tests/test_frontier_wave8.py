from modelome.frontier import FrontierCrawler
from modelome.models import ArtifactKind, Link, SourcePage, SourceRecord
from modelome.storage import Database


def _seed_link(database: Database, url: str, relation: str) -> None:
    database.ingest_page(
        "catalog",
        SourcePage(
            records=(
                SourceRecord(
                    source_record_id="catalog-record",
                    kind=ArtifactKind.CATALOG_RECORD,
                    canonical_url="https://catalog.example/model",
                    title="Model record",
                    raw={},
                    links=(Link(url, relation=relation),),
                ),
            ),
            next_state={"done": True},
            complete=True,
        ),
    )


def test_openreview_checkpoint_attachment_is_materialized_without_download(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://openreview.net/attachment?id=paper-1&name=model_weights"
    _seed_link(database, url, "weights")

    outcome = FrontierCrawler(database, fetchers=[]).crawl(limit=10)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 1
    assert database.list_frontier(status="done")[0]["url"] == url
    checkpoint = next(
        row
        for row in database.table_rows("artifacts")
        if row["source"] == "frontier"
    )
    assert (checkpoint["kind"], checkpoint["title"]) == (
        "weights",
        "model_weights",
    )


def test_openreview_attachment_name_must_identify_checkpoint(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://openreview.net/attachment?id=paper-1&name=paper_pdf"
    _seed_link(database, url, "weights")

    outcome = FrontierCrawler(database, fetchers=[]).crawl(limit=10)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 0
    assert database.list_frontier(status="ignored")[0]["url"] == url
