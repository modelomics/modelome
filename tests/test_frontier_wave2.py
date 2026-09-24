from modelome.frontier import FrontierCrawler, _worth_fetching
from modelome.models import ArtifactKind, Link, SourcePage, SourceRecord
from modelome.storage import Database


def test_huggingface_resolve_checkpoint_is_frontier_fetchable() -> None:
    assert _worth_fetching(
        "https://huggingface.co/lab/model/resolve/main/model.safetensors"
    )
    assert _worth_fetching("https://huggingface.co/lab/model/raw/a1b2c3/README.md")
    assert not _worth_fetching("https://huggingface.co/lab/model")
    assert not _worth_fetching(
        "https://huggingface.co/lab/model/resolve/main/config.json"
    )


def test_source_declared_huggingface_checkpoint_is_materialized(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    checkpoint_url = (
        "https://huggingface.co/lab/model/resolve/main/model.safetensors"
    )
    database.ingest_page(
        "catalog",
        SourcePage(
            records=(
                SourceRecord(
                    source_record_id="catalog-model",
                    kind=ArtifactKind.CATALOG_RECORD,
                    canonical_url="https://catalog.example/models/model",
                    title="Model record",
                    raw={},
                    links=(Link(checkpoint_url, relation="weights"),),
                ),
            ),
            next_state={"done": True},
            complete=True,
        ),
    )

    outcome = FrontierCrawler(database).crawl(limit=10)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 1
    assert database.list_frontier(status="done")[0]["url"] == checkpoint_url
    checkpoint = next(
        row
        for row in database.table_rows("artifacts")
        if row["source"] == "frontier"
    )
    assert checkpoint["kind"] == "weights"
