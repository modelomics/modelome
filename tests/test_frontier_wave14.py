from modelome.frontier import FrontierCrawler
from modelome.models import ArtifactKind, Link, SourcePage, SourceRecord
from modelome.storage import Database


class RecordingFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def accepts(self, url: str) -> bool:
        return True

    def fetch(self, url: str) -> SourceRecord:
        self.calls.append(url)
        return SourceRecord(
            source_record_id=url,
            kind=ArtifactKind.OTHER,
            canonical_url=url,
            title="Fetched page",
            raw={},
        )


def _seed(database: Database, url: str, relation: str) -> None:
    database.ingest_page(
        "model-catalog",
        SourcePage(
            records=(
                SourceRecord(
                    source_record_id="model-1",
                    kind=ArtifactKind.MODEL_CARD,
                    canonical_url="https://catalog.example/models/1",
                    title="Model one",
                    raw={},
                    links=(
                        Link(url, relation=relation, locator="$.description"),
                    ),
                ),
            ),
            next_state={"done": True},
            complete=True,
        ),
    )


def test_source_declared_model_document_pdf_is_materialized_without_download(
    tmp_path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://docs.example/models/model-one/model-card.pdf"
    _seed(database, url, "documentation_reference")
    fetcher = RecordingFetcher()

    outcome = FrontierCrawler(database, fetchers=[fetcher]).crawl(limit=10)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 1
    assert fetcher.calls == []
    document = next(
        row
        for row in database.table_rows("artifacts")
        if row["source"] == "frontier"
    )
    assert (document["kind"], document["title"]) == (
        "web_page",
        "model-card.pdf",
    )


def test_unrelated_pdf_reference_is_not_admitted(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://docs.example/models/model-one/paper.pdf"
    _seed(database, url, "references")
    fetcher = RecordingFetcher()

    outcome = FrontierCrawler(database, fetchers=[fetcher]).crawl(limit=10)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 0
    assert fetcher.calls == []
    assert database.list_frontier(status="ignored")[0]["url"] == url
