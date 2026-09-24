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
        "catalog",
        SourcePage(
            records=(
                SourceRecord(
                    source_record_id="catalog-model",
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


def test_text_mention_of_weights_does_not_override_github_repository_fetcher(
    tmp_path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://github.com/lab/model"
    database.ingest_page(
        "paper",
        SourcePage(
            records=(
                SourceRecord(
                    source_record_id="paper-1",
                    kind=ArtifactKind.PAPER,
                    canonical_url="https://papers.example/paper-1",
                    title="Paper",
                    text=f"Trained weights are available at {url}",
                    raw={},
                ),
            ),
            next_state={"done": True},
            complete=True,
        ),
    )
    fetcher = RecordingFetcher()

    outcome = FrontierCrawler(database, fetchers=[fetcher]).crawl(limit=10)

    assert outcome.status == "complete"
    assert fetcher.calls == [url]


def test_structured_weight_locator_keeps_reference_only_routing(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://github.com/lab/model"
    database.ingest_page(
        "catalog",
        SourcePage(
            records=(
                SourceRecord(
                    source_record_id="catalog-model",
                    kind=ArtifactKind.CATALOG_RECORD,
                    canonical_url="https://catalog.example/model",
                    title="Model",
                    raw={},
                    links=(
                        Link(
                            url,
                            relation="weights",
                            locator="$.checkpoint_url",
                        ),
                    ),
                ),
            ),
            next_state={"done": True},
            complete=True,
        ),
    )

    outcome = FrontierCrawler(database, fetchers=[]).crawl(limit=10)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 1
    checkpoint = next(
        row
        for row in database.table_rows("artifacts")
        if row["source"] == "frontier"
    )
    assert checkpoint["kind"] == "weights"


def test_declared_pdparams_checkpoint_uses_reference_only_path(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://models.example/checkpoints/model_state.pdparams"
    _seed(database, url, "weights")
    fetcher = RecordingFetcher()

    outcome = FrontierCrawler(database, fetchers=[fetcher]).crawl(limit=10)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 1
    assert fetcher.calls == []
    checkpoint = next(
        row
        for row in database.table_rows("artifacts")
        if row["source"] == "frontier"
    )
    assert (checkpoint["kind"], checkpoint["title"]) == (
        "weights",
        "model_state.pdparams",
    )


def test_other_pdparams_relation_keeps_existing_fetch_behavior(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://models.example/checkpoints/model_state.pdparams"
    _seed(database, url, "references")
    fetcher = RecordingFetcher()

    outcome = FrontierCrawler(database, fetchers=[fetcher]).crawl(limit=10)

    assert outcome.status == "complete"
    assert fetcher.calls == [url]
