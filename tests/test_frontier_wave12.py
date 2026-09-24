from modelome.frontier import FrontierCrawler, _worth_fetching
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


def _ingest_weight_link(database: Database, url: str, relation: str = "weights") -> None:
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


def test_declared_huggingface_resolve_file_with_other_suffix_is_reference_only(
    tmp_path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://huggingface.co/lab/model/resolve/a1b2c3/model_state.pdparams"
    _ingest_weight_link(database, url)
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


def test_declared_gitlab_release_archive_is_reference_only(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    url = "https://gitlab.com/lab/model/-/releases/v1/downloads/checkpoint.tar.gz"
    _ingest_weight_link(database, url, "checkpoint")
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
        "checkpoint.tar.gz",
    )


def test_huggingface_model_endpoint_still_requires_a_versioned_file() -> None:
    assert not _worth_fetching("https://huggingface.co/lab/model")
    assert not _worth_fetching(
        "https://huggingface.co/lab/model/resolve/a1b2c3/model_state.pdparams"
    )
    assert _worth_fetching(
        "https://huggingface.co/lab/model/resolve/a1b2c3/model_state.pdparams",
        declared_weight=True,
    )
