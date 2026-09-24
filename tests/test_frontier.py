from modelome.fetchers import (
    AwsBedrockModelCardFetcher,
    GitHubRepositoryFetcher,
    NvidiaNgcModelCardFetcher,
    OpenAIModelDocumentationFetcher,
    PyTorchHubModelPageFetcher,
)
from modelome.frontier import FrontierCrawler, _worth_fetching
from modelome.models import ArtifactKind, Link, ModelHint, SourcePage, SourceRecord
from modelome.storage import Database


class Fetcher:
    def accepts(self, url):
        return url.startswith("https://code.test/")

    def fetch(self, url):
        return SourceRecord(
            source_record_id=url,
            kind=ArtifactKind.CODE_REPOSITORY,
            canonical_url=url,
            title="Dynamically discovered repository",
            raw={"url": url},
            links=(Link("https://provider.test/model-card"),),
        )


class FailingFetcher:
    def accepts(self, url):
        return True

    def fetch(self, url):
        raise RuntimeError("upstream unavailable")


class GitHubFixtureFetcher:
    def accepts(self, url):
        return url == "https://github.com/lab/discovered-model"

    def fetch(self, url):
        return SourceRecord(
            source_record_id="lab/discovered-model",
            kind=ArtifactKind.CODE_REPOSITORY,
            canonical_url=url,
            title="Discovered implementation",
            raw={"full_name": "lab/discovered-model"},
        )


class CodeOnlyGitHubFixtureFetcher:
    def accepts(self, url):
        return url == "https://github.com/lab/code-only-model"

    def fetch(self, url):
        return SourceRecord(
            source_record_id="lab/code-only-model",
            kind=ArtifactKind.CODE_REPOSITORY,
            canonical_url=url,
            title="lab/code-only-model",
            text=(
                "# AuroraFieldNet-8\n\n"
                "A deep neural network for reconstructing physical fields."
            ),
            raw={"full_name": "lab/code-only-model"},
        )


def seed(database: Database, url: str) -> None:
    database.ingest_page(
        "seed",
        SourcePage(
            records=(
                SourceRecord(
                    source_record_id="seed-1",
                    kind=ArtifactKind.PAPER,
                    canonical_url="https://papers.test/one",
                    title="Paper",
                    raw={"title": "Paper"},
                    links=(Link(url),),
                ),
            ),
            next_state={"done": True},
            complete=True,
        ),
    )


def test_frontier_fetches_discovered_repository_and_bounds_next_depth(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    seed(database, "https://code.test/lab/model")
    crawler = FrontierCrawler(database, [Fetcher()])

    outcome = crawler.crawl(limit=10, max_depth=1)

    assert outcome.status == "complete"
    assert outcome.stats["new_artifacts"] == 1
    assert database.list_frontier(status="done")[0]["url"] == (
        "https://code.test/lab/model"
    )
    pending = database.list_frontier()
    assert pending[0]["url"] == "https://provider.test/model-card"
    assert pending[0]["depth"] == 1


def test_default_frontier_passes_environment_token_only_to_github_fetcher(
    tmp_path, monkeypatch
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    monkeypatch.setenv("GITHUB_TOKEN", "public-metadata-token")

    crawler = FrontierCrawler(database)

    github = next(
        fetcher
        for fetcher in crawler.fetchers
        if isinstance(fetcher, GitHubRepositoryFetcher)
    )
    assert github.token == "public-metadata-token"
    assert any(
        isinstance(fetcher, NvidiaNgcModelCardFetcher)
        for fetcher in crawler.fetchers
    )
    assert any(
        isinstance(fetcher, AwsBedrockModelCardFetcher)
        for fetcher in crawler.fetchers
    )
    assert any(
        isinstance(fetcher, OpenAIModelDocumentationFetcher)
        for fetcher in crawler.fetchers
    )
    assert any(
        isinstance(fetcher, PyTorchHubModelPageFetcher)
        for fetcher in crawler.fetchers
    )


def test_paper_text_repository_url_enters_frontier_and_links_back_to_model(
    tmp_path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    paper = SourceRecord(
        source_record_id="paper-with-code",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.test/paper-with-code",
        title="Paper with dynamically discovered code",
        text=(
            "The implementation is available at "
            "https://github.com/lab/discovered-model?utm_source=paper."
        ),
        raw={},
        models=(ModelHint("introduced-model", "Dynamically Discovered Model"),),
    )
    database.ingest_page("papers", (paper,), {}, extractor="fixture")

    pending = database.list_frontier(status="pending")
    assert [item["url"] for item in pending] == [
        "https://github.com/lab/discovered-model"
    ]

    outcome = FrontierCrawler(database, [GitHubFixtureFetcher()]).crawl(limit=10)

    assert outcome.status == "complete"
    model_id = database.search_models("Dynamically Discovered Model")[0]["id"]
    detail = database.model_detail(model_id)
    assert detail is not None
    connected = detail["connected_artifacts"]
    assert len(connected) == 1
    assert connected[0]["kind"] == "code_repository"
    connection = connected[0]["connections"][0]
    assert connection["relation"] == "implementation"
    assert connection["evidence_source"]["source"] == "papers"


def test_repository_inventory_can_discover_a_model_documented_only_in_readme(
    tmp_path,
) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    database.ingest_page(
        "repository-inventory",
        (
            SourceRecord(
                source_record_id="event-1",
                kind=ArtifactKind.CATALOG_RECORD,
                canonical_url="https://inventory.test/events/1",
                title="Public repository observation",
                raw={"event_id": "1"},
                links=(
                    Link(
                        "https://github.com/lab/code-only-model",
                        "repository_observation",
                    ),
                ),
            ),
        ),
        {},
        extractor="fixture",
    )

    outcome = FrontierCrawler(
        database,
        [CodeOnlyGitHubFixtureFetcher()],
    ).crawl(limit=10)

    assert outcome.status == "complete"
    matches = database.search_models("AuroraFieldNet-8")
    assert len(matches) == 1
    detail = database.model_detail(matches[0]["id"])
    assert detail is not None
    assert any(
        artifact["kind"] == "code_repository"
        for artifact in detail["artifacts"]
    )


def test_frontier_materializes_binary_weight_links_without_downloading(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    seed(database, "https://weights.test/checkpoint.safetensors")

    outcome = FrontierCrawler(database).crawl(limit=10)

    assert outcome.stats["records_seen"] == 1
    assert database.list_frontier(status="done")[0]["url"].endswith(".safetensors")
    artifact = next(
        row for row in database.table_rows("artifacts") if row["source"] == "frontier"
    )
    assert (artifact["kind"], artifact["title"]) == (
        "weights",
        "checkpoint.safetensors",
    )


def test_frontier_reports_failed_when_every_fetch_fails(tmp_path) -> None:
    database = Database(tmp_path / "store")
    database.initialize()
    seed(database, "https://broken.test/model-card")

    outcome = FrontierCrawler(database, [FailingFetcher()]).crawl(limit=10)

    assert outcome.status == "failed"
    assert outcome.stats["complete"] is False
    assert outcome.stats["records_seen"] == 0
    assert len(outcome.stats["errors"]) == 1
    frontier_status = next(
        item for item in database.source_status() if item["source"] == "frontier"
    )
    assert frontier_status["last_run"]["status"] == "failed"


def test_frontier_finalizes_run_when_claiming_raises(tmp_path, monkeypatch) -> None:
    database = Database(tmp_path / "store")
    database.initialize()

    def fail_claim(*, limit, lease_seconds):
        raise RuntimeError("injected claim failure")

    monkeypatch.setattr(database, "claim_frontier", fail_claim)

    outcome = FrontierCrawler(database, [Fetcher()]).crawl(limit=10)

    assert outcome.status == "failed"
    assert outcome.run_id > 0
    assert any("injected claim failure" in error for error in outcome.stats["errors"])
    frontier_status = next(
        item for item in database.source_status() if item["source"] == "frontier"
    )
    assert frontier_status["last_run"]["status"] == "failed"
    assert frontier_status["last_run"]["finished_at"] is not None

    # The failure must not retain the source's six-hour single-writer lease.
    next_run = database.start_run("frontier")
    database.finish_run(next_run, "complete")


def test_frontier_fetches_versioned_hub_readmes_but_not_enumerated_model_pages() -> None:
    assert _worth_fetching(
        "https://huggingface.co/lab/model/raw/a1b2c3/README.md"
    )
    assert not _worth_fetching("https://huggingface.co/lab/model")
