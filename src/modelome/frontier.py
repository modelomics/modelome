from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from modelome.extract import IntroductionCueExtractor
from modelome.fetchers import (
    REFERENCE_WEIGHT_SUFFIXES,
    ArtifactFetcher,
    AwsBedrockModelCardFetcher,
    GitHubRepositoryFetcher,
    HuggingFaceModelCardFetcher,
    NvidiaNgcModelCardFetcher,
    OpenAIModelDocumentationFetcher,
    PrivateResourceError,
    PyTorchHubModelPageFetcher,
    WebPageFetcher,
    WeightReferenceFetcher,
)
from modelome.models import SourcePage, SyncStats
from modelome.normalize import identifier_from_url
from modelome.storage import Database

_BINARY_SUFFIXES = {
    ".7z",
    ".bin",
    ".bz2",
    ".ckpt",
    ".dmg",
    ".gguf",
    ".gz",
    ".h5",
    ".iso",
    ".onnx",
    ".parquet",
    ".pdf",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tgz",
    ".whl",
    ".xz",
    ".zip",
}
_ENUMERATED_HOSTS = {
    "api.openalex.org",
    "openalex.org",
    "www.openalex.org",
}


@dataclass(frozen=True, slots=True)
class FrontierOutcome:
    status: str
    run_id: int
    stats: dict[str, Any]


class FrontierCrawler:
    """Enrich links discovered by primary sources with bounded crawl depth."""

    def __init__(
        self,
        database: Database,
        fetchers: Iterable[ArtifactFetcher] | None = None,
        *,
        extractor: Any | None = None,
        max_attempts: int = 3,
        claim_lease_seconds: int = 3600,
        github_token: str | None = None,
    ) -> None:
        self.database = database
        token = (
            os.environ.get("GITHUB_TOKEN", "").strip()
            if github_token is None
            else github_token.strip()
        )
        self.fetchers = tuple(
            fetchers
            if fetchers is not None
            else (
                GitHubRepositoryFetcher(token=token or None),
                WeightReferenceFetcher(),
                HuggingFaceModelCardFetcher(),
                AwsBedrockModelCardFetcher(),
                OpenAIModelDocumentationFetcher(),
                PyTorchHubModelPageFetcher(),
                NvidiaNgcModelCardFetcher(),
                WebPageFetcher(),
            )
        )
        self.extractor = extractor or IntroductionCueExtractor()
        self.max_attempts = max_attempts
        if claim_lease_seconds < 1:
            raise ValueError("claim_lease_seconds must be positive")
        self.claim_lease_seconds = claim_lease_seconds

    def crawl(self, *, limit: int = 200, max_depth: int = 1) -> FrontierOutcome:
        if limit < 1:
            raise ValueError("limit must be positive")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")

        source = "frontier"
        run_id = self.database.start_run(source)
        stats = SyncStats(source=source)
        try:
            items = self.database.claim_frontier(
                limit=limit,
                lease_seconds=self.claim_lease_seconds,
            )

            for item in items:
                url = str(item["url"])
                depth = int(item.get("depth") or 0)
                if depth > max_depth or not _worth_fetching(url):
                    self.database.update_frontier(
                        url, "ignored", error="outside crawl policy"
                    )
                    continue

                fetcher = next(
                    (candidate for candidate in self.fetchers if candidate.accepts(url)),
                    None,
                )
                if fetcher is None:
                    self.database.update_frontier(
                        url, "ignored", error="no artifact fetcher"
                    )
                    continue

                try:
                    record = fetcher.fetch(url)
                    result = self.database.ingest_page(
                        source,
                        SourcePage(
                            records=(record,),
                            next_state={"last_url": url},
                            complete=False,
                        ),
                        run_id=run_id,
                        extractor=self.extractor,
                        enqueue_links=depth < max_depth,
                        link_depth=depth + 1,
                    )
                    stats.records_seen += result["records_seen"]
                    stats.new_artifacts += result["new_artifacts"]
                    stats.new_revisions += result["new_revisions"]
                    stats.models_touched += result["models_touched"]
                    stats.links_discovered += result["links_discovered"]
                    if result.get("errors"):
                        stats.errors.append(f"{url}: record was quarantined")
                        self.database.update_frontier(
                            url, "failed", error="record quarantined"
                        )
                    else:
                        self.database.update_frontier(url, "done")
                except PrivateResourceError as error:
                    self.database.update_frontier(url, "ignored", error=str(error))
                except Exception as error:  # one failed URL must not stop the batch
                    attempts = int(item.get("attempts") or 0)
                    message = f"{type(error).__name__}: {error}"
                    stats.errors.append(f"{url}: {message}")
                    self.database.add_dead_letter(
                        source,
                        url,
                        "frontier_fetch",
                        message,
                        {"url": url, "depth": depth},
                        run_id=run_id,
                    )
                    retry_status = (
                        "pending" if attempts < self.max_attempts else "failed"
                    )
                    self.database.update_frontier(url, retry_status, error=message)

            exhausted_batch = len(items) < limit
            stats.complete = exhausted_batch and not stats.errors
            if stats.errors and stats.records_seen == 0:
                status = "failed"
            elif stats.errors or not exhausted_batch:
                status = "partial"
            else:
                status = "complete"
            self.database.finish_run(run_id, status, stats)
            return FrontierOutcome(status, run_id, stats.as_dict())
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            stats.complete = False
            stats.errors.append(message)
            self.database.finish_run(run_id, "failed", stats, error=message)
            return FrontierOutcome("failed", run_id, stats.as_dict())


def _worth_fetching(url: str) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").casefold()
    if host in _ENUMERATED_HOSTS:
        return False
    suffix = PurePosixPath(parts.path).suffix.casefold()
    if suffix in _BINARY_SUFFIXES and suffix not in REFERENCE_WEIGHT_SUFFIXES:
        return False
    identifier = identifier_from_url(url)
    # The Hub enumerator already captures the model endpoint and metadata.
    if identifier and identifier.namespace == "huggingface:model":
        segments = [segment.casefold() for segment in parts.path.split("/") if segment]
        is_versioned_file = len(segments) >= 5 and segments[2] in {"raw", "resolve"}
        if not is_versioned_file:
            return False
        filename = segments[-1]
        # Hub model pages are enumerated elsewhere, but a source-declared
        # resolve/raw checkpoint URL is a distinct, useful artifact. Preserve
        # the existing README exception and admit only known reference weights.
        return filename == "readme.md" or PurePosixPath(filename).suffix in (
            REFERENCE_WEIGHT_SUFFIXES
        )
    return True


__all__ = ["FrontierCrawler", "FrontierOutcome"]
