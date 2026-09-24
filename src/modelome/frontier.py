from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlsplit

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
    PublicUrlPolicy,
    PyTorchHubModelPageFetcher,
    WebPageFetcher,
    WeightReferenceFetcher,
)
from modelome.models import ArtifactKind, Identifier, SourcePage, SourceRecord, SyncStats
from modelome.normalize import canonicalize_url, identifier_from_url
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
_TEXT_MENTION_LOCATOR = re.compile(r"^text:\d+-\d+$")


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
            declared_weight_urls = _declared_weight_urls(self.database, items)
            reference_fetcher = _DeclaredWeightReferenceFetcher(declared_weight_urls)

            for item in items:
                url = str(item["url"])
                depth = int(item.get("depth") or 0)
                declared_weight_url = url in declared_weight_urls
                if depth > max_depth or not (
                    _worth_fetching(url, declared_weight=declared_weight_url)
                    or (
                        declared_weight_url
                        and _is_reference_only_checkpoint_url(url)
                    )
                ):
                    self.database.update_frontier(
                        url, "ignored", error="outside crawl policy"
                    )
                    continue

                if declared_weight_url:
                    # Keep the existing suffix-aware reference fetcher when it
                    # recognizes the URL. For all other explicitly declared
                    # weights/checkpoints, prefer metadata over a generic web
                    # fetch that could retrieve a large response body.
                    fetcher = next(
                        (
                            candidate
                            for candidate in self.fetchers
                            if isinstance(candidate, WeightReferenceFetcher)
                            and candidate.accepts(url)
                        ),
                        None,
                    )
                    if fetcher is None and reference_fetcher.accepts(url):
                        fetcher = reference_fetcher
                else:
                    fetcher = next(
                        (
                            candidate
                            for candidate in self.fetchers
                            if candidate.accepts(url)
                        ),
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


class _DeclaredWeightReferenceFetcher:
    """Materialize narrowly recognized extensionless checkpoints by URL only."""

    def __init__(self, declared_urls: set[str]) -> None:
        self.declared_urls = declared_urls
        self.url_policy = PublicUrlPolicy()

    def accepts(self, url: str) -> bool:
        return (
            url in self.declared_urls
            and (
                not _is_openreview_attachment(url)
                or _is_openreview_checkpoint_attachment(url)
            )
            and self.url_policy.allows(url, resolve=False)
        )

    def fetch(self, url: str) -> SourceRecord:
        canonical_url = self.url_policy.validate(url, resolve=False)
        parts = urlsplit(canonical_url)
        filename = PurePosixPath(parts.path).name
        if _is_openreview_checkpoint_attachment(canonical_url):
            names = parse_qs(parts.query, keep_blank_values=False).get("name", [])
            if names:
                filename = names[0]
        return SourceRecord(
            source_record_id=canonical_url,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonical_url,
            title=filename or canonical_url,
            raw={
                "reference_only": True,
                "suffix": PurePosixPath(parts.path).suffix.casefold(),
            },
            identifiers=(Identifier("url", canonical_url),),
        )


def _declared_weight_urls(database: Database, items: list[dict[str, Any]]) -> set[str]:
    url_by_id = {
        str(item["id"]): canonicalize_url(str(item["url"]))
        for item in items
        if item.get("id") and item.get("url")
    }
    if not url_by_id:
        return set()
    return {
        url_by_id[str(row["url_id"])]
        for row in database.table_rows("url_discoveries")
        if str(row.get("url_id")) in url_by_id
        and str(row.get("relation", "")).casefold() in {"weights", "checkpoint"}
        and not _TEXT_MENTION_LOCATOR.fullmatch(str(row.get("locator") or ""))
    }


def _is_extensionless_github_release_asset(url: str) -> bool:
    parts = urlsplit(url)
    segments = [segment for segment in parts.path.split("/") if segment]
    return (
        parts.scheme.casefold() == "https"
        and (parts.hostname or "").casefold() == "github.com"
        and len(segments) == 6
        and segments[2:4] == ["releases", "download"]
        and bool(segments[5])
        and not PurePosixPath(segments[5]).suffix
    )


def _is_gitlab_release_asset(url: str) -> bool:
    parts = urlsplit(url)
    if (
        parts.scheme.casefold() != "https"
        or (parts.hostname or "").casefold() != "gitlab.com"
    ):
        return False
    segments = [segment for segment in parts.path.split("/") if segment]
    return any(
        segments[index] == "-"
        and segments[index + 1] == "releases"
        and bool(segments[index + 2])
        and segments[index + 3] == "downloads"
        and index + 4 < len(segments)
        for index in range(max(0, len(segments) - 4))
    )


def _is_huggingface_versioned_file(url: str) -> bool:
    parts = urlsplit(url)
    identifier = identifier_from_url(url)
    segments = [segment for segment in parts.path.split("/") if segment]
    return bool(
        identifier
        and identifier.namespace == "huggingface:model"
        and len(segments) >= 5
        and segments[2].casefold() in {"raw", "resolve"}
        and bool(segments[-1])
    )


def _is_openreview_checkpoint_attachment(url: str) -> bool:
    if not _is_openreview_attachment(url):
        return False
    query = parse_qs(urlsplit(url).query, keep_blank_values=False)
    names = query.get("name", [])
    name = names[0].casefold() if len(names) == 1 else ""
    return bool(name) and ("weight" in name or "checkpoint" in name)


def _is_openreview_attachment(url: str) -> bool:
    parts = urlsplit(url)
    if (
        parts.scheme.casefold() != "https"
        or (parts.hostname or "").casefold() != "openreview.net"
        or parts.path.rstrip("/") != "/attachment"
    ):
        return False
    query = parse_qs(parts.query, keep_blank_values=False)
    ids = query.get("id", [])
    return len(ids) == 1 and bool(ids[0])


def _is_reference_only_checkpoint_url(url: str) -> bool:
    return (
        _is_extensionless_github_release_asset(url)
        or _is_gitlab_release_asset(url)
        or _is_huggingface_versioned_file(url)
        or _is_openreview_checkpoint_attachment(url)
    )


def _worth_fetching(url: str, *, declared_weight: bool = False) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").casefold()
    if host in _ENUMERATED_HOSTS:
        return False
    suffix = PurePosixPath(parts.path).suffix.casefold()
    declared_checkpoint_url = declared_weight and _is_reference_only_checkpoint_url(url)
    if (
        suffix in _BINARY_SUFFIXES
        and suffix not in REFERENCE_WEIGHT_SUFFIXES
        and not declared_checkpoint_url
    ):
        return False
    identifier = identifier_from_url(url)
    # The Hub enumerator already captures the model endpoint and metadata.
    if identifier and identifier.namespace == "huggingface:model":
        segments = [segment.casefold() for segment in parts.path.split("/") if segment]
        is_versioned_file = _is_huggingface_versioned_file(url)
        if not is_versioned_file:
            return False
        filename = segments[-1]
        # Hub model pages are enumerated elsewhere, but a source-declared
        # resolve/raw checkpoint URL is a distinct, useful artifact. Preserve
        # the existing README exception and admit only known reference weights.
        return (
            filename == "readme.md"
            or PurePosixPath(filename).suffix in REFERENCE_WEIGHT_SUFFIXES
            or declared_weight
        )
    return True


__all__ = ["FrontierCrawler", "FrontierOutcome"]
