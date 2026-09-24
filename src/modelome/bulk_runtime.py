from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modelome.bulk import BulkControlOrchestrator, BulkOutcome
from modelome.commoncrawl_bulk import CommonCrawlWetBulkLoader
from modelome.commoncrawl_projection import CommonCrawlWetDiscoveryProjector
from modelome.commoncrawl_search_runtime import run_commoncrawl_search_ingestion
from modelome.gharchive_bulk import GhArchiveEventBulkLoader
from modelome.lake import ParquetLandingZone
from modelome.pubmed_bulk import PubMedBulkLoader
from modelome.semantic_scholar_bulk import SemanticScholarBulkLoader
from modelome.software_heritage_bulk import SoftwareHeritageOriginBulkLoader
from modelome.software_heritage_projection import (
    run_software_heritage_github_shard_projection,
)
from modelome.sources.commoncrawl import CommonCrawlWetSourceAdapter
from modelome.sources.gharchive import GhArchiveSourceAdapter
from modelome.sources.pubmed import PubMedBulkSourceAdapter
from modelome.sources.semantic_scholar import SemanticScholarDatasetSourceAdapter
from modelome.sources.software_heritage import SoftwareHeritageOriginSourceAdapter
from modelome.storage import Database

type BulkControlSource = (
    SemanticScholarDatasetSourceAdapter
    | PubMedBulkSourceAdapter
    | CommonCrawlWetSourceAdapter
    | GhArchiveSourceAdapter
    | SoftwareHeritageOriginSourceAdapter
)


def select_bulk_sources(
    sources: Mapping[str, Any],
    requested_names: list[str] | None = None,
) -> list[BulkControlSource]:
    """Select only adapters that expose a supported complete-corpus data plane."""

    supported = (
        SemanticScholarDatasetSourceAdapter,
        PubMedBulkSourceAdapter,
        CommonCrawlWetSourceAdapter,
        GhArchiveSourceAdapter,
        SoftwareHeritageOriginSourceAdapter,
    )
    names = requested_names or []
    if names:
        selected: list[BulkControlSource] = []
        for name in names:
            source = sources.get(name)
            if source is None:
                raise KeyError(f"unknown or disabled source: {name}")
            if not isinstance(source, supported):
                raise ValueError(f"bulk loading is not implemented for {type(source).__name__}")
            selected.append(source)
        return selected
    return [source for source in sources.values() if isinstance(source, supported)]


def run_bulk_source(
    database: Database,
    lake: ParquetLandingZone,
    source: BulkControlSource,
    *,
    max_new_shards: int | None,
) -> BulkOutcome:
    """Load a bounded prefix and publish only exact, complete release groups."""

    orchestrator = BulkControlOrchestrator(database, lake)
    ordinary_budget = 1 if max_new_shards is None else max_new_shards
    if isinstance(source, SemanticScholarDatasetSourceAdapter):
        loader = SemanticScholarBulkLoader(lake, control_adapter=source)
        return orchestrator.run(
            control_source=source.name,
            loader=loader.load_shard,
            selector=loader.plan_shard,
            order_key=loader.shard_order,
            max_new_shards=ordinary_budget,
        )
    if isinstance(source, PubMedBulkSourceAdapter):
        loader = PubMedBulkLoader(lake)
        return orchestrator.run(
            control_source=source.name,
            loader=loader.load,
            selector=loader.plan_shard,
            order_key=loader.shard_order,
            max_new_shards=ordinary_budget,
        )
    if isinstance(source, CommonCrawlWetSourceAdapter):
        loader = CommonCrawlWetBulkLoader(lake, data_url=source.data_url)
        projector = CommonCrawlWetDiscoveryProjector(lake)
        return orchestrator.run(
            control_source=source.name,
            loader=lambda record: _load_and_project_commoncrawl_shard(
                database,
                loader,
                projector,
                record,
            ),
            selector=loader.plan_shard,
            order_key=loader.shard_order,
            max_new_shards=ordinary_budget,
        )
    if isinstance(source, GhArchiveSourceAdapter):
        loader = GhArchiveEventBulkLoader(lake, data_url=source.data_url)
        return orchestrator.run(
            control_source=source.name,
            loader=loader.load,
            selector=loader.plan_shard,
            order_key=loader.shard_order,
            max_new_shards=loader.shard_budget(max_new_shards),
        )
    if isinstance(source, SoftwareHeritageOriginSourceAdapter):
        loader = SoftwareHeritageOriginBulkLoader(
            lake,
            bucket_url=source.bucket_url,
            graph_prefix=source.graph_prefix,
        )
        return orchestrator.run(
            control_source=source.name,
            loader=lambda record: _load_and_project_software_heritage_shard(
                database,
                lake,
                loader,
                record,
            ),
            selector=loader.plan_shard,
            order_key=loader.shard_order,
            max_new_shards=loader.shard_budget(max_new_shards),
        )
    raise AssertionError(f"unsupported bulk source: {type(source).__name__}")


def _load_and_project_commoncrawl_shard(
    database: Database,
    loader: CommonCrawlWetBulkLoader,
    projector: CommonCrawlWetDiscoveryProjector,
    control_record: Any,
) -> Any:
    """Land one WET shard and finish its bounded searchable projection.

    Every document remains in the immutable discovery projection. Search ingestion
    admits only evidence-backed model candidates and exact GitHub relations, using
    one bounded page and durable checkpoint per loop iteration. A cached shard
    therefore resumes safely after interruption instead of being treated as done.
    """

    bulk_receipt = loader.load(control_record)
    shard_receipt = bulk_receipt.shard_receipt
    if shard_receipt is None:
        raise ValueError("Common Crawl bulk receipt is missing its shard receipt")
    discovery_receipt = projector.materialize(shard_receipt)
    previous_row = -1
    while True:
        outcome = run_commoncrawl_search_ingestion(
            database,
            projector,
            discovery_receipt,
        )
        if outcome.status == "failed":
            raise RuntimeError(outcome.error or "Common Crawl search ingestion failed")
        if outcome.complete:
            return bulk_receipt
        if outcome.next_row <= previous_row:
            raise RuntimeError("Common Crawl search ingestion did not advance")
        previous_row = outcome.next_row


def _load_and_project_software_heritage_shard(
    database: Database,
    landing_zone: ParquetLandingZone,
    loader: SoftwareHeritageOriginBulkLoader,
    control_record: Any,
) -> Any:
    """Land one origin shard and finish its GitHub frontier projection."""

    bulk_receipt = loader.load(control_record)
    shard_receipt = bulk_receipt.shard_receipt
    if shard_receipt is None:
        raise ValueError("Software Heritage bulk receipt is missing its shard receipt")
    previous_row = -1
    while True:
        outcome = run_software_heritage_github_shard_projection(
            database,
            landing_zone,
            source_receipt=shard_receipt,
        )
        if outcome.status == "failed":
            raise RuntimeError(outcome.error or "Software Heritage GitHub projection failed")
        if outcome.complete:
            return bulk_receipt
        if outcome.next_row <= outcome.start_row or outcome.next_row <= previous_row:
            raise RuntimeError("Software Heritage GitHub projection did not advance")
        previous_row = outcome.next_row


__all__ = [
    "BulkControlSource",
    "run_bulk_source",
    "select_bulk_sources",
]
