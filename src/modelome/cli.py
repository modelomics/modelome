from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from modelome import __version__
from modelome.alphaxiv import AlphaXivExactIdEnricher, AlphaXivMcpClient
from modelome.alphaxiv_runtime import run_alphaxiv_enrichment
from modelome.artifact_relation_runtime import run_artifact_relation_projection
from modelome.backfill import (
    DEFAULT_BACKFILL_MAX_PAGES,
    run_arxiv_backfill,
    run_biorxiv_backfill,
    run_crossref_backfill,
    run_datacite_backfill,
    run_eartharxiv_backfill,
    run_europe_pmc_backfill,
    run_hal_backfill,
    run_openalex_backfill,
    run_osf_preprints_backfill,
)
from modelome.benchmark import evaluate_epoch_benchmark
from modelome.bootstrap import DEFAULT_BOOTSTRAP_MAX_PAGES, run_arxiv_bootstrap
from modelome.bulk_runtime import run_bulk_source, select_bulk_sources
from modelome.config import Settings
from modelome.coverage import evaluate_manifest
from modelome.daily import run_daily
from modelome.entries import (
    build_entries,
    plan_entry_seed,
    read_entry_seeds,
    write_entry_bundle,
)
from modelome.entry_seed_export import (
    assess_entry_readiness,
    export_current_entry_seeds,
    source_tags_from_configs,
)
from modelome.export import export_public_metadata
from modelome.frontier import FrontierCrawler
from modelome.institutional import ingest_institutional_text
from modelome.lake import ParquetLandingZone, ReleaseReceipt
from modelome.paper_ingestion import ingest_paper, prepare_paper_ingestion
from modelome.pipeline import SyncEngine
from modelome.pmc_bootstrap import run_pmc_bootstrap
from modelome.projection_runtime import run_semantic_scholar_projection
from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.sources.biorxiv import BioRxivPublicationSourceAdapter, BioRxivSourceAdapter
from modelome.sources.catalog import (
    load_benchmark_configs,
    load_source_configs,
    load_sources,
)
from modelome.sources.crossref import CrossrefSourceAdapter
from modelome.sources.datacite import DataCiteSourceAdapter
from modelome.sources.eartharxiv import EarthArxivSourceAdapter
from modelome.sources.europe_pmc import EuropePmcSourceAdapter
from modelome.sources.hal import HalSourceAdapter
from modelome.sources.openalex import OpenAlexSourceAdapter
from modelome.sources.osf_preprints import OsfPreprintSourceAdapter
from modelome.sources.pmc import PmcSourceAdapter
from modelome.storage import Database as ParquetStore
from modelome.text_import import (
    extract_doi,
    ingest_locally_authorized_text,
)


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="modelome",
        description="Build and query an evidence-backed deep-learning model registry.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--store",
        type=Path,
        default=settings.store,
        help=f"Parquet store directory (default: {settings.store})",
    )
    parser.add_argument(
        "--sources-file",
        type=Path,
        default=settings.sources,
        help=f"source catalog path (default: {settings.sources})",
    )
    parser.add_argument(
        "--lake",
        type=Path,
        default=settings.lake,
        help=f"bulk Parquet landing-zone directory (default: {settings.lake})",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="initialize the registry store")
    commands.add_parser("sources", help="list configured discovery sources")
    commands.add_parser("benchmarks", help="list configured external benchmarks")
    lake_status = commands.add_parser(
        "lake-status",
        help="show sealed bulk releases and physical Parquet row counts",
    )
    lake_status.add_argument(
        "--verify-shards",
        action="store_true",
        help="rehash and validate every referenced Parquet shard",
    )
    project = commands.add_parser(
        "project",
        help="materialize a sealed bulk corpus into source-normalized Parquet",
    )
    project.add_argument("--source", default="semantic-scholar")
    project.add_argument(
        "--release",
        help="exact sealed release (default: newest complete release)",
    )
    project.add_argument(
        "--base-artifact-id",
        help="exact prior projection artifact for an ambiguous diff lineage",
    )
    project.add_argument(
        "--target-artifact-id",
        help="reopen one exact already-materialized target artifact",
    )

    sync = commands.add_parser("sync", help="run resumable source discovery")
    sync.add_argument(
        "--source",
        action="append",
        default=[],
        help="source name; repeat or comma-separate (default: every enabled source)",
    )
    sync.add_argument(
        "--max-pages",
        type=_positive_int,
        default=1000,
        help="per-source page budget for this run (default: 1000)",
    )
    sync.add_argument("--fail-fast", action="store_true")
    sync.add_argument(
        "--frontier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enrich URLs found by the primary sources (default: enabled)",
    )
    sync.add_argument(
        "--frontier-limit", type=_positive_int, default=settings.frontier_limit
    )
    sync.add_argument(
        "--frontier-max-depth", type=_nonnegative_int, default=settings.frontier_max_depth
    )

    crawl = commands.add_parser("crawl", help="enrich pending paper/code/card links")
    crawl.add_argument("--limit", type=_positive_int, default=settings.frontier_limit)
    crawl.add_argument(
        "--max-depth", type=_nonnegative_int, default=settings.frontier_max_depth
    )

    paper_ingest = commands.add_parser(
        "ingest-paper",
        help="persist and plan one exact paper observation without materializing entries",
    )
    paper_ingest.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "one normalized paper JSON/JSONL object with an exact source identity; "
            "no network paper search is performed"
        ),
    )
    paper_ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the one-paper resource/entry plan without opening the store",
    )

    institutional = commands.add_parser(
        "ingest-institutional-text",
        help="ingest one manually obtained, institutionally authorized paper text",
    )
    institutional.add_argument("--doi", required=True, help="article DOI")
    institutional.add_argument("--title", required=True, help="article title")
    institutional.add_argument(
        "--text-file",
        type=Path,
        required=True,
        help="locally extracted UTF-8 text; PDFs and browser sessions are never read directly",
    )
    institutional.add_argument(
        "--landing-url",
        help="optional publisher landing page retained as a non-crawling locator",
    )
    institutional.add_argument(
        "--access-confirmed",
        action="store_true",
        help="affirm that your institutional license permits this particular local use",
    )

    resolve = commands.add_parser(
        "resolve-paper-text",
        help="report lawful full-text access routes for a paper DOI (OA, repository, publisher)",
    )
    resolve.add_argument("--doi", required=True, help="article DOI")
    resolve.add_argument(
        "--institutional-access",
        action="store_true",
        help="check Crossref links for institutional entitlements (e.g., UC Berkeley)",
    )

    text = commands.add_parser(
        "ingest-text",
        help="ingest locally extracted UTF-8 text (never PDFs, never remote fetches)",
    )
    text.add_argument("--doi", help="article DOI (auto-detected from text if omitted)")
    text.add_argument("--title", required=True, help="article title (metadata only)")
    text.add_argument(
        "--text-file",
        type=Path,
        required=True,
        help="locally extracted UTF-8 text file; PDFs are not parsed, and no remote fetch occurs",
    )
    text.add_argument(
        "--access-confirmed",
        action="store_true",
        help="affirm that your institutional/publisher license permits this particular local use",
    )

    bulk = commands.add_parser(
        "bulk-load",
        help="transfer verified shards selected by synced bulk control records",
    )
    bulk.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "bulk control source name; repeat or comma-separate "
            "(default: every enabled bulk source)"
        ),
    )
    bulk.add_argument(
        "--max-new-shards",
        type=_nonnegative_int,
        default=None,
        help=(
            "new-transfer budget per control source "
            "(default: 1; GH Archive: 48 closed hours; Software Heritage: 4 shards)"
        ),
    )

    daily = commands.add_parser(
        "daily",
        help="advance current feeds, bulk corpora, history, and link enrichment",
    )
    daily.add_argument(
        "--max-pages",
        type=_positive_int,
        default=1_000,
        help="per-source current-feed page budget (default: 1000)",
    )
    daily.add_argument(
        "--max-new-shards",
        type=_nonnegative_int,
        default=None,
        help=(
            "new-transfer budget per bulk control source "
            "(default: 1; GH Archive: 48 closed hours; Software Heritage: 4 shards)"
        ),
    )
    daily.add_argument(
        "--bootstrap-max-pages",
        type=_positive_int,
        default=DEFAULT_BOOTSTRAP_MAX_PAGES,
        help=(
            "per-source historical bootstrap page budget "
            f"(default: {DEFAULT_BOOTSTRAP_MAX_PAGES})"
        ),
    )
    daily.add_argument(
        "--frontier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enrich discovered paper/code/card links (default: enabled)",
    )
    daily.add_argument(
        "--frontier-limit",
        type=_positive_int,
        default=settings.frontier_limit,
    )
    daily.add_argument(
        "--frontier-max-depth",
        type=_nonnegative_int,
        default=settings.frontier_max_depth,
    )
    daily.add_argument(
        "--alphaxiv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enrich already-discovered arXiv IDs when ALPHAXIV_API_KEY is set "
            "(default: enabled)"
        ),
    )
    daily.add_argument(
        "--alphaxiv-limit",
        type=_positive_int,
        default=100,
        help="exact-ID alphaXiv enrichment budget (default: 100)",
    )
    daily.add_argument(
        "--artifact-relations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="publish evidence-backed paper/code/card/weights relations (default: enabled)",
    )

    backfill = commands.add_parser(
        "backfill", help="run a resumable historical source window"
    )
    backfill.add_argument("--source", default="openalex")
    backfill.add_argument("--from", dest="from_date", required=True, metavar="YYYY-MM-DD")
    backfill.add_argument("--to", dest="to_date", required=True, metavar="YYYY-MM-DD")
    backfill.add_argument(
        "--max-pages",
        type=_positive_int,
        default=DEFAULT_BACKFILL_MAX_PAGES,
        help=f"page budget for this run (default: {DEFAULT_BACKFILL_MAX_PAGES})",
    )
    backfill.add_argument(
        "--namespace",
        help="optional durable checkpoint namespace (normally derived from the window)",
    )

    bootstrap = commands.add_parser(
        "bootstrap",
        help="build complete history from source-declared boundaries",
    )
    bootstrap.add_argument("--source", default="arxiv")
    bootstrap.add_argument(
        "--max-pages",
        type=_positive_int,
        default=DEFAULT_BOOTSTRAP_MAX_PAGES,
        help=f"page budget for this run (default: {DEFAULT_BOOTSTRAP_MAX_PAGES})",
    )
    bootstrap.add_argument(
        "--namespace",
        help="optional durable checkpoint namespace",
    )

    alphaxiv = commands.add_parser(
        "enrich-alphaxiv",
        help="enrich exact arXiv IDs already discovered by primary sources",
    )
    alphaxiv.add_argument("--limit", type=_positive_int, default=100)
    alphaxiv.add_argument(
        "--refresh-after-days",
        type=_nonnegative_int,
        help="also revisit successful evidence at or beyond this age",
    )

    commands.add_parser(
        "link-artifacts",
        help="publish the current evidence-backed artifact relation projection",
    )

    entry_plan = commands.add_parser(
        "entry-plan",
        help="validate source-entry seeds and print no-side-effect resource actions",
    )
    entry_plan.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON/JSONL source-entry seed file; never reads or writes the registry store",
    )
    entry_build = commands.add_parser(
        "build-entry-corpus",
        help="materialize a reviewed source-entry seed stream only when explicitly invoked",
    )
    entry_build.add_argument("--input", type=Path, required=True, help="JSON/JSONL seed file")
    entry_build.add_argument(
        "--output",
        type=Path,
        help="new output directory (required unless --dry-run is supplied)",
    )
    entry_build.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and summarize the build without writing any entry output",
    )
    entry_seed_export = commands.add_parser(
        "export-entry-seeds",
        help="write current explicit model/technique assertions as a later-build seed stream",
    )
    entry_seed_export.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new JSONL seed file; reads the current registry but never changes it",
    )
    entry_seed_export.add_argument(
        "--source",
        action="append",
        default=[],
        help="source name; repeat or comma-separate (default: every current source)",
    )
    entry_seed_export.add_argument(
        "--relation-root",
        type=Path,
        help=(
            "root containing a sealed current artifact-relation projection; "
            "read only and never materialized by this command"
        ),
    )
    entry_seed_export.add_argument(
        "--link-current-resources",
        action="store_true",
        help=(
            "attach exact current URL/identifier evidence with a read-only targeted join; "
            "cannot be combined with --relation-root"
        ),
    )
    entry_readiness = commands.add_parser(
        "entry-readiness",
        help="audit current evidence prerequisites for a future entry-corpus build",
    )
    entry_readiness.add_argument(
        "--relation-root",
        type=Path,
        help=(
            "root containing a sealed current artifact-relation projection; "
            "default: STORE/projections/artifact-relations"
        ),
    )

    commands.add_parser("stats", help="show registry and backlog counts")
    commands.add_parser("status", help="show per-source cursor, coverage, and run status")

    export = commands.add_parser(
        "export-metadata",
        help="write a metadata-only, provenance-preserving public dataset bundle",
    )
    export.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new local directory for a Hub-uploadable Parquet bundle",
    )

    coverage = commands.add_parser(
        "coverage", help="measure recall against an external acceptance corpus"
    )
    coverage.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="UTF-8 CSV with Model and Bucket columns; never used for discovery",
    )

    benchmark = commands.add_parser(
        "benchmark", help="measure recall against a configured live benchmark"
    )
    benchmark.add_argument(
        "--name",
        help="benchmark name (default: the sole enabled benchmark)",
    )
    benchmark.add_argument(
        "--minimum-recall",
        type=_unit_interval,
        default=1.0,
        help="exit nonzero below this recall, from 0 through 1 (default: 1.0)",
    )

    search = commands.add_parser("search", help="search canonical model names and aliases")
    search.add_argument("query")
    search.add_argument("--limit", type=_positive_int, default=20)

    show = commands.add_parser("show", help="show one model and all supporting evidence")
    show.add_argument("model_id")

    dead = commands.add_parser("dead-letters", help="show quarantined records and fetches")
    dead.add_argument("--source")
    dead.add_argument("--limit", type=_positive_int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code = _dispatch(args)
    except (KeyError, ValueError, OSError, RuntimeError) as error:
        if args.json:
            _emit({"error": f"{type(error).__name__}: {error}"}, as_json=True)
        else:
            parser.error(str(error))
        exit_code = 2
    raise SystemExit(exit_code)


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "sources":
        configs = load_source_configs(args.sources_file)
        rows = [
            {
                "name": config["name"],
                "adapter": config["adapter"],
                "enabled": config.get("enabled", True),
                "schedule": config.get("schedule"),
                "url": config.get("url") or config.get("api_v2_url"),
            }
            for config in configs
        ]
        _emit(rows, as_json=args.json)
        return 0

    if args.command == "benchmarks":
        configs = load_benchmark_configs(args.sources_file)
        rows = [
            {
                "name": config["name"],
                "adapter": config["adapter"],
                "enabled": config.get("enabled", True),
                "schedule": config.get("schedule"),
                "url": config.get("url"),
            }
            for config in configs
        ]
        _emit(rows, as_json=args.json)
        return 0

    if args.command == "lake-status":
        lake = ParquetLandingZone(args.lake)
        releases = lake.list_releases(verify_shards=args.verify_shards)
        _emit(
            _lake_status(
                lake,
                releases,
                payload_verified=args.verify_shards,
            ),
            as_json=args.json,
        )
        return 0

    if args.command == "project":
        if args.source != "semantic-scholar":
            raise ValueError(f"projection is not implemented for {args.source!r}")
        outcome = run_semantic_scholar_projection(
            ParquetLandingZone(args.lake),
            release=args.release,
            base_artifact_id=args.base_artifact_id,
            target_artifact_id=args.target_artifact_id,
        )
        _emit(asdict(outcome), as_json=args.json)
        return 0

    # Entry planning/building deliberately stays outside the registry store.  It
    # can be used to prepare or execute a later identity-centered corpus build
    # without opening, downloading, or mutating the existing evidence corpus.
    if args.command == "entry-plan":
        seeds = read_entry_seeds(args.input)
        _emit([plan_entry_seed(seed) for seed in seeds], as_json=args.json)
        return 0

    if args.command == "build-entry-corpus":
        result = build_entries(read_entry_seeds(args.input))
        if args.dry_run:
            _emit({**result.manifest(), "dry_run": True}, as_json=args.json)
            return 0
        if args.output is None:
            raise ValueError("build-entry-corpus requires --output unless --dry-run is used")
        _emit(write_entry_bundle(result, args.output), as_json=args.json)
        return 0

    if args.command == "ingest-paper" and args.dry_run:
        prepared = prepare_paper_ingestion(_one_paper_seed(args.input))
        _emit(
            {
                "source": prepared.entry_seed["source"],
                "source_record_id": prepared.record.source_record_id,
                "canonical_url": prepared.record.canonical_url,
                "candidate_count": prepared.candidate_count,
                "direct_resource_count": prepared.direct_resource_count,
                "extractor": prepared.extractor_name,
                "entry_seed": prepared.entry_seed,
                "plan": prepared.plan,
                "dry_run": True,
                "materialized_entries": 0,
            },
            as_json=args.json,
        )
        return 0

    if args.command == "export-entry-seeds":
        configs = load_source_configs(args.sources_file)
        selected = _source_names(args.source)
        configured_names = {str(config["name"]) for config in configs}
        unknown = sorted(set(selected) - configured_names)
        if unknown:
            raise ValueError("unknown source: " + ", ".join(unknown))
        receipt = export_current_entry_seeds(
            args.store,
            args.output,
            source_tags=source_tags_from_configs(configs),
            sources=selected or None,
            relation_root=args.relation_root,
            link_current_resources=args.link_current_resources,
        )
        _emit(asdict(receipt), as_json=args.json)
        return 0

    if args.command == "entry-readiness":
        _emit(
            assess_entry_readiness(
                args.store,
                load_source_configs(args.sources_file),
                relation_root=args.relation_root,
            ),
            as_json=args.json,
        )
        return 0

    if args.command == "benchmark":
        head = Path(args.store).expanduser().resolve() / "HEAD.json"
        if not head.is_file():
            raise ValueError(
                f"registry store is not initialized: {args.store}; run 'modelome init' first"
            )

    store = ParquetStore(args.store)
    store.initialize()
    try:
        if args.command == "init":
            _emit(
                {"store": str(args.store), "initialized": True},
                as_json=args.json,
            )
            return 0
        if args.command == "sync":
            sources = load_sources(args.sources_file)
            selected = _source_names(args.source)
            outcomes = SyncEngine(store, sources).sync(
                selected or None,
                max_pages=args.max_pages,
                fail_fast=args.fail_fast,
            )
            payload: dict[str, Any] = {
                "sources": [asdict(outcome) for outcome in outcomes]
            }
            frontier_failed = False
            if args.frontier:
                frontier_outcome = FrontierCrawler(store).crawl(
                    limit=args.frontier_limit,
                    max_depth=args.frontier_max_depth,
                )
                payload["frontier"] = asdict(frontier_outcome)
                frontier_failed = bool(frontier_outcome.stats.get("errors"))
            _emit(payload, as_json=args.json)
            return (
                1
                if frontier_failed or any(item.status == "failed" for item in outcomes)
                else 0
            )
        if args.command == "crawl":
            outcome = FrontierCrawler(store).crawl(
                limit=args.limit,
                max_depth=args.max_depth,
            )
            _emit(asdict(outcome), as_json=args.json)
            return 1 if outcome.stats.get("errors") else 0
        if args.command == "ingest-paper":
            outcome = ingest_paper(store, _one_paper_seed(args.input))
            _emit(asdict(outcome), as_json=args.json)
            return 1 if outcome.stats.get("errors") else 0
        if args.command == "ingest-institutional-text":
            outcome = ingest_institutional_text(
                store,
                doi=args.doi,
                title=args.title,
                text_path=args.text_file,
                landing_url=args.landing_url,
                access_confirmed=args.access_confirmed,
            )
            _emit(asdict(outcome), as_json=args.json)
            return 0
        if args.command == "resolve-paper-text":
            from modelome.text_import import resolve_lawful_access_routes

            outcome = resolve_lawful_access_routes(args.doi)
            outcome["institutional_access_requested"] = args.institutional_access
            if args.institutional_access and outcome.get("open_access") is False:
                outcome["institutional_advice"] = (
                    "On UC Berkeley's network, open the landing URL in a browser; if the "
                    "institutional session entitles you to the PDF, download it, extract "
                    "text locally, then run `modelome ingest-text` to import the authorized copy"
                )
            _emit(outcome, as_json=args.json)
            return 0
        if args.command == "ingest-text":
            from modelome.text_import import inventory_text_artifacts, read_text_file

            text_content = read_text_file(args.text_file)
            if not args.access_confirmed:
                raise SystemExit(
                    "ingest-text requires --access-confirmed to confirm that "
                    "your institutional/publisher license permits this local use"
                )
            doi = args.doi or extract_doi(text_content) or "unknown"
            inventory = inventory_text_artifacts(text_content)
            outcome = ingest_locally_authorized_text(
                store,
                doi=doi,
                title=args.title,
                text=text_content,
                access_confirmed=True,
            )
            payload = asdict(outcome)
            payload["artifact_urls"] = inventory.artifact_urls
            payload["all_urls"] = inventory.urls
            _emit(payload, as_json=args.json)
            return 0
        if args.command == "bulk-load":
            sources = load_sources(args.sources_file)
            selected = select_bulk_sources(sources, _source_names(args.source))
            lake = ParquetLandingZone(args.lake)
            lake.initialize()
            outcomes = [
                run_bulk_source(
                    store,
                    lake,
                    source,
                    max_new_shards=args.max_new_shards,
                )
                for source in selected
            ]
            _emit(
                {
                    "lake": str(lake.root),
                    "sources": [asdict(outcome) for outcome in outcomes],
                },
                as_json=args.json,
            )
            return 1 if any(outcome.errors for outcome in outcomes) else 0
        if args.command == "daily":
            alphaxiv_enricher = _configured_alphaxiv_enricher(args.alphaxiv)
            outcome = run_daily(
                store,
                ParquetLandingZone(args.lake),
                load_sources(args.sources_file),
                max_pages=args.max_pages,
                max_new_shards=args.max_new_shards,
                bootstrap_max_pages=args.bootstrap_max_pages,
                frontier=args.frontier,
                frontier_limit=args.frontier_limit,
                frontier_max_depth=args.frontier_max_depth,
                alphaxiv_enricher=alphaxiv_enricher,
                alphaxiv_limit=args.alphaxiv_limit,
                artifact_relations=args.artifact_relations,
            )
            _emit(asdict(outcome), as_json=args.json)
            return 1 if outcome.status == "failed" else 0
        if args.command == "link-artifacts":
            outcome = run_artifact_relation_projection(store.root)
            _emit(asdict(outcome), as_json=args.json)
            return 0
        if args.command == "enrich-alphaxiv":
            outcome = run_alphaxiv_enrichment(
                store,
                limit=args.limit,
                refresh_after_days=args.refresh_after_days,
            )
            _emit(asdict(outcome), as_json=args.json)
            return 1 if outcome.status == "failed" else 0
        if args.command == "backfill":
            sources = load_sources(args.sources_file)
            source = sources.get(args.source)
            if source is None:
                raise KeyError(f"unknown or disabled source: {args.source}")
            if isinstance(source, ArxivSourceAdapter):
                outcome = run_arxiv_backfill(
                    store,
                    source,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            elif isinstance(source, CrossrefSourceAdapter):
                outcome = run_crossref_backfill(
                    store,
                    source,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            elif isinstance(source, EuropePmcSourceAdapter):
                outcome = run_europe_pmc_backfill(
                    store,
                    source,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            elif isinstance(source, DataCiteSourceAdapter):
                outcome = run_datacite_backfill(
                    store,
                    source,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            elif isinstance(source, OpenAlexSourceAdapter):
                outcome = run_openalex_backfill(
                    store,
                    source,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            elif isinstance(
                source,
                (BioRxivSourceAdapter, BioRxivPublicationSourceAdapter),
            ):
                outcome = run_biorxiv_backfill(
                    store,
                    source,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            elif isinstance(source, OsfPreprintSourceAdapter):
                outcome = run_osf_preprints_backfill(
                    store,
                    source,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            elif isinstance(source, HalSourceAdapter):
                outcome = run_hal_backfill(
                    store,
                    source,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            elif isinstance(source, EarthArxivSourceAdapter):
                outcome = run_eartharxiv_backfill(
                    store,
                    source,
                    from_date=args.from_date,
                    to_date=args.to_date,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            else:
                raise ValueError(
                    f"historical window backfill is not implemented for "
                    f"{type(source).__name__}"
                )
            _emit(asdict(outcome), as_json=args.json)
            return 1 if outcome.status == "failed" else 0
        if args.command == "bootstrap":
            sources = load_sources(args.sources_file)
            source = sources.get(args.source)
            if source is None:
                raise KeyError(f"unknown or disabled source: {args.source}")
            if isinstance(source, ArxivSourceAdapter):
                outcome = run_arxiv_bootstrap(
                    store,
                    source,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            elif isinstance(source, PmcSourceAdapter):
                outcome = run_pmc_bootstrap(
                    store,
                    source,
                    max_pages=args.max_pages,
                    namespace=args.namespace,
                )
            else:
                raise ValueError(
                    f"complete bootstrap is not implemented for "
                    f"{type(source).__name__}"
                )
            _emit(asdict(outcome), as_json=args.json)
            return 1 if outcome.status == "failed" else 0
        if args.command == "stats":
            _emit(store.stats(), as_json=args.json)
            return 0
        if args.command == "status":
            _emit(store.source_status(), as_json=args.json)
            return 0
        if args.command == "export-metadata":
            receipt = export_public_metadata(
                store,
                args.output,
                source_configs=load_source_configs(args.sources_file),
            )
            _emit(asdict(receipt), as_json=args.json)
            return 0
        if args.command == "coverage":
            report = evaluate_manifest(store, args.manifest)
            _emit(report, as_json=args.json)
            return 0 if report["missing"] == 0 else 1
        if args.command == "benchmark":
            config = _select_benchmark(
                load_benchmark_configs(args.sources_file),
                args.name,
            )
            report = evaluate_epoch_benchmark(store, config)
            report["minimum_recall"] = args.minimum_recall
            integrity_ok = report.get("integrity_ok", report.get("invalid_count", 0) == 0)
            report["integrity_ok"] = bool(integrity_ok)
            report["passed"] = report["integrity_ok"] and report["recall"] >= args.minimum_recall
            _emit_benchmark_report(report, as_json=args.json)
            return 0 if report["passed"] else 1
        if args.command == "search":
            _emit(store.search_models(args.query, args.limit), as_json=args.json)
            return 0
        if args.command == "show":
            detail = store.model_detail(args.model_id)
            if detail is None:
                raise KeyError(f"model not found: {args.model_id}")
            _emit(detail, as_json=args.json)
            return 0
        if args.command == "dead-letters":
            _emit(
                store.list_dead_letters(args.source, args.limit),
                as_json=args.json,
            )
            return 0
    finally:
        store.close()
    raise AssertionError(f"unhandled command: {args.command}")


def _configured_alphaxiv_enricher(enabled: bool) -> AlphaXivExactIdEnricher | None:
    if not enabled or not os.environ.get("ALPHAXIV_API_KEY", "").strip():
        return None
    return AlphaXivExactIdEnricher(AlphaXivMcpClient())


def _source_names(values: list[str]) -> list[str]:
    return list(
        dict.fromkeys(name.strip() for value in values for name in value.split(",") if name.strip())
    )


def _lake_status(
    lake: ParquetLandingZone,
    releases: tuple[ReleaseReceipt, ...],
    *,
    payload_verified: bool,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    for release in releases:
        key = (release.source, release.dataset)
        group = groups.setdefault(
            key,
            {
                "source": release.source,
                "dataset": release.dataset,
                "sealed_releases": 0,
                "sealed_shards": 0,
                "physical_rows": 0,
            },
        )
        group["sealed_releases"] += 1
        group["sealed_shards"] += release.shard_count
        group["physical_rows"] += release.row_count
        rows.append(
            {
                "source": release.source,
                "dataset": release.dataset,
                "release": release.release,
                "application_mode": release.application_mode,
                "shards": release.shard_count,
                "physical_rows": release.row_count,
                "seal": release.path,
            }
        )
    return {
        "lake": lake.root,
        "payload_verified": payload_verified,
        "count_semantics": (
            "physical snapshot/diff rows; not deduplicated papers or model entities"
        ),
        "sealed_releases": len(rows),
        "sealed_shards": sum(row["shards"] for row in rows),
        "physical_rows": sum(row["physical_rows"] for row in rows),
        "sources": sorted(groups.values(), key=lambda row: (row["source"], row["dataset"])),
        "releases": rows,
    }


def _one_paper_seed(path: Path) -> dict[str, Any]:
    """Read exactly one normalized paper observation for ``ingest-paper``."""

    seeds = read_entry_seeds(path)
    if len(seeds) != 1:
        raise ValueError("ingest-paper input must contain exactly one paper object")
    return seeds[0]


def _select_benchmark(
    configs: tuple[dict[str, Any], ...],
    requested_name: str | None,
) -> dict[str, Any]:
    enabled = [config for config in configs if config.get("enabled", True) is not False]
    if requested_name:
        selected = next(
            (config for config in enabled if config["name"] == requested_name),
            None,
        )
        if selected is None:
            raise KeyError(f"unknown or disabled benchmark: {requested_name}")
        return selected
    if not enabled:
        raise ValueError("no enabled benchmarks are configured")
    if len(enabled) > 1:
        names = ", ".join(str(config["name"]) for config in enabled)
        raise ValueError(f"multiple benchmarks are enabled ({names}); specify --name")
    return enabled[0]


def _emit_benchmark_report(report: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        _emit(report, as_json=True)
        return
    missing_names = [
        item["name"] for item in report["expectations"] if not item["found"]
    ]
    preview_limit = 25
    summary = {
        "benchmark": report["benchmark"],
        "retrieved_at": report["retrieval"]["retrieved_at"],
        "corpus_sha256": report["retrieval"]["sha256"],
        "raw_rows": report["raw_rows"],
        "unique_expectations": report["unique_expectations"],
        "duplicates": report["duplicate_count"],
        "invalid": report["invalid_count"],
        "integrity_ok": report["integrity_ok"],
        "found": report["found"],
        "missing": report["missing"],
        "recall": report["recall"],
        "minimum_recall": report["minimum_recall"],
        "passed": report["passed"],
        "missing_names_preview": missing_names[:preview_limit],
        "missing_names_omitted": max(0, len(missing_names) - preview_limit),
    }
    _emit(summary, as_json=False)


def _emit(value: Any, *, as_json: bool) -> None:
    value = _jsonable(value)
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if isinstance(value, list):
        if not value:
            print("No records.")
            return
        for item in value:
            print(_human_line(item))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                print(f"{key}: {json.dumps(item, ensure_ascii=False, sort_keys=True)}")
            else:
                print(f"{key}: {item}")
        return
    print(value)


def _human_line(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)
    preferred = ["name", "canonical_name", "source", "status", "adapter", "url", "id"]
    parts = [f"{key}={value[key]}" for key in preferred if value.get(key) is not None]
    if not parts:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "  ".join(parts)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


if __name__ == "__main__":
    main(sys.argv[1:])
