"""Build a versioned dataset from paper observations, adapters, or a local store."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from modelome import __version__
from modelome.entries import build_entries, read_entry_seeds, write_entry_bundle
from modelome.entry_seed_export import export_current_entry_seeds, source_tags_from_configs
from modelome.export import export_public_metadata
from modelome.paper_ingestion import ingest_paper
from modelome.pipeline import SyncEngine
from modelome.sources.base import SourceAdapter
from modelome.sources.catalog import load_source_configs
from modelome.storage import Database


@dataclass(frozen=True, slots=True)
class DatasetReceipt:
    output: str
    status: str
    source_commit: str
    entry_count: int
    seed_count: int
    papers_ingested: int


def build_dataset(
    store: str | Path,
    output: str | Path,
    *,
    papers: Iterable[Mapping[str, Any]] | None = None,
    sources: Mapping[str, SourceAdapter] | None = None,
    source_configs: Sequence[Mapping[str, Any]] | None = None,
    max_pages: int = 1,
) -> DatasetReceipt:
    """Ingest bounded work and atomically export the current cumulative dataset.

    Supply paper observations OR explicit source adapters; omit both to export
    an existing store offline. Source scans retain page checkpoints between
    calls. Paper observations are idempotent by source identity and content.
    A page budget yields a ``partial`` bundle; ingestion errors raise without
    publishing a bundle. Completed ingestion remains available for retry.

    Every bundle includes all current admitted entries in the store, reusable
    entry seeds, public metadata Parquet tables, and a manifest with checksums.
    No frontier crawl, full-text fetch, or bulk archive download is launched.
    Concurrent writers during export are detected and require an export retry.
    """

    root = Path(store).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"dataset output already exists: {destination}")
    if root == destination or root in destination.parents or destination in root.parents:
        raise ValueError("dataset output and registry store must not overlap")
    if papers is not None and sources is not None:
        raise ValueError("choose paper observations or source adapters, not both")
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    if sources is not None and not sources:
        raise ValueError("at least one source adapter is required")
    if papers is None and sources is None and not (root / "HEAD.json").is_file():
        raise ValueError(f"registry store is not initialized: {root}")
    configs = tuple(load_source_configs() if source_configs is None else source_configs)
    database = Database(root)
    staging: Path | None = None
    try:
        database.initialize()
        outcomes = []
        papers_ingested = 0
        if papers is not None:
            for paper in papers:
                result = ingest_paper(database, paper)
                if result.stats.get("errors"):
                    raise RuntimeError(f"paper ingestion failed: {result.source_record_id}")
                papers_ingested += 1
            if papers_ingested == 0:
                raise ValueError("paper worklist is empty")
        if sources is not None:
            outcomes = SyncEngine(database, sources).sync(max_pages=max_pages)
            failed = [
                outcome.source for outcome in outcomes
                if outcome.status == "failed" or outcome.stats.get("errors")
            ]
            if failed:
                raise RuntimeError("source ingestion failed: " + ", ".join(failed))

        head_before = (root / "HEAD.json").read_bytes()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        seeds = export_current_entry_seeds(
            root,
            staging / "seeds.jsonl",
            source_tags=source_tags_from_configs(configs),
            link_current_resources=True,
        )
        entries = build_entries(read_entry_seeds(Path(seeds.output)) if seeds.seed_count else ())
        write_entry_bundle(entries, staging / "entries")
        metadata = export_public_metadata(database, staging / "metadata", source_configs=configs)
        if (root / "HEAD.json").read_bytes() != head_before:
            raise RuntimeError(
                "registry changed during dataset export; retry with no active writers"
            )
        if metadata.source_commit != seeds.commit:
            raise RuntimeError("dataset exports refer to different registry commits; retry")

        # Keep receipts relocatable: staging paths cease to exist after publication.
        metadata_receipt = asdict(metadata)
        metadata_receipt["output"] = "metadata"
        _write_json(staging / "metadata" / "export-receipt.json", metadata_receipt)
        status = "partial" if any(item.status != "complete" for item in outcomes) else "complete"
        receipt = DatasetReceipt(
            output=str(destination),
            status=status,
            source_commit=seeds.commit,
            entry_count=len(entries.entries),
            seed_count=seeds.seed_count,
            papers_ingested=papers_ingested,
        )
        files = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files[path.relative_to(staging).as_posix()] = {
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
        _write_json(staging / "manifest.json", {
            "format": "modelome-dataset-v1",
            "modelome_version": __version__,
            "status": status,
            "mode": "papers" if papers is not None else "sources" if sources else "store",
            "source_commit": seeds.commit,
            "papers_ingested": papers_ingested,
            "entries": entries.manifest(),
            "sync": [asdict(item) for item in outcomes],
            "files": files,
        })
        if destination.exists():
            raise FileExistsError(f"dataset output already exists: {destination}")
        os.rename(staging, destination)
        staging = None
        return receipt
    finally:
        database.close()
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
