# Dataset builds

`modelome.dataset.build_dataset` composes source ingestion, admitted-entry seed
export, deterministic entry construction, and public metadata export. The
`modelome build-dataset` command calls this same library function.

## Inputs

Choose one CLI mode:

| Mode | Behavior |
| --- | --- |
| `--input papers.jsonl` | Ingest normalized paper observations one at a time, then export. |
| `--source NAME` | Sync the explicitly selected adapters, then export. Repeat or comma-separate names. |
| `--from-store` | Export an initialized local store without fetching upstream resources. |

All three modes export the **cumulative current store**, including earlier runs
and other sources. Use a separate store for an isolated dataset. `--max-pages`
defaults to one page per selected source; it does not limit offline export or
the number of supplied paper observations.

Global options such as `--store`, `--sources-file`, and `--json` precede the
subcommand. `--output` must name a new directory outside the store.

A minimal paper observation has this shape:

```json
{
  "source": "example",
  "source_record_id": "paper-1",
  "kind": "paper",
  "canonical_url": "https://example.org/papers/1",
  "title": "Example technique",
  "models": [{"local_id": "technique", "name": "Example technique"}],
  "links": [{"url": "https://example.org/code/1", "relation": "implementation"}]
}
```

This is a synthetic example. Supply source-backed model declarations and resource
links for real work. The input accepts a JSON object, array, `{"seeds": [...]}`
wrapper, or JSONL. Optional identifiers, timestamps, text, and model metadata
follow the [entry contract](entries.md). A paper without an admitted model or
technique assertion can remain evidence without creating an entry.

## Python API

```python
from modelome.dataset import build_dataset
from modelome.entries import read_entry_seeds
from modelome.sources.catalog import load_source_configs, load_sources
from pathlib import Path

# Process a local worklist.
receipt = build_dataset(
    "data/store", "data/exports/papers-v1",
    papers=read_entry_seeds(Path("papers.jsonl")),
)

# Advance one adapter using the same page checkpoints as the CLI.
adapters = load_sources()
receipt = build_dataset(
    "data/store", "data/exports/sources-v1",
    sources={"huggingface": adapters["huggingface"]},
    source_configs=load_source_configs(),
    max_pages=1,
)

# Export the current store offline.
receipt = build_dataset("data/store", "data/exports/snapshot-v1")
```

`papers` accepts an iterable of observations. `sources` accepts a mapping of
names to adapters implementing `fetch_page(state)`. These inputs are mutually
exclusive. `source_configs` supplies catalog metadata and entry tags; it defaults
to the bundled catalog or `MODELOME_SOURCES` override. The returned
`DatasetReceipt` contains the output path, status, immutable source commit,
entry and seed counts, and number of papers ingested in this invocation.

## Output and recovery

```text
dataset/
  manifest.json
  seeds.jsonl
  seeds.jsonl.manifest.json
  entries/
    entries.jsonl
    manifest.json
  metadata/
    *.parquet
    source-manifest.json
    export-receipt.json
    README.md
```

The root manifest binds every child file to its SHA-256 digest and byte count.
Entries and metadata must reference the same immutable store commit. If the
store changes during export, the build aborts publication; retry with no active
writers. Export receipts use relative paths so the bundle can be moved.

Exit code 0 means the requested work and export completed. Exit code 1 means a
source scan reached its page budget and the published bundle is partial. Exit
code 2 reports invalid arguments or a failed build. A `complete` offline export
can contain incomplete source inventories; consult `metadata/source-manifest.json`
for source coverage. None of these statuses certifies universal completeness.

Paper ingestion is idempotent by source identity and observed content. If a
worklist fails, fix the input and rerun it against the same store. Source scans
resume at committed page checkpoints. A failed export leaves these durable
ingestion results available and removes temporary bundle files. Existing output
directories are never intentionally replaced.

## Scope and scale

The builder stores resource references and direct evidence. It does not resolve
arbitrary paper IDs, search by title, run bulk archive ingestion, fetch linked
full text, crawl the frontier, or download weights. Those operations remain
separate source-specific capabilities. Entry seeds omit raw source payloads and
retained text; public metadata also excludes abstracts and model-card bodies.

The current store and entry deduplication operate in memory, and JSON/JSONL
worklist reading loads the input file. This first library workflow is intended
for bounded batches and current-store exports; full-corpus builds still require
partitioning and a streaming identity/materialization path. Use the existing
bulk landing-zone and projection commands independently when that work is needed.
