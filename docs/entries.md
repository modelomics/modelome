# Entry construction

MODELOME's eventual public surface is a list of **entries**, not a download of
papers. An entry is one evidence-backed technique, architecture, model, or release
family with every known related resource attached: paper, source code, checkpoint,
model card, data, evaluation, documentation, and any other source-declared link.

The entry builder is intentionally offline and opt-in. It does not read the current
registry store, crawl a source, retrieve a paper, or download a checkpoint. It is the
deterministic last stage that a future worker calls after processing one source record
at a time.

`export-entry-seeds` is the explicit bridge from the current registry to that offline
stage. It streams only current source records with an admitted model or technique claim
(whether source-declared or conservatively derived by an extractor), writes a new JSONL
seed file plus a hash manifest, and never modifies the registry. It does not make a
source catalog's ordinary papers into entry candidates.

A source-catalog row may carry optional `entry_tags = ["..."]`. The export copies those
strings onto that source's seeds exactly as written; they are navigation metadata, not
identity assertions or merge keys.

## Resource-scoped catalog rows

An `html_catalog` table rule can declare `row_link_rules`: URL-pattern rules for
resources exposed in the **same selected row** as a model. Each rule names the
resource relation (for example `paper`, `official_implementation`, `model_card`,
or `weights`), may restrict the link text, and can mark a binary reference as
non-crawlable. The adapter attaches a matching URL only to that row's exact
model-local ID and preserves the row-cell locator.

When a table's model label is followed by a citation anchor in the same cell, a
rule may use `name_source = "before_first_anchor"`. It retains only the
source-declared leading model text, rather than accidentally making the cited
authors and year part of the model identity.

This is intentionally different from a document-wide shared link. A repository,
paper, or checkpoint in one catalog row is never broadcast to other rows, and a
page-wide navigation link is not silently promoted to a model resource. A configured
row-link rule that finds no link in the entire selected catalog fails closed, exposing
source-layout drift instead of emitting a misleading resource-complete snapshot.

## Resumable model-detail resources

Some catalogs expose only one model-specific documentation link per index row, while
the page itself declares the paper, repository, model card, or artifact URLs. An
`html_catalog` source can opt into `detail_resource_rules` for that case. The catalog
page first freezes its exact model-page worklist into the source checkpoint. Later
source pages resolve a bounded `detail_batch_size` slice at a time, respecting normal
HTTP retry-after handling and retaining only URLs matched by the configured rules.

Every resulting detail record repeats the catalog's exact model identifier and scopes
the selected resources to that one model-local ID. This is deliberately not a generic
web-page inheritance rule: unmatched links—including site navigation—do not become
entry resources, and a detail page without a matching paper or repository remains a
valid, documented model rather than a fabricated incomplete record. A source sync is
authoritative only after its final detail batch succeeds, so a transient detail-page
failure cannot tombstone the preceding catalog snapshot.

## Contract

`src/modelome/entries.py` accepts a JSON or JSONL stream of **entry seeds**. One seed
is a lossless source observation with a source/record identity, a canonical artifact
URL, declared identifiers, model/technique assertions, resource links, and arbitrary
tags. A source adapter can produce that exact shape with
`source_record_to_entry_seed(record, source=..., tags=...)`.

Seeds with no declared model/technique assertion intentionally produce no entry. This
prevents a full bibliographic source from treating every paper as a model. Such a
paper remains available as resource evidence; a resolver or curator can later add one
explicit technique assertion when the evidence supports it.

An entry keeps:

- every source observation that asserted it;
- all exact external identifiers;
- every declared resource URL, relation, locator, source, record ID, and local-model
  assertion that supplied it;
- source-declared release/checkpoint identifiers, versions, revisions, metadata,
  confidence, and locators, including artifacts that have no downloadable URL;
- source-declared model lineage such as `base_model`, preserving the target's name,
  identifiers, confidence, and locator. A target receives an entry ID only when its
  exact model identifier (or its local ID in the same source record) resolves; lineage
  never merges two entries and is never inferred from a shared name;
- loose string tags such as `field:computer-vision`, `department:physics`,
  `domain:protein-design`, or an unnamespaced project tag. There is no controlled
  vocabulary or tag-based merge behavior.

Only a shared **exact namespaced identifier** can combine two candidates. Names,
aliases, title similarity, tags, organization names, repository hosts, and broad
project URLs never combine entries. If the evidence is insufficient, the builder
leaves two entries instead of guessing. A later source can contribute an exact bridge
identifier and a complete rerun will merge them deterministically.

Links retain their original relation. Checkpoint links are represented as reference-only
resources and are never planned for binary download. Other crawlable resources become
explicit `resolve_resource` actions, so an operator can use the existing bounded
resource crawler one URL at a time rather than treating the entry build itself as a
fetch job.

## Review and future execution

To prepare a later full build from an existing registry snapshot, explicitly export a
seed stream. This read-only step refuses to overwrite a prior export:

```bash
uv run modelome --json export-entry-seeds --output data/entry-seeds.jsonl
```

URLs found in the current text of a candidate record are direct, first-hop resource
evidence and are included in every export (unless they duplicate a source-declared
link). For an entry to include resources discovered in *other* records—such as a
generic paper that points at a separately ingested model card—use the recommended
cross-record path: a targeted, read-only join over current exact URL and
artifact-identifier evidence. It does not create a global graph, write to the registry,
use fuzzy matching, or download anything. It adds incoming URL mentions and exact
same-artifact bridges from other current records.

```bash
uv run modelome --store data/store --json export-entry-seeds \
  --link-current-resources \
  --output data/entry-seeds.jsonl
```

If a current sealed artifact-relation projection already exists, `--relation-root` is an
optional alternative. The exporter verifies that it exactly matches the current immutable
registry commit and refuses stale output; it never builds or updates that graph.

Before a full future run, inspect its actual prerequisites without creating an export:

```bash
uv run modelome --store data/store --json entry-readiness
```

The report distinguishes “the current evidence can be materialized into entries” from
“every configured source has been observed and completed.” Neither result claims that
private, deleted, or otherwise undiscoverable models have been covered.

Create or receive a seed file, then review the actions without touching the registry:

```bash
uv run modelome --json entry-plan --input data/entry-seeds.jsonl
uv run modelome --json build-entry-corpus --input data/entry-seeds.jsonl --dry-run
```

Only an explicit non-dry run writes an entry bundle, and it refuses to replace a
destination:

```bash
uv run modelome --json build-entry-corpus \
  --input data/entry-seeds.jsonl \
  --output data/entry-corpus-2026-09
```

The bundle contains `entries.jsonl` and a content-hashed `manifest.json`. It is a
portable interface for a future incremental worker and eventual Hub publication;
neither command downloads full text, scans the existing Parquet registry, or mutates
the registry store.

## One-record worker shape

The intended future execution loop is deliberately modest:

```text
source record or paper with an explicit model/technique assertion
  -> emit one entry seed (or export current assertions to a seed stream)
  -> attach every declared link with provenance
  -> attach every source-declared model-to-model relation without using it as identity
  -> attach targeted exact URL/identifier evidence from other current records
     (or optionally use an already-sealed relation projection)
  -> plan only allowed resource-resolution actions
  -> resolve those resources through bounded evidence ingestion when desired
  -> re-export the resulting explicit assertions and links
  -> periodically build or compact the finite seed stream
```

This permits a complete long-running crawl without a prerequisite corpus download:
each paper/catalog record advances independently, all resource links survive partial
failures, and the final entry build is reproducible from the seed stream.
